# Tên file: ast_models.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple, trunc_normal_

# Override the timm package to allow using it with newer pytorch versions
# (Fix lỗi tương thích giữa timm 0.4.5 và torch mới)
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    def __init__(self, label_dim=527, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=True):
        super(ASTModel, self).__init__()
        
        # Override timm's PatchEmbed if needed inside the vit model construction
        # But here we just use the standard vit_deit_base_distilled_patch16_384 from timm
        
        if verbose:
            print('---------------AST Model Summary---------------')
            print('ImageNet Pretraining: {:s}, AudioSet Pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))

        # input_fdim: mel bins (e.g. 128)
        # input_tdim: time frames (e.g. 1024)
        
        # Config cho base384 (giống trong notebook của bạn)
        if model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            self.original_num_patches = 576
            
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            self.original_num_patches = 196
        else:
            raise Exception('Model size must be base384 or base224')

        # Tự động điều chỉnh Positional Embedding cho input âm thanh chữ nhật (thay vì vuông như ảnh)
        f_dim, t_dim = self.get_shape(input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        # --- FIX: Update img_size validation for timm ---
        if hasattr(self.v.patch_embed, 'img_size'):
            self.v.patch_embed.img_size = (input_fdim, input_tdim)

        # --- FIX: Adjust Patch Embedding for usage with Audio Checkpoint ---
        # 1. Channel fix: 3 (RGB) -> 1 (Spectrogram)
        # 2. Stride fix: Default (16, 16) -> (10, 10) (AST Paper/Checkpoint)
        new_stride = (10, 10)
        
        # Luôn tạo lại Conv2d để đảm bảo stride đúng là (10, 10) và in_channels=1
        # Checkpoint của bạn train với fstride=10, tstride=10 -> Stride=(10, 10)
        new_proj = nn.Conv2d(1, self.v.patch_embed.proj.out_channels, 
                             kernel_size=self.v.patch_embed.proj.kernel_size, 
                             stride=new_stride)
        self.v.patch_embed.proj = new_proj
        
        if verbose:
            print('f_dim: {:d}, t_dim: {:d}, num_patches: {:d}'.format(f_dim, t_dim, num_patches))

        # Re-initialize positional embedding
        new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.v.pos_embed.shape[2]))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

        # Định nghĩa Head (Lớp cuối cùng)
        # AST gốc dùng MLP Head: Norm -> Linear
        self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

        # Tự động tải Pretrain AudioSet nếu cần (Web demo thì thường set False vì ta load từ file .pth rồi)
        if audioset_pretrain == True:
            self.load_audioset_pretrain(model_size)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        # Stride phải khớp với checkpoint (10, 10)
        test_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # Hàm load weights thông minh cho Positional Embedding
    def load_audioset_pretrain(self, model_size):
        # (Lược bỏ code download để gọn nhẹ cho Web App vì bạn không dùng tính năng này trực tiếp)
        pass

    def forward(self, x):
        # x shape: [Batch, Time, Freq] -> cần chuyển về [Batch, 1, Freq, Time] cho Conv2d
        # Lưu ý: Code notebook của bạn output từ make_features là (Batch, Time, Freq)
        # Conv2d của PatchEmbed cần (Batch, Channel, Height, Width)
        
        # AST Model trong repo gốc mong đợi input là (Batch, Time, Freq)
        # Sau đó nó tự unsqueeze và transpose bên trong? 
        # KHÔNG, timm Vision Transformer mong đợi (Batch, 3, 224, 224).
        # Nhưng AST đã override PatchEmbed.
        
        # Để an toàn nhất với input từ hàm make_features của bạn (Batch, 1024, 128):
        # Ta cần thêm dimension channel = 1 -> (Batch, 1, 1024, 128)
        # Nhưng đợi đã, AST gốc input chuẩn là (Batch, 1024, 128).
        
        x = x.unsqueeze(1) # (Batch, 1, 1024, 128)
        x = x.transpose(2, 3) # (Batch, 1, 128, 1024) -> (Batch, 1, Freq, Time)
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Thêm cls_token và dist_token
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        
        # Lấy output (trung bình cộng của cls_token và dist_token)
        x = (x[:, 0] + x[:, 1]) / 2
        
        x = self.mlp_head(x)
        return x