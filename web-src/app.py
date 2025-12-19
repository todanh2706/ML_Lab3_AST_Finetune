import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
import io
import os
import timm
import matplotlib.pyplot as plt
import librosa.display

# --- IMPORT MODEL TỪ FILE LOCAL ---
from ast_models import ASTModel 

# 1. CẤU HÌNH & MODEL VERSIONS
st.set_page_config(page_title="AST - UrbanSound8K", layout="wide")

# Định nghĩa các phiên bản Model tại đây
MODEL_VERSIONS = {
    "AST-P": {
        "path": "./models/AST-P",
        "model_size": "base384", # Cần khớp với lúc train (base384/base224)
        "description": "Model AST-P (Pretrained ImageNet + AudioSet)"
    },
    "AST-S": {
        "path": "./models/AST-S",
        "model_size": "base384", # <--- QUAN TRỌNG: Nếu AST-S là kiến trúc nhỏ hơn, hãy sửa thành 'base224' hoặc config tương ứng
        "description": "Pretrained ImageNet"
    }
}

LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark",
    "Drilling", "Engine Idling", "Gun Shot", "Jackhammer",
    "Siren", "Street Music"
]

def list_available_models(folder_path):
    """Return sorted relative paths of .pth files under the given folder."""
    if not os.path.isdir(folder_path):
        return []
    model_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pth"):
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                model_files.append(rel_path)
    return sorted(model_files)

# 2. HÀM XỬ LÝ ÂM THANH
def make_features(waveform, sr, target_length=1024, mel_bins=128):
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_target_samples = int(10 * sr)
    if waveform.shape[1] < num_target_samples:
        waveform = nn.functional.pad(waveform, (0, num_target_samples - waveform.shape[1]))
    else:
        waveform = waveform[:, :num_target_samples]

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
    )
    
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:target_length, :]

    spectrogram_vis = fbank.t().cpu().numpy() 
    fbank = (fbank - (-4.2677393)) / (2 * 4.5689974)
    
    return fbank.unsqueeze(0), spectrogram_vis

# 3. HÀM LOAD MODEL ĐỘNG (Dựa theo model_size)
# Không cache cứng model nữa mà cache theo tham số size để linh hoạt
@st.cache_resource
def load_architecture_skeleton(model_size='base384'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo khung model dựa trên size được chọn
    model = ASTModel(
        label_dim=527, 
        input_tdim=1024, 
        input_fdim=128, 
        model_size=model_size, 
        audioset_pretrain=False
    )
    
    # Thay Head (Giữ nguyên cấu trúc Sequential: Norm -> Linear)
    # mlp_head là Sequential(LayerNorm, Linear)
    # Chúng ta chỉ thay thế lớp Linear cuối cùng (index 1)
    in_features = model.mlp_head[1].in_features
    model.mlp_head[1] = nn.Linear(in_features, 10)
    
    model.to(device)
    model.eval()
    return model, device

# 4. HÀM PREDICT (Nhận cấu hình version)
def ensemble_predict(audio_input, version_config, selected_models):
    if not selected_models:
        st.error("Vui long chon it nhat mot model truoc khi chay.")
        return None, 0, np.zeros(len(LABELS)), None, {
            "vote_mode": False,
            "model_count": 0,
            "majority_count": 0,
            "used_models": [],
            "vote_counts": None,
            "per_model_results": []
        }
    # Lấy thông tin từ config
    folder_path = version_config["path"]
    model_size = version_config["model_size"]
    
    # Load kiến trúc phù hợp (Base hoặc Small)
    base_model, device = load_architecture_skeleton(model_size)
    
    # Xử lý audio
    if isinstance(audio_input, bytes):
        file_obj = io.BytesIO(audio_input)
    else:
        file_obj = audio_input
        file_obj.seek(0)
        
    try:
        # Sử dụng soundfile để đọc file thay vì torchaudio.load
        audio_data, sr = sf.read(file_obj)
        
        # Chuyển đổi sang Tensor của PyTorch
        # soundfile trả về (samples, channels) hoặc (samples,)
        # torchaudio mong đợi (channels, samples)
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # (1, samples)
        else:
            waveform = waveform.t() # (channels, samples)
            
    except Exception as e:
        st.error(f"Lỗi đọc file âm thanh: {str(e)}")
        return None, 0, np.zeros(len(LABELS)), None, {
            "vote_mode": False,
            "model_count": 0,
            "majority_count": 0,
            "used_models": [],
            "vote_counts": None,
            "per_model_results": []
        }
    input_tensor, spec_vis = make_features(waveform, sr)
    input_tensor = input_tensor.to(device)
    
    sum_probs = torch.zeros(1, len(LABELS)).to(device)
    
    # Tạo progress bar riêng cho từng lần chạy
    progress_text = st.empty()
    bar = st.progress(0)
    
    total_models = len(selected_models)
    used_models = []
    votes = []
    vote_counts = None
    majority_count = 0
    per_model_results = []
    
    for idx, model_name in enumerate(selected_models, start=1):
        pth_path = os.path.join(folder_path, model_name)
        
        if not os.path.exists(pth_path): 
            st.warning(f"Không tìm thấy file: {pth_path}")
            bar.progress(min(int(idx / total_models * 100), 100))
            continue
            
        progress_text.text(f"Đang chạy {model_name} ({idx}/{total_models}) từ {folder_path}...")
        bar.progress(min(int(idx / total_models * 100), 100))
        
        try:
            state_dict = torch.load(pth_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
                
            base_model.load_state_dict(new_state_dict, strict=True)
            
            with torch.no_grad():
                output = base_model(input_tensor)
                probs = torch.softmax(output, dim=-1)
                sum_probs += probs
                pred_idx = probs.argmax(-1).item()
                pred_prob = probs[0, pred_idx].item()
                votes.append(pred_idx)
                used_models.append(model_name)
                per_model_results.append({
                    "model": model_name,
                    "label": LABELS[pred_idx],
                    "prob": pred_prob
                })
                
        except Exception as e:
            st.warning(f"Lỗi tải {pth_path}: {e}")

    bar.empty()
    progress_text.empty()
    
    valid_models_count = len(used_models)
    
    if valid_models_count == 0:
        st.error(f"Không tìm thấy file model nào đủ điều kiện chạy trong thư mục: {folder_path}")
        return None, 0, np.zeros(len(LABELS)), spec_vis, {
            "vote_mode": False,
            "model_count": 0,
            "majority_count": 0,
            "used_models": [],
            "vote_counts": None,
            "per_model_results": []
        }

    avg_probs = sum_probs / valid_models_count # Chia cho số model thực tế tìm thấy
    vote_mode = valid_models_count > 1
    
    if vote_mode:
        vote_counts = np.bincount(votes, minlength=len(LABELS))
        majority_count = int(vote_counts.max())
        top_indices = np.flatnonzero(vote_counts == majority_count)
        
        if len(top_indices) == 1:
            final_idx = int(top_indices[0])
        else:
            top_idx_list = top_indices.tolist()
            candidate_probs = avg_probs[0, top_idx_list]
            final_idx = int(top_idx_list[int(candidate_probs.argmax().item())])
            
        conf_value = majority_count / valid_models_count
    else:
        final_idx = avg_probs.argmax(-1).item()
        conf_value = avg_probs[0][final_idx].item()
        majority_count = 1
    
    return LABELS[final_idx], conf_value, avg_probs[0].cpu().numpy(), spec_vis, {
        "vote_mode": vote_mode,
        "model_count": valid_models_count,
        "majority_count": majority_count,
        "used_models": used_models,
        "vote_counts": vote_counts.tolist() if vote_counts is not None else None,
        "per_model_results": per_model_results
    }

# --- GIAO DIỆN CHÍNH ---
st.title("UrbanSound8K Analysis System")
st.markdown("---")

# 1. Sidebar chọn Model
st.sidebar.header("Cấu hình Model")
selected_version_name = st.sidebar.selectbox(
    "Chọn phiên bản Model:",
    list(MODEL_VERSIONS.keys())
)
current_config = MODEL_VERSIONS[selected_version_name]

st.sidebar.info(f"**Mô tả:** {current_config['description']}")
st.sidebar.warning(f"**Đường dẫn:** `{current_config['path']}`")

available_models = list_available_models(current_config["path"])
st.sidebar.subheader("Chọn model sử dụng")
selected_models = []
if not available_models:
    st.sidebar.error(f"Không tìm thấy file .pth trong thư mục: {current_config['path']}")
else:
    for model_name in available_models:
        safe_key = model_name.replace(os.sep, "__")
        checkbox_key = f"{selected_version_name}_{safe_key}"
        display_name = model_name.replace(os.sep, "/")
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = True
        if st.sidebar.checkbox(display_name, key=checkbox_key):
            selected_models.append(model_name)
    if not selected_models:
        st.sidebar.warning("Chọn ít nhất một model để chạy.")

# 2. Input Audio
col1, col2 = st.columns([1, 2])

# Initialize session state to track the last used input source
if "last_input_source" not in st.session_state:
    st.session_state.last_input_source = None

# Callback functions to track which input was just used
def on_upload_change():
    st.session_state.last_input_source = "upload"

def on_mic_change():
    st.session_state.last_input_source = "mic"

with col1:
    st.subheader("1. Nhập liệu")
    tab_up, tab_mic = st.tabs(["Upload File", "Ghi âm"])
    input_data = None

    with tab_up:
        up_file = st.file_uploader("Chọn file .wav", type=["wav"], key="uploader", on_change=on_upload_change)
        
    with tab_mic:
        mic_file = st.audio_input("Bắt đầu ghi âm", key="mic", on_change=on_mic_change)

    # Determine which input to use based on last interaction
    if st.session_state.last_input_source == "upload" and up_file:
        input_data = up_file.read()
        up_file.seek(0)
    elif st.session_state.last_input_source == "mic" and mic_file:
        input_data = mic_file
    elif up_file and not mic_file:
        input_data = up_file.read()
        up_file.seek(0)
    elif mic_file and not up_file:
        input_data = mic_file

with col2:
    st.subheader("2. Kết quả phân tích")
    if input_data:
        # Nút bấm phân tích
        if st.button("Chạy mô hình " + selected_version_name, type="primary"):
            if not selected_models:
                st.warning("Hãy chọn ít nhất một model ở phần cấu hình trước khi chạy.")
            else:
                with st.spinner(f"Đang xử lý với {selected_version_name}..."):
                    lbl, conf, probs, spec_img, vote_info = ensemble_predict(input_data, current_config, selected_models)
                    
                    if lbl:
                        label_idx = LABELS.index(lbl)
                        # Hàng 1: Kết quả text + Audio player
                        r1_c1, r1_c2 = st.columns(2)
                        with r1_c1:
                            st.success(f"Dự đoán: **{lbl}**")
                            metric_label = "Tỷ lệ đồng thuận" if vote_info.get("vote_mode") else "Độ tin cậy"
                            st.metric(metric_label, f"{conf:.1%}")
                        if len(selected_models) > 1 and vote_info.get("per_model_results"):
                            st.markdown("**Kết quả từng model:**")
                            for item in vote_info["per_model_results"]:
                                display_name = item["model"].replace(os.sep, "/")
                                st.markdown(f"- `{display_name}`: {item['label']} ({item['prob']:.1%})")
                        if vote_info.get("vote_mode"):
                            st.caption(f"Council voting: {vote_info.get('majority_count', 0)}/{vote_info.get('model_count', 0)} model đồng ý.")
                        with r1_c2:
                            st.audio(input_data, format='audio/wav')

                        # Hàng 2: Spectrogram + Biểu đồ cột
                        st.write("---")
                        sub_c1, sub_c2 = st.columns([3, 2])
                        
                        with sub_c1:
                            st.write("**Mel Spectrogram**")
                            if spec_img is not None:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                # hop_length = sr * frame_shift_ms / 1000 = 16000 * 10 / 1000 = 160
                                img = librosa.display.specshow(spec_img, sr=16000, hop_length=160, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                                ax.set_xlabel("Thời gian (s)")
                                ax.set_ylabel("Tần số Mel")
                                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        with sub_c2:
                            st.write("**Xác suất các lớp**")
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            colors = ['#4CAF50' if i == label_idx else '#2196F3' for i in range(len(LABELS))]
                            ax2.barh(LABELS, probs, color=colors)
                            ax2.set_xlim(0, 1)
                            ax2.set_xlabel("Xác suất")
                            for i, v in enumerate(probs):
                                ax2.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig2)
                            plt.close(fig2)
    else:
        st.info("Vui lòng tải file âm thanh hoặc ghi âm để phân tích.")
