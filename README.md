# ML_Lab3_AST_Finetune

# Quy trình Fine-tuning AST Model với UrbanSound8K trên Google Colab

Tài liệu này hướng dẫn chi tiết các bước chuẩn bị dữ liệu, xây dựng pipeline và thực hiện tinh chỉnh (fine-tuning) mô hình Audio Spectrogram Transformer (AST) pre-trained trên AudioSet cho tác vụ phân loại âm thanh môi trường (UrbanSound8K).

## 1. Chuẩn bị môi trường và Dữ liệu

### 1.1. Thiết lập Google Colab

Sử dụng Google Colab để tận dụng GPU miễn phí (NVIDIA T4).

-   Runtime: Chọn "Change runtime type" -> Chọn "T4 GPU".
-   Mount Google Drive: Để lưu trữ dữ liệu bền vững và lưu checkpoint của model.

### 1.2. Cài đặt thư viện cần thiết

Cài đặt các thư viện Python chuyên dụng cho xử lý âm thanh và Deep Learning.

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install datasets
pip install librosa
pip install pandas
```

### 1.3. Tải và Cấu trúc dữ liệu

Bộ dữ liệu UrbanSound8K (US8K) có kích thước khoảng 6GB.

1.  Tải bộ dữ liệu từ nguồn chính thức hoặc Kaggle về Google Drive.
2.  Giải nén vào thư mục làm việc trên Drive (ví dụ: `/content/drive/MyDrive/US8K/`).
3.  Đảm bảo cấu trúc thư mục như sau:
    -   `UrbanSound8K/audio/`: Chứa các thư mục con `fold1` đến `fold10`.
    -   `UrbanSound8K/metadata/UrbanSound8K.csv`: File chứa nhãn và thông tin file.

## 2\. Xử lý dữ liệu (Data Preprocessing)

Đây là bước quan trọng nhất để chuyển đổi âm thanh thô thành định dạng Spectrogram mà AST có thể hiểu được.

### 2.1. Cấu hình tham số âm thanh

Thiết lập các tham số cố định để khớp với kiến trúc AST pre-trained.

-   Tần số lấy mẫu (Sampling Rate): 16,000 Hz (16kHz).
-   Thời lượng (Duration): 10 giây.
-   Số lượng Mel bins: 128.
-   Số lượng Time frames: 1024 (tương ứng 10s với window size 25ms, hop size 10ms).
-   Mean: 0 (Chuẩn hóa).
-   Std: 0.5 (Chuẩn hóa).

### 2.2. Xây dựng Class Dataset

Tạo một class kế thừa từ `torch.utils.data.Dataset` để xử lý từng file âm thanh on-the-fly (trong lúc train mới xử lý để tiết kiệm RAM).
Các bước xử lý trong hàm `__getitem__`:

1.  **Load Audio**: Dùng `torchaudio.load()`.
2.  **Resample**: Chuyển đổi về 16kHz.
3.  **Mix-to-Mono**: Nếu file là Stereo (2 kênh), chuyển về Mono (1 kênh) bằng cách lấy trung bình.
4.  **Padding/Truncation**:
    -   Nếu file ngắn hơn 10s: Thêm padding (số 0) vào cuối cho đủ độ dài.
    -   Nếu file dài hơn 10s: Cắt bỏ phần thừa.
5.  **Feature Extraction**: Chuyển đổi sóng âm thành Log Mel Spectrogram (fbank).
6.  **Normalization**: Áp dụng công thức `(input - mean) / (2 * std)` để đưa dữ liệu về khoảng giá trị mong muốn.

## 3\. Chuẩn bị Mô hình (Model Setup)

Sử dụng thư viện `transformers` của Hugging Face để tải mô hình AST một cách dễ dàng và chuẩn xác.

### 3.1. Tải Pre-trained Weights

Sử dụng checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`. Đây là mô hình đã học rất tốt trên AudioSet.

### 3.2. Thay đổi lớp phân loại (Classifier Head)

Mô hình gốc có đầu ra là 527 lớp (AudioSet). Cần thay thế bằng lớp Linear mới phù hợp với US8K.

-   **Input Features**: 768 (Kích thước vector đặc trưng của AST).
-   **Output Features**: 10 (Số lớp của UrbanSound8K).
-   **Khởi tạo**: Trọng số của lớp mới này sẽ được khởi tạo ngẫu nhiên.

<!-- end list -->

```python
from transformers import ASTForAudioClassification

# Tải model, bỏ qua sự không khớp kích thước ở lớp cuối
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=10,
    ignore_mismatched_sizes=True
)
```

## 4\. Chiến thuật Huấn luyện (Training Strategy)

### 4.1. Thiết lập 10-Fold Cross-Validation

UrbanSound8K chia sẵn 10 fold. Tuyệt đối không xáo trộn (shuffle) file giữa các fold để tránh rò rỉ dữ liệu (data leakage).

-   **Vòng lặp**: Chạy 10 lần thực nghiệm.
-   **Trong mỗi vòng**: Chọn 1 fold làm Test Set (ví dụ Fold 10), 9 fold còn lại làm Training Set.

### 4.2. Siêu tham số (Hyperparameters)

-   **Loss Function**: CrossEntropyLoss (cho bài toán phân loại đơn nhãn).
-   **Optimizer**: AdamW.
-   **Learning Rate**: Sử dụng LR nhỏ (ví dụ: 1e-5 hoặc 5e-5) để không phá vỡ trọng số pre-trained.
-   **Batch Size**: 8 hoặc 16 (tùy thuộc vào VRAM của Colab). Nếu gặp lỗi OOM (Out of Memory), hãy giảm Batch Size xuống.
-   **Epochs**: 5 - 10 epochs (vì là fine-tuning nên mô hình hội tụ khá nhanh).

### 4.3. Quy trình Training Loop

1.  **Chuyển Model sang GPU**: `model.to(device)`.
2.  **Train**:
    -   Đưa batch dữ liệu qua model.
    -   Tính Loss.
    -   Backpropagation (Lan truyền ngược).
    -   Cập nhật trọng số (Optimizer step).
3.  **Validate**:
    -   Sau mỗi epoch, chạy đánh giá trên tập Test (Fold được chọn).
    -   Tính Accuracy.

## 5\. Đánh giá và Báo cáo (Evaluation)

### 5.1. Thu thập kết quả

Lưu lại độ chính xác (Accuracy) của từng Fold (ví dụ: Fold 1: 95%, Fold 2: 92%...).

### 5.2. Tính toán tổng hợp

Tính giá trị trung bình (Mean) và độ lệch chuẩn (Standard Deviation) của Accuracy trên 10 Fold.

-   **Kết quả cuối cùng = Mean Accuracy ± Std.**

### 5.3. Lưu Checkpoint

Lưu lại trọng số của mô hình có kết quả tốt nhất (Best Model) vào Google Drive để sử dụng sau này cho việc demo hoặc inference.
