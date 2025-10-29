# EfficientNetB2 for CIFAR-100 Classification (Transfer Learning)
Dự án này sử dụng mô hình tiền huấn luyện EfficientNetB2 (pre-trained trên ImageNet) và 
áp dụng kỹ thuật Transfer Learning (Học Chuyển giao) để phân loại tập dữ liệu CIFAR-100. 
Mô hình được huấn luyện theo hai giai đoạn: Feature Extractor (Trích xuất đặc trưng) và Fine-Tuning (Điều chỉnh chi tiết), 
kết hợp với kỹ thuật Early Stopping để đạt hiệu suất tối ưu.

# Mục tiêu Dự án
Mục tiêu là xây dựng một mô hình phân loại hiệu quả cho 100 lớp (classes) của bộ dữ liệu CIFAR-100 
bằng cách tận dụng kiến trúc mạng đã được học từ tập dữ liệu lớn ImageNet, 
từ đó rút ngắn thời gian huấn luyện và tăng cường độ chính xác.

# Công nghệ và Thư viện
Ngôn ngữ: Python

Framework: PyTorch

Mô hình nền: EfficientNetB2 (pre-trained on ImageNet)

Tập dữ liệu: CIFAR-100 https://www.kaggle.com/datasets/fedesoriano/cifar100/data

Tối ưu hóa: Adam

Hàm mất mát: nn.CrossEntropyLoss

# Kết quả Đánh giá (Evaluation Results)
Sau khi hoàn tất quá trình huấn luyện và tải lại checkpoint tốt nhất mô hình đạt độ chính xác cao **83.25%** trên tập kiểm thử CIFAR-100.
