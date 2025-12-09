Deepfake Detection using CNNs
Đồ án môn học: Xây dựng mô hình CNN phát hiện ảnh giả mạo (Real vs. AI-generated)

👤 Thông tin sinh viên
Họ và tên: Nguyễn Đức Thịnh

MSSV: 23110156

Trường: Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)

Giảng viên hướng dẫn: TS. Lê Văn Vinh

📖 Giới thiệu
Dự án này xây dựng một mô hình Convolutional Neural Network (CNN) thủ công (không sử dụng pre-trained models) để phân loại hình ảnh thành hai nhóm:

REAL: Ảnh thật.

FAKE: Ảnh do AI tạo ra (Deepfake).

Mô hình được huấn luyện trên bộ dữ liệu CIFAKE và đạt độ chính xác khoảng 94%.

📂 Dữ liệu (Dataset)
Dự án sử dụng dataset từ Kaggle: CIFAKE: Real and AI-Generated Synthetic Images.

Training: 100,000 ảnh.

Testing: 20,000 ảnh.

Kích thước: 32x32 (được resize lên 224x224 trong quá trình huấn luyện).

🧠 Kiến trúc Mô hình
Mô hình MyNet được thiết kế với 4 khối tích chập (Convolutional Blocks):

Input: 3x224x224

Layers: 4 lớp Conv2d (tăng dần channels: 32 -> 64 -> 128 -> 256), mỗi lớp đi kèm BatchNorm, ReLU và MaxPool.

Classifier: Lớp Fully Connected (Linear) đầu ra 2 lớp (Real/Fake) có sử dụng Dropout (0.5) để chống overfitting.

🛠 Cài đặt & Hướng dẫn sử dụng
1. Yêu cầu hệ thống
Python 3.x

Thư viện: torch, torchvision, matplotlib, kaggle

Bash

pip install torch torchvision matplotlib kaggle
2. Tải dữ liệu
Dự án được cấu hình để chạy trên Google Colab và tải dữ liệu trực tiếp từ Kaggle API.

Tạo API Token trên Kaggle (kaggle.json).

Upload file kaggle.json khi chạy notebook.

3. Chạy dự án
Mở file notebook CNNs_Deepfake_Detection.ipynb và chạy lần lượt các cell để:

Tải dữ liệu.

Huấn luyện mô hình (Training).

Đánh giá kết quả (Evaluation).

Kiểm tra thử trên ảnh ngẫu nhiên (Inference).

📊 Kết quả
Sau 10 epochs huấn luyện:

Training Loss: ~0.17

Test Accuracy: ~94.6%

Biểu đồ Loss và Accuracy sẽ được hiển thị trực tiếp trong notebook sau khi quá trình huấn luyện hoàn tất.

Dự án phục vụ mục đích học tập và nghiên cứu.
