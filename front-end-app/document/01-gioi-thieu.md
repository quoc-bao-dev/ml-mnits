# Giới thiệu Đồ án CNN từ đầu bằng NumPy

Chào mừng bạn đến với Dashboard mô phỏng **Mạng Tích Chập (CNN) nhận diện chữ số viết tay MNIST**! Đây là đồ án thực hiện cho môn **Toán cho Khoa học Máy tính** (giảng dạy bởi TS. Dương Việt Hằng).

## 🎯 Mục tiêu đồ án

1. **Không sử dụng framework Deep Learning**: Không sử dụng TensorFlow, Keras, PyTorch, Jax... Toàn bộ kiến trúc mạng nơ-ron (Forward pass, Backward pass, Cập nhật trọng số) được lập trình hoàn toàn bằng thư viện **NumPy**.
2. **Làm rõ toán học**: Trực quan hóa cách các công thức Toán học (Đại số tuyến tính, Giải tích nhiều biến, Đạo hàm ma trận) được chuyển hóa thành các dòng code Python chạy thực tế.
3. **Trực quan hóa sinh động**: Giúp người học hiểu được bản chất của tích chập (Convolution), gộp cực đại (MaxPooling), lan truyền ngược (Backpropagation) thông qua các giao diện đồ họa trực tiếp.

---

## 🛠️ Kiến trúc Hệ thống

Hệ thống được thiết kế theo hướng module hóa cao, chia thành hai phần chính:

### 1. Python Backend Engine (`src/`)
Đóng vai trò là công cụ tính toán hiệu năng cao chạy trên dòng lệnh:
*   `src/data.py`: Tự động tải, giải nén và chuẩn hóa bộ dữ liệu MNIST thô từ Google Storage.
*   `src/model.py`: Chứa các lớp toán học của CNN (`Conv3x3`, `MaxPool2`, `Softmax`).
*   `src/trainer.py`: Chạy vòng lặp huấn luyện, tính toán Cross-Entropy loss và ghi log chi tiết.
*   `src/visualize.py`: Kết xuất các biểu đồ trực quan hóa dữ liệu và kết quả huấn luyện.

### 2. Next.js Web App (`front-end-app/`)
Giao diện người dùng hiện đại phong cách AI Agent để:
*   Theo dõi tiến trình huấn luyện thời gian thực (Real-time).
*   Chạy dự đoán trực quan bằng cách vẽ tay hoặc tải ảnh lên.
*   Xem tài liệu toán học định dạng Markdown có hỗ trợ công thức $\LaTeX$.

---

## 📚 Hướng dẫn sử dụng nhanh

1. **Tab Huấn luyện Model**: Bấm nút **"Bắt đầu Huấn luyện"** để chạy script huấn luyện. Bạn sẽ thấy log hệ thống chảy liên tục kèm theo biểu đồ Loss & Accuracy được cập nhật ngay sau khi huấn luyện xong.
2. **Tab Dự đoán Chữ số**: Sau khi model đã được huấn luyện thành công, bạn có thể tải ảnh chữ số viết tay của riêng mình lên hoặc vẽ trực tiếp trên canvas, hoặc đơn giản là chọn ngẫu nhiên một ảnh từ tập test MNIST để hệ thống phân tích.
3. **Tab Tài liệu**: Nơi lưu trữ các công thức toán học và giải thích chi tiết nhất.
