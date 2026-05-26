# Đồ án: Nhận diện chữ số viết tay (MNIST) với Mạng tích chập (CNN) thuần NumPy

**Môn học:** Toán cho Khoa học Máy tính
**Mục tiêu:** Xây dựng một mạng nơ-ron nhân tạo từ con số không, áp dụng các kiến thức Đại số tuyến tính và Giải tích (Quy tắc chuỗi) để tự viết thuật toán Backpropagation mà không phụ thuộc vào các framework Deep Learning (TensorFlow/PyTorch).

---

## 1. Trực giác bài toán (The Intuition)
Hãy tưởng tượng cỗ máy của chúng ta là một **"nhà máy phân loại thư"**. 
- **Đầu vào (Nguyên liệu):** Một bức ảnh đen trắng chứa một chữ số viết tay. Máy tính không thấy hình ảnh, nó chỉ thấy một lưới ma trận các con số thể hiện sắc độ sáng tối.
- **Xử lý (Dây chuyền):** Bức ảnh đi qua các trạm kiểm duyệt (Layers). Các trạm này dùng những chiếc "kính lúp" (Filter ma trận) để quét qua bức ảnh, tìm kiếm các nét dọc, nét ngang, đường cong.
- **Học hỏi (Tự sửa sai):** Nếu nhà máy phân loại sai, người quản lý (Hàm Loss) sẽ báo lỗi. Nhà máy tự động dùng **Đạo hàm (Calculus)** để lần ngược lại dây chuyền, tìm xem trạm nào làm sai và "vặn" lại các thông số (Trọng số) cho đúng.

---

## 2. Giai đoạn 1: Chuẩn bị dữ liệu (Data Preparation)

Chúng ta sẽ sử dụng bộ dữ liệu kinh điển **MNIST**. Để tập trung vào phần Toán, ta sẽ dùng hàm có sẵn của Keras chỉ để tải dữ liệu thô, sau đó hoàn toàn xử lý bằng NumPy.

*   **Định dạng Input ($X$):** Ma trận 3D kích thước `(N, 28, 28)`. Trong đó $N$ là số lượng ảnh, $28 \times 28$ là kích thước (pixel) của mỗi bức ảnh.
*   **Định dạng Label ($Y$):** Mảng 1D chứa các số nguyên từ $0$ đến $9$.
*   **Tiền xử lý (Preprocessing):**
    *   **Giới hạn dữ liệu:** Trích xuất $1000$ ảnh để huấn luyện (giúp thuật toán thuần vòng lặp Python chạy mượt mà trên laptop cá nhân).
    *   **Chuẩn hóa (Normalization):** Đưa các giá trị pixel từ khoảng $[0, 255]$ về đoạn $[0, 1]$ bằng phép chia ma trận cho $255$. Việc này giúp các phép toán Gradient Descent hội tụ nhanh hơn.

---

## 3. Giai đoạn 2: Kiến trúc phần mềm (The Architecture)

Dự án được thiết kế theo mô hình Hướng đối tượng (OOP). Mỗi lớp (Layer) trong mạng nơ-ron là một class độc lập, tuân thủ nghiêm ngặt 2 phương thức:
1.  `forward(input)`: Tính toán luồng đi tới, sinh ra kết quả.
2.  `backward(d_L_d_out, learning_rate)`: Lan truyền lỗi ngược, tự cập nhật ma trận trọng số và trả về gradient cho lớp đứng trước nó.

**Các thành phần (Class) cần xây dựng:**
*   `Conv3x3`: Lớp tích chập quét bộ lọc $3 \times 3$ qua ma trận ảnh.
*   `MaxPool2`: Lớp gộp giúp giảm nửa kích thước không gian ma trận (lấy giá trị max trong các ô $2 \times 2$).
*   `Softmax`: Lớp kích hoạt cuối cùng, dùng hàm mũ để chuyển đổi ma trận điểm số thành phân phối xác suất.

---

## 4. Giai đoạn 3: Phân tích Toán học cốt lõi (The Core Mathematics)

Trái tim của đồ án nằm ở thuật toán **Backpropagation** tại lớp Convolution. Chúng ta áp dụng Quy tắc chuỗi (Chain Rule) trên không gian ma trận.

Công thức tính đạo hàm của hàm mất mát $L$ theo từng phần tử trong ma trận trọng số (Filter) $W$:

$$ \frac{\partial L}{\partial W_{m,n}} = \sum_{i} \sum_{j} X_{i+m, j+n} \frac{\partial L}{\partial Y_{i,j}} $$

**Phân tích các thành phần (Visual Breakdown):**
- $\frac{\partial L}{\partial W_{m,n}}$: Gradient của lỗi theo trọng số của bộ lọc (chỉ báo hướng cần điều chỉnh).
- $X_{i+m, j+n}$: Các điểm ảnh (pixels) của ma trận đầu vào tại vị trí đang được chập.
- $\frac{\partial L}{\partial Y_{i,j}}$: Độ lỗi (Gradient) được truyền ngược từ lớp ngay phía sau về lớp hiện tại.

*Nhận xét Toán học:* Đạo hàm của một phép chập thực chất lại chính là một phép chập giữa vùng ảnh đầu vào và ma trận lỗi.

---

## 5. Giai đoạn 4: Kế hoạch Lập trình (Code Plan)

### Luồng hoạt động (Pipeline)
1.  **Khởi tạo:** Thiết lập các ma trận ngẫu nhiên (dùng `np.random.randn`).
2.  **Huấn luyện (Training Loop - Epochs):** Lặp qua từng bức ảnh.
    *   *Bước 1:* Chạy `forward()` qua toàn bộ các class.
    *   *Bước 2:* Tính Cross-Entropy Loss.
    *   *Bước 3:* Chạy `backward()` ngược từ class cuối cùng lên class đầu tiên. Trọng số tự động cập nhật bên trong.
3.  **Kiểm thử (Inference):** Đưa ảnh mới (chưa từng huấn luyện) qua hàm `forward()` để kiểm tra độ chính xác (Accuracy).

### Bản nháp mã nguồn cốt lõi (Core Snippet)
```python
import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Khởi tạo ma trận ngẫu nhiên cho filter
        self.filters = np.random.randn(num_filters, 3, 3) / 9 
        
    def forward(self, input_image):
        self.last_input = input_image
        h, w = input_image.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        
        # Phép chập (Nhân chập ma trận)
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = input_image[i:(i + 3), j:(j + 3)]
                output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        
        # Ứng dụng công thức Toán: Phép chập giữa Input và Gradient lỗi
        for i in range(self.last_input.shape[0] - 2):
            for j in range(self.last_input.shape[1] - 2):
                im_region = self.last_input[i:(i + 3), j:(j + 3)]
                for f in range(self.num_filters):
                    d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Cập nhật trọng số (Gradient Descent)
        self.filters -= learning_rate * d_L_d_filters
        return None
```
## 6. Giai đoạn 5: Chạy thử nghiệm và Đánh giá (Execution & Testing)
Môi trường:

Yêu cầu: Python 3.8+

Cài đặt dependencies: pip install numpy keras matplotlib

Cách khởi chạy:

Chạy script chính: python cnn_numpy_mnist.py

Kết quả mong đợi:

Terminal sẽ in ra quá trình huấn luyện theo thời gian thực.

Giá trị Loss giảm dần và Accuracy tăng dần theo từng Epoch, chứng minh thuật toán Backpropagation và các công thức giải tích đạo hàm đã được code chính xác.