# Kiến trúc Mạng tích chập (CNN) MNIST

Kiến trúc mạng CNN của đồ án gồm 3 lớp cơ bản được thiết kế theo hướng đối tượng (OOP).

---

## 🏗️ Sơ đồ Luồng dữ liệu (Data Pipeline)

```
[Ảnh Đầu Vào] (28 × 28)
       │
       ▼
 [Lớp Tích Chập] (Conv3x3)
 ── Kích thước bộ lọc: 3×3
 ── Số bộ lọc (filters): 8
 ── Không dùng padding (valid)
 ── Output: 8 kênh ảnh kích thước (26 × 26 × 8)
       │
       ▼
 [Lớp Gộp Cực Đại] (MaxPool2)
 ── Cửa sổ gộp: 2×2, bước nhảy (stride): 2
 ── Thu nhỏ kích thước ảnh đi 2 lần
 ── Output: 8 kênh ảnh kích thước (13 × 13 × 8)
       │
       ▼
 [Lớp Phẳng Hóa] (Flatten)
 ── Chuyển ma trận 3D thành vector 1D
 ── Kích thước vector: 13 × 13 × 8 = 1352 phần tử
       │
       ▼
 [Lớp Kết Nối Đầy Đủ + Softmax] (Softmax Layer)
 ── Kích thước ma trận trọng số: (1352 × 10)
 ── Output: Vector gồm 10 xác suất tương ứng với chữ số 0-9
```

---

## 🛡️ Chi tiết các Lớp (OOP Implementation)

### 1. Lớp Conv3x3
Lớp này trích xuất các đặc trưng biên (edges), góc (corners) của chữ số viết tay.
*   **Tham số**: `filters` kích thước `(8, 3, 3)` khởi tạo ngẫu nhiên theo phân phối Gaussian và chia cho 9 để chuẩn hóa.
*   **Hàm bổ trợ**: `iterate_regions()` trượt cửa sổ $3 \times 3$ qua ảnh đầu vào.

### 2. Lớp MaxPool2
Lớp này giúp giảm số lượng tham số tính toán và tạo ra tính bất biến dịch chuyển nhẹ (translation invariance).
*   **Cơ chế**: Lấy giá trị lớn nhất trong mỗi vùng $2 \times 2$.
*   **Lan truyền ngược**: Lưu trữ giá trị đầu vào để xác định vị trí của pixel lớn nhất, chỉ truyền gradient về duy nhất vị trí đó (các vị trí khác nhận gradient bằng 0).

### 3. Lớp Softmax
Lớp phân loại tích hợp nhân ma trận với vector.
*   **Trọng số**: `weights` kích thước `(1352, 10)` và `biases` kích thước `(10,)`.
*   **Trick chống tràn số (Numerical Stability)**: Khi tính $e^z$, nếu $z$ quá lớn sẽ gây ra lỗi `NaN` (Not a Number). Lớp Softmax của chúng ta tự động trừ đi giá trị lớn nhất trước khi tính hàm mũ:
    ```python
    totals = np.dot(input_flat, self.weights) + self.biases
    exp_values = np.exp(totals - np.max(totals))
    ```
