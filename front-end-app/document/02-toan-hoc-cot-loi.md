# Cơ sở Toán học của mạng CNN & Backpropagation

Mạng Nơ-ron Tích chập (CNN) hoạt động dựa trên các nguyên lý giải tích nhiều biến, đại số tuyến tính và tối ưu hóa số học. Dưới đây là các công thức toán học chi tiết được áp dụng trong chương trình.

---

## 1. Phép Tích Chập (Convolution) & Quy tắc chuỗi (Chain Rule)

Lớp tích chập thực hiện phép chập giữa ma trận ảnh đầu vào $X$ kích thước $H \times W$ và bộ lọc (filter) $W$ kích thước $3 \times 3$:

$$Y_{i,j,f} = \sum_{m=0}^{2} \sum_{n=0}^{2} X_{i+m, j+n} \cdot W_{f, m, n} + b_f$$

Trong đó:
*   $Y_{i,j,f}$: Giá trị pixel đầu ra tại hàng $i$, cột $j$, filter $f$.
*   $W_{f, m, n}$: Trọng số tại vị trí $(m, n)$ của filter $f$.
*   $b_f$: Độ lệch (bias) của filter $f$.

### Lan truyền ngược qua lớp Conv (Backpropagation)
Khi nhận được gradient lỗi $\frac{\partial L}{\partial Y}$ từ lớp tiếp theo, ta cần tính gradient lỗi của các tham số filter để cập nhật bằng Gradient Descent:

$$\frac{\partial L}{\partial W_{f, m, n}} = \sum_{i} \sum_{j} X_{i+m, j+n} \cdot \frac{\partial L}{\partial Y_{i, j, f}}$$

> **Nhận xét**: Gradient của filter thực chất là phép chập giữa vùng ảnh đầu vào và ma trận lỗi đầu ra. Công thức này tương đương với:
> ```python
> d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
> ```

---

## 2. Hàm kích hoạt Softmax & Hàm mất mát Cross-Entropy

Để phân loại chữ số từ 0 đến 9, ta kết hợp hàm kích hoạt Softmax ở lớp cuối cùng với hàm mất mát Entropy chéo (Cross-Entropy).

### Hàm Softmax
Biến đổi đầu ra tuyến tính $z$ thành phân phối xác suất:

$$p_i = \frac{e^{z_i}}{\sum_{k=0}^{9} e^{z_k}}$$

### Hàm mất mát Cross-Entropy
Lượng hóa sai số giữa phân phối xác suất dự đoán $p$ và nhãn thực tế $y$ (dưới dạng One-hot vector):

$$L = - \sum_{k=0}^{9} y_k \ln(p_k) = - \ln(p_{correct})$$

### Đạo hàm liên hợp (Softmax + Cross-Entropy)
Khi tính toán lan truyền ngược, đạo hàm của hàm mất mát Cross-Entropy theo đầu vào $z_i$ của lớp Softmax có dạng cực kỳ rút gọn:

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

Nói cách cách khác:
*   Nếu $i$ là nhãn đúng ($y_i = 1$): $\frac{\partial L}{\partial z_i} = p_i - 1$
*   Nếu $i$ là nhãn sai ($y_i = 0$): $\frac{\partial L}{\partial z_i} = p_i$

Đây là điểm khởi đầu cho dòng gradient lan truyền ngược qua toàn bộ mạng!

---

## 3. Thuật toán tối ưu Gradient Descent

Sau khi tính toán được các gradient lỗi $\frac{\partial L}{\partial W}$ và $\frac{\partial L}{\partial b}$ tại mỗi lớp, ta tiến hành cập nhật trọng số theo hướng ngược lại với gradient:

$$W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

Trong đó $\eta$ là tốc độ học (learning rate, mặc định $\eta = 0.005$).
