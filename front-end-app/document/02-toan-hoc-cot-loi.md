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

### Chứng minh: Gradient của Conv là một phép Convolution

So sánh công thức gradient filter với định nghĩa phép chập $(\text{conv}(X, G))_{m,n} = \sum_i \sum_j X_{i+m,j+n} \cdot G_{i,j}$, với $G_{i,j} = \frac{\partial L}{\partial Y_{i,j,f}}$, hai biểu thức **giống hệt nhau**.

Kiểm chứng số học: cho $X$ kích thước $3 \times 3$, filter $W$ kích thước $2 \times 2$:

```python
# Forward: Y[i,j] = sum(X[i:i+2, j:j+2] * W)
# Backward qua Chain Rule (cộng dồn qua mọi vị trí (i,j)):
dL_dW_chain = sum over (i,j): dL_dY[i,j] * X[i:i+2, j:j+2]

# Tương đương conv(X, dL_dY):
dL_dW_conv[m,n] = sum over (i,j): X[i+m, j+n] * dL_dY[i,j]

# → Hai cách cho kết quả bằng nhau (sai số máy tính ≈ 1e-16)
```

Tính chất này không phải ngẫu nhiên — nó xuất phát từ cấu trúc tuyến tính của phép chập: đạo hàm của một phép tuyến tính theo tham số của nó vẫn là một phép tuyến tính cùng cấu trúc.

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

### Chứng minh: Tại sao $\frac{\partial L}{\partial z_i} = p_i - y_i$?

**Bước 1 — Jacobian của Softmax:**

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & i = j \\ -p_i p_j & i \neq j \end{cases}$$

**Bước 2 — Chain Rule với Cross-Entropy** (chỉ nhãn đúng $c$ có $\frac{\partial L}{\partial p_i} \neq 0$):

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j} = \frac{-1}{p_c} \cdot \frac{\partial p_c}{\partial z_j}$$

- Khi $j = c$: $\displaystyle\frac{\partial L}{\partial z_c} = \frac{-1}{p_c} \cdot p_c(1 - p_c) = p_c - 1 = p_c - y_c$
- Khi $j \neq c$: $\displaystyle\frac{\partial L}{\partial z_j} = \frac{-1}{p_c} \cdot (-p_c p_j) = p_j = p_j - y_j$

Gộp lại cho mọi $i$: $\dfrac{\partial L}{\partial z_i} = p_i - y_i$ $\blacksquare$

---

## 3. Thuật toán tối ưu Gradient Descent

Sau khi tính toán được các gradient lỗi $\frac{\partial L}{\partial W}$ và $\frac{\partial L}{\partial b}$ tại mỗi lớp, ta tiến hành cập nhật trọng số theo hướng ngược lại với gradient:

$$W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

Trong đó $\eta$ là tốc độ học (learning rate, mặc định $\eta = 0.005$).
