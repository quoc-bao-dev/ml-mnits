# Nền tảng Toán học để xây dựng CNN

Trước khi đi vào kiến trúc CNN, ta cần nắm vững ba trụ cột toán học: **Đại số tuyến tính**, **Giải tích nhiều biến**, và **Quy tắc chuỗi**. Đây là những công cụ thực sự chạy bên dưới mọi lớp của mạng nơ-ron.

---

## 1. Đại số tuyến tính (Linear Algebra)

### 1.1 Vector và Ma trận

Một **vector** cột $\mathbf{x} \in \mathbb{R}^n$ là một mảng $n$ số thực được sắp xếp theo chiều dọc:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

Một **ma trận** $A \in \mathbb{R}^{m \times n}$ là mảng hình chữ nhật gồm $m$ hàng và $n$ cột:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Trong CNN**: ảnh đầu vào MNIST là ma trận $X \in \mathbb{R}^{28 \times 28}$. Mỗi phần tử $x_{i,j} \in [0, 1]$ là giá trị độ sáng của pixel tại hàng $i$, cột $j$. Ma trận trọng số của lớp Softmax là $W \in \mathbb{R}^{1352 \times 10}$.

### 1.2 Tích Vô hướng (Dot Product)

Tích vô hướng giữa hai vector $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ được định nghĩa:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n \in \mathbb{R}$$

Kết quả luôn là một số thực (scalar). Ý nghĩa hình học: tích vô hướng đo **độ tương đồng hướng** giữa hai vector. Khi hai vector cùng chiều, tích vô hướng đạt giá trị lớn nhất; khi vuông góc, tích vô hướng bằng 0.

**Trong CNN**: tại mỗi vị trí $(i, j)$, lớp Conv tính tích vô hướng giữa vùng ảnh $3 \times 3$ và bộ lọc $3 \times 3$:

$$Y_{i,j,f} = \sum_{m=0}^{2} \sum_{n=0}^{2} X_{i+m,\, j+n} \cdot W_{f,m,n} + b_f$$

Đây chính là tích vô hướng giữa hai ma trận $3 \times 3$ được "trải phẳng" thành vector độ dài 9.

### 1.3 Nhân Ma trận (Matrix Multiplication)

Với $A \in \mathbb{R}^{m \times k}$ và $B \in \mathbb{R}^{k \times n}$, tích $C = AB \in \mathbb{R}^{m \times n}$ được tính:

$$c_{ij} = \sum_{l=1}^{k} a_{il} \cdot b_{lj}, \quad i = 1,\ldots,m; \quad j = 1,\ldots,n$$

Mỗi phần tử $c_{ij}$ là tích vô hướng giữa **hàng $i$** của $A$ và **cột $j$** của $B$. Điều kiện bắt buộc: số cột của $A$ phải bằng số hàng của $B$.

**Trong CNN**: lớp Softmax thực hiện:

$$\mathbf{z} = W^\top \mathbf{x} + \mathbf{b}, \quad W^\top \in \mathbb{R}^{10 \times 1352},\; \mathbf{x} \in \mathbb{R}^{1352},\; \mathbf{z} \in \mathbb{R}^{10}$$

Tương đương trong NumPy:
```python
totals = np.dot(input_flat, self.weights) + self.biases  # (1352,) × (1352, 10) → (10,)
```

Mỗi phần tử $z_k$ của vector đầu ra là "điểm số" (logit) của chữ số $k$, được tính bằng tổng trọng số của toàn bộ 1352 đặc trưng.

### 1.4 Chuyển vị Ma trận (Transpose)

Ma trận chuyển vị $A^\top \in \mathbb{R}^{n \times m}$ của $A \in \mathbb{R}^{m \times n}$ được xây dựng bằng cách đổi hàng thành cột:

$$(A^\top)_{ij} = A_{ji}$$

**Trong Backpropagation**: khi truyền gradient ngược qua lớp Softmax, ta cần gradient của loss theo vector đặc trưng đầu vào:

$$\frac{\partial L}{\partial \mathbf{x}} = W \cdot \frac{\partial L}{\partial \mathbf{z}} \in \mathbb{R}^{1352}$$

Chuyển vị xuất hiện tự nhiên khi đảo chiều phép nhân ma trận trong backward pass.

---

## 2. Giải tích (Calculus)

### 2.1 Đạo hàm một biến

Cho hàm $f: \mathbb{R} \to \mathbb{R}$, **đạo hàm** tại điểm $x$ đo tốc độ thay đổi của $f$ khi $x$ thay đổi một lượng vô cùng nhỏ $\Delta x$:

$$f'(x) = \frac{df}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

**Ý nghĩa thực tế**:
- $f'(x) > 0$: tăng $x$ một chút sẽ làm $f$ tăng.
- $f'(x) < 0$: tăng $x$ một chút sẽ làm $f$ giảm.
- $f'(x) = 0$: $f$ đang ở cực trị cục bộ hoặc điểm bằng phẳng.

Gradient Descent khai thác thông tin này: nếu ta muốn giảm $f$ (tức là giảm loss $L$), ta dịch chuyển $x$ theo hướng **ngược dấu** của đạo hàm.

Một số đạo hàm thường gặp trong CNN:

| Hàm $f(x)$ | Đạo hàm $f'(x)$ | Xuất hiện ở |
|---|---|---|
| $x^n$ | $nx^{n-1}$ | Tính gradient tổng quát |
| $e^x$ | $e^x$ | Hàm Softmax |
| $\ln(x)$ | $1/x$ | Hàm mất mát Cross-Entropy |
| $\max(x, 0)$ | $\mathbf{1}[x > 0]$ | Hàm ReLU (nếu dùng) |

### 2.2 Đạo hàm riêng (Partial Derivative)

Khi $f$ phụ thuộc vào nhiều biến $f(x_1, x_2, \ldots, x_n)$, **đạo hàm riêng** theo $x_i$ được tính bằng cách xem tất cả các biến còn lại là hằng số:

$$\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(\ldots, x_i + \Delta x_i, \ldots) - f(\ldots, x_i, \ldots)}{\Delta x_i}$$

**Ví dụ cụ thể**: $f(x_1, x_2) = x_1^2 + 3x_1 x_2 - x_2^3$

$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2 \qquad \text{(giữ } x_2 \text{ cố định)}$$

$$\frac{\partial f}{\partial x_2} = 3x_1 - 3x_2^2 \qquad \text{(giữ } x_1 \text{ cố định)}$$

**Trong CNN**: hàm mất mát $L$ phụ thuộc vào hàng nghìn tham số (mỗi phần tử của filter $W_{f,m,n}$, mỗi trọng số Softmax $w_{ij}$). Ta cần tính đạo hàm riêng của $L$ theo từng tham số đó để biết cần cập nhật theo hướng nào.

### 2.3 Gradient

**Gradient** của $f: \mathbb{R}^n \to \mathbb{R}$ tại điểm $\mathbf{x}$ là vector gồm toàn bộ đạo hàm riêng:

$$\nabla_{\mathbf{x}} f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n$$

**Tính chất quan trọng**: gradient luôn trỏ theo **hướng tăng nhanh nhất** của $f$. Ngược lại, $-\nabla f$ trỏ theo hướng giảm nhanh nhất — đó chính là hướng mà Gradient Descent di chuyển.

Quy tắc cập nhật tham số:

$$W \leftarrow W - \eta \cdot \nabla_W L$$

trong đó $\eta$ là learning rate kiểm soát độ lớn của mỗi bước dịch chuyển.

### 2.4 Đạo hàm của hàm hợp (dạng một biến)

Nếu $y = f(u)$ và $u = g(x)$, tức là $y = f(g(x))$, thì đạo hàm của $y$ theo $x$ là:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Ví dụ**: $y = e^{3x^2}$. Đặt $u = 3x^2$, $y = e^u$:

$$\frac{dy}{dx} = e^u \cdot 6x = 6x \cdot e^{3x^2}$$

Đây là tiền đề trực tiếp dẫn đến Quy tắc chuỗi trong mạng nhiều lớp.

---

## 3. Quy tắc Chuỗi (Chain Rule)

### 3.1 Dạng nhiều biến

Đây là công cụ trung tâm của Backpropagation. Nếu $L$ phụ thuộc vào $\mathbf{y} = (y_1, \ldots, y_k)$, và mỗi $y_j$ lại phụ thuộc vào $x$, thì:

$$\frac{\partial L}{\partial x} = \sum_{j=1}^{k} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x}$$

Nói trực quan: ảnh hưởng của $x$ lên $L$ bằng **tổng** ảnh hưởng của $x$ lên từng $y_j$ nhân với ảnh hưởng của $y_j$ lên $L$ — qua từng "con đường" mà $x$ có thể tác động đến $L$.

### 3.2 Dạng ma trận (Jacobian)

Khi $\mathbf{y} \in \mathbb{R}^m$ và $\mathbf{x} \in \mathbb{R}^n$, **ma trận Jacobian** $J \in \mathbb{R}^{m \times n}$ chứa toàn bộ đạo hàm riêng:

$$J_{ij} = \frac{\partial y_i}{\partial x_j}$$

Chain Rule dạng ma trận:

$$\frac{\partial L}{\partial \mathbf{x}} = J^\top \cdot \frac{\partial L}{\partial \mathbf{y}}$$

Đây là công thức chung để truyền gradient ngược qua bất kỳ lớp nào trong mạng nơ-ron.

### 3.3 Minh họa qua toàn bộ luồng CNN

Xét luồng tính toán từ đầu vào đến loss:

$$X \xrightarrow{\text{Conv}} Y \xrightarrow{\text{MaxPool}} P \xrightarrow{\text{Flatten}} \mathbf{p}_{\text{flat}} \xrightarrow{\text{Softmax+CE}} L$$

Để tính gradient của loss theo filter tích chập $W_{\text{conv}}$, Chain Rule được áp dụng lần lượt từng lớp:

$$\frac{\partial L}{\partial W_{\text{conv}}} = \underbrace{\frac{\partial L}{\partial \mathbf{p}_{\text{flat}}}}_{\text{từ Softmax}} \cdot \underbrace{\frac{\partial \mathbf{p}_{\text{flat}}}{\partial P}}_{\text{qua MaxPool}} \cdot \underbrace{\frac{\partial P}{\partial Y}}_{\text{qua MaxPool}} \cdot \underbrace{\frac{\partial Y}{\partial W_{\text{conv}}}}_{\text{qua Conv}}$$

Mỗi nhân tử ứng với hàm `backprop()` của một lớp:

```python
# Softmax: tính gradient của loss theo vector đầu vào phẳng
gradient = self.last_input_shape  # ∂L/∂p_flat

# MaxPool: chỉ truyền gradient về đúng vị trí có giá trị max
d_L_d_input[r, c, f] = d_L_d_out[r // 2, c // 2, f]  # ∂p_flat/∂P

# Conv: tích lũy gradient cho từng filter
d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region   # ∂Y/∂W_conv
```

Chain Rule biến một bài toán tối ưu hóa với hàng nghìn tham số liên kết phức tạp thành **chuỗi nhân đạo hàm có thể tính tuần tự từng lớp** — đó là lý do tại sao Backpropagation có thể triển khai được trong thực tế.

---

## Tóm tắt mối liên hệ

| Khái niệm | Vai trò trong CNN |
|---|---|
| Tích vô hướng | Tính một điểm đầu ra của Conv tại $(i,j)$ |
| Nhân ma trận | Tính toàn bộ 10 logit trong lớp Softmax |
| Đạo hàm riêng | Đo mức độ ảnh hưởng của một tham số lên loss |
| Gradient | Vector chứa đạo hàm riêng của loss theo mọi tham số |
| Chain Rule | Truyền gradient ngược qua nhiều lớp liên tiếp |
