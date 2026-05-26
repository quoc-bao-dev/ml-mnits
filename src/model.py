"""
src/model.py – Giai đoạn 2-3: Kiến trúc CNN & Toán học cốt lõi
================================================================
Xây dựng các lớp (Layer) của mạng nơ-ron tích chập bằng NumPy thuần.
Mỗi class tuân thủ giao diện forward(input) / backward(gradient, lr).

Pipeline:  Input(28×28) → Conv3x3 → MaxPool2 → Softmax → Output(10 classes)
"""

import numpy as np
import os


class Conv3x3:
    """
    Lớp Tích chập (Convolution Layer) với bộ lọc 3×3.

    ─── Toán học cốt lõi (Giai đoạn 3) ───

    FORWARD:
      Y[i, j, f] = Σ_m Σ_n  X[i+m, j+n] · W_f[m, n]
      → Phép nhân chập (convolution) giữa ma trận ảnh và bộ lọc.

    BACKWARD (Quy tắc chuỗi – Chain Rule trên ma trận):
      ∂L/∂W_f[m, n] = Σ_i Σ_j  X[i+m, j+n] · ∂L/∂Y[i, j, f]
      → Gradient của filter = phép chập giữa vùng ảnh đầu vào và ma trận lỗi.
    """

    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Khởi tạo ma trận trọng số ngẫu nhiên, chia cho 9 để giá trị nhỏ
        # → giúp Gradient Descent ổn định ở giai đoạn đầu.
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """Sinh ra tất cả vùng ảnh 3×3 khi bộ lọc trượt qua ảnh đầu vào."""
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                yield image[i:(i + 3), j:(j + 3)], i, j

    def forward(self, input_image):
        """
        Lan truyền tới: (28, 28) → (26, 26, num_filters)
        Kích thước giảm 2 vì bộ lọc 3×3 không pad viền.
        """
        self.last_input = input_image
        h, w = input_image.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input_image):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backward(self, d_L_d_out, learning_rate):
        """
        Lan truyền ngược – Áp dụng Quy tắc chuỗi.
        ∂L/∂W_f[m,n] = Σ_i Σ_j  X[i+m, j+n] · ∂L/∂Y[i, j, f]
        """
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Cập nhật trọng số (Gradient Descent)
        self.filters -= learning_rate * d_L_d_filters
        return None


class MaxPool2:
    """
    Lớp Gộp cực đại (Max Pooling) với cửa sổ 2×2, bước nhảy 2.

    FORWARD:  Y[i,j,f] = max( X[2i:2i+2, 2j:2j+2, f] )
    BACKWARD: Gradient chỉ chảy về vị trí có giá trị max (winner-take-all).
    """

    def iterate_regions(self, image):
        """Sinh ra các vùng 2×2 không chồng lấp."""
        h, w, _ = image.shape
        for i in range(h // 2):
            for j in range(w // 2):
                yield image[(i*2):(i*2+2), (j*2):(j*2+2)], i, j

    def forward(self, input_volume):
        """Forward: (26, 26, num_filters) → (13, 13, num_filters)"""
        self.last_input = input_volume
        h, w, num_filters = input_volume.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input_volume):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out, learning_rate):
        """Backward: Truyền gradient chỉ về vị trí max."""
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h2, w2, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h2):
                for j2 in range(w2):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i*2+i2, j*2+j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input


class Softmax:
    """
    Lớp Kết nối đầy đủ + Kích hoạt Softmax (Fully Connected + Softmax).

    FORWARD:
      z = W · x + b                    (Đại số tuyến tính)
      p_i = exp(z_i) / Σ_k exp(z_k)   (Hàm Softmax)

    BACKWARD (Chain Rule):
      ∂L/∂W = ∂L/∂z ⊗ x^T
      ∂L/∂b = ∂L/∂z
      ∂L/∂x = W^T · ∂L/∂z

    Hàm mất mát: Cross-Entropy Loss  L = -ln(p_correct)
    """

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input_volume):
        """Forward: Flatten → Linear → Softmax → (10,) xác suất."""
        self.last_input_shape = input_volume.shape
        input_flat = input_volume.flatten()
        self.last_input = input_flat

        totals = np.dot(input_flat, self.weights) + self.biases
        self.last_totals = totals

        # Softmax với trick trừ max để tránh overflow
        exp_values = np.exp(totals - np.max(totals))
        self.last_exp = exp_values
        self.last_sum = np.sum(exp_values)

        return exp_values / self.last_sum

    def backward(self, d_L_d_out, learning_rate):
        """Backward qua Softmax + Fully Connected."""
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            t_exp = self.last_exp
            S = self.last_sum

            # Jacobian của Softmax
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Chain Rule
            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t
            d_L_d_inputs = self.weights @ d_L_d_t

            # Gradient Descent
            self.weights -= learning_rate * d_L_d_w
            self.biases -= learning_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)


# ============================================================================
# Lưu / Tải trọng số model
# ============================================================================

def save_model(conv, pool, softmax, save_dir="output/model"):
    """Lưu trọng số của tất cả các lớp ra file .npz."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "cnn_weights.npz")
    np.savez(path,
             conv_filters=conv.filters,
             softmax_weights=softmax.weights,
             softmax_biases=softmax.biases)
    print(f"[INFO] Đã lưu model → {path}")
    return path


def load_model(model_path="output/model/cnn_weights.npz"):
    """Tải trọng số và khởi tạo lại mạng CNN."""
    data = np.load(model_path)

    conv = Conv3x3(data["conv_filters"].shape[0])
    conv.filters = data["conv_filters"]

    pool = MaxPool2()

    softmax = Softmax(data["softmax_weights"].shape[0],
                      data["softmax_weights"].shape[1])
    softmax.weights = data["softmax_weights"]
    softmax.biases = data["softmax_biases"]

    print(f"[INFO] Đã tải model ← {model_path}")
    return conv, pool, softmax


def predict(conv, pool, softmax, image):
    """
    Dự đoán chữ số từ một ảnh 28×28.

    Parameters
    ----------
    image : np.ndarray, shape (28, 28), giá trị [0, 1]

    Returns
    -------
    (predicted_digit, confidence, probabilities) : tuple
    """
    out = conv.forward(image)
    out = pool.forward(out)
    out = softmax.forward(out)

    predicted = np.argmax(out)
    confidence = out[predicted]

    return predicted, confidence, out
