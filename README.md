# 🧠 Nhận diện chữ số viết tay (MNIST) với CNN thuần NumPy

> **Đồ án môn Toán cho Khoa học Máy tính – TS. Dương Việt Hằng**

Xây dựng mạng nơ-ron tích chập (Convolutional Neural Network) **từ con số không** bằng NumPy, áp dụng **Đại số tuyến tính** và **Giải tích** (Quy tắc chuỗi) để tự viết thuật toán Backpropagation mà không phụ thuộc vào các framework Deep Learning (TensorFlow/PyTorch).

---

## 📁 Cấu trúc thư mục

```
do-an/
├── README.md              ← Hướng dẫn sử dụng (file này)
├── plan.md                ← Kế hoạch đồ án chi tiết
├── requirements.txt       ← Danh sách thư viện cần cài
│
├── train.py               ← 🚀 Entry point: Huấn luyện model
├── predict.py             ← 🔮 Entry point: Dự đoán từ ảnh
│
├── src/                   ← Mã nguồn chính
│   ├── __init__.py
│   ├── data.py            ← Giai đoạn 1: Tải & chuẩn bị dữ liệu MNIST
│   ├── model.py           ← Giai đoạn 2-3: Kiến trúc CNN + Toán học
│   ├── trainer.py         ← Giai đoạn 4-5: Vòng lặp huấn luyện + logging
│   └── visualize.py       ← Vẽ biểu đồ và minh họa kết quả
│
├── output/                ← Kết quả sinh ra (tự tạo khi chạy)
│   ├── charts/            ← Biểu đồ Loss, Accuracy, Predictions
│   ├── logs/              ← File log huấn luyện (training_YYYYMMDD_HHMMSS.log)
│   └── model/             ← Trọng số model đã train (cnn_weights.npz)
│
└── mnist_data/            ← Cache dữ liệu MNIST (tự tải lần đầu)
```

---

## ⚙️ Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **Thư viện**: NumPy, Matplotlib (Pillow nếu muốn dùng `predict.py` với ảnh tự chụp)

---

## 🚀 Hướng dẫn chạy từ đầu

### Bước 1: Cài đặt thư viện

```bash
pip3 install -r requirements.txt
```

### Bước 2: Huấn luyện model

```bash
python3 train.py
```

Chương trình sẽ:
1. Tự động tải bộ dữ liệu MNIST (lần đầu, ~11MB)
2. Trích xuất 1.000 ảnh train + 1.000 ảnh test
3. Chuẩn hóa pixel từ `[0, 255]` về `[0, 1]`
4. Huấn luyện CNN qua 3 epoch với Backpropagation
5. Lưu kết quả vào thư mục `output/`

**Thời gian chạy:** ~30-60 giây trên laptop thông thường.

### Bước 3: Dự đoán chữ số

```bash
# Dùng ảnh mẫu ngẫu nhiên từ MNIST
python3 predict.py

# Dùng ảnh tự chụp (28×28 hoặc sẽ tự resize)
python3 predict.py path/to/my_digit.png
```

---

## 🏗️ Kiến trúc mạng CNN

```
Input (28×28)
    ↓
Conv3x3 (8 filters)     → Output: (26×26×8)
    ↓
MaxPool2 (2×2)           → Output: (13×13×8)
    ↓
Softmax (1352 → 10)      → Output: 10 xác suất (chữ số 0-9)
```

| Lớp | Mô tả | Toán học |
|-----|--------|----------|
| **Conv3x3** | Tích chập bộ lọc 3×3 | `Y[i,j,f] = Σ X[i+m,j+n] · W[m,n]` |
| **MaxPool2** | Gộp cực đại 2×2 | `Y[i,j] = max(X[2i:2i+2, 2j:2j+2])` |
| **Softmax** | FC + Softmax | `p_i = exp(z_i) / Σ exp(z_k)` |

---

## 📊 Kết quả mong đợi

| Epoch | Train Loss | Train Acc | Test Acc |
|-------|-----------|-----------|----------|
| 1     | ~1.15     | ~65%      | ~74%     |
| 2     | ~0.50     | ~84%      | ~80%     |
| 3     | ~0.40     | ~88%      | ~81%     |

- **Loss giảm dần** → Gradient Descent hoạt động đúng
- **Accuracy tăng dần** → Backpropagation + Chain Rule được code chính xác

---

## 📂 Output sinh ra

Sau khi chạy `train.py`, thư mục `output/` sẽ chứa:

| File | Mô tả |
|------|--------|
| `charts/sample_digits.png` | 20 ảnh mẫu MNIST |
| `charts/pixel_matrix.png` | Minh họa ma trận pixel |
| `charts/training_history.png` | Biểu đồ Loss & Accuracy |
| `charts/predictions.png` | Dự đoán 20 ảnh test |
| `logs/training_*.log` | Log chi tiết quá trình huấn luyện |
| `model/cnn_weights.npz` | Trọng số model đã train |

---

## 🔬 Toán học cốt lõi

### Backpropagation tại lớp Convolution (Chain Rule)

$$\frac{\partial L}{\partial W_{m,n}} = \sum_{i} \sum_{j} X_{i+m, j+n} \cdot \frac{\partial L}{\partial Y_{i,j}}$$

- $\frac{\partial L}{\partial W_{m,n}}$: Gradient của lỗi theo trọng số bộ lọc
- $X_{i+m, j+n}$: Vùng ảnh đầu vào đang được chập
- $\frac{\partial L}{\partial Y_{i,j}}$: Gradient lỗi truyền ngược từ lớp sau

> **Nhận xét:** Đạo hàm của phép chập thực chất là một phép chập giữa input và gradient lỗi.

### Hàm mất mát: Cross-Entropy

$$L = -\ln(p_{correct})$$

### Hàm Softmax

$$p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$$
