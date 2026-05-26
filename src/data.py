"""
src/data.py – Giai đoạn 1: Chuẩn bị dữ liệu (Data Preparation)
================================================================
Tải bộ dữ liệu MNIST trực tiếp từ internet, chuẩn hóa và trích xuất
dữ liệu con phục vụ huấn luyện/kiểm thử.
"""

import numpy as np
import os
import gzip
import struct
import urllib.request
import ssl

# URL nguồn dữ liệu MNIST (mirror chính thức của Google)
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download_file(url, save_path):
    """Tải file từ URL nếu chưa tồn tại trên đĩa."""
    if os.path.exists(save_path):
        return
    print(f"  ↓ Đang tải: {os.path.basename(save_path)} ...")
    # Bỏ qua xác minh SSL (macOS thường thiếu root certificates cho Python)
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx) as response:
        with open(save_path, 'wb') as f:
            f.write(response.read())


def _parse_images(filepath):
    """Đọc file ảnh MNIST (IDX3 format) → np.ndarray shape (N, 28, 28)."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def _parse_labels(filepath):
    """Đọc file nhãn MNIST (IDX1 format) → np.ndarray shape (N,)."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist_data(data_dir="mnist_data", num_train=1000, num_test=1000):
    """
    Tải, trích xuất và chuẩn hóa bộ dữ liệu MNIST.

    Parameters
    ----------
    data_dir : str
        Thư mục cache dữ liệu MNIST thô.
    num_train : int
        Số ảnh huấn luyện cần trích xuất.
    num_test : int
        Số ảnh kiểm thử cần trích xuất.

    Returns
    -------
    (train_images, train_labels, test_images, test_labels) : tuple of np.ndarray
    """
    print("=" * 60)
    print("       GIAI ĐOẠN 1 – CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)")
    print("=" * 60)

    # --- Bước 1: Tải dữ liệu thô từ internet --------------------------------
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n[INFO] Tải dữ liệu MNIST (cache: {data_dir}/) ...")
    for key, filename in MNIST_FILES.items():
        _download_file(MNIST_URL + filename, os.path.join(data_dir, filename))

    # Đọc file nhị phân IDX → numpy array
    X_train_full = _parse_images(os.path.join(data_dir, MNIST_FILES["train_images"]))
    y_train_full = _parse_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    X_test_full  = _parse_images(os.path.join(data_dir, MNIST_FILES["test_images"]))
    y_test_full  = _parse_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))

    print(f"\n[INFO] Kích thước bộ dữ liệu gốc:")
    print(f"  • Train : {X_train_full.shape}  (N={X_train_full.shape[0]})")
    print(f"  • Test  : {X_test_full.shape}   (N={X_test_full.shape[0]})")
    print(f"  • Kiểu pixel : {X_train_full.dtype}, Giá trị: [{X_train_full.min()}, {X_train_full.max()}]")

    # --- Bước 2: Giới hạn dữ liệu --------------------------------------------
    train_images = X_train_full[:num_train]
    train_labels = y_train_full[:num_train]
    test_images  = X_test_full[:num_test]
    test_labels  = y_test_full[:num_test]

    print(f"\n[INFO] Trích xuất: {num_train} train + {num_test} test")

    # --- Bước 3: Chuẩn hóa (Normalization) -----------------------------------
    # Đưa giá trị pixel từ [0, 255] về [0, 1] → Gradient Descent hội tụ nhanh hơn
    train_images = train_images.astype(np.float64) / 255.0
    test_images  = test_images.astype(np.float64) / 255.0

    print(f"[INFO] Chuẩn hóa: pixel / 255 → [{train_images.min():.1f}, {train_images.max():.1f}]")

    # --- Bước 4: Thống kê phân bố nhãn ----------------------------------------
    print(f"\n[INFO] Phân bố nhãn (train):")
    unique, counts = np.unique(train_labels, return_counts=True)
    for digit, count in zip(unique, counts):
        bar = "█" * (count // 2)
        print(f"  Chữ số {digit}: {count:4d} ảnh  {bar}")

    return train_images, train_labels, test_images, test_labels
