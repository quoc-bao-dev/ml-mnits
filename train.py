"""
train.py – Entry point: Huấn luyện mạng CNN
=============================================
Chạy: python3 train.py

Thực hiện:
  1. Tải & chuẩn bị dữ liệu MNIST
  2. Huấn luyện mạng CNN (3 epochs)
  3. Lưu model, biểu đồ, và log
"""

import os
import sys

# Đảm bảo import được module src/ dù chạy từ thư mục nào
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_mnist_data
from src.trainer import train_and_test
from src.visualize import (
    visualize_samples,
    visualize_pixel_matrix,
    visualize_training,
    visualize_predictions,
)

# Thư mục gốc của dự án
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "output")
CHARTS = os.path.join(OUTPUT, "charts")


def load_env(env_path=os.path.join(ROOT, ".env")):
    """Đọc file .env thủ công để tránh dependencies phụ thuộc."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()


def main():
    # Nạp cấu hình từ .env
    load_env()
    
    # Đọc tham số với giá trị mặc định là 1000 nếu không được cấu hình
    # GIỚI HẠN TỐI ĐA:
    # - MNIST_NUM_TRAIN: tối đa 60000 mẫu
    # - MNIST_NUM_TEST: tối đa 10000 mẫu
    num_train = int(os.environ.get("MNIST_NUM_TRAIN", 1000))
    num_test = int(os.environ.get("MNIST_NUM_TEST", 1000))

    # Giới hạn an toàn để không vượt quá kích thước thật của bộ dữ liệu MNIST
    num_train = min(max(1, num_train), 60000)
    num_test = min(max(1, num_test), 10000)

    # ═══════════════════════════════════════════════════
    # Giai đoạn 1: Chuẩn bị dữ liệu
    # ═══════════════════════════════════════════════════
    train_images, train_labels, test_images, test_labels = load_mnist_data(
        data_dir=os.path.join(ROOT, "mnist_data"),
        num_train=num_train,
        num_test=num_test
    )

    visualize_samples(
        train_images, train_labels,
        save_path=os.path.join(CHARTS, "sample_digits.png")
    )

    visualize_pixel_matrix(
        train_images[0], train_labels[0],
        save_path=os.path.join(CHARTS, "pixel_matrix.png")
    )

    # ═══════════════════════════════════════════════════
    # Giai đoạn 2-5: Huấn luyện & Kiểm thử
    # ═══════════════════════════════════════════════════
    history, conv, pool, softmax = train_and_test(
        train_images, train_labels,
        test_images, test_labels,
        num_epochs=3,
        learning_rate=0.005,
        output_dir=OUTPUT,
    )

    # ═══════════════════════════════════════════════════
    # Lưu biểu đồ kết quả
    # ═══════════════════════════════════════════════════
    visualize_training(
        history,
        save_path=os.path.join(CHARTS, "training_history.png")
    )
    visualize_predictions(
        conv, pool, softmax, test_images, test_labels,
        save_path=os.path.join(CHARTS, "predictions.png")
    )

    print("\n🎉 Hoàn tất! Xem kết quả trong thư mục output/")


if __name__ == "__main__":
    main()
