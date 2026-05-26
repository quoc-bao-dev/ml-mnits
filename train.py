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


def main():
    # ═══════════════════════════════════════════════════
    # Giai đoạn 1: Chuẩn bị dữ liệu
    # ═══════════════════════════════════════════════════
    train_images, train_labels, test_images, test_labels = load_mnist_data(
        data_dir=os.path.join(ROOT, "mnist_data")
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
