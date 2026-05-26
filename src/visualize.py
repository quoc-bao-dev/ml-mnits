"""
src/visualize.py – Visualization Utilities
============================================
Các hàm vẽ biểu đồ và minh họa kết quả.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def visualize_samples(images, labels, save_path="output/charts/sample_digits.png"):
    """Hiển thị 20 ảnh mẫu (2 hàng × 10 cột) cùng nhãn."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    fig.suptitle("Mẫu dữ liệu MNIST – 20 ảnh chữ số viết tay đầu tiên",
                 fontsize=14, fontweight='bold', y=1.02)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap='gray')
        ax.set_title(f"Label: {labels[idx]}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Đã lưu → {save_path}")


def visualize_pixel_matrix(image, label, save_path="output/charts/pixel_matrix.png"):
    """Minh họa song song ảnh gốc và ma trận giá trị pixel."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(image, cmap='gray')
    ax1.set_title(f"Ảnh gốc (Label = {label})", fontsize=12, fontweight='bold')
    ax1.axis('off')

    center = image[7:21, 7:21]
    im = ax2.imshow(center, cmap='hot', interpolation='nearest')
    ax2.set_title("Ma trận pixel (vùng trung tâm 14×14)", fontsize=12, fontweight='bold')

    for i in range(center.shape[0]):
        for j in range(center.shape[1]):
            val = center[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax2.text(j, i, f"{val:.1f}", ha='center', va='center',
                     fontsize=6, color=color)

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Đã lưu → {save_path}")


def visualize_training(history, save_path="output/charts/training_history.png"):
    """Vẽ biểu đồ Loss và Accuracy qua các epoch."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Kết quả Huấn luyện CNN thuần NumPy",
                 fontsize=14, fontweight='bold')

    ax1.plot(epochs, history["loss"], 'ro-', linewidth=2, markersize=8,
             label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Loss giảm dần → Mạng đang học", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["accuracy"], 'bo-', linewidth=2, markersize=8,
             label="Train Accuracy")
    ax2.plot(epochs, history["test_accuracy"], 'gs-', linewidth=2, markersize=8,
             label="Test Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy tăng dần → Backpropagation hoạt động đúng",
                  fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Đã lưu → {save_path}")


def visualize_predictions(conv, pool, softmax, test_images, test_labels,
                          save_path="output/charts/predictions.png"):
    """Hiển thị 20 ảnh test cùng dự đoán (Xanh = Đúng, Đỏ = Sai)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from src.model import predict

    fig, axes = plt.subplots(2, 10, figsize=(16, 4.5))
    fig.suptitle("Dự đoán trên tập kiểm thử (Xanh = Đúng, Đỏ = Sai)",
                 fontsize=14, fontweight='bold', y=1.02)

    for idx, ax in enumerate(axes.flat):
        image = test_images[idx]
        label = test_labels[idx]

        pred, conf, _ = predict(conv, pool, softmax, image)

        ax.imshow(image, cmap='gray')
        color = 'green' if pred == label else 'red'
        ax.set_title(f"Pred:{pred} ({conf*100:.0f}%)\nTrue:{label}",
                     fontsize=8, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Đã lưu → {save_path}")
