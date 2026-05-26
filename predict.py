"""
predict.py – Dự đoán chữ số từ ảnh
=====================================
Cách dùng:
  python3 predict.py <đường_dẫn_ảnh>
  python3 predict.py                    # Dùng ảnh mẫu từ MNIST

Ví dụ:
  python3 predict.py my_digit.png
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model, predict

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "output", "model", "cnn_weights.npz")


def load_image(image_path):
    """
    Đọc ảnh từ file và chuyển về dạng 28×28 grayscale, giá trị [0, 1].
    Hỗ trợ PNG, JPG, BMP...
    """
    from PIL import Image

    img = Image.open(image_path).convert('L')   # Chuyển sang grayscale
    img = img.resize((28, 28))                   # Resize về 28×28

    # Chuyển thành numpy array, chuẩn hóa
    arr = np.array(img, dtype=np.float64) / 255.0

    # MNIST dùng nền đen (0) và chữ trắng (1).
    # Nếu ảnh đầu vào có nền trắng → đảo ngược
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def load_sample_from_mnist():
    """Lấy ngẫu nhiên 1 ảnh từ bộ test MNIST."""
    from src.data import load_mnist_data
    _, _, test_images, test_labels = load_mnist_data(
        data_dir=os.path.join(ROOT, "mnist_data"),
        num_train=100, num_test=100
    )
    idx = np.random.randint(len(test_images))
    return test_images[idx], test_labels[idx]


def main():
    # Kiểm tra model đã train chưa
    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa có model! Hãy chạy train trước:")
        print("   python3 train.py")
        sys.exit(1)

    # Tải model
    conv, pool, softmax = load_model(MODEL_PATH)

    # Xác định ảnh đầu vào
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy file: {image_path}")
            sys.exit(1)
        print(f"\n[INFO] Đọc ảnh: {image_path}")
        image = load_image(image_path)
        true_label = None
    else:
        print("\n[INFO] Không có ảnh đầu vào → dùng ảnh mẫu từ MNIST")
        image, true_label = load_sample_from_mnist()

    # Dự đoán
    pred, conf, probs = predict(conv, pool, softmax, image)

    # Hiển thị kết quả
    print("\n" + "=" * 50)
    print(f"  🔢 KẾT QUẢ DỰ ĐOÁN")
    print("=" * 50)
    print(f"  Chữ số dự đoán : {pred}")
    print(f"  Độ tin cậy     : {conf * 100:.1f}%")
    if true_label is not None:
        status = "✅ ĐÚNG" if pred == true_label else "❌ SAI"
        print(f"  Nhãn thật      : {true_label}  ({status})")

    print(f"\n  Xác suất từng chữ số:")
    for i in range(10):
        bar = "█" * int(probs[i] * 40)
        marker = " ◄" if i == pred else ""
        print(f"    {i}: {probs[i]*100:5.1f}% {bar}{marker}")
    print("=" * 50)

    # Lưu ảnh kết quả
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        out_path = os.path.join(ROOT, "output", "charts", "prediction_single.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.imshow(image, cmap='gray')
        title = f"Dự đoán: {pred} ({conf*100:.1f}%)"
        if true_label is not None:
            title += f"\nNhãn thật: {true_label}"
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.axis('off')

        colors = ['#2196F3'] * 10
        colors[pred] = '#4CAF50'
        ax2.barh(range(10), [p * 100 for p in probs], color=colors)
        ax2.set_yticks(range(10))
        ax2.set_xlabel("Xác suất (%)")
        ax2.set_title("Phân phối xác suất", fontsize=13, fontweight='bold')
        ax2.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n[INFO] Đã lưu kết quả → {out_path}")
    except Exception as e:
        print(f"\n[WARN] Không thể lưu hình: {e}")


if __name__ == "__main__":
    main()
