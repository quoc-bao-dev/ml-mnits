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
    Đọc ảnh từ file, tự động cắt biên thừa (bounding box),
    căn giữa (centering) và chuyển về dạng 28×28 grayscale chuẩn MNIST, giá trị [0, 1].
    """
    from PIL import Image

    img = Image.open(image_path).convert('L')   # Chuyển sang grayscale
    
    # MNIST dùng nền đen (0) và chữ trắng (255).
    # Kiểm tra nếu là nền trắng chữ đen thì đảo ngược màu.
    arr = np.array(img, dtype=np.float64)
    if arr.mean() > 127:
        arr = 255.0 - arr
        img = Image.fromarray(arr.astype(np.uint8))

    # Tìm Bounding Box của nét vẽ để loại bỏ phần lề đen thừa
    bbox = img.getbbox()
    if bbox is not None:
        # Cắt lấy riêng vùng nét vẽ
        cropped = img.crop(bbox)
        
        # Giữ tỉ lệ ảnh vẽ gốc và resize về tối đa 20x20 để tránh chữ số quá to sát viền 28x28
        w, h = cropped.size
        ratio = min(20.0 / w, 20.0 / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Tạo khung ảnh đen 28x28 và dán ảnh nét vẽ vào tâm
        centered_img = Image.new('L', (28, 28), 0)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        centered_img.paste(resized, (offset_x, offset_y))
        img = centered_img
    else:
        # Nếu ảnh trống trơn, resize trực tiếp về 28x28
        img = img.resize((28, 28))

    # Chuyển thành numpy array, chuẩn hóa về [0, 1]
    arr = np.array(img, dtype=np.float64) / 255.0
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

    visualize = "--visualize" in sys.argv
    if visualize:
        sys.argv.remove("--visualize")

    # Xác định ảnh đầu vào
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy file: {image_path}")
            sys.exit(1)
        print(f"\n[INFO] Đọc ảnh: {image_path}")
        image = load_image(image_path)
        
        # Nhãn thật có thể được truyền qua tham số thứ 2 (đối với ảnh mẫu MNIST)
        true_label = None
        if len(sys.argv) > 2:
            try:
                true_label = int(sys.argv[2])
            except ValueError:
                pass
    else:
        print("\n[INFO] Không có ảnh đầu vào → dùng ảnh mẫu từ MNIST")
        image, true_label = load_sample_from_mnist()
        # Lưu lại ảnh mẫu ngẫu nhiên để server/frontend có thể đọc và hiển thị
        try:
            from PIL import Image
            temp_out = os.path.join(ROOT, "output", "temp_predict.png")
            os.makedirs(os.path.dirname(temp_out), exist_ok=True)
            img_to_save = Image.fromarray((image * 255).astype(np.uint8))
            img_to_save.save(temp_out)
        except Exception as e:
            print(f"[WARN] Không thể lưu ảnh mẫu tạm thời: {e}")



    # Dự đoán
    if visualize:
        from train_visual import capture_conv_forward, capture_maxpool_forward, capture_softmax_forward, NumpyEncoder
        import json
        
        steps = []
        conv_step, conv_out = capture_conv_forward(conv, image)
        steps.append(conv_step)
        
        pool_step, pool_out = capture_maxpool_forward(pool, conv_out)
        steps.append(pool_step)
        
        # Softmax forward
        # Determine a label to use for loss calculation in capture_softmax_forward
        temp_probs = softmax.forward(pool_out)
        pred = int(np.argmax(temp_probs))
        label_to_use = true_label if true_label is not None else pred
        
        sm_step, probs, _ = capture_softmax_forward(softmax, pool_out, label_to_use)
        steps.append(sm_step)
        
        conf = float(probs[pred])
        
        result = {
            "input_image": image,
            "label": label_to_use,
            "learning_rate": 0.005,
            "total_steps": len(steps),
            "steps": steps,
        }
        
        visual_path = os.path.join(ROOT, "output", "visual_logs", "predict_visual_data.json")
        os.makedirs(os.path.dirname(visual_path), exist_ok=True)
        with open(visual_path, "w", encoding="utf-8") as f:
            json.dump(result, f, cls=NumpyEncoder, ensure_ascii=False)
        print(f"VISUAL_DATA_PATH:{visual_path}")
    else:
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
