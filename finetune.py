# finetune.py
import os
import sys
import numpy as np

# Đảm bảo import được module src/ dù chạy từ thư mục nào
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model, save_model, predict
from predict import load_image

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "output", "model", "cnn_weights.npz")
OUTPUT_DIR = os.path.join(ROOT, "output")

def main():
    if len(sys.argv) < 3:
        print("❌ Thiếu đối số! Cách dùng: python3 finetune.py <đường_dẫn_ảnh> <nhãn_đúng>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        correct_label = int(sys.argv[2])
    except ValueError:
        print("❌ Nhãn đúng phải là số nguyên từ 0 đến 9.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa có model weights để tinh chỉnh! Hãy chạy huấn luyện trước.")
        sys.exit(1)

    # 1. Tải model và ảnh
    conv, pool, softmax = load_model(MODEL_PATH)
    image = load_image(image_path)

    # 2. Dự đoán thử lần đầu tiên để xem trạng thái trước khi sửa
    pred_before, conf_before, probs_before = predict(conv, pool, softmax, image)
    loss_before = -np.log(max(1e-15, probs_before[correct_label]))

    print(f"BEFORE_PRED:{pred_before}")
    print(f"BEFORE_CONF:{conf_before * 100:.1f}")
    print(f"BEFORE_LOSS:{loss_before:.4f}")

    # 3. Thực hiện huấn luyện bổ sung (Fine-tuning)
    # Lặp lại 5 lần (epoch cục bộ) để mạng "học thuộc" nét vẽ này ngay lập tức
    lr = 0.02
    for epoch in range(5):
        # Forward pass
        out = conv.forward(image)
        out = pool.forward(out)
        out = softmax.forward(out)
        
        # Tạo gradient cho hàm mất mát Cross-Entropy
        # dL/dout = 0 cho mọi lớp ngoại trừ lớp nhãn đúng = -1 / out[correct_label]
        gradient = np.zeros(10)
        gradient[correct_label] = -1 / max(1e-15, out[correct_label])
        
        # Backward pass & Cập nhật trọng số
        grad = softmax.backward(gradient, lr)
        grad = pool.backward(grad, lr)
        conv.backward(grad, lr)

    # 4. Dự đoán lại sau khi sửa đổi để xem kết quả mới
    pred_after, conf_after, probs_after = predict(conv, pool, softmax, image)
    loss_after = -np.log(max(1e-15, probs_after[correct_label]))

    print(f"AFTER_PRED:{pred_after}")
    print(f"AFTER_CONF:{conf_after * 100:.1f}")
    print(f"AFTER_LOSS:{loss_after:.4f}")

    # 5. Lưu lại model weights đã được tinh chỉnh
    save_model(conv, pool, softmax, os.path.join(OUTPUT_DIR, "model"))
    print("✅ Cập nhật trọng số mạng nơ-ron thành công!")

if __name__ == "__main__":
    main()
