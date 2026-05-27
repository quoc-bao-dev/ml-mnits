"""
train_visual.py – Trực quan hóa quá trình huấn luyện CNN step-by-step
=====================================================================
Chạy forward + backward trên 1 ảnh mẫu MNIST, xuất JSON chi tiết
cho Frontend visualize từng bước toán học.

Chạy: python3 train_visual.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_mnist_data
from src.model import Conv3x3, MaxPool2, Softmax

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output", "visual_logs")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder xử lý các kiểu dữ liệu NumPy."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return np.round(obj, 6).tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        return super().default(obj)


def capture_conv_forward(conv, image):
    """Chạy Conv3x3 forward và thu thập dữ liệu trực quan."""
    h, w = image.shape
    output = np.zeros((h - 2, w - 2, conv.num_filters))
    conv.last_input = image

    for im_region, i, j in conv.iterate_regions(image):
        output[i, j] = np.sum(im_region * conv.filters, axis=(1, 2))

    return {
        "id": 1,
        "type": "conv_forward",
        "phase": "forward",
        "title": "Tích chập Conv3x3 (Forward)",
        "description": "Trượt 8 bộ lọc 3×3 qua ảnh 28×28 → Output 26×26×8",
        "formula": "Y[i, j, f] = Σ_m Σ_n X[i+m, j+n] · W_f[m, n]",
        "input_shape": [int(h), int(w)],
        "output_shape": [int(h - 2), int(w - 2), int(conv.num_filters)],
        "filters": conv.filters,
        "output": output,
    }, output


def capture_maxpool_forward(pool, conv_output):
    """Chạy MaxPool2 forward và thu thập dữ liệu trực quan."""
    pool.last_input = conv_output
    h, w, nf = conv_output.shape
    output = np.zeros((h // 2, w // 2, nf))

    max_masks = np.zeros_like(conv_output)  # 1 tại vị trí max, 0 ở nơi khác

    for im_region, i, j in pool.iterate_regions(conv_output):
        output[i, j] = np.amax(im_region, axis=(0, 1))
        for f in range(nf):
            max_val = output[i, j, f]
            for i2 in range(2):
                for j2 in range(2):
                    if im_region[i2, j2, f] == max_val:
                        max_masks[i * 2 + i2, j * 2 + j2, f] = 1
                        break
                else:
                    continue
                break

    return {
        "id": 2,
        "type": "maxpool_forward",
        "phase": "forward",
        "title": "Gộp cực đại MaxPool 2×2 (Forward)",
        "description": "Lấy giá trị lớn nhất trong mỗi vùng 2×2, giảm kích thước 50%",
        "formula": "Y[i, j, f] = max(X[2i:2i+2, 2j:2j+2, f])",
        "input_shape": list(conv_output.shape),
        "output_shape": list(output.shape),
        "output": output,
        # Chỉ lưu max_masks cho filter 0 để giảm kích thước JSON
        "max_mask_f0": max_masks[:, :, 0],
    }, output


def capture_softmax_forward(softmax_layer, pool_output, label):
    """Chạy Softmax forward và thu thập dữ liệu trực quan."""
    probs = softmax_layer.forward(pool_output)
    prediction = int(np.argmax(probs))
    loss = -np.log(probs[label])

    return {
        "id": 3,
        "type": "softmax_forward",
        "phase": "forward",
        "title": "Lớp Softmax (Forward)",
        "description": "Flatten → Nhân ma trận → Softmax → 10 xác suất",
        "formula_linear": "z = W · x + b",
        "formula_softmax": "p_i = exp(z_i - max(z)) / Σ exp(z_k - max(z))",
        "input_length": int(np.prod(pool_output.shape)),
        "logits": softmax_layer.last_totals,
        "probabilities": probs,
        "prediction": prediction,
        "confidence": float(probs[prediction]),
        "correct_label": int(label),
        "is_correct": prediction == int(label),
    }, probs, float(loss)


def capture_loss(probs, label, loss_val):
    """Thu thập dữ liệu hàm mất mát."""
    return {
        "id": 4,
        "type": "loss",
        "phase": "forward",
        "title": "Hàm mất mát Cross-Entropy",
        "description": f"Đo sai lệch giữa xác suất dự đoán và nhãn đúng (label={label})",
        "formula": "L = -ln(p_correct)",
        "loss_value": loss_val,
        "correct_label": int(label),
        "correct_prob": float(probs[label]),
        "probabilities": probs,
    }


def capture_backward(conv, pool, softmax_layer, probs, label, lr):
    """Chạy backward pass và thu thập dữ liệu gradient."""
    filters_before = conv.filters.copy()

    # Initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / probs[label]

    # Softmax backward
    grad_pool = softmax_layer.backward(gradient, lr)

    target_one_hot = np.zeros(10)
    target_one_hot[label] = 1.0
    d_L_d_z = probs - target_one_hot

    softmax_step = {
        "id": 5,
        "type": "softmax_backward",
        "phase": "backward",
        "title": "Lan truyền ngược qua Softmax",
        "description": "Tính ∂L/∂z qua Jacobian Softmax, cập nhật W và b",
        "formula": "∂L/∂W = ∂L/∂z ⊗ xᵀ  ;  W ← W - lr · ∂L/∂W",
        "initial_gradient": gradient,
        "d_L_d_z": d_L_d_z,
        "output_gradient_shape": list(grad_pool.shape) if grad_pool is not None else None,
        "grad_pool": grad_pool,
    }

    # MaxPool backward
    grad_conv = pool.backward(grad_pool, lr)

    maxpool_step = {
        "id": 6,
        "type": "maxpool_backward",
        "phase": "backward",
        "title": "Lan truyền ngược qua MaxPool",
        "description": "Gradient chỉ chảy về vị trí đã có giá trị max (winner-take-all)",
        "formula": "∂L/∂X[i,j,f] = ∂L/∂Y[i÷2, j÷2, f] nếu X[i,j,f] là max, ngược lại = 0",
        "output_gradient_shape": list(grad_conv.shape),
        # Gradient heatmap cho filter 0 (giữ nguyên để tránh break giao diện cũ nếu có)
        "gradient_heatmap_f0": grad_conv[:, :, 0],
        "grad_conv": grad_conv,
    }

    # Conv backward
    conv.backward(grad_conv, lr)
    filters_after = conv.filters.copy()
    filter_deltas = filters_after - filters_before

    conv_step = {
        "id": 7,
        "type": "conv_backward",
        "phase": "backward",
        "title": "Lan truyền ngược qua Conv3x3",
        "description": "Tính gradient bộ lọc và cập nhật trọng số qua Gradient Descent",
        "formula": "∂L/∂W_f = Σ X[i+m, j+n] · ∂L/∂Y[i,j,f]  ;  W ← W - lr · ∂L/∂W",
        "filters_before": filters_before,
        "filters_after": filters_after,
        "filter_deltas": filter_deltas,
        "learning_rate": lr,
    }

    return [softmax_step, maxpool_step, conv_step]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dữ liệu - chỉ cần vài ảnh
    train_images, train_labels, _, _ = load_mnist_data(
        data_dir=os.path.join(ROOT, "mnist_data"),
        num_train=10, num_test=1
    )

    image = train_images[0]
    label = int(train_labels[0])

    # Khởi tạo mạng
    conv = Conv3x3(8)
    pool = MaxPool2()
    softmax_layer = Softmax(13 * 13 * 8, 10)
    lr = 0.005

    steps = []

    # === FORWARD ===
    print("STEP:1:conv_forward")
    conv_step, conv_out = capture_conv_forward(conv, image)
    steps.append(conv_step)

    print("STEP:2:maxpool_forward")
    pool_step, pool_out = capture_maxpool_forward(pool, conv_out)
    steps.append(pool_step)

    print("STEP:3:softmax_forward")
    sm_step, probs, loss_val = capture_softmax_forward(softmax_layer, pool_out, label)
    steps.append(sm_step)

    print("STEP:4:loss")
    loss_step = capture_loss(probs, label, loss_val)
    steps.append(loss_step)

    # === BACKWARD ===
    print("STEP:5-7:backward")
    backward_steps = capture_backward(conv, pool, softmax_layer, probs, label, lr)
    steps.extend(backward_steps)

    # Build output
    result = {
        "input_image": image,
        "label": label,
        "learning_rate": lr,
        "total_steps": len(steps),
        "steps": steps,
    }

    output_path = os.path.join(OUTPUT_DIR, "visual_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, cls=NumpyEncoder, ensure_ascii=False)

    print(f"VISUAL_DATA_PATH:{output_path}")
    print(f"VISUAL_DATA_SIZE:{os.path.getsize(output_path)}")
    print("VISUAL_DONE")


if __name__ == "__main__":
    main()
