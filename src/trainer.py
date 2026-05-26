"""
src/trainer.py – Giai đoạn 4-5: Huấn luyện & Kiểm thử
=======================================================
Vòng lặp huấn luyện CNN với logging ra file và console.
"""

import numpy as np
import os
import time
from datetime import datetime

from src.model import Conv3x3, MaxPool2, Softmax, save_model


class Logger:
    """Ghi log đồng thời ra console và file."""

    def __init__(self, log_dir="output/logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"training_{timestamp}.log")
        self.file = open(self.log_path, "w", encoding="utf-8")
        self.log(f"=== Training Log – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def log(self, message):
        """Ghi một dòng ra cả console và file."""
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def train_and_test(train_images, train_labels, test_images, test_labels,
                   num_epochs=3, learning_rate=0.005,
                   output_dir="output"):
    """
    Huấn luyện mạng CNN và ghi log chi tiết.

    Pipeline mỗi ảnh:
      Image(28×28) → Conv3x3(8) → MaxPool2 → Softmax(10) → Prediction

    Returns
    -------
    history : dict
        Lịch sử loss và accuracy qua từng epoch.
    conv, pool, softmax : các lớp đã huấn luyện.
    """
    logger = Logger(os.path.join(output_dir, "logs"))

    # ── Khởi tạo mạng ──
    conv = Conv3x3(8)
    pool = MaxPool2()
    softmax = Softmax(13 * 13 * 8, 10)

    logger.log("=" * 60)
    logger.log("       HUẤN LUYỆN CNN THUẦN NUMPY")
    logger.log("=" * 60)
    logger.log(f"  Kiến trúc : Conv3x3(8) → MaxPool2 → Softmax(10)")
    logger.log(f"  Epochs    : {num_epochs}")
    logger.log(f"  Lr        : {learning_rate}")
    logger.log(f"  Train set : {len(train_images)} ảnh")
    logger.log(f"  Test set  : {len(test_images)} ảnh")
    logger.log("=" * 60)

    history = {"loss": [], "accuracy": [], "test_accuracy": []}
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.log(f"\n{'─' * 60}")
        logger.log(f"  EPOCH {epoch + 1}/{num_epochs}")
        logger.log(f"{'─' * 60}")

        # Xáo trộn dữ liệu
        perm = np.random.permutation(len(train_images))
        imgs = train_images[perm]
        lbls = train_labels[perm]

        total_loss = 0
        num_correct = 0

        for i, (image, label) in enumerate(zip(imgs, lbls)):
            # ── FORWARD ──
            out = conv.forward(image)
            out = pool.forward(out)
            out = softmax.forward(out)

            # Cross-Entropy Loss
            loss = -np.log(out[label])
            total_loss += loss

            predicted = np.argmax(out)
            if predicted == label:
                num_correct += 1

            # ── BACKWARD ──
            gradient = np.zeros(10)
            gradient[label] = -1 / out[label]

            grad = softmax.backward(gradient, learning_rate)
            grad = pool.backward(grad, learning_rate)
            conv.backward(grad, learning_rate)

            # Log mỗi 100 ảnh
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                acc = num_correct / (i + 1) * 100
                logger.log(f"    [{i+1:4d}/{len(train_images)}]  "
                           f"Loss: {avg_loss:.4f}  |  Acc: {acc:.1f}%")

        # ── Thống kê epoch ──
        epoch_loss = total_loss / len(train_images)
        epoch_acc = num_correct / len(train_images) * 100
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        # ── Kiểm thử ──
        test_correct = 0
        for image, label in zip(test_images, test_labels):
            out = conv.forward(image)
            out = pool.forward(out)
            out = softmax.forward(out)
            if np.argmax(out) == label:
                test_correct += 1
        test_acc = test_correct / len(test_images) * 100
        history["test_accuracy"].append(test_acc)

        elapsed = time.time() - epoch_start
        logger.log(f"\n  ▸ Epoch {epoch+1} Summary ({elapsed:.1f}s):")
        logger.log(f"    Train Loss    : {epoch_loss:.4f}")
        logger.log(f"    Train Accuracy: {epoch_acc:.1f}%")
        logger.log(f"    Test Accuracy : {test_acc:.1f}%")

    # ── Lưu model ──
    save_model(conv, pool, softmax, os.path.join(output_dir, "model"))

    total_time = time.time() - total_start
    logger.log(f"\n{'=' * 60}")
    logger.log(f"  ✅ HUẤN LUYỆN HOÀN TẤT ({total_time:.1f}s)")
    logger.log(f"{'=' * 60}")
    logger.log(f"  Final Train Accuracy : {history['accuracy'][-1]:.1f}%")
    logger.log(f"  Final Test Accuracy  : {history['test_accuracy'][-1]:.1f}%")
    logger.log(f"  Final Loss           : {history['loss'][-1]:.4f}")
    logger.log(f"  Log file             : {logger.log_path}")
    logger.log(f"{'=' * 60}")

    logger.close()
    return history, conv, pool, softmax
