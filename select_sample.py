# select_sample.py
import os
import sys
import numpy as np
from PIL import Image

# Đảm bảo import được module src/ dù chạy từ thư mục nào
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_mnist_data

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    # Tải một lượng nhỏ dữ liệu test để lấy mẫu ngẫu nhiên cho nhanh
    _, _, test_images, test_labels = load_mnist_data(
        data_dir=os.path.join(ROOT, "mnist_data"),
        num_train=10, num_test=100
    )
    idx = np.random.randint(len(test_images))
    image = test_images[idx]
    label = int(test_labels[idx])
    
    # Lưu thành temp_predict.png
    temp_out = os.path.join(ROOT, "output", "temp_predict.png")
    os.makedirs(os.path.dirname(temp_out), exist_ok=True)
    
    # Chuyển đổi từ float [0, 1] sang uint8 [0, 255]
    img_to_save = Image.fromarray((image * 255).astype(np.uint8))
    img_to_save.save(temp_out)
    
    # In ra nhãn để Next.js API bắt được
    print(f"LABEL:{label}")

if __name__ == "__main__":
    main()
