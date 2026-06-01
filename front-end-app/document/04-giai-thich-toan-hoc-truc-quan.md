


## Bài toán: Dạy máy tính nhận ra chữ số viết tay

Cho máy tính xem 1000 ảnh chữ số viết tay, mỗi ảnh đã biết nhãn (0–9). Làm sao để máy học được cách nhận ra chữ số từ những ảnh đó?

Câu trả lời: **thử — đo sai — sửa — lặp lại.** Đó là toàn bộ bí mật của việc huấn luyện mạng nơ-ron.

---

## Lớp Conv: "Bộ lọc tìm đặc trưng"

Hình dung bạn đang tìm chữ "1" trong ảnh. Chữ "1" có đặc trưng gì? — một nét thẳng đứng. Bạn có thể dùng một tấm "kính lọc" hình 3×3 được thiết kế để sáng lên khi gặp nét thẳng đứng, tối đi khi không có.

Lớp Conv làm đúng điều đó — nhưng thay vì bạn tự thiết kế kính lọc, **mạng tự học ra các kính lọc hữu ích** trong quá trình huấn luyện. 8 filter trong đồ án này sẽ tự học 8 loại đặc trưng khác nhau.

--- 

## Lớp MaxPool: "Nén thông tin, giữ cái quan trọng"

Sau Conv, ta có một ảnh đặc trưng khá lớn (26×26). MaxPool chia ảnh thành các ô 2×2 và chỉ giữ lại **giá trị lớn nhất** mỗi ô — tức là chỉ giữ "tín hiệu mạnh nhất" và bỏ phần còn lại.

Kết quả: ảnh nhỏ đi một nửa (13×13), nhưng thông tin quan trọng vẫn còn. Ít số hơn = tính toán nhanh hơn, mạng gọn hơn.

---

## Softmax: "Biến điểm số thành phần trăm"

Sau khi ảnh qua Conv và MaxPool, mạng tính ra 10 con số — một con số cho mỗi chữ số 0–9. Các số này có thể là 3.2, -1.5, 7.8... không phải xác suất.

Softmax biến 10 con số đó thành 10 phần trăm cộng lại bằng 100%. Số nào lớn hơn sẽ được phần trăm lớn hơn — theo kiểu "bình chọn có trọng số".

> Ví dụ: [3.2, -1.5, 7.8] → [8%, 1%, 91%] — mạng đang nghĩ chữ số thứ 3 (index 2) với 91% tự tin.

---

## Cross-Entropy Loss: "Điểm phạt khi đoán sai"

Sau khi có xác suất, ta cần đo "mạng đang sai bao nhiêu". Quy tắc đơn giản:

- Mạng đoán đúng với **xác suất cao** → phạt **ít** (loss nhỏ)
- Mạng đoán đúng với **xác suất thấp** → phạt **nhiều** (loss lớn)
- Mạng đoán đúng với **xác suất = 100%** → loss = 0, hoàn hảo

> Ví dụ: ảnh là chữ "3", mạng nói "tôi nghĩ là chữ 3 với 91% tự tin" → loss nhỏ. Nhưng nếu chỉ "12% tự tin" → loss lớn, cần sửa nhiều hơn.

---

## Backpropagation: "Truy ngược lỗi về từng trọng số"

Đây là bước quan trọng nhất. Sau khi biết loss, ta cần biết **ai chịu trách nhiệm** cho lỗi đó — cụ thể là mỗi trọng số (filter, weight) đã góp phần vào lỗi như thế nào.

Hình dung một dây chuyền sản xuất: nếu sản phẩm cuối bị lỗi, ta truy ngược từng công đoạn để tìm nguyên nhân. Backpropagation làm đúng vậy — đi ngược từ loss về từng lớp, tính xem mỗi tham số đóng góp bao nhiêu vào lỗi. Đó chính là **Quy tắc chuỗi (Chain Rule)** từ Giải tích.

---

## Gradient Descent: "Sửa từng bước nhỏ"

Sau khi biết mỗi trọng số đóng góp bao nhiêu vào lỗi, ta sửa nó theo hướng **giảm lỗi** — một bước nhỏ, không sửa quá mạnh để tránh phá vỡ những gì đã học được.

Bước nhỏ hay lớn do **learning rate** ($\eta = 0.005$) quyết định. Sửa xong một ảnh, chuyển sang ảnh tiếp theo — lặp đi lặp lại 1000 lần — mạng dần dần học được cách nhận ra chữ số.

---

## Tóm tắt một vòng huấn luyện

```
Ảnh đầu vào (28×28)
    ↓  Conv: tìm đặc trưng (cạnh, nét)
    ↓  MaxPool: nén, giữ cái quan trọng
    ↓  Softmax: ra 10 xác suất
    ↓  Loss: đo sai số
    ↓  Backprop: truy ngược lỗi về từng tham số
    ↓  Gradient Descent: sửa từng bước nhỏ
    → Lặp lại với ảnh tiếp theo
```

Sau 3 epoch trên 1000 ảnh train (~3000 bước), mạng đạt ~81% accuracy trên tập test MNIST — không phải vì được lập trình sẵn cách nhận chữ số, mà vì **tự học ra từ dữ liệu**.
