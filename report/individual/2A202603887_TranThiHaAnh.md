# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603887
- **Họ và tên:**Tran Thi Ha Anh
- **SEED** (in ở ô đầu notebook): 8919

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| 1. BoW + LogReg | 0.902 | 0.788 | 0.247 | 0.000 |
| 2. BoW + MLP | 0.929 | 0.919 | 6.236 | 0.000 |
| 3a. MiniLM + kNN | 0.888 | 0.974 | 122.217 | 0.000 |
| 3b. MiniLM + LogReg | 0.944 | 0.950 | 37.951 | 0.000 |
| 3c. MiniLM + MLP head | 0.949 | 0.974 | 40.890 | 0.000 |

## 3. Dự đoán và thực tế
- ml_f1: Dự đoán F1 khoảng 0.8, precision cao hơn recall vì dữ liệu mất cân bằng và ngưỡng 0.5 có thể bỏ sót spam. Kết quả F1 test 0.902, precision 1.000, recall 0.822. Dự đoán đúng chiều chênh lệch precision–recall nhưng đánh giá thấp F1 khoảng 0.102; đặc trưng BoW kết hợp LogReg phân biệt nhiều tin spam tốt hơn dự kiến.

- dl_relu: Dự đoán bỏ ReLU sẽ làm F1 giảm vì hai lớp tuyến tính liên tiếp không tạo ra tính phi tuyến mới. Thực tế mô hình không ReLU có F1 = 0.929, bằng F1 test của MLP có ReLU khi làm tròn ba chữ số. Lý giải về tính tuyến tính vẫn đúng, nhưng trên dữ liệu và cấu hình huấn luyện này lợi thế của ReLU không thể hiện rõ qua F1.

- hard_eval: Dự đoán mô hình đơn giản đạt khoảng 50–60%, encoder 70–80%, LLM 90–95% trên 50 case khó. Tuy nhiên ô stress test chưa có bảng kết quả thực thi trong notebook được cung cấp. Vì vậy chưa thể xác định dự đoán lệch bao nhiêu; đây là dự đoán cần kiểm chứng, không phải kết quả thực tế. Một đối chiếu đã có dữ liệu là emb_tricky: dự đoán MiniLM + LogReg tốt hơn BoW, thực tế đúng 6/6 so với 4/6.

## 4. Phân tích 50 case khó

Hai case cụ thể có trong bộ dữ liệu để phân tích cơ chế rủi ro:
- Case #1 — ham_looks_spam: “Congrats!!! You WON the bet lol, I owe you a beer. Call me later” (nhãn thật: ham). BoW có nguy cơ dự đoán nhầm spam vì các token Congrats, WON, Call thường xuất hiện trong tin quảng cáo. Muốn phân loại đúng phải nhận ra đây là lời nhắn giữa bạn bè về một vụ cá cược.
- Case #2 — ham_looks_spam: “FREE pizza in the office kitchen right now, come quick before it's gone” (nhãn thật: ham). Các từ FREE, now, quick tạo tín hiệu giống spam, nhưng ngữ cảnh thực tế là đồng nghiệp báo có pizza miễn phí trong văn phòng. Mô hình phụ thuộc từ khóa có nguy cơ dương tính giả.

## 5. Thử thách cá nhân

Dùng knn_predict với k = 1, 5, 25, báo F1 và giải thích vì sao k quá nhỏ hoặc quá lớn có thể bất lợi.

Cách làm: Giữ nguyên embedding MiniLM, dữ liệu train/test và hàm knn_predict; chỉ thay tham số k. Dùng f1_score(y_te, preds) để đánh giá trên tập test.
k	F1 test
1	0.8925
5	0.8877
25	0.8973

Giải thích: Khi k = 1, dự đoán phụ thuộc một hàng xóm nên dễ bị nhiễu hoặc một mẫu không đại diện chi phối. Khi k quá lớn, nhiều hàng xóm xa hơn và có thể khác lớp được đưa vào biểu quyết, làm mờ tín hiệu của các ví dụ gần nhất; với dữ liệu mất cân bằng, điều này còn có thể thiên về lớp ham. Trong ba mức đã thử, k = 25 tốt nhất, nhưng mức chênh nhỏ và không chứng minh rằng tăng k mãi sẽ tốt hơn.

## 6. Phương án cho sản phẩm thật
Yêu cầu 10 triệu SMS/ngày tương đương khoảng 115.7 tin/giây trung bình, với p95 < 50 ms/tin. Bốn mô hình đáp ứng ngưỡng p95 trong notebook là BoW + LogReg (0.247 ms), BoW + MLP (6.236 ms), MiniLM + LogReg (37.951 ms) và MiniLM + MLP head (40.890 ms). MiniLM + kNN (122.217 ms) không đáp ứng.

Lựa chọn đề xuất: Dùng BoW + MLP làm bộ lọc mặc định vì F1 test 0.929, p95 6.236 ms và tốc độ đo cả lô khoảng 5,815 tin/giây, tạo khoảng dự phòng độ trễ lớn. Với tin có xác suất spam ở vùng không chắc chắn, ví dụ [0.3, 0.7], chuyển sang MiniLM + MLP head để đánh giá sâu hơn (F1 test 0.949, p95 riêng mô hình 40.890 ms). Chỉ áp dụng nhánh hai sau khi đo p95 toàn tuyến, vì cộng hai độ trễ p95 riêng lẻ không bảo đảm p95 hệ thống dưới 50 ms.

Đánh đổi: Cách lai tăng độ phức tạp triển khai, sử dụng RAM/CPU cho encoder và có thể tạo thêm độ trễ cho tin không chắc chắn. Ngược lại, đa số tin được xử lý nhanh bằng mô hình nhẹ. 

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
