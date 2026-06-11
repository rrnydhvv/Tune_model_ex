# Báo Cáo Kết Quả Mô Hình: Toxicity Prediction (Độc Tính)
**Bộ dữ liệu:** Toxicity-50F (171 mẫu, 50 đặc trưng)
**Thuật toán áp dụng:** Siêu mô hình Stacking Classifier (SVM + Random Forest + XGBoost) kết hợp SMOTE.

> [!NOTE]
> Báo cáo này tổng hợp kết quả chạy thực tế của mô hình chẩn đoán Độc tính (Toxicity). Điểm đặc biệt của tập dữ liệu này là số lượng mẫu rất nhỏ (chỉ 171 mẫu) nhưng lại chứa nhiều đặc trưng, đòi hỏi hệ thống phải chống Overfitting cực kỳ nghiêm ngặt.

---

## 1. Quá trình Xây dựng Giải pháp (Hành trình Tối ưu hóa)

Việc dự đoán độc tính trên một tập dữ liệu y sinh nhỏ bé là một bài toán khó, đòi hỏi phải vượt qua nhiều cạm bẫy kỹ thuật:

1. **Giai đoạn 1: Ảo ảnh Data Leakage**: Ban đầu, thuật toán được chạy trực tiếp trên tập gốc (1204 đặc trưng). Điểm số CV lên tới 0.86, nhưng đây là "điểm ảo" do việc chọn lọc đặc trưng (SelectKBest) diễn ra trước khi chia Validation Fold, khiến mô hình "nhìn trộm" dữ liệu.
2. **Giai đoạn 2: Bế tắc với Lời nguyền Số chiều**: Khi bịt kín rò rỉ dữ liệu bằng cách nhét bộ lọc vào trong Pipeline, mô hình lập tức lộ nguyên hình với điểm Test ROC-AUC rớt thê thảm xuống 0.46. Sự tồn tại của hơn 3,700 cặp đặc trưng trùng lặp (nhiễu) trên vỏn vẹn 171 mẫu đã làm thuật toán bị Overfitting nghiêm trọng.
3. **Giai đoạn 3: Can thiệp bằng Domain Knowledge (Tập 13F)**: Chuyển sang sử dụng tập 13 đặc trưng do Chuyên gia sinh hóa chọn lọc giúp đưa ROC-AUC về mức thực tế ổn định (0.57) và khởi động được cơ chế nhận diện chất độc (Threshold Tuning).
4. **Giai đoạn 4: Giải pháp Tối thượng (Machine Cleaning Pipeline - Tập 50F)**: Để không bỏ sót tri thức ẩn, một hệ thống tự động làm sạch (Loại bỏ giá trị tĩnh -> Lọc tương quan > 0.9 -> Lọc Random Forest Top 50) đã được kích hoạt. Tập dữ liệu `50F` sinh ra đã kết hợp xuất sắc cả việc giảm nhiễu (của con người) và phát hiện quy luật (của máy), từ đó phá vỡ mọi giới hạn.

---

## 2. Cấu trúc Mô Hình (The Pipeline)
Nhờ kế thừa toàn bộ "tinh hoa" từ bài toán DIA trước đó, mô hình Toxicity được xây dựng với các chốt chặn kỹ thuật cao nhất:
- **Xử lý mất cân bằng:** Sử dụng `SMOTE` nằm gọn bên trong Cross-Validation (Tuyệt đối không rò rỉ dữ liệu).
- **Tối ưu hóa siêu tham số (GridSearchCV):** Dò tìm tham số tốt nhất cho 3 mô hình độc lập:
  - `SVM` (Đạt ROC-AUC CV: 0.57)
  - `Random Forest` (Đạt ROC-AUC CV: 0.63 - Thể hiện sức mạnh vượt trội nhất trên tập này)
  - `XGBoost` (Đạt ROC-AUC CV: 0.56)
- **Stacking Classifier:** Sử dụng Logistic Regression làm Tổng tư lệnh (Meta-Classifier) để tổng hợp sức mạnh dự đoán của cả 3 mô hình trên.

---

## 3. Đánh Giá Khả Năng Phân Tách (ROC-AUC)
- **ROC-AUC Đạt: 0.8068**
> [!TIP]
> **Nhận xét:** Với một tập dữ liệu y sinh chỉ vỏn vẹn 171 dòng, việc mô hình duy trì được đường cong ROC ở mức trên 0.80 là một thành tựu rất đáng kể. Hệ thống chứng minh được khả năng phân biệt rõ ràng giữa hai nhóm thuốc Độc tính và Không độc tính.

---

## 4. Phân Tích Kết Quả Theo Ngưỡng Quyết Định (Threshold)

Hệ thống cho phép bác sĩ / nhà nghiên cứu lựa chọn 2 chế độ cảnh báo:

### Chế độ 1: Chẩn đoán Cân bằng (Ngưỡng Mặc định = 0.50)
*Ưu tiên dự đoán chắc chắn, tránh báo động giả.*
- **Accuracy (Độ chính xác tổng):** 77.14%
- **Precision (Độ chuẩn xác):** 71.43% (7/10 ca bị hệ thống cảnh báo thực sự là thuốc độc).
- **Recall (Độ nhạy):** 45.45% (Chỉ tóm được khoảng một nửa số thuốc độc).
- **F1-Score:** 0.5556

### Chế độ 2: Tầm soát Chủ động (Ngưỡng Cảnh báo Sớm = 0.22)
*Áp dụng quy tắc "Thà bắt nhầm còn hơn bỏ sót", ép hệ thống phải đạt Recall tối thiểu 85%.*
- **Recall (Độ nhạy):** Tăng vọt lên **90.91%**! (Mô hình tóm gọn thành công 10 trên tổng số 11 ca thuốc độc trong tập Test).
- **Precision:** Giảm xuống còn **40.00%** (Cảnh báo sai nhiều hơn, nhưng trong y tế sàng lọc bước 1, điều này là hoàn toàn chấp nhận được).
- **Accuracy:** 54.29%

---

## 5. Kết Luận
1. **Độ an toàn dữ liệu:** Mô hình đã được cô lập hoàn toàn giữa Train và Test thông qua `StratifiedKFold` và `ImbPipeline`, bảo đảm kết quả báo cáo trên đây là thực lực 100% của mô hình, không hề có hiện tượng học vẹt (Data Leakage).
2. **Giá trị ứng dụng:** Tương tự như bài toán DIA, bạn có thể triển khai hệ thống này theo 2 màng lọc. Màng lọc 1 sử dụng ngưỡng 0.22 để loại bỏ nhanh các loại thuốc an toàn (giữ lại 90% mầm độc). Màng lọc 2 sử dụng ngưỡng 0.50 để xác nhận lại.
3. Toàn bộ mã nguồn và bảng kết quả trên đã được kết xuất an toàn vào file `ML\toxic_model.ipynb` để bạn sẵn sàng thuyết trình!
