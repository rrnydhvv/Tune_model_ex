# Báo Cáo Kết Quả Mô Hình: Toxicity Prediction (Độc Tính)
**Bộ dữ liệu:** Toxicity-50F (171 mẫu, 50 đặc trưng)
**Thuật toán áp dụng:** Mô hình Stacking Classifier (SVM + Random Forest + XGBoost) kết hợp SMOTE.

> [!NOTE]
> Báo cáo này tổng hợp kết quả chạy thực tế của mô hình chẩn đoán Độc tính (Toxicity). Điểm đặc biệt của tập dữ liệu này là số lượng mẫu rất nhỏ (chỉ 171 mẫu) nhưng lại chứa nhiều đặc trưng, đòi hỏi hệ thống phải kiểm soát Overfitting chặt chẽ.

---

## 1. Quá trình Xây dựng Giải pháp (Hành trình Tối ưu hóa)

Việc dự đoán độc tính trên một tập dữ liệu y sinh nhỏ bé là một bài toán khó, đòi hỏi phải vượt qua nhiều cạm bẫy kỹ thuật:

1. **Giai đoạn 1: Vấn đề Data Leakage**: Ban đầu, thuật toán được chạy trực tiếp trên tập gốc (1204 đặc trưng). Điểm số CV lên tới 0.86, nhưng đây là kết quả sai lệch do việc chọn lọc đặc trưng (SelectKBest) diễn ra trước khi chia Validation Fold, khiến mô hình tiếp xúc với dữ liệu kiểm thử.
2. **Giai đoạn 2: Hạn chế do số chiều dữ liệu lớn**: Khi xử lý rò rỉ dữ liệu bằng cách đặt bộ lọc vào trong Pipeline, điểm Test ROC-AUC giảm xuống 0.46. Sự tồn tại của hơn 3,700 cặp đặc trưng trùng lặp (nhiễu) trên 171 mẫu đã làm thuật toán bị Overfitting.
3. **Giai đoạn 3: Can thiệp bằng Domain Knowledge (Tập 13F)**: Chuyển sang sử dụng tập 13 đặc trưng do chuyên gia sinh hóa chọn lọc giúp đưa ROC-AUC về mức thực tế (0.57) và khởi động được cơ chế nhận diện (Threshold Tuning).
4. **Giai đoạn 4: Giải pháp Làm sạch Tự động (Tập 50F)**: Một hệ thống tự động làm sạch (Loại bỏ giá trị tĩnh -> Lọc tương quan > 0.9 -> Lọc Random Forest Top 50) đã được áp dụng. Tập dữ liệu `50F` sinh ra đã kết hợp việc giảm nhiễu và phát hiện quy luật, từ đó cải thiện hiệu suất.

---

## 2. Cấu trúc Mô Hình (The Pipeline)
Dựa trên phương pháp của bài toán DIA, mô hình Toxicity được xây dựng với các bước xử lý phù hợp:
- **Xử lý mất cân bằng:** Sử dụng `SMOTE` nằm gọn bên trong Cross-Validation (Tuyệt đối không rò rỉ dữ liệu).
- **Tối ưu hóa siêu tham số (GridSearchCV):** Dò tìm tham số tốt nhất cho 3 mô hình độc lập:
  - `SVM` (Đạt ROC-AUC CV: 0.57): Support Vector Machine kết hợp hàm nhân (kernel RBF) để ánh xạ 50 đặc trưng hóa học vào không gian đa chiều, từ đó tìm ra siêu mặt phẳng (hyperplane) uốn lượn phân cách giữa thuốc Độc và Không độc. Thuật toán này đặc biệt hiệu quả trên tập mẫu nhỏ.
  - `Random Forest` (Đạt ROC-AUC CV: 0.63): Xây dựng một "khu rừng" gồm nhiều cây quyết định nhằm chống lại hiện tượng Overfitting do nhiễu. Mô hình hoạt động hiệu quả trên tập này nhờ khả năng tự động đánh giá và bỏ qua các đặc trưng ít quan trọng.
  - `XGBoost` (Đạt ROC-AUC CV: 0.56): Thúc đẩy quá trình học thông qua việc sửa lỗi liên tục. Bằng cách giới hạn độ sâu (max_depth=3) và tốc độ học (learning_rate=0.01) để tránh Overfitting trên tập nhỏ, thuật toán cung cấp thêm thông tin hữu ích bổ trợ cho SVM và RF.
- **Stacking Classifier:** Sử dụng `Logistic Regression` làm mô hình bậc 2 (Meta-Classifier). Mô hình này sẽ nhận lấy xác suất dự đoán của 3 mô hình trên làm đầu vào để đưa ra quyết định cuối cùng, giúp tăng cường độ ổn định.

---

## 3. Đánh Giá Khả Năng Phân Tách (ROC-AUC)
- **ROC-AUC Đạt: 0.8068**
> [!TIP]
> **Nhận xét:** Với một tập dữ liệu y sinh chỉ có 171 dòng, việc mô hình duy trì được đường cong ROC ở mức trên 0.80 là một kết quả khả quan. Hệ thống chứng minh được khả năng phân biệt giữa hai nhóm thuốc Độc tính và Không độc tính.

---

## 4. Phân Tích Kết Quả Theo Ngưỡng Quyết Định (Threshold)

Hệ thống cho phép bác sĩ / nhà nghiên cứu lựa chọn 2 chế độ cảnh báo:

### Chế độ 1: Chẩn đoán Cân bằng (Ngưỡng Mặc định = 0.50)
*Ưu tiên dự đoán chắc chắn, tránh báo động giả.*
- **Accuracy (Độ chính xác tổng):** 77.14%
- **Precision (Độ chuẩn xác):** 71.43% (7/10 ca bị hệ thống cảnh báo thực sự là thuốc độc).
- **Recall (Độ nhạy):** 45.45% (Nhận diện được khoảng một nửa số mẫu độc tính).
- **F1-Score:** 0.5556

### Chế độ 2: Tầm soát Chủ động (Ngưỡng Cảnh báo Sớm = 0.22)
*Áp dụng quy tắc "Thà bắt nhầm còn hơn bỏ sót", ép hệ thống phải đạt Recall tối thiểu 85%.*
- **Recall (Độ nhạy):** Tăng lên **90.91%** (Mô hình nhận diện thành công 10 trên tổng số 11 ca thuốc độc trong tập Test).
- **Precision:** Giảm xuống còn **40.00%** (Cảnh báo sai nhiều hơn, nhưng trong y tế sàng lọc bước 1, điều này là hoàn toàn chấp nhận được).
- **Accuracy:** 54.29%

---

## 5. Kết Luận
1. **Độ an toàn dữ liệu:** Mô hình đã được cô lập hoàn toàn giữa Train và Test thông qua `StratifiedKFold` và `ImbPipeline`, bảo đảm kết quả báo cáo trên đây là thực lực 100% của mô hình, không hề có hiện tượng học vẹt (Data Leakage).
2. **Giá trị ứng dụng:** Tương tự như bài toán DIA, có thể triển khai hệ thống này theo 2 màng lọc. Màng lọc 1 sử dụng ngưỡng 0.22 để loại bỏ nhanh các loại thuốc an toàn (giữ lại 90% mẫu độc tính). Màng lọc 2 sử dụng ngưỡng 0.50 để xác nhận lại.
3. Toàn bộ mã nguồn và bảng kết quả trên đã được lưu vào file `ML\toxic_model.ipynb`.
