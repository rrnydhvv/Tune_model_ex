# Báo Cáo Hành Trình Tối Ưu Hóa Thuật Toán Học Máy
**Bài toán:** Dự đoán thuốc gây tự miễn dịch (Drug Induced Autoimmunity - DIA)
**Dữ liệu:** 477 mẫu huấn luyện, 198 đặc trưng phân tử (RDKit)

> [!NOTE]
> Báo cáo này tổng hợp quá trình nâng cấp mô hình từ một phiên bản Baseline cơ bản lên một hệ thống Ensemble, nhằm giải quyết các thách thức về mất cân bằng dữ liệu và đáp ứng các mục tiêu y tế khác nhau.

---

## 1. Thách Thức Ban Đầu
- **Dữ liệu nhỏ & Rộng:** Tập huấn luyện có 477 mẫu nhưng mang đến 198 đặc trưng, dễ gây hiện tượng Nhiễu (Noise) và Quá khớp (Overfitting).
- **Mất cân bằng lớp (Class Imbalance):** Tỷ lệ nhãn Risk (Nguy cơ) thấp hơn hẳn so với nhãn Safe (An toàn). Các thuật toán thông thường có xu hướng bỏ qua nhãn Risk để đạt Accuracy cao giả tạo.

---

## 2. Các Giai Đoạn Tiến Hóa Của Mô Hình

### Giai đoạn 1: Thiết lập Baseline (Random Forest)
- Áp dụng `RandomForestClassifier` cơ bản kết hợp với trích xuất đặc trưng `SelectFromModel`.
- **Kết quả:** Xảy ra hiện tượng Overfitting nhẹ. Mô hình học rất tốt trên tập Train nhưng không khái quát hóa tốt trên tập Test.

### Giai đoạn 2: Khắc phục Mất Cân Bằng bằng SMOTE
- Bổ sung thuật toán nội suy `SMOTE` để sinh thêm dữ liệu nhân tạo cho nhóm thiểu số.
- Thiết lập siêu tham số chuẩn mực để khống chế độ sâu của cây (`max_depth`).
- **Kết quả:** Mô hình đạt trạng thái cân bằng. F1-Score chạm mốc **0.67**, với Precision = 67% và Recall = 67%. Đây là một nền tảng vững chắc.

### Giai đoạn 3: Tối ưu Độ Chuẩn Xác bằng XGBoost
- Nâng cấp từ Random Forest của Scikit-Learn lên `XGBRFClassifier` để tạo ra một thuật toán mạnh mẽ hơn với khả năng bắt ranh giới phức tạp trên CPU.
- Mục tiêu: Tối ưu hóa khả năng nhận diện chính xác.
- **Kết quả:** 
  - Tại ngưỡng xác suất 0.54, mô hình đạt **Precision 89%**. 
  - Nghĩa là 89% các loại thuốc bị cảnh báo thực sự mang rủi ro. F1-Score đạt **0.69**. 
  - **Ứng dụng:** Có thể sử dụng làm bước xác nhận cuối cùng nhằm loại bỏ báo động giả.

### Giai đoạn 4: Tối ưu Độ Nhạy (Max Recall trên CPU)
- Khi mục tiêu y khoa thay đổi sang "Thà bắt nhầm còn hơn bỏ sót", chúng tôi thay đổi chiến thuật tập trung vào thuật toán.
- Ép trọng số cực gắt (`class_weight={0: 1, 1: 10}`) và tối ưu hóa hệ số **F2-Score**.
- **Kết quả:**
  - Mô hình đạt **Recall 100%** ở ngưỡng tự nhiên và **93.3%** ở ngưỡng F2 tối ưu. Nhận diện được hầu hết các trường hợp rủi ro.
  - **Đánh đổi:** Precision sụt giảm xuống còn khoảng 26-32%.
  - **Ứng dụng:** Phù hợp làm màng lọc thô (Screening) ở tuyến đầu.

---

## 3. Giai Đoạn 3: Mô Hình Kết Hợp (Ensemble)
> [!IMPORTANT]
> Thay vì phải chịu sự đánh đổi giữa Precision và Recall, chúng tôi đã sử dụng kỹ thuật **Stacking Classifier** để kết hợp các mô hình.

**Cấu trúc Stacking:**
1. **XGBoost:** Mô hình giúp cải thiện Precision. Thuật toán này sử dụng kỹ thuật Gradient Boosting, xây dựng tuần tự các cây quyết định mà cây sau sẽ cố gắng sửa lỗi của cây trước. Nhờ vậy, nó hiệu quả trong việc tìm ra các ranh giới phân loại phức tạp, hạn chế dương tính giả (False Positive).
2. **Random Forest:** Mô hình hỗ trợ tăng cường Recall. Hoạt động dựa trên cơ chế Bagging, sinh ra hàng trăm cây quyết định độc lập trên các tập dữ liệu phụ. Tính đa dạng này giúp thuật toán bao quát không gian dữ liệu, ít bị Overfitting và hạn chế bỏ sót các ca rủi ro (giảm False Negative).
3. **Logistic Regression (L2):** Chuyên gia xử lý dữ liệu nhiều chiều. Đóng vai trò như một màng lọc tuyến tính ổn định. Việc áp dụng hình phạt L2 (Ridge Regularization) giúp thuật toán kiểm soát tốt 198 đặc trưng phân tử mà không bị nhiễu.
4. **Meta-Classifier (Logistic Regression):** Đóng vai trò mô hình bậc 2 (meta-classifier) để tổng hợp và đưa ra quyết định cuối cùng.

### Kết Quả Đạt Được
- **Chỉ số Phân Tách (ROC-AUC):** Đạt **0.9089**.
- **Trạng thái cân bằng (Ngưỡng 0.50):** 
  - Đạt **F1-Score: 0.73**.
  - Precision: **80%** | Recall: **67%** | Accuracy: **87.5%**.
- **Trạng thái cảnh báo sớm (Ngưỡng 0.15):** 
  - Chỉ cần hạ nhẹ ngưỡng quyết định, mô hình đạt **Recall 90%** (Nhận diện được 27/30 ca).
  - Vẫn duy trì được **Precision 56.25%** và **Accuracy 80%**. Cải thiện sự sụt giảm Precision của phiên bản Random Forest trước đó.

> [!TIP]
> **Kết Luận:** Hệ thống Stacking Classifier này là phiên bản phù hợp cho bộ dữ liệu DIA. Toàn bộ mã nguồn đã được lưu vào file `ML\drug_model.ipynb`.
