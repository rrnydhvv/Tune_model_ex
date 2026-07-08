# Báo Cáo Tổng Quan Mô Hình Học Máy

Dự án này bao gồm hai mô hình học máy phân loại phục vụ cho mục đích dự đoán y sinh (Độc tính và Nguy cơ Tự miễn dịch). Dưới đây là kết quả của các mô hình và cách mà luồng thuật toán tác động để giải quyết các khó khăn của dữ liệu.

---

## 1. Cách Các Thuật Toán Tác Động Đến Bài Toán (Luồng Xử Lý Tuần Tự)

Dữ liệu y sinh thường có đặc thù là **số lượng mẫu rất ít** nhưng **số lượng đặc trưng lại cực kỳ nhiều** (ví dụ: cấu trúc phân tử), dẫn đến tình trạng mất cân bằng lớp và rất dễ bị học vẹt (Overfitting). Để giải quyết, mô hình hoạt động theo các bước sau:

*   **Bước 1 - Chuẩn hoá dữ liệu (StandardScaler):** Đưa toàn bộ các đặc trưng hóa học về cùng một thang đo. Giúp các thuật toán (đặc biệt là SVM và Logistic Regression) không bị thiên vị bởi các đặc trưng có giá trị quá lớn.
*   **Bước 2 - Xử lý mất cân bằng (SMOTE):** Vì số lượng mẫu độc tính/rủi ro thường rất ít so với mẫu an toàn, thuật toán `SMOTE` sẽ nội suy và sinh thêm các dữ liệu nhân tạo cho nhóm thiểu số. Nhờ vậy, mô hình sẽ học được cách nhận diện nhóm rủi ro thay vì bỏ qua chúng.
*   **Bước 3 - Chọn lọc đặc trưng (Feature Selection):** (Áp dụng trên bài toán DIA). Sử dụng `RandomForest` làm màng lọc để tự động đánh giá tầm quan trọng của hàng trăm đặc trưng phân tử, qua đó vứt bỏ các đặc trưng nhiễu, giúp giải quyết lời nguyền số chiều.
*   **Bước 4 - Trí tuệ tập thể (Stacking Classifier):** Thay vì dùng 1 thuật toán, hệ thống kết hợp nhiều thuật toán lại để bù trừ điểm yếu cho nhau:
    *   **XGBoost:** Tìm kiếm các ranh giới siêu phức tạp, mang lại độ Chuẩn xác (Precision) rất cao.
    *   **Random Forest:** Cấu trúc nhiều cây quyết định giúp quét diện rộng, chống Overfitting tốt và đẩy mạnh độ Nhạy (Recall).
    *   **SVM / Logistic Regression:** Hoạt động rất ổn định trên không gian dữ liệu nhiều chiều.
    *   **Meta-Classifier:** Cuối cùng, một mô hình Tổng tư lệnh (`Logistic Regression`) sẽ lấy dự đoán của tất cả các mô hình trên để đưa ra quyết định chốt hạ.
*   **Bước 5 - Dịch chuyển ngưỡng quyết định (Threshold Tuning):** Mặc định, máy tính kết luận "CÓ BỆNH" nếu xác suất > 50%. Tuy nhiên trong y khoa, chúng ta ưu tiên "Thà bắt nhầm còn hơn bỏ sót". Bằng cách hạ ngưỡng quyết định (xuống 0.15 hoặc 0.22), thuật toán bị ép phải cảnh báo ngay cả khi rủi ro còn thấp, giúp tóm gọn thành công >90% ca bệnh.

---

## 2. Kết Quả Mô Hình Dự Đoán Độc Tính (Toxicity)
*Bài toán: Phân loại một loại thuốc có khả năng gây độc hay không dựa trên 50 đặc trưng hoá học.*
- **Dữ liệu:** 171 mẫu (sau khi được tự động làm sạch).
- **Thuật toán:** Stacking (SVM + Random Forest + XGBoost).
- **Chỉ số đánh giá tổng thể (ROC-AUC):** **0.8068**
- **Khả năng Ứng dụng:**
  - **Chế độ cân bằng (Ngưỡng 0.50):** Độ chính xác 77.1%. Ưu tiên chẩn đoán chắc chắn, ít báo động giả.
  - **Chế độ cảnh báo sớm (Ngưỡng 0.22):** Độ nhạy (Recall) đạt **90.9%**. Hệ thống tóm gọn gần như toàn bộ thuốc độc, phù hợp làm màng lọc bước 1.

---

## 3. Kết Quả Mô Hình Dự Đoán Tự Miễn Dịch (DIA)
*Bài toán: Dự đoán nguy cơ gây tự miễn dịch (Drug Induced Autoimmunity) của thuốc.*
- **Dữ liệu:** 477 mẫu (198 đặc trưng phân tử).
- **Thuật toán:** SMOTE + SelectFromModel + Stacking (XGBoost + Random Forest + Logistic Regression).
- **Chỉ số đánh giá tổng thể (ROC-AUC):** **0.9089** (Mức độ cực kỳ xuất sắc).
- **Khả năng Ứng dụng:**
  - **Chế độ cân bằng (Ngưỡng 0.50):** F1-Score đạt 0.73, Accuracy 87.5%.
  - **Chế độ cảnh báo sớm (Ngưỡng 0.15):** Độ nhạy (Recall) đạt **90.0%**. Hệ thống phát hiện được 27/30 ca bệnh, giữ được độ chuẩn xác (Precision) rất tốt ở mức 56.25%.
