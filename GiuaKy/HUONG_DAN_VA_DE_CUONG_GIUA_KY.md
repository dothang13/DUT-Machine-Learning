# ĐỀ CƯƠNG ÔN TẬP & CẨM NANG THỰC CHIẾN THI GIỮA KỲ HỌC MÁY
**Sinh viên:** Nguyễn Đỗ Thắng - **MSSV:** 102230046 - **Lớp:** 23T_NHAT1  
**Tài liệu tham khảo tại chỗ (Phòng thi):** Sử dụng các file trong thư mục `GiuaKy/`

---

## 1. TỔNG QUAN BÀI THI & CÁC FILE ĐÃ CHUẨN BỊ SẴN
Trong thư mục `GiuaKy/`, Bách đã chuẩn bị đầy đủ 100% các nội dung theo đúng 3 bức ảnh note dặn dò của bạn bè:
1. **File Notebook hoàn chỉnh:** [ON_TAP_GIUA_KY_HOC_MAY_102230046.ipynb](file:///f:/DAIHOC/Nam4-ky1/HocMay/GiuaKy/ON_TAP_GIUA_KY_HOC_MAY_102230046.ipynb)
   - Chứa toàn bộ code tự viết (from scratch bằng NumPy thuần).
   - Chạy được độc lập từng cell hoặc bấm `Run All`.
   - Đầy đủ biểu đồ so sánh Side-by-Side (Ảnh gốc, Custom, Sklearn).
   - Đầy đủ bảng số liệu so sánh rõ ràng từng tiêu chí.
2. **File Thư viện module tiện ích:** [ml_midterm_toolkit.py](file:///f:/DAIHOC/Nam4-ky1/HocMay/GiuaKy/ml_midterm_toolkit.py)
   - Chứa sẵn các class `CustomKMeans`, `CustomFCM`, `CustomKNN` và các hàm phân tích, vẽ biểu đồ tự động.
3. **Ảnh mẫu thực hành:** [sample_color_image.jpg](file:///f:/DAIHOC/Nam4-ky1/HocMay/GiuaKy/sample_color_image.jpg)

---

## 2. TRẢ LỜI CÁC CÂU HỎI LÝ THUYẾT & VẤN ĐÁP CỐT LÕI

### Câu 1: Phân biệt bản chất giữa Ảnh màu (RGB) và Ảnh trắng đen (Grayscale) trong phân cụm ảnh?
* **Về số chiều đặc trưng:**
  * **Ảnh màu (RGB):** Kích thước $(H, W, 3)$. Mỗi điểm ảnh là 1 vector 3 chiều $[R, G, B]^T \in [0, 255]^3$. Khi phân cụm, ma trận dữ liệu có kích thước $(H \cdot W, 3)$. Tâm cụm $\mu_k$ là 1 vector màu 3D $[\bar{R}, \bar{G}, \bar{B}]^T$. Phân cụm giúp gom các điểm có **màu sắc tương đồng** (Color Quantization / Palette).
  * **Ảnh trắng đen / Grayscale:** Kích thước $(H, W)$. Mỗi điểm ảnh chỉ là 1 giá trị vô hướng cường độ sáng $I \in [0, 255]$ (1 chiều). Khi phân cụm, ma trận dữ liệu có kích thước $(H \cdot W, 1)$. Tâm cụm $\mu_k$ là một mức xám $\bar{I}$. Phân cụm giúp phân vùng ảnh theo **độ tương phản sáng tối (vùng tối, vùng xám, vùng sáng)**.
* **Về công thức khoảng cách:**
  * Ảnh màu dùng khoảng cách không gian 3D: $d = \sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (B_1 - B_2)^2}$.
  * Ảnh xám dùng độ chênh lệch 1D: $d = |I_1 - I_2|$.

---

### Câu 2: Tại sao chọn K, chọn C? Làm sao chứng minh bằng biểu đồ?
Thầy rất hay hỏi câu này, trong notebook đã vẽ sẵn 3 biểu đồ:
1. **Với K-Means (Phương pháp Elbow & Silhouette):**
   * **Phương pháp khuỷu tay (Elbow Method):** Vẽ đồ thị giá trị $WCSS$ (Inertia - tổng bình phương khoảng cách nội cụm) theo các giá trị $K = 2, 3, \dots, 8$.
     * $WCSS = \sum_{k=1}^K \sum_{x \in C_k} ||x - \mu_k||^2$.
     * Điểm khuỷu tay (Elbow point) là điểm mà sau đó tốc độ giảm của WCSS bắt đầu chậm lại rõ rệt. Đó là số cụm $K$ tối ưu (đạt sự cân bằng giữa độ nén dữ liệu và số lượng cụm).
   * **Chỉ số Silhouette Score:** Đo lường độ gắn kết nội cụm so với độ tách biệt ngoại cụm, nhận giá trị trong $[-1, 1]$. Điểm $K$ có Silhouette Score cao nhất là điểm phân cụm rõ ràng nhất.
2. **Với Fuzzy C-Means (Hệ số FPC & Silhouette):**
   * **Hệ số phân hoạch mờ FPC (Fuzzy Partition Coefficient):** $FPC(c) = \frac{1}{N} \sum_{k=1}^N \sum_{i=1}^c u_{ik}^2 \in [\frac{1}{c}, 1]$. Giá trị FPC càng tiến gần về 1 chứng tỏ các điểm dữ liệu thuộc về các cụm rõ ràng, ít bị phân vân giữa các ranh giới mờ.
   * Kết hợp chọn giá trị $c$ có FPC cao và Silhouette Score tối ưu.
3. **Với K-NN (Đồ thị Tỷ lệ lỗi Error Rate / Accuracy theo K):**
   * **Nếu $K$ quá nhỏ ($K = 1$):** Mô hình bị **Overfitting** (quá khớp), rất nhạy cảm với nhiễu và ngoại lai.
   * **Nếu $K$ quá lớn ($K \to N$):** Mô hình bị **Underfitting** (thiếu khớp), ranh giới quyết định bị làm mờ, lớp chiếm đa số lấn át các lớp khác.
   * **Biểu đồ:** Vẽ đồ thị $Error\ Rate = 1 - Accuracy$ trên tập kiểm thử theo $K = 1, 3, 5, \dots, 21$. Giá trị $K$ lẻ có tỷ lệ lỗi thấp nhất là $K$ tối ưu.

---

## 3. CÁCH XỬ LÝ NHANH TRONG PHÒNG THI (KHI THẦY PHÁT ĐỀ MỚI)

### Trường hợp 1: Thầy phát file ảnh màu mới (ví dụ `anh_de_thi.png`)
* Trong file [ON_TAP_GIUA_KY_HOC_MAY_102230046.ipynb](file:///f:/DAIHOC/Nam4-ky1/HocMay/GiuaKy/ON_TAP_GIUA_KY_HOC_MAY_102230046.ipynb), tìm đến **Cell 5**, sửa dòng:
  ```python
  IMAGE_PATH = 'anh_de_thi.png'  # Đổi thành tên file hoặc đường dẫn ảnh thầy phát
  ```
* Bấm chạy lại các Cell tiếp theo, toàn bộ phân đoạn ảnh màu, ảnh xám $K=5$, $K=7$, biểu đồ so sánh Custom vs Sklearn sẽ tự động vẽ ra ngay lập tức.

### Trường hợp 2: Thầy phát file dữ liệu bảng mới (ví dụ `dataset.csv` hoặc `data.xlsx`)
* Trong file [ON_TAP_GIUA_KY_HOC_MAY_102230046.ipynb](file:///f:/DAIHOC/Nam4-ky1/HocMay/GiuaKy/ON_TAP_GIUA_KY_HOC_MAY_102230046.ipynb), tìm đến **Cell 10**, bỏ comment dòng:
  ```python
  df_custom = pd.read_csv('dataset.csv')
  # Tách đặc trưng X và nhãn y (nếu có):
  X_raw = df_custom.iloc[:, :-1].values
  y_true = df_custom.iloc[:, -1].values
  ```
* Bấm chạy tiếp các Cell sau: biểu đồ Elbow chọn K, chọn C, phân lớp K-NN và bảng so sánh với Sklearn sẽ cập nhật tự động.
