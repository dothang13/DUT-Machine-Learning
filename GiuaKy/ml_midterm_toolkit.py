"""
THƯ VIỆN THUẬT TOÁN HỌC MÁY ÔN TẬP GIỮA KỲ
Bao gồm:
1. CustomKMeans (from scratch) + Sklearn Comparison
2. CustomFCM (Fuzzy C-Means from scratch) + Skfuzzy Comparison
3. CustomKNN (from scratch) + Sklearn Comparison
4. Image Segmentation (Phân đoạn ảnh màu & ảnh mức xám)
5. Phương pháp chọn K, chọn C tối ưu (Elbow Method, Silhouette, Validation Curve)
6. Bảng so sánh và biểu đồ trực quan hóa Side-by-Side
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Thư viện scikit-learn để đối sánh
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, classification_report
try:
    import skfuzzy as fuzz
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


# ==============================================================================
# PHẦN 1: CUSTOM K-MEANS TỪ ĐẦU (FROM SCRATCH)
# ==============================================================================
class CustomKMeans:
    """
    Thuật toán K-Means phân cụm tự cài đặt hoàn toàn bằng NumPy.
    - Hỗ trợ khoảng cách: 'euclidean' hoặc 'manhattan'.
    - Khởi tạo tâm: 'random' hoặc 'k-means++' hoặc truyền mảng tâm ban đầu chỉ định.
    """
    def __init__(self, k=3, max_iters=300, tol=1e-4, metric='euclidean', random_state=42, init_centroids=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.metric = metric
        self.random_state = random_state
        self.init_centroids = init_centroids
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # WCSS (Tổng bình phương khoảng cách tới tâm cụm)
        self.n_iter_ = 0

    def _compute_distance(self, X, centroids):
        """
        Tính ma trận khoảng cách giữa các điểm X (N, D) và các tâm cụm (K, D).
        Trả về ma trận (N, K).
        """
        if self.metric == 'euclidean':
            # Euclidean distance: sqrt(sum((x - c)^2))
            return np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        elif self.metric == 'manhattan':
            # Manhattan distance: sum(|x - c|)
            return np.sum(np.abs(X[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Khoảng cách '{self.metric}' không được hỗ trợ. Chọn 'euclidean' hoặc 'manhattan'.")

    def _init_centroids(self, X):
        if self.init_centroids is not None:
            return np.array(self.init_centroids, dtype=float)
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        # Khởi tạo ngẫu nhiên k điểm từ X
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        return X[random_indices].astype(float)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.centroids = self._init_centroids(X)

        for it in range(self.max_iters):
            self.n_iter_ = it + 1
            # 1. Bước gán cụm (Assignment Step)
            dists = self._compute_distance(X, self.centroids)
            labels = np.argmin(dists, axis=1)

            # 2. Bước cập nhật tâm cụm (Update Step)
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.k):
                members = X[labels == j]
                if len(members) > 0:
                    if self.metric == 'manhattan':
                        # Với L1 norm, trung vị (median) tối thiểu hóa tổng độ lệch tuyệt đối
                        # Nhưng trong giáo trình cơ bản thường dùng mean, hỗ trợ cả 2 (ở đây dùng mean theo chuẩn K-Means)
                        new_centroids[j] = np.mean(members, axis=0)
                    else:
                        new_centroids[j] = np.mean(members, axis=0)
                else:
                    # Nếu cụm rỗng, tái tạo tâm tại 1 điểm bất kỳ
                    new_centroids[j] = X[np.random.choice(n_samples)]

            # Kiểm tra hội tụ
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = np.argmin(self._compute_distance(X, self.centroids), axis=1)
        # Tính Inertia / WCSS (Within-Cluster Sum of Squares)
        euc_dists = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        self.inertia_ = float(np.sum(euc_dists ** 2))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = self._compute_distance(X, self.centroids)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ==============================================================================
# PHẦN 2: CUSTOM FUZZY C-MEANS (FCM) TỪ ĐẦU (FROM SCRATCH)
# ==============================================================================
class CustomFCM:
    """
    Thuật toán Phân cụm mờ Fuzzy C-Means (FCM) từ đầu.
    - c: Số lượng cụm (clusters)
    - m: Fuzziness exponent (thường chọn m = 2.0)
    - max_iters: Số vòng lặp tối đa
    - tol: Ngưỡng dừng biến thiên ma trận thuộc tính U (epsilon)
    """
    def __init__(self, c=3, m=2.0, max_iters=300, tol=1e-5, random_state=42):
        self.c = c
        self.m = m
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.u = None           # Ma trận mức độ thuộc (N, c)
        self.labels_ = None     # Nhãn cứng thu được bằng argmax(u)
        self.obj_history = []   # Lịch sử hàm mục tiêu J_m
        self.n_iter_ = 0

    def _init_u(self, n_samples):
        np.random.seed(self.random_state)
        # Khởi tạo ma trận U ngẫu nhiên sao cho tổng mỗi hàng bằng 1
        u = np.random.dirichlet(np.ones(self.c), size=n_samples)
        return u

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.u = self._init_u(n_samples)

        for it in range(self.max_iters):
            self.n_iter_ = it + 1
            u_old = self.u.copy()

            # 1. Tính toán tâm cụm v_i:
            # v_i = sum_k (u_ik^m * x_k) / sum_k (u_ik^m)
            um = self.u ** self.m  # (N, c)
            self.centers = (um.T @ X) / np.sum(um, axis=0)[:, np.newaxis] # (c, n_features)

            # 2. Tính ma trận khoảng cách d_ij = ||x_j - v_i||
            # dists shape: (N, c)
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
            dists = np.fmax(dists, 1e-10) # Tránh chia cho 0

            # 3. Tính hàm mục tiêu J_m = sum_k sum_i (u_ik^m * d_ik^2)
            j_m = float(np.sum((self.u ** self.m) * (dists ** 2)))
            self.obj_history.append(j_m)

            # 4. Cập nhật ma trận mức độ thuộc u_ik:
            # u_ik = 1 / sum_j ( (d_ik / d_jk) ^ (2 / (m - 1)) )
            power = 2.0 / (self.m - 1.0)
            inv_dists = 1.0 / dists
            inv_dists_power = inv_dists ** power
            self.u = inv_dists_power / np.sum(inv_dists_power, axis=1, keepdims=True)

            # Kiểm tra điều kiện dừng: max(|u_new - u_old|) < tol
            if np.max(np.abs(self.u - u_old)) < self.tol:
                break

        self.labels_ = np.argmax(self.u, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
        dists = np.fmax(dists, 1e-10)
        power = 2.0 / (self.m - 1.0)
        inv_dists = 1.0 / dists
        inv_dists_power = inv_dists ** power
        u_new = inv_dists_power / np.sum(inv_dists_power, axis=1, keepdims=True)
        return np.argmax(u_new, axis=1)

    def compute_fpc(self):
        """Fuzzy Partition Coefficient (FPC) nằm trong [1/c, 1], càng gần 1 phân cụm càng rõ nét"""
        return float(np.sum(self.u ** 2) / len(self.u))


# ==============================================================================
# PHẦN 3: CUSTOM K-NEAREST NEIGHBORS (K-NN) TỪ ĐẦU (FROM SCRATCH)
# ==============================================================================
class CustomKNN:
    """
    Thuật toán phân lớp K-NN tự cài đặt hoàn toàn bằng NumPy.
    - Hỗ trợ khoảng cách: 'euclidean', 'manhattan'.
    - Hỗ trợ weights: 'uniform' (bỏ phiếu đều) hoặc 'distance' (nghịch đảo khoảng cách).
    """
    def __init__(self, k=3, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distance(self, x):
        if self.metric == 'euclidean':
            return np.linalg.norm(self.X_train - x, axis=1)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Metric '{self.metric}' không hợp lệ.")

    def predict_one(self, x):
        dists = self._compute_distance(x)
        k_indices = np.argsort(dists)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_dists = dists[k_indices]

        if self.weights == 'uniform':
            # Bỏ phiếu số đông (Majority Vote)
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            # Tie-break: chọn nhãn có tổng khoảng cách nhỏ hơn nếu hòa phiếu
            max_count = np.max(counts)
            candidates = labels[counts == max_count]
            if len(candidates) == 1:
                return candidates[0]
            else:
                candidate_dists = {c: np.sum(k_nearest_dists[k_nearest_labels == c]) for c in candidates}
                return min(candidate_dists, key=candidate_dists.get)
        elif self.weights == 'distance':
            weights = 1.0 / (k_nearest_dists + 1e-10)
            score_dict = {c: 0.0 for c in self.classes_}
            for lbl, w in zip(k_nearest_labels, weights):
                score_dict[lbl] += w
            return max(score_dict, key=score_dict.get)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x) for x in X])

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == np.asarray(y))


# ==============================================================================
# PHẦN 4: CÔNG CỤ CHỌN K VÀ CHỌN C TỐI ƯU (CÓ BIỂU ĐỒ MINH CHỨNG)
# ==============================================================================
def find_optimal_k_kmeans(X, k_range=range(2, 9), plot=True):
    """
    Tìm K tối ưu cho K-Means bằng 2 phương pháp:
    1. Elbow Method (Inertia / WCSS)
    2. Silhouette Score
    """
    inertias = []
    silhouettes = []
    k_list = list(k_range)

    for k in k_list:
        model = CustomKMeans(k=k, random_state=42)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)

    best_k_silhouette = k_list[np.argmax(silhouettes)]

    if plot:
        fig, ax1 = plt.subplots(1, 2, figsize=(13, 4.5))

        # Đồ thị Elbow
        ax1[0].plot(k_list, inertias, 'bo-', linewidth=2, markersize=7)
        ax1[0].set_title("Phương pháp Elbow (Inertia/WCSS theo K)", fontsize=12, fontweight='bold')
        ax1[0].set_xlabel("Số cụm K", fontsize=11)
        ax1[0].set_ylabel("Inertia (Tổng bình phương KC nội cụm)", fontsize=11)
        ax1[0].grid(True, linestyle='--', alpha=0.6)

        # Đồ thị Silhouette
        ax1[1].plot(k_list, silhouettes, 'rs--', linewidth=2, markersize=7)
        ax1[1].scatter([best_k_silhouette], [max(silhouettes)], color='green', s=140, zorder=5, 
                       label=f'K tối ưu Silhouette = {best_k_silhouette}')
        ax1[1].set_title("Chỉ số Silhouette Score theo K", fontsize=12, fontweight='bold')
        ax1[1].set_xlabel("Số cụm K", fontsize=11)
        ax1[1].set_ylabel("Silhouette Score (càng cao càng tốt)", fontsize=11)
        ax1[1].legend()
        ax1[1].grid(True, linestyle='--', alpha=0.6)

        plt.suptitle("PHÂN TÍCH LỰA CHỌN SỐ CỤM K TỐI ƯU (K-MEANS)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return {
        'k_list': k_list,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'best_k_silhouette': best_k_silhouette
    }


def find_optimal_c_fcm(X, c_range=range(2, 9), plot=True):
    """
    Tìm C tối ưu cho Fuzzy C-Means (FCM) bằng FPC và Silhouette Score.
    """
    fpcs = []
    silhouettes = []
    c_list = list(c_range)

    for c in c_list:
        model = CustomFCM(c=c, random_state=42)
        model.fit(X)
        fpcs.append(model.compute_fpc())
        sil = silhouette_score(X, model.labels_)
        silhouettes.append(sil)

    best_c = c_list[np.argmax(silhouettes)]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

        # Đồ thị FPC
        ax[0].plot(c_list, fpcs, 'mo-', linewidth=2, markersize=7)
        ax[0].set_title("Hệ số phân hoạch mờ FPC theo C", fontsize=12, fontweight='bold')
        ax[0].set_xlabel("Số cụm C", fontsize=11)
        ax[0].set_ylabel("FPC (càng gần 1 càng phân hoạch rõ)", fontsize=11)
        ax[0].grid(True, linestyle='--', alpha=0.6)

        # Đồ thị Silhouette
        ax[1].plot(c_list, silhouettes, 'gd--', linewidth=2, markersize=7)
        ax[1].scatter([best_c], [max(silhouettes)], color='red', s=140, zorder=5, label=f'C tối ưu = {best_c}')
        ax[1].set_title("Chỉ số Silhouette Score theo C", fontsize=12, fontweight='bold')
        ax[1].set_xlabel("Số cụm C", fontsize=11)
        ax[1].set_ylabel("Silhouette Score", fontsize=11)
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

        plt.suptitle("PHÂN TÍCH LỰA CHỌN SỐ CỤM C TỐI ƯU (FUZZY C-MEANS)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return {'c_list': c_list, 'fpcs': fpcs, 'silhouettes': silhouettes, 'best_c': best_c}


def find_optimal_k_knn(X_train, y_train, X_val, y_val, k_range=range(1, 21, 2), metric='euclidean', plot=True):
    """
    Tìm K tối ưu cho K-NN dựa trên Accuracy & Error Rate trên tập Validation/Test.
    """
    accuracies = []
    error_rates = []
    k_list = list(k_range)

    for k in k_list:
        knn = CustomKNN(k=k, metric=metric)
        knn.fit(X_train, y_train)
        acc = knn.score(X_val, y_val)
        accuracies.append(acc)
        error_rates.append(1.0 - acc)

    best_k = k_list[np.argmax(accuracies)]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

        # Đồ thị Error Rate
        ax[0].plot(k_list, error_rates, 'r-o', linewidth=2, markersize=7)
        ax[0].scatter([best_k], [min(error_rates)], color='black', s=140, zorder=5, label=f'K tối ưu (Error min) = {best_k}')
        ax[0].set_title(f"Tỷ lệ lỗi (Error Rate) theo K ({metric})", fontsize=12, fontweight='bold')
        ax[0].set_xlabel("Giá trị K lân cận", fontsize=11)
        ax[0].set_ylabel("Tỷ lệ phân loại sai (Error)", fontsize=11)
        ax[0].legend()
        ax[0].grid(True, linestyle='--', alpha=0.6)

        # Đồ thị Accuracy
        ax[1].plot(k_list, accuracies, 'b-s', linewidth=2, markersize=7)
        ax[1].scatter([best_k], [max(accuracies)], color='green', s=140, zorder=5, label=f'K tối ưu (Acc max) = {best_k}')
        ax[1].set_title(f"Độ chính xác (Accuracy) theo K ({metric})", fontsize=12, fontweight='bold')
        ax[1].set_xlabel("Giá trị K lân cận", fontsize=11)
        ax[1].set_ylabel("Accuracy", fontsize=11)
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

        plt.suptitle("PHÂN TÍCH LỰA CHỌN K TỐI ƯU CHO K-NN", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return {'k_list': k_list, 'accuracies': accuracies, 'best_k': best_k}


# ==============================================================================
# PHẦN 5: PHÂN ĐOẠN ẢNH MÀU & ẢNH TRẮNG ĐEN (IMAGE SEGMENTATION)
# ==============================================================================
def load_and_preprocess_image(image_input, is_gray=False, max_size=(300, 300)):
    """
    Đọc và tiền xử lý ảnh:
    - Nếu là chuỗi đường dẫn: mở ảnh. Nếu đã là np.ndarray thì giữ nguyên.
    - is_gray=True: chuyển sang Grayscale (ảnh 1 kênh).
    - is_gray=False: chuyển sang RGB (ảnh 3 kênh).
    - Resize ảnh nhỏ lại để tốc độ xử lý nhanh gọn phù hợp thi cử.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype('uint8'))
    else:
        img = image_input

    if is_gray:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    # Resize để phân đoạn nhanh
    if max_size is not None:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    img_np = np.array(img, dtype=float)
    return img_np


def segment_image_kmeans(image_input, k=5, is_gray=False, metric='euclidean', compare_with_sklearn=True):
    """
    Phân đoạn ảnh bằng K-Means:
    1. Chạy Custom K-Means
    2. Chạy Sklearn K-Means để đối sánh
    3. Vẽ biểu đồ so sánh: Ảnh gốc | Ảnh Custom K-Means | Ảnh Sklearn K-Means
    4. Trả về bảng so sánh chi tiết
    """
    img_np = load_and_preprocess_image(image_input, is_gray=is_gray)
    h, w = img_np.shape[0], img_np.shape[1]

    # Chuẩn bị dữ liệu điểm ảnh:
    # Nếu ảnh màu: shape (h, w, 3) -> reshape thành (h*w, 3)
    # Nếu ảnh xám:  shape (h, w)    -> reshape thành (h*w, 1)
    if is_gray:
        pixel_data = img_np.reshape(-1, 1)
    else:
        pixel_data = img_np.reshape(-1, 3)

    # 1. Custom K-Means
    t0 = time.time()
    custom_km = CustomKMeans(k=k, metric=metric, random_state=42, max_iters=100)
    custom_km.fit(pixel_data)
    t_custom = time.time() - t0

    # Tái tạo ảnh phân đoạn từ tâm cụm
    custom_segmented_pixels = custom_km.centroids[custom_km.labels_]
    if is_gray:
        custom_segmented_img = custom_segmented_pixels.reshape(h, w).astype('uint8')
    else:
        custom_segmented_img = custom_segmented_pixels.reshape(h, w, 3).astype('uint8')

    results = {
        'is_gray': is_gray,
        'k': k,
        'custom_time': t_custom,
        'custom_inertia': custom_km.inertia_,
        'custom_centroids': custom_km.centroids
    }

    # 2. Sklearn K-Means
    if compare_with_sklearn:
        t0 = time.time()
        sk_km = SklearnKMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        sk_km.fit(pixel_data)
        t_sk = time.time() - t0

        sk_segmented_pixels = sk_km.cluster_centers_[sk_km.labels_]
        if is_gray:
            sk_segmented_img = sk_segmented_pixels.reshape(h, w).astype('uint8')
        else:
            sk_segmented_img = sk_segmented_pixels.reshape(h, w, 3).astype('uint8')

        results['sklearn_time'] = t_sk
        results['sklearn_inertia'] = sk_km.inertia_
        results['sklearn_centroids'] = sk_km.cluster_centers_

        # Vẽ ảnh so sánh 3 cột
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        cmap = 'gray' if is_gray else None

        axes[0].imshow(img_np.astype('uint8'), cmap=cmap)
        axes[0].set_title(f"Ảnh gốc ({'Mức xám' if is_gray else 'Màu RGB'})\nKích thước: {w}x{h}", fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(custom_segmented_img, cmap=cmap)
        axes[1].set_title(f"Custom K-Means (K = {k})\nThời gian: {t_custom:.3f}s | WCSS: {custom_km.inertia_:.1e}", fontweight='bold', color='navy')
        axes[1].axis('off')

        axes[2].imshow(sk_segmented_img, cmap=cmap)
        axes[2].set_title(f"Sklearn K-Means (K = {k})\nThời gian: {t_sk:.3f}s | WCSS: {sk_km.inertia_:.1e}", fontweight='bold', color='darkgreen')
        axes[2].axis('off')

        title_type = "ẢNH TRẮNG ĐEN (GRAYSCALE)" if is_gray else "ẢNH MÀU (RGB)"
        plt.suptitle(f"KẾT QUẢ PHÂN ĐOẠN {title_type} VỚI K = {k}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    return results, custom_segmented_img


def segment_image_fcm(image_input, c=5, is_gray=False, max_pixels=5000):
    """
    Phân đoạn ảnh bằng Fuzzy C-Means (FCM).
    Do FCM tính toán ma trận mờ (N, C) nên lấy mẫu hoặc resize nhỏ để tối ưu tốc độ thi.
    """
    img_np = load_and_preprocess_image(image_input, is_gray=is_gray, max_size=(150, 150))
    h, w = img_np.shape[0], img_np.shape[1]

    if is_gray:
        pixel_data = img_np.reshape(-1, 1)
    else:
        pixel_data = img_np.reshape(-1, 3)

    t0 = time.time()
    fcm = CustomFCM(c=c, m=2.0, max_iters=50, tol=1e-3, random_state=42)
    fcm.fit(pixel_data)
    t_fcm = time.time() - t0

    segmented_pixels = fcm.centers[fcm.labels_]
    if is_gray:
        segmented_img = segmented_pixels.reshape(h, w).astype('uint8')
    else:
        segmented_img = segmented_pixels.reshape(h, w, 3).astype('uint8')

    # Vẽ hiển thị
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cmap = 'gray' if is_gray else None

    axes[0].imshow(img_np.astype('uint8'), cmap=cmap)
    axes[0].set_title("Ảnh đầu vào", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(segmented_img, cmap=cmap)
    axes[1].set_title(f"Custom Fuzzy C-Means (c = {c})\nThời gian: {t_fcm:.3f}s | FPC: {fcm.compute_fpc():.3f}", fontweight='bold', color='purple')
    axes[1].axis('off')

    plt.suptitle(f"PHÂN ĐOẠN ẢNH BẰNG FUZZY C-MEANS (C = {c})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {'fcm_time': t_fcm, 'fpc': fcm.compute_fpc(), 'centers': fcm.centers}, segmented_img


# ==============================================================================
# PHẦN 6: BẢNG SO SÁNH CUSTOM VS THƯ VIỆN (BENCHMARK & COMPARISON TABLE)
# ==============================================================================
def compare_kmeans_tabular(X, k=3, metric='euclidean'):
    """
    So sánh toàn diện Custom K-Means vs Sklearn KMeans trên tập dữ liệu bảng.
    Xuất ra bảng đối sánh các chỉ số: Thời gian, WCSS (Inertia), Silhouette Score.
    """
    # 1. Custom
    t0 = time.time()
    cust = CustomKMeans(k=k, metric=metric, random_state=42)
    cust_labels = cust.fit_predict(X)
    t_cust = time.time() - t0
    sil_cust = silhouette_score(X, cust_labels)

    # 2. Sklearn
    t0 = time.time()
    sk = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    sk_labels = sk.fit_predict(X)
    t_sk = time.time() - t0
    sil_sk = silhouette_score(X, sk_labels)

    df_comp = pd.DataFrame({
        'Tiêu chí đánh giá': [
            'Thuật toán / Thư viện',
            'Số cụm K',
            'Độ đo khoảng cách',
            'Thời gian thực thi (giây)',
            'WCSS / Inertia (càng thấp càng tốt)',
            'Silhouette Score (càng cao càng tốt)',
            'Số vòng lặp hội tụ'
        ],
        'Thuật toán Custom (Tự viết)': [
            'CustomKMeans (NumPy)',
            k,
            metric,
            f"{t_cust:.4f} s",
            f"{cust.inertia_:.4f}",
            f"{sil_cust:.4f}",
            f"{cust.n_iter_}"
        ],
        'Thư viện Scikit-Learn': [
            'sklearn.cluster.KMeans',
            k,
            'euclidean',
            f"{t_sk:.4f} s",
            f"{sk.inertia_:.4f}",
            f"{sil_sk:.4f}",
            f"{sk.n_iter_}"
        ]
    })

    return df_comp, cust_labels, sk_labels


def compare_knn_tabular(X_train, y_train, X_test, y_test, k=5, metric='euclidean'):
    """
    So sánh toàn diện Custom KNN vs Sklearn KNeighborsClassifier.
    Xuất bảng so sánh Accuracy, Runtime và Confusion Matrix side-by-side.
    """
    # 1. Custom
    t0 = time.time()
    cust_knn = CustomKNN(k=k, metric=metric)
    cust_knn.fit(X_train, y_train)
    cust_preds = cust_knn.predict(X_test)
    t_cust = time.time() - t0
    cust_acc = accuracy_score(y_test, cust_preds)

    # 2. Sklearn
    t0 = time.time()
    sk_knn = SklearnKNN(n_neighbors=k, metric=metric)
    sk_knn.fit(X_train, y_train)
    sk_preds = sk_knn.predict(X_test)
    t_sk = time.time() - t0
    sk_acc = accuracy_score(y_test, sk_preds)

    df_comp = pd.DataFrame({
        'Tiêu chí đánh giá': [
            'Mô hình',
            'Số lân cận K',
            'Khoảng cách',
            'Thời gian dự đoán (giây)',
            'Độ chính xác (Accuracy)',
            'Độ khớp kết quả (Custom vs Sklearn)'
        ],
        'Thuật toán Custom (Tự viết)': [
            'CustomKNN (NumPy)',
            k,
            metric,
            f"{t_cust:.4f} s",
            f"{cust_acc * 100:.2f}%",
            "-"
        ],
        'Thư viện Scikit-Learn': [
            'sklearn.neighbors.KNeighborsClassifier',
            k,
            metric,
            f"{t_sk:.4f} s",
            f"{sk_acc * 100:.2f}%",
            f"{np.mean(cust_preds == sk_preds) * 100:.2f}%"
        ]
    })

    return df_comp, cust_preds, sk_preds
