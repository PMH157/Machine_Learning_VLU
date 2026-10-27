import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

# Chia dữ liệu thành tập học và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

def gini(nhan):
    """Tính Gini từ tỷ lệ các lớp trong một nút."""
    nhan = np.asarray(nhan)

    if len(nhan) == 0:
        return 0.0

    _, so_luong = np.unique(nhan, return_counts=True)
    ty_le = so_luong / len(nhan)

    return 1.0 - np.sum(ty_le ** 2)


def tim_cau_hoi(X, y):
    """Trả về cột và ngưỡng làm Gini giảm nhiều nhất."""
    y = np.asarray(y)
    n = len(y)

    gini_goc = gini(y)
    giam_tot_nhat = -np.inf
    cot_tot_nhat = None
    nguong_tot_nhat = None

    for cot in X.columns:
        gia_tri = X[cot].to_numpy()
        cac_muc = np.sort(np.unique(gia_tri))

        # Chỉ cần thử trung điểm giữa hai giá trị liền nhau
        for a, b in zip(cac_muc[:-1], cac_muc[1:]):
            nguong = (float(a) + float(b)) / 2
            di_trai = gia_tri <= nguong

            y_trai = y[di_trai]
            y_phai = y[~di_trai]

            gini_chia = (
                len(y_trai) / n * gini(y_trai)
                + len(y_phai) / n * gini(y_phai)
            )
            giam = gini_goc - gini_chia

            if giam > giam_tot_nhat:
                giam_tot_nhat = giam
                cot_tot_nhat = cot
                nguong_tot_nhat = nguong

    if cot_tot_nhat is None:
        raise ValueError("Khong co nguong chia hop le.")

    return cot_tot_nhat, nguong_tot_nhat


# Tìm câu hỏi trên tập học
cot_tot, nguong_tot = tim_cau_hoi(X_tr, y_tr)

di_trai = X_tr[cot_tot].to_numpy() <= nguong_tot
nhan = y_tr.to_numpy()

gini_chia = (
    di_trai.mean() * gini(nhan[di_trai])
    + (~di_trai).mean() * gini(nhan[~di_trai])
)

print(f"Cau hoi tu tinh: {cot_tot} <= {nguong_tot:.4f}")
print(f"Gini nut goc: {gini(nhan):.4f}")
print(f"Gini sau chia: {gini_chia:.4f}")
print(f"Gini giam: {gini(nhan) - gini_chia:.4f}")

# Đối chiếu với câu hỏi gốc do scikit-learn chọn
cay_goc = DecisionTreeClassifier(max_depth=1, random_state=0)
cay_goc.fit(X_tr, y_tr)

cot_sklearn = X.columns[cay_goc.tree_.feature[0]]
nguong_sklearn = cay_goc.tree_.threshold[0]

print(
    f"Cau hoi sklearn: "
    f"{cot_sklearn} <= {nguong_sklearn:.4f}"
)
print(
    "Hai ket qua trung nhau:",
    cot_tot == cot_sklearn
    and np.isclose(nguong_tot, nguong_sklearn)
)