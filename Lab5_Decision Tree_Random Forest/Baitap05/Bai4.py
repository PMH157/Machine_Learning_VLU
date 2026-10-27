# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


# Đọc và chia dữ liệu
df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

cac_gia_tri = [1, 2, 5, 10, 20, 40]
bang_leaf = []

# Không giới hạn độ sâu, chỉ thay min_samples_leaf
for m in cac_gia_tri:
    mh = DecisionTreeClassifier(
        min_samples_leaf=m,
        random_state=0
    )

    diem_cv = cross_val_score(
        mh,
        X_tr,
        y_tr,
        cv=5,
        scoring="accuracy"
    ).mean()

    mh.fit(X_tr, y_tr)

    bang_leaf.append({
        "min_samples_leaf": m,
        "depth": mh.get_depth(),
        "n_leaves": mh.get_n_leaves(),
        "accuracy_train": accuracy_score(
            y_tr, mh.predict(X_tr)
        ),
        "accuracy_cv": diem_cv
    })

bang_leaf = pd.DataFrame(bang_leaf)

print("KET QUA KIEM DINH CHEO")
print(bang_leaf.round(4).to_string(index=False))

# Chọn tham số theo điểm kiểm định chéo cao nhất
hang_tot = bang_leaf.loc[
    bang_leaf["accuracy_cv"].idxmax()
]
m_tot = int(hang_tot["min_samples_leaf"])

cac_mo_hinh = {
    f"min_samples_leaf={m_tot}": DecisionTreeClassifier(
        min_samples_leaf=m_tot,
        random_state=0
    ),
    "max_depth=5": DecisionTreeClassifier(
        max_depth=5,
        random_state=0
    )
}

bang_cuoi = []

# Chấm mỗi mô hình cuối cùng một lần trên tập kiểm tra
for ten, mh in cac_mo_hinh.items():
    mh.fit(X_tr, y_tr)

    bang_cuoi.append({
        "model": ten,
        "depth": mh.get_depth(),
        "n_leaves": mh.get_n_leaves(),
        "accuracy_test": accuracy_score(
            y_te, mh.predict(X_te)
        )
    })

ket_qua = pd.DataFrame(bang_cuoi)

print("\nMin_samples_leaf tot nhat:", m_tot)
print("\nSO SANH HAI CACH CAT TIA")
print(ket_qua.round(4).to_string(index=False))

acc_leaf = ket_qua.iloc[0]["accuracy_test"]
acc_depth = ket_qua.iloc[1]["accuracy_test"]

if acc_leaf > acc_depth:
    print(
        "\nNhan xet: Gioi han so khach toi thieu o la "
        "cho accuracy kiem tra cao hon."
    )
elif acc_leaf < acc_depth:
    print(
        "\nNhan xet: Gioi han do sau bang 5 "
        "cho accuracy kiem tra cao hon."
    )
else:
    print(
        "\nNhan xet: Hai cach cho accuracy "
        "kiem tra bang nhau."
    )