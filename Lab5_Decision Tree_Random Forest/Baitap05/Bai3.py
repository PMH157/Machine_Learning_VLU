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

bang_do_sau = []
bang_so_sanh = []

for tieu_chi in ["gini", "entropy"]:
    diem_theo_do_sau = {}

    # Chọn độ sâu chỉ trên tập học
    for d in range(1, 16):
        mh = DecisionTreeClassifier(
            criterion=tieu_chi,
            max_depth=d,
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
        diem_hoc = accuracy_score(y_tr, mh.predict(X_tr))

        diem_theo_do_sau[d] = diem_cv

        bang_do_sau.append({
            "criterion": tieu_chi,
            "max_depth": d,
            "accuracy_train": diem_hoc,
            "accuracy_cv": diem_cv
        })

    # Nếu bằng điểm, chọn độ sâu nhỏ hơn
    d_tot = max(
        diem_theo_do_sau,
        key=diem_theo_do_sau.get
    )

    mh_tot = DecisionTreeClassifier(
        criterion=tieu_chi,
        max_depth=d_tot,
        random_state=0
    )
    mh_tot.fit(X_tr, y_tr)

    # Đánh giá mô hình đã chọn trên tập kiểm tra
    acc_test = accuracy_score(
        y_te,
        mh_tot.predict(X_te)
    )

    bang_so_sanh.append({
        "criterion": tieu_chi,
        "best_depth": d_tot,
        "accuracy_cv": diem_theo_do_sau[d_tot],
        "accuracy_test": acc_test
    })

print("KET QUA THEO DO SAU")
print(
    pd.DataFrame(bang_do_sau)
    .round(4)
    .to_string(index=False)
)

ket_qua = pd.DataFrame(bang_so_sanh)

print("\nSO SANH GINI VA ENTROPY")
print(ket_qua.round(4).to_string(index=False))

acc_gini = ket_qua.loc[
    ket_qua["criterion"] == "gini",
    "accuracy_test"
].iloc[0]

acc_entropy = ket_qua.loc[
    ket_qua["criterion"] == "entropy",
    "accuracy_test"
].iloc[0]

chenh_lech = abs(acc_entropy - acc_gini) * 100

print(
    f"\nNhan xet: Doi Gini sang entropy lam accuracy "
    f"kiem tra chenh {chenh_lech:.2f} diem phan tram."
)