# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


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

# Thêm cột ngẫu nhiên cho cả hai tập
rng = np.random.default_rng(0)

X_tr_nn = X_tr.copy()
X_te_nn = X_te.copy()

X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr_nn))
X_te_nn["so_ngau_nhien"] = rng.random(len(X_te_nn))

# Huấn luyện rừng
rung_nn = RandomForestClassifier(
    n_estimators=200,
    random_state=0
)
rung_nn.fit(X_tr_nn, y_tr)

# Độ quan trọng dựa trên mức giảm Gini
mdi = pd.Series(
    rung_nn.feature_importances_,
    index=X_tr_nn.columns
)

# Độ quan trọng bằng xáo trộn trên tập kiểm tra
hoan_vi = permutation_importance(
    rung_nn,
    X_te_nn,
    y_te,
    scoring="accuracy",
    n_repeats=20,
    random_state=0
)

bang_qt = pd.DataFrame({
    "MDI": mdi,
    "permutation_mean": hoan_vi.importances_mean,
    "permutation_std": hoan_vi.importances_std
})

# Các cột bằng điểm được tính đồng hạng
bang_qt["rank_MDI"] = (
    bang_qt["MDI"]
    .rank(ascending=False, method="min")
    .astype(int)
)

bang_qt["rank_permutation"] = (
    bang_qt["permutation_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

acc = accuracy_score(
    y_te,
    rung_nn.predict(X_te_nn)
)

print(f"Accuracy tap kiem tra: {acc:.4f}")

print("\nDO QUAN TRONG CUA NAM COT")
print(
    bang_qt
    .sort_values("permutation_mean", ascending=False)
    .round(4)
    .to_string()
)

cot_nn = bang_qt.loc["so_ngau_nhien"]

print(
    f"\nCot ngau nhien xep hang "
    f"{int(cot_nn['rank_MDI'])} theo MDI."
)
print(
    f"Cot ngau nhien xep hang "
    f"{int(cot_nn['rank_permutation'])} theo permutation."
)
print(
    f"Permutation mean cua cot ngau nhien: "
    f"{cot_nn['permutation_mean']:.4f}"
)

print(
    "\nGiai thich: MDI co the danh gia cao cot ngau nhien "
    "vi cot nay co nhieu nguong chia."
)
print(
    "Permutation do muc giam accuracy tren tap kiem tra, "
    "nen cot vo dung thuong co diem gan 0 hoac am."
)