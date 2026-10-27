import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

print("So khach tap hoc:", len(X_tr))
print("So khach tap kiem tra:", len(X_te))
# Huấn luyện và vẽ cây năm lá
cay_nam_la = DecisionTreeClassifier(
    max_leaf_nodes=5,
    random_state=0
)
cay_nam_la.fit(X_tr, y_tr)

plt.figure(figsize=(16, 9))
plot_tree(
    cay_nam_la,
    feature_names=list(X.columns),
    class_names=["Tre han", "Dung han"],
    filled=True,
    rounded=True,
    precision=3,
    fontsize=10
)
plt.title("Cay quyet dinh nam la")
plt.tight_layout()
plt.savefig("cay_nam_la.png", dpi=300, bbox_inches="tight")
plt.show()

# Huấn luyện một cây mới có độ sâu tối đa bằng 2
cay_sau_2 = DecisionTreeClassifier(
    max_depth=2,
    random_state=0
)
cay_sau_2.fit(X_tr, y_tr)

plt.figure(figsize=(14, 7))
plot_tree(
    cay_sau_2,
    feature_names=list(X.columns),
    class_names=["Tre han", "Dung han"],
    filled=True,
    rounded=True,
    precision=3,
    fontsize=10
)
plt.title("Cay quyet dinh sau 2")
plt.tight_layout()
plt.savefig("cay_sau_2.png", dpi=300, bbox_inches="tight")
plt.show()

# Đọc kết luận và tỷ lệ lớp của hai lá thuộc nhánh trái
nut_trai = cay_sau_2.tree_.children_left[0]
hai_la = [
    cay_sau_2.tree_.children_left[nut_trai],
    cay_sau_2.tree_.children_right[nut_trai]
]

ket_luan = []

for i, nut in enumerate(hai_la, start=1):
    so_luong = cay_sau_2.tree_.value[nut][0]
    ty_le = so_luong / so_luong.sum()
    nhan = cay_sau_2.classes_[np.argmax(ty_le)]
    ket_luan.append(nhan)

    print(
        f"La trai {i}: class={nhan}, "
        f"ty le tre han={ty_le[0]:.4f}, "
        f"ty le dung han={ty_le[1]:.4f}"
    )

print("Hai la cung ket luan:", ket_luan[0] == ket_luan[1])