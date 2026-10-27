# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

# Chia dữ liệu giống mục 8 của tài liệu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

cac_mo_hinh = {
    "Cay sau 3": DecisionTreeRegressor(
        max_depth=3,
        random_state=0
    ),
    "Duong thang": LinearRegression(),
    "Rung 200 cay": RandomForestRegressor(
        n_estimators=200,
        random_state=0
    )
}

can_moi = pd.DataFrame({
    "dien_tich": [150.0, 250.0]
})

bang_ket_qua = []

# Huấn luyện và so sánh ba mô hình
for ten, mh in cac_mo_hinh.items():
    mh.fit(X_tr, y_tr)

    du_doan_test = mh.predict(X_te)
    gia_moi = mh.predict(can_moi)

    bang_ket_qua.append({
        "model": ten,
        "MAE_test": mean_absolute_error(
            y_te, du_doan_test
        ),
        "gia_150m2": gia_moi[0],
        "gia_250m2": gia_moi[1]
    })

ket_qua = pd.DataFrame(bang_ket_qua)

print("SO SANH MO HINH, DON VI TY DONG")
print(ket_qua.round(4).to_string(index=False))

print(
    "\nDien tich lon nhat trong tap hoc:",
    X_tr["dien_tich"].max()
)

# Kiểm tra cách rừng tổng hợp dự đoán của từng cây
rung = cac_mo_hinh["Rung 200 cay"]
can_250 = pd.DataFrame({"dien_tich": [250.0]})

gia_tung_cay = np.array([
    cay.predict(can_250.to_numpy())[0]
    for cay in rung.estimators_
])

gia_rung = rung.predict(can_250)[0]

print(f"\nGia 250m2 do rung du doan: {gia_rung:.4f}")
print(
    f"Trung binh du doan cua 200 cay: "
    f"{gia_tung_cay.mean():.4f}"
)
print(
    "Hai ket qua trung nhau:",
    np.isclose(gia_rung, gia_tung_cay.mean())
)

tot_nhat = ket_qua.loc[
    ket_qua["MAE_test"].idxmin(),
    "model"
]

print("\nMo hinh co MAE thap nhat:", tot_nhat)

print(
    "\nKet luan: Rung khong ngoai suy duoc xu huong "
    "tang gia ngoai mien dien tich da hoc."
)
print(
    "Moi cay du doan bang gia trung binh tai la; "
    "rung lay trung binh du doan cua 200 cay."
)