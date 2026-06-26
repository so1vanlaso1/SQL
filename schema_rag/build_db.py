"""Build and populate the Vietnamese-style FMCG SQLite demo database."""
from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

from . import config, joined_tables, schema_def

SEED = 42
START = dt.date(2024, 1, 1)
DAYS = 540


def _d(offset: int) -> str:
    return (START + dt.timedelta(days=offset)).isoformat()


def _create_schema(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA foreign_keys = OFF;")
    for name in reversed(schema_def.all_table_names()):
        con.execute(f"DROP TABLE IF EXISTS {name};")
    con.execute("PRAGMA foreign_keys = ON;")
    for name in schema_def.all_table_names():
        con.executescript(schema_def.ddl_for(name))


def _insert(con: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    con.executemany(sql, [[r[c] for c in cols] for r in rows])


def _create_indexes(con: sqlite3.Connection) -> None:
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_don_hang_ban_ngay_dat_hang ON don_hang_ban(ngay_dat_hang)",
        "CREATE INDEX IF NOT EXISTS idx_don_hang_ban_khach_hang_id ON don_hang_ban(khach_hang_id)",
        "CREATE INDEX IF NOT EXISTS idx_don_hang_ban_nhan_vien_id ON don_hang_ban(nhan_vien_id)",
        "CREATE INDEX IF NOT EXISTS idx_don_hang_ban_nha_phan_phoi_id ON don_hang_ban(nha_phan_phoi_id)",
        "CREATE INDEX IF NOT EXISTS idx_chi_tiet_don_hang_ban_don_hang_id ON chi_tiet_don_hang_ban(don_hang_id)",
        "CREATE INDEX IF NOT EXISTS idx_chi_tiet_don_hang_ban_san_pham_id ON chi_tiet_don_hang_ban(san_pham_id)",
        "CREATE INDEX IF NOT EXISTS idx_khach_hang_loai_khach_hang_id ON khach_hang(loai_khach_hang_id)",
        "CREATE INDEX IF NOT EXISTS idx_khach_hang_vi_tri_id ON khach_hang(vi_tri_id)",
        "CREATE INDEX IF NOT EXISTS idx_npp_khach_hang_khach_hang_id ON nha_phan_phoi_khach_hang(khach_hang_id)",
        "CREATE INDEX IF NOT EXISTS idx_npp_khach_hang_nha_phan_phoi_id ON nha_phan_phoi_khach_hang(nha_phan_phoi_id)",
        "CREATE INDEX IF NOT EXISTS idx_npp_khach_hang_tuyen_id ON nha_phan_phoi_khach_hang(tuyen_id)",
        "CREATE INDEX IF NOT EXISTS idx_tuyen_ban_hang_vi_tri_id ON tuyen_ban_hang(vi_tri_id)",
        "CREATE INDEX IF NOT EXISTS idx_vi_tri_tinh_thanh ON vi_tri(tinh_thanh)",
        "CREATE INDEX IF NOT EXISTS idx_lich_su_vieng_tham_ngay ON lich_su_vieng_tham(ngay_vieng_tham)",
        "CREATE INDEX IF NOT EXISTS idx_lich_su_vieng_tham_khach_hang_id ON lich_su_vieng_tham(khach_hang_id)",
        "CREATE INDEX IF NOT EXISTS idx_lich_su_vieng_tham_nha_phan_phoi_id ON lich_su_vieng_tham(nha_phan_phoi_id)",
    ]
    for sql in indexes:
        con.execute(sql)


def _id(prefix: str, n: int) -> str:
    return f"{prefix}_{n:03d}"


def populate(con: sqlite3.Connection) -> dict:
    rnd = random.Random(SEED)
    counts: dict = {}

    cong_ty = [
        {"cong_ty_id": "CTY_001", "ten_cong_ty": "Cong ty FMCG An Phat", "nganh_hang": "FMCG"},
        {"cong_ty_id": "CTY_002", "ten_cong_ty": "Nuoc Giai Khat Sao Viet", "nganh_hang": "Beverage"},
    ]
    _insert(con, "cong_ty", cong_ty)

    vung = [
        {"vung_id": "VUNG_BAC", "ten_vung": "Mien Bac", "quoc_gia": "Viet Nam"},
        {"vung_id": "VUNG_TRUNG", "ten_vung": "Mien Trung", "quoc_gia": "Viet Nam"},
        {"vung_id": "VUNG_NAM", "ten_vung": "Mien Nam", "quoc_gia": "Viet Nam"},
        {"vung_id": "VUNG_TAY_NGUYEN", "ten_vung": "Tay Nguyen", "quoc_gia": "Viet Nam"},
        {"vung_id": "VUNG_MEKONG", "ten_vung": "Mekong", "quoc_gia": "Viet Nam"},
    ]
    _insert(con, "vung", vung)

    tinh = [
        ("Ha Noi", "Cau Giay", "Dich Vong", 21.03, 105.79),
        ("Hai Phong", "Le Chan", "Niem Nghia", 20.84, 106.68),
        ("Da Nang", "Hai Chau", "Hoa Cuong", 16.05, 108.22),
        ("Khanh Hoa", "Nha Trang", "Loc Tho", 12.24, 109.19),
        ("Ho Chi Minh", "Quan 7", "Tan Phong", 10.73, 106.70),
        ("Can Tho", "Ninh Kieu", "An Hoa", 10.04, 105.78),
        ("Dong Nai", "Bien Hoa", "Tan Hiep", 10.95, 106.86),
        ("An Giang", "Long Xuyen", "My Binh", 10.38, 105.44),
        ("Nghe An", "Vinh", "Hung Binh", 18.67, 105.68),
        ("Lam Dong", "Da Lat", "Phuong 1", 11.94, 108.44),
    ]
    vi_tri = []
    for i, (province, district, ward, lat, lng) in enumerate(tinh, 1):
        vi_tri.append(
            {
                "vi_tri_id": _id("VT", i),
                "tinh_thanh": province,
                "quan_huyen": district,
                "phuong_xa": ward,
                "vi_do": lat + rnd.uniform(-0.04, 0.04),
                "kinh_do": lng + rnd.uniform(-0.04, 0.04),
            }
        )
    _insert(con, "vi_tri", vi_tri)

    nha_phan_phoi = []
    for i in range(8):
        nha_phan_phoi.append(
            {
                "nha_phan_phoi_id": _id("NPP", i + 1),
                "cong_ty_id": rnd.choice(cong_ty)["cong_ty_id"],
                "vung_id": rnd.choice(vung)["vung_id"],
                "ten_nha_phan_phoi": f"Nha phan phoi {chr(65 + i)}",
                "trang_thai": "ACTIVE" if rnd.random() > 0.1 else "INACTIVE",
            }
        )
    _insert(con, "nha_phan_phoi", nha_phan_phoi)

    tuyen_ban_hang = []
    tuyen_id = 1
    for npp in nha_phan_phoi:
        for j in range(rnd.randint(2, 4)):
            vt = rnd.choice(vi_tri)
            tuyen_ban_hang.append(
                {
                    "tuyen_id": tuyen_id,
                    "nha_phan_phoi_id": npp["nha_phan_phoi_id"],
                    "vung_id": npp["vung_id"],
                    "vi_tri_id": vt["vi_tri_id"],
                    "ma_tuyen": f"T{tuyen_id:03d}",
                    "ten_tuyen": f"Tuyen {vt['quan_huyen']} {j + 1}",
                    "trang_thai": "ACTIVE",
                }
            )
            tuyen_id += 1
    _insert(con, "tuyen_ban_hang", tuyen_ban_hang)

    nhan_vien = []
    for npp in nha_phan_phoi:
        for j in range(rnd.randint(2, 4)):
            idx = len(nhan_vien) + 1
            nhan_vien.append(
                {
                    "nhan_vien_id": _id("NV", idx),
                    "nha_phan_phoi_id": npp["nha_phan_phoi_id"],
                    "ten_nhan_vien": f"Nhan vien ban hang {idx}",
                    "ngay_vao_lam": _d(rnd.randint(0, 180)),
                    "trang_thai": "ACTIVE",
                }
            )
    _insert(con, "nhan_vien", nhan_vien)

    phan_cong_tuyen = []
    for i, route in enumerate(tuyen_ban_hang, 1):
        staff_pool = [s for s in nhan_vien if s["nha_phan_phoi_id"] == route["nha_phan_phoi_id"]]
        staff = rnd.choice(staff_pool)
        phan_cong_tuyen.append(
            {
                "phan_cong_id": i,
                "nha_phan_phoi_id": route["nha_phan_phoi_id"],
                "nhan_vien_id": staff["nhan_vien_id"],
                "tuyen_id": route["tuyen_id"],
                "ngay_bat_dau": _d(rnd.randint(0, 90)),
                "ngay_ket_thuc": None,
            }
        )
    _insert(con, "phan_cong_tuyen", phan_cong_tuyen)

    loai_khach_hang = [
        {"loai_khach_hang_id": "GROCERY", "ten_loai": "Tap hoa", "mo_ta": "Traditional grocery outlet"},
        {"loai_khach_hang_id": "MINI_SUPERMARKET", "ten_loai": "Sieu thi mini", "mo_ta": "Mini supermarket"},
        {"loai_khach_hang_id": "WHOLESALE_SHOP", "ten_loai": "Dai ly si", "mo_ta": "Wholesale shop"},
        {"loai_khach_hang_id": "CONVENIENCE_STORE", "ten_loai": "Cua hang tien loi", "mo_ta": "Convenience store"},
        {"loai_khach_hang_id": "HORECA", "ten_loai": "Nha hang khach san", "mo_ta": "Hotel restaurant cafe"},
    ]
    _insert(con, "loai_khach_hang", loai_khach_hang)

    khach_hang = []
    mapping = []
    customer_id = 1
    for npp in nha_phan_phoi:
        routes = [r for r in tuyen_ban_hang if r["nha_phan_phoi_id"] == npp["nha_phan_phoi_id"]]
        staff = [s for s in nhan_vien if s["nha_phan_phoi_id"] == npp["nha_phan_phoi_id"]]
        for _ in range(rnd.randint(12, 18)):
            route = rnd.choice(routes)
            vt = rnd.choice(vi_tri)
            kh_id = _id("KH", customer_id)
            khach_hang.append(
                {
                    "khach_hang_id": kh_id,
                    "loai_khach_hang_id": rnd.choice(loai_khach_hang)["loai_khach_hang_id"],
                    "vi_tri_id": vt["vi_tri_id"],
                    "ten_khach_hang": f"Cua hang {customer_id}",
                    "dia_chi": f"{customer_id} duong chinh, {vt['phuong_xa']}, {vt['tinh_thanh']}",
                    "so_dien_thoai": f"09{rnd.randint(10000000, 99999999)}",
                    "ngay_tao": _d(rnd.randint(0, 120)),
                }
            )
            mapping.append(
                {
                    "phan_phoi_khach_hang_id": customer_id,
                    "nha_phan_phoi_id": npp["nha_phan_phoi_id"],
                    "khach_hang_id": kh_id,
                    "nhan_vien_id": rnd.choice(staff)["nhan_vien_id"],
                    "tuyen_id": route["tuyen_id"],
                    "ngay_mo": _d(rnd.randint(0, 160)),
                    "trang_thai": "OPEN",
                }
            )
            customer_id += 1
    _insert(con, "khach_hang", khach_hang)
    _insert(con, "nha_phan_phoi_khach_hang", mapping)

    danh_muc_san_pham = [
        {"danh_muc_id": "CAT_BEV", "ten_danh_muc": "Do uong"},
        {"danh_muc_id": "CAT_SNACK", "ten_danh_muc": "Banh keo"},
        {"danh_muc_id": "CAT_DAIRY", "ten_danh_muc": "Sua"},
        {"danh_muc_id": "CAT_HOME", "ten_danh_muc": "Gia dung"},
        {"danh_muc_id": "CAT_CARE", "ten_danh_muc": "Cham soc ca nhan"},
        {"danh_muc_id": "CAT_FROZEN", "ten_danh_muc": "Dong lanh"},
    ]
    _insert(con, "danh_muc_san_pham", danh_muc_san_pham)

    san_pham = []
    for i in range(60):
        san_pham.append(
            {
                "san_pham_id": _id("SP", i + 1),
                "cong_ty_id": rnd.choice(cong_ty)["cong_ty_id"],
                "danh_muc_id": rnd.choice(danh_muc_san_pham)["danh_muc_id"],
                "ten_san_pham": f"San pham FMCG {i + 1}",
                "don_vi_tinh": rnd.choice(["thung", "chai", "goi", "hop"]),
                "trang_thai": "ACTIVE",
            }
        )
    _insert(con, "san_pham", san_pham)

    bang_gia = []
    for i, sp in enumerate(san_pham, 1):
        bang_gia.append(
            {
                "bang_gia_id": i,
                "san_pham_id": sp["san_pham_id"],
                "gia_ban": round(rnd.uniform(5_000, 180_000), -2),
                "ngay_bat_dau": _d(0),
                "ngay_ket_thuc": _d(DAYS),
            }
        )
    _insert(con, "bang_gia_san_pham", bang_gia)

    khuyen_mai = []
    for i in range(8):
        start = rnd.randint(0, DAYS - 80)
        khuyen_mai.append(
            {
                "khuyen_mai_id": _id("KM", i + 1),
                "cong_ty_id": rnd.choice(cong_ty)["cong_ty_id"],
                "ten_khuyen_mai": f"Khuyen mai quy {i + 1}",
                "phan_tram_giam": rnd.choice([5, 10, 15, 20]),
                "ngay_bat_dau": _d(start),
                "ngay_ket_thuc": _d(start + rnd.randint(20, 70)),
            }
        )
    _insert(con, "khuyen_mai", khuyen_mai)

    km_sp = []
    rel_id = 1
    for km in khuyen_mai:
        for sp in rnd.sample(san_pham, 6):
            km_sp.append(
                {
                    "khuyen_mai_san_pham_id": rel_id,
                    "khuyen_mai_id": km["khuyen_mai_id"],
                    "san_pham_id": sp["san_pham_id"],
                }
            )
            rel_id += 1
    _insert(con, "khuyen_mai_san_pham", km_sp)

    visits = []
    orders = []
    items = []
    deliveries = []
    returns = []
    visit_id = 1
    order_id = 1
    item_id = 1
    delivery_id = 1
    return_id = 1
    result_choices = ["VISITED", "ORDERED", "NO_ORDER", "STORE_CLOSED", "CUSTOMER_BUSY", "NOT_FOUND"]

    price_by_product = {p["san_pham_id"]: next(bg["gia_ban"] for bg in bang_gia if bg["san_pham_id"] == p["san_pham_id"]) for p in san_pham}

    for mp in mapping:
        declining = rnd.random() < 0.35
        visit_count = rnd.randint(10, 28)
        for _ in range(visit_count):
            if declining:
                day = int(rnd.triangular(0, DAYS, 0))
            else:
                day = rnd.randint(0, DAYS)
            ordered = rnd.random() < 0.55
            result = "ORDERED" if ordered else rnd.choice(result_choices)
            visits.append(
                {
                    "vieng_tham_id": visit_id,
                    "khach_hang_id": mp["khach_hang_id"],
                    "nha_phan_phoi_id": mp["nha_phan_phoi_id"],
                    "nhan_vien_id": mp["nhan_vien_id"],
                    "tuyen_id": mp["tuyen_id"],
                    "ngay_vieng_tham": _d(day),
                    "ket_qua": result,
                    "ghi_chu": "co don hang" if ordered else "khong phat sinh don",
                }
            )
            if ordered:
                order_rows = []
                promo = rnd.choice(khuyen_mai) if rnd.random() < 0.25 else None
                for sp in rnd.sample(san_pham, rnd.randint(1, 5)):
                    qty = rnd.randint(1, 25)
                    price = price_by_product[sp["san_pham_id"]]
                    if promo:
                        price = round(price * (1 - promo["phan_tram_giam"] / 100), 2)
                    order_rows.append((sp, qty, price, promo))
                total = round(sum(qty * price for _, qty, price, _ in order_rows), 2)
                dh_id = _id("DH", order_id)
                npp = next(n for n in nha_phan_phoi if n["nha_phan_phoi_id"] == mp["nha_phan_phoi_id"])
                orders.append(
                    {
                        "don_hang_id": dh_id,
                        "cong_ty_id": npp["cong_ty_id"],
                        "nha_phan_phoi_id": mp["nha_phan_phoi_id"],
                        "nhan_vien_id": mp["nhan_vien_id"],
                        "khach_hang_id": mp["khach_hang_id"],
                        "vieng_tham_id": visit_id,
                        "ngay_dat_hang": _d(day),
                        "trang_thai": "NORMAL" if rnd.random() > 0.08 else "CANCELLED",
                        "tong_tien": total,
                    }
                )
                for sp, qty, price, promo in order_rows:
                    items.append(
                        {
                            "chi_tiet_id": item_id,
                            "don_hang_id": dh_id,
                            "san_pham_id": sp["san_pham_id"],
                            "khuyen_mai_id": promo["khuyen_mai_id"] if promo else None,
                            "so_luong": qty,
                            "don_gia": price,
                            "thanh_tien": round(qty * price, 2),
                        }
                    )
                    item_id += 1
                    if rnd.random() < 0.04:
                        returns.append(
                            {
                                "tra_ve_id": return_id,
                                "don_hang_id": dh_id,
                                "san_pham_id": sp["san_pham_id"],
                                "ngay_tra": _d(min(DAYS, day + rnd.randint(1, 20))),
                                "so_luong": rnd.randint(1, max(1, qty // 2)),
                                "ly_do": rnd.choice(["hang_hong", "can_date", "giao_sai"]),
                            }
                        )
                        return_id += 1
                if orders[-1]["trang_thai"] == "NORMAL":
                    deliveries.append(
                        {
                            "giao_hang_id": delivery_id,
                            "don_hang_id": dh_id,
                            "nha_phan_phoi_id": mp["nha_phan_phoi_id"],
                            "ngay_xuat_kho": _d(min(DAYS, day + 1)),
                            "ngay_giao": _d(min(DAYS, day + rnd.randint(2, 6))),
                            "trang_thai": "DELIVERED" if rnd.random() > 0.08 else "FAILED",
                        }
                    )
                    delivery_id += 1
                order_id += 1
            visit_id += 1

    _insert(con, "lich_su_vieng_tham", visits)
    _insert(con, "don_hang_ban", orders)
    _insert(con, "chi_tiet_don_hang_ban", items)
    _insert(con, "don_giao_hang", deliveries)
    _insert(con, "hang_tra_ve", returns)

    _create_indexes(con)
    con.commit()

    for name in schema_def.all_table_names():
        counts[name] = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    return counts


def build(db_path: Path | None = None) -> dict:
    db_path = Path(db_path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        _create_schema(con)
        counts = populate(con)
        joined_counts = joined_tables.refresh(con)
        counts.update(joined_counts)
    finally:
        con.close()
    total = sum(counts.values())
    print(f"[db] built {db_path} ({len(counts)} tables, {total} rows)")
    for name, n in counts.items():
        print(f"     {name:<28} {n:>6}")
    return counts


if __name__ == "__main__":
    build()
