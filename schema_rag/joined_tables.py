"""Materialized joined feature tables used by the chat pipeline."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class JoinedTable:
    name: str
    purpose: str
    sources: tuple[str, ...]
    sql: str
    indexes: tuple[str, ...] = ()


JOINED_TABLES: tuple[JoinedTable, ...] = (
    JoinedTable(
        name="jt_don_hang_day_du",
        purpose="Phân tích đơn hàng bán đã được làm giàu với khách hàng, nhà phân phối, nhân viên, tuyến, vùng, viếng thăm và giao hàng.",
        sources=("don_hang_ban", "khach_hang", "loai_khach_hang", "nha_phan_phoi", "vung", "cong_ty", "nhan_vien", "tuyen_ban_hang", "vi_tri", "lich_su_vieng_tham", "don_giao_hang"),
        sql="""
            CREATE TABLE jt_don_hang_day_du AS
            SELECT
                dh.don_hang_id,
                dh.ngay_dat_hang,
                substr(dh.ngay_dat_hang, 1, 7) AS thang_dat_hang,
                dh.trang_thai AS trang_thai_don_hang,
                dh.tong_tien,
                dh.cong_ty_id,
                ct.ten_cong_ty,
                ct.nganh_hang,
                dh.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.trang_thai AS trang_thai_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                dh.khach_hang_id,
                kh.ten_khach_hang,
                lkh.ten_loai AS loai_khach_hang,
                kh.dia_chi,
                vt.tinh_thanh,
                vt.quan_huyen,
                vt.phuong_xa,
                dh.nhan_vien_id,
                nv.ten_nhan_vien,
                dh.vieng_tham_id,
                lsvt.ngay_vieng_tham,
                lsvt.ket_qua AS ket_qua_vieng_tham,
                lsvt.tuyen_id,
                tbh.ma_tuyen,
                tbh.ten_tuyen,
                gh.giao_hang_id,
                gh.ngay_xuat_kho,
                gh.ngay_giao,
                gh.trang_thai AS trang_thai_giao_hang,
                CASE
                    WHEN gh.ngay_giao IS NOT NULL AND gh.ngay_xuat_kho IS NOT NULL
                    THEN CAST(julianday(gh.ngay_giao) - julianday(gh.ngay_xuat_kho) AS INTEGER)
                    ELSE NULL
                END AS so_ngay_giao_hang
            FROM don_hang_ban dh
            LEFT JOIN cong_ty ct ON ct.cong_ty_id = dh.cong_ty_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = dh.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = dh.khach_hang_id
            LEFT JOIN loai_khach_hang lkh ON lkh.loai_khach_hang_id = kh.loai_khach_hang_id
            LEFT JOIN vi_tri vt ON vt.vi_tri_id = kh.vi_tri_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = dh.nhan_vien_id
            LEFT JOIN lich_su_vieng_tham lsvt ON lsvt.vieng_tham_id = dh.vieng_tham_id
            LEFT JOIN tuyen_ban_hang tbh ON tbh.tuyen_id = lsvt.tuyen_id
            LEFT JOIN don_giao_hang gh ON gh.don_hang_id = dh.don_hang_id
        """,
        indexes=(
            "CREATE INDEX IF NOT EXISTS idx_jt_don_hang_day_du_ngay ON jt_don_hang_day_du(ngay_dat_hang)",
            "CREATE INDEX IF NOT EXISTS idx_jt_don_hang_day_du_kh ON jt_don_hang_day_du(khach_hang_id)",
            "CREATE INDEX IF NOT EXISTS idx_jt_don_hang_day_du_npp ON jt_don_hang_day_du(nha_phan_phoi_id)",
        ),
    ),
    JoinedTable(
        name="jt_chi_tiet_ban_hang_day_du",
        purpose="Phân tích doanh số theo dòng hàng, sản phẩm, danh mục, khuyến mãi, khách hàng và nhà phân phối.",
        sources=("chi_tiet_don_hang_ban", "don_hang_ban", "san_pham", "danh_muc_san_pham", "khuyen_mai", "khach_hang", "nha_phan_phoi", "nhan_vien", "vung"),
        sql="""
            CREATE TABLE jt_chi_tiet_ban_hang_day_du AS
            SELECT
                ctdh.chi_tiet_id,
                ctdh.don_hang_id,
                dh.ngay_dat_hang,
                substr(dh.ngay_dat_hang, 1, 7) AS thang_dat_hang,
                dh.trang_thai AS trang_thai_don_hang,
                dh.khach_hang_id,
                kh.ten_khach_hang,
                dh.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                dh.nhan_vien_id,
                nv.ten_nhan_vien,
                ctdh.san_pham_id,
                sp.ten_san_pham,
                sp.don_vi_tinh,
                sp.danh_muc_id,
                dm.ten_danh_muc,
                ctdh.khuyen_mai_id,
                km.ten_khuyen_mai,
                km.phan_tram_giam,
                ctdh.so_luong,
                ctdh.don_gia,
                ctdh.thanh_tien
            FROM chi_tiet_don_hang_ban ctdh
            LEFT JOIN don_hang_ban dh ON dh.don_hang_id = ctdh.don_hang_id
            LEFT JOIN san_pham sp ON sp.san_pham_id = ctdh.san_pham_id
            LEFT JOIN danh_muc_san_pham dm ON dm.danh_muc_id = sp.danh_muc_id
            LEFT JOIN khuyen_mai km ON km.khuyen_mai_id = ctdh.khuyen_mai_id
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = dh.khach_hang_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = dh.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = dh.nhan_vien_id
        """,
        indexes=(
            "CREATE INDEX IF NOT EXISTS idx_jt_ctbh_ngay ON jt_chi_tiet_ban_hang_day_du(ngay_dat_hang)",
            "CREATE INDEX IF NOT EXISTS idx_jt_ctbh_sp ON jt_chi_tiet_ban_hang_day_du(san_pham_id)",
            "CREATE INDEX IF NOT EXISTS idx_jt_ctbh_dm ON jt_chi_tiet_ban_hang_day_du(danh_muc_id)",
        ),
    ),
    JoinedTable(
        name="jt_vieng_tham_khach_hang_day_du",
        purpose="Phân tích viếng thăm khách hàng theo kết quả, tuyến, nhân viên, nhà phân phối và địa bàn.",
        sources=("lich_su_vieng_tham", "khach_hang", "loai_khach_hang", "nha_phan_phoi", "nhan_vien", "tuyen_ban_hang", "vi_tri", "vung"),
        sql="""
            CREATE TABLE jt_vieng_tham_khach_hang_day_du AS
            SELECT
                vt.vieng_tham_id,
                vt.ngay_vieng_tham,
                substr(vt.ngay_vieng_tham, 1, 7) AS thang_vieng_tham,
                vt.ket_qua,
                vt.ghi_chu,
                vt.khach_hang_id,
                kh.ten_khach_hang,
                lkh.ten_loai AS loai_khach_hang,
                vt.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                vt.nhan_vien_id,
                nv.ten_nhan_vien,
                vt.tuyen_id,
                tbh.ma_tuyen,
                tbh.ten_tuyen,
                loc.tinh_thanh,
                loc.quan_huyen,
                loc.phuong_xa
            FROM lich_su_vieng_tham vt
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = vt.khach_hang_id
            LEFT JOIN loai_khach_hang lkh ON lkh.loai_khach_hang_id = kh.loai_khach_hang_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = vt.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = vt.nhan_vien_id
            LEFT JOIN tuyen_ban_hang tbh ON tbh.tuyen_id = vt.tuyen_id
            LEFT JOIN vi_tri loc ON loc.vi_tri_id = kh.vi_tri_id
        """,
        indexes=(
            "CREATE INDEX IF NOT EXISTS idx_jt_vt_ngay ON jt_vieng_tham_khach_hang_day_du(ngay_vieng_tham)",
            "CREATE INDEX IF NOT EXISTS idx_jt_vt_kq ON jt_vieng_tham_khach_hang_day_du(ket_qua)",
        ),
    ),
    JoinedTable(
        name="jt_khach_hang_phan_phoi_day_du",
        purpose="Hồ sơ khách hàng và quan hệ phục vụ hiện tại với nhà phân phối, nhân viên và tuyến.",
        sources=("nha_phan_phoi_khach_hang", "khach_hang", "loai_khach_hang", "nha_phan_phoi", "nhan_vien", "tuyen_ban_hang", "vi_tri", "vung"),
        sql="""
            CREATE TABLE jt_khach_hang_phan_phoi_day_du AS
            SELECT
                map.phan_phoi_khach_hang_id,
                map.ngay_mo,
                map.trang_thai AS trang_thai_quan_he,
                map.khach_hang_id,
                kh.ten_khach_hang,
                kh.dia_chi,
                kh.so_dien_thoai,
                kh.ngay_tao AS ngay_tao_khach_hang,
                lkh.ten_loai AS loai_khach_hang,
                map.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                map.nhan_vien_id,
                nv.ten_nhan_vien,
                map.tuyen_id,
                tbh.ma_tuyen,
                tbh.ten_tuyen,
                loc.tinh_thanh,
                loc.quan_huyen,
                loc.phuong_xa
            FROM nha_phan_phoi_khach_hang map
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = map.khach_hang_id
            LEFT JOIN loai_khach_hang lkh ON lkh.loai_khach_hang_id = kh.loai_khach_hang_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = map.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = map.nhan_vien_id
            LEFT JOIN tuyen_ban_hang tbh ON tbh.tuyen_id = map.tuyen_id
            LEFT JOIN vi_tri loc ON loc.vi_tri_id = kh.vi_tri_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_khpp_kh ON jt_khach_hang_phan_phoi_day_du(khach_hang_id)",),
    ),
    JoinedTable(
        name="jt_giao_hang_day_du",
        purpose="Phân tích giao hàng, thời gian giao, trạng thái giao và thông tin đơn hàng liên quan.",
        sources=("don_giao_hang", "don_hang_ban", "khach_hang", "nha_phan_phoi", "nhan_vien", "vung"),
        sql="""
            CREATE TABLE jt_giao_hang_day_du AS
            SELECT
                gh.giao_hang_id,
                gh.don_hang_id,
                gh.ngay_xuat_kho,
                gh.ngay_giao,
                gh.trang_thai AS trang_thai_giao_hang,
                CAST(julianday(gh.ngay_giao) - julianday(gh.ngay_xuat_kho) AS INTEGER) AS so_ngay_giao_hang,
                dh.ngay_dat_hang,
                dh.trang_thai AS trang_thai_don_hang,
                dh.tong_tien,
                dh.khach_hang_id,
                kh.ten_khach_hang,
                gh.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                dh.nhan_vien_id,
                nv.ten_nhan_vien
            FROM don_giao_hang gh
            LEFT JOIN don_hang_ban dh ON dh.don_hang_id = gh.don_hang_id
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = dh.khach_hang_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = gh.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = dh.nhan_vien_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_giao_hang_dh ON jt_giao_hang_day_du(don_hang_id)",),
    ),
    JoinedTable(
        name="jt_hang_tra_ve_day_du",
        purpose="Phân tích hàng trả về theo lý do, sản phẩm, danh mục, khách hàng, nhà phân phối và đơn gốc.",
        sources=("hang_tra_ve", "don_hang_ban", "san_pham", "danh_muc_san_pham", "khach_hang", "nha_phan_phoi", "vung"),
        sql="""
            CREATE TABLE jt_hang_tra_ve_day_du AS
            SELECT
                htv.tra_ve_id,
                htv.ngay_tra,
                substr(htv.ngay_tra, 1, 7) AS thang_tra,
                htv.ly_do,
                htv.so_luong AS so_luong_tra,
                htv.don_hang_id,
                dh.ngay_dat_hang,
                dh.trang_thai AS trang_thai_don_hang,
                dh.tong_tien,
                htv.san_pham_id,
                sp.ten_san_pham,
                sp.danh_muc_id,
                dm.ten_danh_muc,
                dh.khach_hang_id,
                kh.ten_khach_hang,
                dh.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung
            FROM hang_tra_ve htv
            LEFT JOIN don_hang_ban dh ON dh.don_hang_id = htv.don_hang_id
            LEFT JOIN san_pham sp ON sp.san_pham_id = htv.san_pham_id
            LEFT JOIN danh_muc_san_pham dm ON dm.danh_muc_id = sp.danh_muc_id
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = dh.khach_hang_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = dh.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_tra_ve_ngay ON jt_hang_tra_ve_day_du(ngay_tra)",),
    ),
    JoinedTable(
        name="jt_san_pham_gia_khuyen_mai",
        purpose="Tra cứu sản phẩm, danh mục, công ty, giá bán và khuyến mãi liên quan.",
        sources=("san_pham", "danh_muc_san_pham", "cong_ty", "bang_gia_san_pham", "khuyen_mai_san_pham", "khuyen_mai"),
        sql="""
            CREATE TABLE jt_san_pham_gia_khuyen_mai AS
            SELECT
                sp.san_pham_id,
                sp.ten_san_pham,
                sp.don_vi_tinh,
                sp.trang_thai AS trang_thai_san_pham,
                sp.cong_ty_id,
                ct.ten_cong_ty,
                sp.danh_muc_id,
                dm.ten_danh_muc,
                bg.gia_ban,
                bg.ngay_bat_dau AS ngay_bat_dau_gia,
                bg.ngay_ket_thuc AS ngay_ket_thuc_gia,
                km.khuyen_mai_id,
                km.ten_khuyen_mai,
                km.phan_tram_giam,
                km.ngay_bat_dau AS ngay_bat_dau_khuyen_mai,
                km.ngay_ket_thuc AS ngay_ket_thuc_khuyen_mai
            FROM san_pham sp
            LEFT JOIN cong_ty ct ON ct.cong_ty_id = sp.cong_ty_id
            LEFT JOIN danh_muc_san_pham dm ON dm.danh_muc_id = sp.danh_muc_id
            LEFT JOIN bang_gia_san_pham bg ON bg.san_pham_id = sp.san_pham_id
            LEFT JOIN khuyen_mai_san_pham kmsp ON kmsp.san_pham_id = sp.san_pham_id
            LEFT JOIN khuyen_mai km ON km.khuyen_mai_id = kmsp.khuyen_mai_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_spgm_sp ON jt_san_pham_gia_khuyen_mai(san_pham_id)",),
    ),
    JoinedTable(
        name="jt_doanh_so_theo_ngay_khach_hang",
        purpose="Tổng hợp doanh số hằng ngày theo khách hàng, nhà phân phối, nhân viên và địa bàn.",
        sources=("don_hang_ban", "khach_hang", "nha_phan_phoi", "nhan_vien", "vi_tri", "vung"),
        sql="""
            CREATE TABLE jt_doanh_so_theo_ngay_khach_hang AS
            SELECT
                dh.ngay_dat_hang AS ngay,
                substr(dh.ngay_dat_hang, 1, 7) AS thang,
                dh.khach_hang_id,
                kh.ten_khach_hang,
                dh.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                dh.nhan_vien_id,
                nv.ten_nhan_vien,
                loc.tinh_thanh,
                COUNT(*) AS so_don_hang,
                SUM(CASE WHEN dh.trang_thai = 'NORMAL' THEN 1 ELSE 0 END) AS so_don_binh_thuong,
                SUM(CASE WHEN dh.trang_thai = 'CANCELLED' THEN 1 ELSE 0 END) AS so_don_huy,
                ROUND(SUM(CASE WHEN dh.trang_thai = 'NORMAL' THEN dh.tong_tien ELSE 0 END), 2) AS doanh_so
            FROM don_hang_ban dh
            LEFT JOIN khach_hang kh ON kh.khach_hang_id = dh.khach_hang_id
            LEFT JOIN vi_tri loc ON loc.vi_tri_id = kh.vi_tri_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = dh.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = dh.nhan_vien_id
            GROUP BY dh.ngay_dat_hang, dh.khach_hang_id, dh.nha_phan_phoi_id, dh.nhan_vien_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_ds_ngay_kh_ngay ON jt_doanh_so_theo_ngay_khach_hang(ngay)",),
    ),
    JoinedTable(
        name="jt_hieu_suat_nhan_vien_ngay",
        purpose="Tổng hợp hiệu suất nhân viên theo ngày: viếng thăm, đơn hàng, khách có đơn và doanh số.",
        sources=("lich_su_vieng_tham", "don_hang_ban", "nhan_vien", "nha_phan_phoi", "vung"),
        sql="""
            CREATE TABLE jt_hieu_suat_nhan_vien_ngay AS
            SELECT
                vt.ngay_vieng_tham AS ngay,
                substr(vt.ngay_vieng_tham, 1, 7) AS thang,
                vt.nhan_vien_id,
                nv.ten_nhan_vien,
                vt.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                COUNT(*) AS so_luot_vieng_tham,
                COUNT(DISTINCT vt.khach_hang_id) AS so_khach_hang_vieng_tham,
                SUM(CASE WHEN vt.ket_qua = 'ORDERED' THEN 1 ELSE 0 END) AS so_luot_co_don,
                COUNT(DISTINCT dh.don_hang_id) AS so_don_hang,
                ROUND(SUM(CASE WHEN dh.trang_thai = 'NORMAL' THEN dh.tong_tien ELSE 0 END), 2) AS doanh_so
            FROM lich_su_vieng_tham vt
            LEFT JOIN don_hang_ban dh ON dh.vieng_tham_id = vt.vieng_tham_id
            LEFT JOIN nhan_vien nv ON nv.nhan_vien_id = vt.nhan_vien_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = vt.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            GROUP BY vt.ngay_vieng_tham, vt.nhan_vien_id, vt.nha_phan_phoi_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_hsnv_ngay ON jt_hieu_suat_nhan_vien_ngay(ngay)",),
    ),
    JoinedTable(
        name="jt_doanh_so_san_pham_thang",
        purpose="Tổng hợp doanh số, số lượng bán theo tháng, sản phẩm, danh mục, nhà phân phối và vùng.",
        sources=("chi_tiet_don_hang_ban", "don_hang_ban", "san_pham", "danh_muc_san_pham", "nha_phan_phoi", "vung"),
        sql="""
            CREATE TABLE jt_doanh_so_san_pham_thang AS
            SELECT
                substr(dh.ngay_dat_hang, 1, 7) AS thang,
                ctdh.san_pham_id,
                sp.ten_san_pham,
                sp.danh_muc_id,
                dm.ten_danh_muc,
                dh.nha_phan_phoi_id,
                npp.ten_nha_phan_phoi,
                npp.vung_id,
                v.ten_vung,
                SUM(ctdh.so_luong) AS so_luong_ban,
                ROUND(SUM(ctdh.thanh_tien), 2) AS doanh_so_dong_hang,
                COUNT(DISTINCT dh.don_hang_id) AS so_don_hang,
                COUNT(DISTINCT dh.khach_hang_id) AS so_khach_hang
            FROM chi_tiet_don_hang_ban ctdh
            LEFT JOIN don_hang_ban dh ON dh.don_hang_id = ctdh.don_hang_id
            LEFT JOIN san_pham sp ON sp.san_pham_id = ctdh.san_pham_id
            LEFT JOIN danh_muc_san_pham dm ON dm.danh_muc_id = sp.danh_muc_id
            LEFT JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = dh.nha_phan_phoi_id
            LEFT JOIN vung v ON v.vung_id = npp.vung_id
            WHERE dh.trang_thai != 'CANCELLED'
            GROUP BY substr(dh.ngay_dat_hang, 1, 7), ctdh.san_pham_id, dh.nha_phan_phoi_id
        """,
        indexes=("CREATE INDEX IF NOT EXISTS idx_jt_dssp_thang ON jt_doanh_so_san_pham_thang(thang)",),
    ),
)

BY_NAME = {t.name: t for t in JOINED_TABLES}


def names() -> list[str]:
    return [t.name for t in JOINED_TABLES]


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def refresh(con: sqlite3.Connection) -> dict[str, int]:
    """Drop/rebuild all materialized joined tables and return row counts."""
    for table in JOINED_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {_quote_ident(table.name)}")
    counts: dict[str, int] = {}
    for table in JOINED_TABLES:
        con.executescript(table.sql)
        for index_sql in table.indexes:
            con.execute(index_sql)
        counts[table.name] = int(con.execute(f"SELECT COUNT(*) FROM {_quote_ident(table.name)}").fetchone()[0])
    con.commit()
    return counts


def refresh_db(db_path: Path | None = None) -> dict[str, int]:
    db_path = Path(db_path or config.DB_PATH)
    con = sqlite3.connect(db_path)
    try:
        return refresh(con)
    finally:
        con.close()
