import pandas as pd

def load_data():
    # 1️⃣ Dosyaları oku
    sales = pd.read_csv("data/sales.csv", parse_dates=["date"])
    inventory = pd.read_csv("data/inventory.csv")
    staff = pd.read_csv("data/staff.csv", parse_dates=["date"])

    # 2️⃣ Satış + Stok birleştir
    df = sales.merge(inventory, on="product_id", how="left")

    # 3️⃣ Personel bilgisini ekle (tarihe göre)
    df = df.merge(staff[["date", "sales_team_size", "overtime_hours"]], on="date", how="left")

    # 4️⃣ Hesaplamalar (kar, gelir vs.)
    df["revenue"] = df["unit_price"] * df["units_sold"] * (1 - df["discount_rate"])
    df["cogs"] = df["units_sold"] * df["unit_cost"]
    df["profit"] = df["revenue"] - df["cogs"]
    df["profit_margin"] = (df["profit"] / df["revenue"]).round(3)

    print("\n--- BİRLEŞMİŞ VERİLER ---")
    print(df.head())
    return df

if __name__ == "__main__":
    load_data()
