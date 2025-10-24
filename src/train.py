import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.features import load_data  # ✅ doğru import (Cloud uyumlu)

def train_model():
    """Yapay zekâ modelini eğitir ve kaydeder."""
    print("📊 Veriler yükleniyor...")
    df = load_data()

    # Girdi (X) ve hedef (y) değişkenlerini ayır
    X = df[[
        "unit_price", "units_sold", "discount_rate",
        "stock_qty", "unit_cost", "lead_time_days",
        "holding_cost_rate", "sales_team_size", "overtime_hours"
    ]]
    y = df["profit_margin"]

    # Eğitim ve test verisini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modeli oluştur ve eğit
    print("🧠 Model eğitiliyor...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Tahminleri yap ve performansı ölç
    y_pred = model.predict(X_test)
    r2 = round(r2_score(y_test, y_pred), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 3)

    print("\n--- MODEL BAŞARISI ---")
    print(f"R2 Skoru: {r2}")
    print(f"MAE: {mae}")

    # models klasörü yoksa oluştur
    os.makedirs("models", exist_ok=True)

    # Modeli kaydet
    model_path = os.path.join("models", "profit_model.joblib")
    joblib.dump(model, model_path)

    print(f"✅ Model kaydedildi: {model_path}")
    return model

# Manuel çalıştırmak istersen:
if __name__ == "__main__":
    train_model()
