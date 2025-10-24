import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from features import load_data

def train_model():
    #1️⃣ Veriyi al
    df = load_data()

    # 2️⃣ Girdi ve hedefi ayır
    X = df[["unit_price", "units_sold", "discount_rate", "stock_qty", "unit_cost", "lead_time_days", "holding_cost_rate", "sales_team_size", "overtime_hours"]]
    y = df["profit_margin"]

    # 3️⃣ Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4️⃣ Modeli oluştur ve eğit
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 5️⃣ Test verisiyle ölç
    y_pred = model.predict(X_test)
    print("\n--- MODEL BAŞARISI ---")
    print("R2 Skoru:", round(r2_score(y_test, y_pred), 3))
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))

    # 6️⃣ Modeli kaydet
    joblib.dump(model, "../models/profit_model.joblib")
    print("Model kaydedildi: ../models/profit_model.joblib")

if __name__ == "__main__":
    train_model()
