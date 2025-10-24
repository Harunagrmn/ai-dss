import joblib
import numpy as np

FEATURE_ORDER = [
    "unit_price", "units_sold", "discount_rate",
    "stock_qty", "unit_cost", "lead_time_days",
    "holding_cost_rate", "sales_team_size", "overtime_hours"
]

def load_model(path="models/profit_model.joblib"):
    return joblib.load(path)

def predict_profit(model, inputs: dict):
    x = np.array([[inputs[k] for k in FEATURE_ORDER]])
    return float(model.predict(x)[0])

def optimize_profit(model, base_inputs: dict):
    best_profit = -999
    best_combo = None

    for price_change in [-0.05, 0, 0.05, 0.1]:   # fiyat değişimi
        for discount_change in [-0.05, 0, 0.05]: # indirim değişimi
            for stock_change in [-0.2, 0, 0.1]:  # stok değişimi
                test = base_inputs.copy()
                test["unit_price"] *= (1 + price_change)
                test["discount_rate"] = max(0, test["discount_rate"] + discount_change)
                test["stock_qty"] *= (1 + stock_change)
                profit = predict_profit(model, test)
                if profit > best_profit:
                    best_profit = profit
                    best_combo = {
                        "unit_price_change": f"{price_change*100:+.0f}%",
                        "discount_change": f"{discount_change*100:+.0f}%",
                        "stock_change": f"{stock_change*100:+.0f}%",
                        "predicted_profit": round(best_profit*100, 2)
                    }
    return best_combo
