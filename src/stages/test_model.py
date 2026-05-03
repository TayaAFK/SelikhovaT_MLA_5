import pandas as pd
import joblib
import json
import numpy as np
import yaml
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.getcwd())

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def test_model():
    config = load_config("src/config.yaml")
    
    df_test = pd.read_csv(config['data_split']['testset_path'])
    target = config['train']['target_column']
    
    X_test = df_test.drop(columns=[target]).values
    y_test_actual = df_test[target].values
    
    model = joblib.load(config['train']['model_path'])
    power_trans = joblib.load(config['train']['power_path'])
    
    y_pred_scaled = model.predict(X_test)
    y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    (rmse, mae, r2) = eval_metrics(y_test_actual, y_pred)
    
    print(f"Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    print("First 3 predictions:", y_pred[:3].flatten())
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    test_model()
