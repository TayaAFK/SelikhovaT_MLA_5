import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import sys
import os
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature

sys.path.append(os.getcwd())

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(config):
    df_train = pd.read_csv(config['data_split']['trainset_path'])
    df_test  = pd.read_csv(config['data_split']['testset_path'])
    target = config['train']['target_column']

    X_train, y_train = df_train.drop(columns=[target]).values, df_train[target].values
    X_val, y_val = df_test.drop(columns=[target]).values, df_test[target].values

    power_trans = PowerTransformer()
    y_train_scaled = power_trans.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = power_trans.transform(y_val.reshape(-1, 1))

    mlflow.set_experiment("sleep_disorder_model_experiment")
    
    with mlflow.start_run():
        if config['train']['model_type'] == "tree":
            lr_pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', ExtraTreesRegressor(random_state=42))
            ])
            params = {'model__n_estimators': config['train']['n_estimators']}
        else:
            lr_pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', SGDRegressor(random_state=42))
            ])
            params = {
                'model__alpha': config['train']['alpha'],
                'model__fit_intercept': [False, True]
            }

        clf = GridSearchCV(lr_pipe, params, cv=config['train']['cv'], n_jobs=4)
        clf.fit(X_train, y_train_scaled.reshape(-1))
        
        best_model = clf.best_estimator_
        
        y_pred_scaled = best_model.predict(X_val)
        y_pred_original = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        (rmse, mae, r2) = eval_metrics(y_val, y_pred_original)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        print(f"R2 Score: {r2}")

        predictions = best_model.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        with open(config['train']['model_path'], "wb") as file:
            pickle.dump(best_model, file)

        with open(config['train']['power_path'], "wb") as file:
            pickle.dump(power_trans, file)

    print("Training stage completed.")
