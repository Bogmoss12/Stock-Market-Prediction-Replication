import xgboost as xgb

def build_xgboost_model():
    # Construct XGBoost Regressor using parameters
    model = xgb.XGBRegressor(
        n_estimators=2000,
        colsample_bytree=1,
        learning_rate=0.2,
        max_depth=3,
        gamma=0,
        objective='reg:squarederror',
        n_jobs=-1
    )
    return model