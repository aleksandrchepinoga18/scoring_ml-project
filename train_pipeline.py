from src.data_preparation import *
from src.eda import run_eda
from src.train import train_lightgbm
from src.evaluate import evaluate_model
from src.evaluate import shap_analysis, get_risky_wallets  # если они в том же файле — см. примечание ниже
import pandas as pd

if __name__ == "__main__":
    # Загрузка и очистка
    df = load_and_clean_data("data/dataset.parquet")
    df = remove_high_corr_features(df)

    # EDA
    run_eda(df)

    # Разделение
    train, val, test = split_data(df, random_state=42)
    feature_cols = prepare_features(df)
    
    X_train_raw = train[feature_cols].copy()
    y_train = train['target'].copy()
    X_val_raw = val[feature_cols].copy()
    y_val = val['target'].copy()
    X_test_raw = test[feature_cols].copy()
    y_test = test['target'].copy()

    # Заполнение пропусков
    X_train, X_val, X_test = fill_missing_with_median(X_train_raw, X_val_raw, X_test_raw)

    # 🔍 Отладочная информация (оставь, если хочешь видеть структуру данных)
    print("\n🔍 Оставленные признаки:", len(X_train.columns))
    print(X_train.columns.tolist())

    print("✅ Разделение завершено:")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Обучение
    model, best_params, numeric_features = train_lightgbm(X_train, y_train)

    # Оценка с оптимальным порогом
    best_threshold = evaluate_model(
        model, 
        X_train, 
        X_val[numeric_features], 
        X_test[numeric_features], 
        y_train, y_val, y_test,
        name="LightGBM"
    )

    # SHAP и рискованные кошельки
    shap_analysis(model, X_test[numeric_features], name="LightGBM")
    X_full = pd.concat([X_train, X_val, X_test])[numeric_features]
    df_full = pd.concat([train, val, test]).reset_index(drop=True)
    get_risky_wallets(model, X_full, df_full, name="LightGBM")

    # Сохраняем порог
    import joblib
    joblib.dump(best_threshold, "models/lightgbm_best_threshold.pkl")
    print(f"\n✅ Лучший порог сохранён: {best_threshold:.4f}")
    print(f"\n✅ Использовано признаков: {len(numeric_features)}")
    print("✅ Обучение завершено!")

    print("\n🔍 Пример входа для API:")
    print(X_test.iloc[0].to_dict())