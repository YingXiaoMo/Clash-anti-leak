import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from go_parser import GoTransformParser
from typing import Tuple, List, Optional

# ===================================================================
# 1. 配置中心
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "../data/smart_weight_data.csv")
CLEAN_DATA_FILE = os.path.join(BASE_DIR, "../data/smart_weight_data_clean.csv")
GO_FILE = os.path.join(BASE_DIR, "../go_transform/transform.go")
MODEL_FILE = os.path.join(BASE_DIR, "../models/Model.bin")

STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes',
    'last_used_seconds', 'traffic_density'
]
ROBUST_SCALER_FEATURES = ['success', 'failure']

LGBM_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.03, 'random_state': 42, 'n_jobs': -1, 'device': 'gpu'
}
EARLY_STOPPING_ROUNDS = 100
EXPECTED_COLS = 37  # 数据列数，必须和 CSV 实际列数一致


# ===================================================================
# 2. 数据清洗
# ===================================================================
def clean_csv(input_file: str, output_file: str) -> None:
    print(f"--> 清洗 CSV 文件: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if line.count(",") == EXPECTED_COLS - 1:  # 列数 = 逗号数 + 1
                f_out.write(line)
            else:
                print(f"    跳过第 {i + 1} 行 (列数异常)")
    print(f"    清洗完成，新文件: {output_file}")


# ===================================================================
# 3. 数据加载与特征处理
# ===================================================================
def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(file_path)
        data.dropna(subset=['weight'], inplace=True)
        data = data[data['weight'] > 0].copy()
        print(f"    数据加载成功，共 {len(data)} 条有效记录。")
        return data
    except Exception as e:
        print(f"    数据加载失败: {e}")
        return None


def extract_features_from_preprocessed(data: pd.DataFrame, feature_order: List[str]) -> Optional[
    Tuple[pd.DataFrame, pd.Series]]:
    try:
        X = data[feature_order]
        y = data['weight']
        return X, y
    except KeyError as e:
        print(f"    缺少必要特征列: {e}")
        return None, None


def apply_feature_transforms(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, RobustScaler]:
    X_scaled = X.copy()

    std_scaler = StandardScaler()
    X_scaled[STD_SCALER_FEATURES] = std_scaler.fit_transform(X_scaled[STD_SCALER_FEATURES])

    robust_scaler = RobustScaler()
    X_scaled[ROBUST_SCALER_FEATURES] = robust_scaler.fit_transform(X_scaled[ROBUST_SCALER_FEATURES])

    return X_scaled, std_scaler, robust_scaler


# ===================================================================
# 4. 模型训练与保存
# ===================================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.Booster:
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)]
    )
    return model


def save_model_and_config(model: lgb.Booster, std_scaler: StandardScaler, robust_scaler: RobustScaler,
                          feature_order: List[str]):
    model.save_model(MODEL_FILE, num_iteration=model.best_iteration)
    print(f"    模型已保存: {MODEL_FILE}")


# ===================================================================
# 5. 主执行流程
# ===================================================================
def main():
    print("--- Mihomo Smart 模型训练开始 ---")

    # 1. 清洗 CSV
    clean_csv(DATA_FILE, CLEAN_DATA_FILE)

    # 2. 初始化特征顺序
    parser = GoTransformParser(GO_FILE)
    feature_order = parser.get_feature_order()

    # 3. 加载数据
    data = load_and_clean_data(CLEAN_DATA_FILE)
    if data is None:
        return

    # 4. 提取特征和目标
    X, y = extract_features_from_preprocessed(data, feature_order)
    if X is None:
        return

    # 5. 特征变换
    X_scaled, std_scaler, robust_scaler = apply_feature_transforms(X)

    # 6. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 7. 训练模型
    model = train_model(X_train, y_train, X_test, y_test)

    # 8. 保存模型
    save_model_and_config(model, std_scaler, robust_scaler, feature_order)

    print("\n🎉 --- 训练完成 --- 🎉")


if __name__ == "__main__":
    main()
