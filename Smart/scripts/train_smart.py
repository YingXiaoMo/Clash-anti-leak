import os
import subprocess
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from go_parser import GoTransformParser 
from typing import Tuple, List, Optional
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))



GO_FILE = os.path.join(BASE_DIR, "../go_transform/transform.go")
MODEL_FILE = os.path.join(BASE_DIR, "../../models/Model.bin")
DATA_DIR = os.path.join(BASE_DIR, "../../data") 

STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes',
    'last_used_seconds', 'traffic_density'
]
ROBUST_SCALER_FEATURES = ['success', 'failure']

LGBM_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.03,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'cpu'
}
EARLY_STOPPING_ROUNDS = 100


def load_and_clean_all_csvs(data_dir: str) -> Optional[pd.DataFrame]:
    """批量加载 CSV 文件并进行基本的编码清洗。"""
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not all_files:
        print("    未找到 CSV 文件")
        return None

    df_list = []
    for file in all_files:
        print(f"--> 处理文件: {file}")
        try:
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            print("    UTF-8 解码失败，尝试 Latin-1 编码")
            df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
        
        df_list.append(df)

    if not df_list:
        print("    没有有效数据")
        return None

    data = pd.concat(df_list, ignore_index=True)
    print(f"    总共 {len(data)} 条有效记录")
    return data


def extract_features_from_preprocessed(data: pd.DataFrame, feature_order: List[str]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
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

def save_model_and_config(model: lgb.Booster, std_scaler: StandardScaler, robust_scaler: RobustScaler, feature_order: List[str]):
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    model.save_model(MODEL_FILE, num_iteration=model.best_iteration)
    print(f"    模型已保存: {MODEL_FILE}")


def main():
    print("--- Mihomo Smart 模型训练开始 ---")
    print(f"📂 使用数据文件夹: {DATA_DIR}")
    print(f"⚙️  Go 配置路径: {os.path.abspath(GO_FILE)}") 

    # 1. 加载数据
    data = load_and_clean_all_csvs(DATA_DIR)
    if data is None:
        return

    # 2. 特征工程
    parser = GoTransformParser(GO_FILE) 
    feature_order = parser.get_feature_order()

    X, y = extract_features_from_preprocessed(data, feature_order)
    if X is None:
        return

    X_scaled, std_scaler, robust_scaler = apply_feature_transforms(X)

    # 3. 训练模型
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, X_test, y_test)

    # 4. 保存模型
    save_model_and_config(model, std_scaler, robust_scaler, feature_order)
    
    print("\n🎉 --- 训练完成 --- 🎉")

if __name__ == "__main__":
    main()
