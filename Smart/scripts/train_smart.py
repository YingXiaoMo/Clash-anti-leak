import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from go_parser import GoTransformParser
from typing import Tuple, List, Optional

try:
    from SmartStore_Lite_Creator import SmartStoreCreator
    CREATOR_AVAILABLE = True
except ImportError:
    CREATOR_AVAILABLE = False
    print("FATAL: SmartStore-Lite-Creator library not found. Cannot encode V3 model.")


# File path configurations
DATA_FILE = 'smart_weight_data.csv' 
GO_FILE = 'transform.go'
MODEL_FILE = 'Model.bin'

# Feature transformation configurations (KEEP THESE AS THEY ARE)
STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 
    'last_used_seconds', 'traffic_density'
]
ROBUST_SCALER_FEATURES = ['success', 'failure']

# LightGBM Model parameters
LGBM_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.03, 'random_state': 42, 'n_jobs': -1, 'device': 'gpu'
}
EARLY_STOPPING_ROUNDS = 100




def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads and cleans data from a CSV file."""
    print(f"--> Loading data: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"    Original data loaded successfully, {len(data)} records.")
    except FileNotFoundError:
        print(f"    ERROR: Data file '{file_path}' not found!")
        return None

    data.dropna(subset=['weight'], inplace=True)
    data = data[data['weight'] > 0].copy()
    print(f"    Clean records remaining: {len(data)}")
    return data

def extract_features_from_preprocessed(data: pd.DataFrame, feature_order: List[str]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Extracts X and y from preprocessed CSV."""
    print("--> Extracting features (X) and target (y)...")
    try:
        X = data[feature_order]
        y = data['weight']
        print("    Successfully extracted features and target.")
        return X, y
    except KeyError as e:
        print(f"    ERROR: Missing required feature column in data: {e}")
        return None, None

def apply_feature_transforms(X: pd.DataFrame, feature_order: List[str]) -> Tuple[pd.DataFrame, StandardScaler, RobustScaler]:
    """Applies StandardScaler and RobustScaler to the feature matrix."""
    print("--> Applying feature transformations...")
    X_scaled = X.copy()
    
    std_scaler = StandardScaler()
    X_scaled[STD_SCALER_FEATURES] = std_scaler.fit_transform(X_scaled[STD_SCALER_FEATURES])
    print(f"    Applied StandardScaler to {len(STD_SCALER_FEATURES)} features.")

    robust_scaler = RobustScaler()
    X_scaled[ROBUST_SCALER_FEATURES] = robust_scaler.fit_transform(X_scaled[ROBUST_SCALER_FEATURES])
    print(f"    Applied RobustScaler to {len(ROBUST_SCALER_FEATURES)} features.")
    
    return X_scaled, std_scaler, robust_scaler

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.Booster:
    """Trains the LightGBM model."""
    print("--> Training LightGBM model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    print(f"    Training complete. Best iteration: {model.best_iteration}")
    return model


def save_model_and_config(model: lgb.Booster, std_scaler: StandardScaler, robust_scaler: RobustScaler, feature_order: List[str]):
    """Saves the model by encoding it into the V3 binary format using SmartStoreCreator."""
    print("--> Encoding LightGBM model into Mihomo V3 binary format...")

    if not CREATOR_AVAILABLE:
        raise Exception("❌ SmartStore-Lite-Creator 库未导入，无法执行 V3 编码。")


    temp_lgbm_model_path = "lgbm_model.txt"
    model.save_model(temp_lgbm_model_path, num_iteration=model.best_iteration)
    

    feature_to_index = {name: i for i, name in enumerate(feature_order)}
    
    config_data = {
        # StandardScaler config
        'std': {
            'features': [feature_to_index[f] for f in STD_SCALER_FEATURES],
            'mean': std_scaler.mean_.tolist(),
            'scale': std_scaler.scale_.tolist(),
        },
        # RobustScaler config
        'robust': {
            'features': [feature_to_index[f] for f in ROBUST_SCALER_FEATURES],
            'center': robust_scaler.center_.tolist(),
            'scale': robust_scaler.scale_.tolist(),
        }
    }

    try:

        creator = SmartStoreCreator(
            lgbm_model_path=temp_lgbm_model_path,
            feature_order=feature_order,
            scaler_config=config_data,
            output_bin_path=MODEL_FILE
        )

        creator.create_smartstore(version=3) 
        

        file_size = os.path.getsize(MODEL_FILE)
        print(f"🎉 V3 二进制模型已成功保存到: {MODEL_FILE} ({file_size} 字节)")

    except Exception as e:
        raise Exception(f"❌ V3 二进制模型编码失败: {e}")
    finally:
        # 5. 清理临时文件
        if os.path.exists(temp_lgbm_model_path):
            os.remove(temp_lgbm_model_path)




def main():
    """Main function to execute all steps."""
    print("--- Mihomo V3 Model Training Start ---")
    

    if not CREATOR_AVAILABLE:
        print("FATAL ERROR: 'SmartStore-Lite-Creator' not installed. Please check your GitHub Actions setup.")
        return


    if not os.path.exists(GO_FILE):
        print(f"FATAL ERROR: Go file '{GO_FILE}' not found. Cannot determine feature order.")
        return

    try:
        parser = GoTransformParser(GO_FILE)
        feature_order = parser.get_feature_order()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return
        
    full_data = load_and_clean_data(DATA_FILE)
    if full_data is None:
        return

    result = extract_features_from_preprocessed(full_data, feature_order)
    if result[0] is None:
        return
    X, y = result

    X_scaled, std_scaler, robust_scaler = apply_feature_transforms(X, feature_order)

    # Note: Using the scaled data X_scaled for train_test_split as features are now ready
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, X_test, y_test)
    
    try:
        save_model_and_config(model, std_scaler, robust_scaler, feature_order)
    except Exception as e:
        print(f"Model Save Error: {e}")

        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        return
        
    print("\n🎉 --- Training Completed Successfully --- 🎉")
    print(f"Final Model '{MODEL_FILE}' is V3 encoded and ready for deployment!")

if __name__ == "__main__":
    main()
