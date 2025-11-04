# ==============================================================================
# Mihomo V3 智能权重模型训练 (已集成 V3 二进制编码)
# 出品：安格视界
# 功能：基于历史数据训练 LightGBM 回归模型，并将模型和Scaler配置打包为V3格式。
# ==============================================================================

import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional, Dict, Any

# ⚡ 重点：导入 smart_store_creator 库，用于生成 V3 二进制文件
# 假设 PyPI 包名 'smart-store-creator' 对应的 Python 模块名是 'smart_store_creator'
try:
    from smart_store_creator import SmartStoreCreator
    CREATOR_AVAILABLE = True
except ImportError:
    CREATOR_AVAILABLE = False
    print("FATAL: smart_store_creator library not found. V3 encoding will fail.")

# ==============================================================================
# 1. Go 源码解析模块 (GoTransformParser)
# ==============================================================================

class GoTransformParser:
    """
    Go 源码解析器
    
    负责解析 Go 语言源文件中的特征顺序定义，提取 getDefaultFeatureOrder 函数中
    的特征映射关系。
    """
    
    def __init__(self, go_file_path: str):
        """
        初始化解析器
        """
        try:
            with open(go_file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            print(f"成功加载 Go 源文件: {go_file_path}")
        except FileNotFoundError:
            # 在 GitHub Actions 中，如果 transform.go 不在 Smart/scripts/ 中，这里会失败
            raise FileNotFoundError(
                f"Go 源文件 '{go_file_path}' 没找到。请确保文件存在于 Smart/scripts/ 目录中。"
            )
        
        self.feature_order = self._parse_feature_order()
    
    def _parse_feature_order(self) -> List[str]:
        """
        解析特征顺序
        """
        print("开始解析 getDefaultFeatureOrder 函数...")
        
        function_pattern = r'func getDefaultFeatureOrder\(\) map\[int\]string \{\s*return map\[int\]string\{(.*?)\}\s*\}'
        match = re.search(function_pattern, self.content, re.DOTALL)
        
        if not match:
            print("警告: 没找到 getDefaultFeatureOrder 函数，使用预定义特征顺序")
            return self._get_fallback_feature_order()
        
        function_body = match.group(1)
        feature_pairs = re.findall(r'(\d+):\s*"([^"]+)"', function_body)
        
        if not feature_pairs:
            print("警告: 函数体中无有效特征定义，使用预定义特征顺序")
            return self._get_fallback_feature_order()
        
        feature_dict = {int(index): name for index, name in feature_pairs}
        sorted_features = [feature_dict[i] for i in sorted(feature_dict.keys())]
        
        print(f"成功解析 {len(sorted_features)} 个特征的顺序定义")
        return sorted_features
    
    def get_feature_order(self) -> List[str]:
        """
        获取特征顺序列表
        """
        return self.feature_order
    
    def _get_fallback_feature_order(self) -> List[str]:
        """
        预定义特征顺序 (作为优雅降级)
        """
        # 完整的特征列表，用于 Go 源码解析失败时的备选
        return [
            'success', 'failure', 'connect_time', 'latency', 'upload_mb', 'download_mb', 
            'duration_minutes', 'last_used_seconds', 'is_udp', 'is_tcp', 'asn_feature', 
            'country_feature', 'address_feature', 'port_feature', 'traffic_ratio', 
            'traffic_density', 'connection_type_feature', 'asn_hash', 'host_hash', 
            'ip_hash', 'geoip_hash'
        ]

# ==============================================================================
# 2. 系统配置参数
# ==============================================================================

# 文件路径配置 (已修正为 GitHub Actions 环境的相对路径)
# 脚本在 Smart/scripts/ 中运行
DATA_FILE = '../data/smart_weight_data.csv'  # 数据位于仓库根目录的 data 文件夹
GO_FILE = 'transform.go'                     # transform.go 位于 Smart/scripts/ 目录
MODEL_FILE = '../../models/Model.bin'        # 模型输出到仓库根目录的 models 文件夹

# 特征预处理配置
STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 
    'last_used_seconds', 'traffic_density'
]
ROBUST_SCALER_FEATURES = ['success', 'failure']

# LightGBM模型超参数配置
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'random_state': 42,
    'n_jobs': -1 
}

EARLY_STOPPING_ROUNDS = 100

# ==============================================================================
# 3. 核心功能模块
# ==============================================================================

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    数据加载与预处理
    """
    print(f"开始加载数据文件: {file_path}")
    
    # 检查文件路径是否正确 (在 GitHub Actions 中，需要确保相对路径正确)
    absolute_data_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_data_path):
        # 这里的 os.getcwd() 应该是 Smart/scripts/
        print(f"错误: 数据文件未找到，预期路径: {absolute_data_path}")
        return None

    try:
        data = pd.read_csv(absolute_data_path, on_bad_lines='skip')
        print(f"数据加载完成，原始记录数: {len(data)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

    original_count = len(data)
    data.dropna(subset=['weight'], inplace=True)
    data = data[data['weight'] > 0].copy()
    final_count = len(data)
    filtered_count = original_count - final_count
    
    print(f"数据清洗完成: {original_count} → {final_count} 条记录 (过滤 {filtered_count} 条)")
    return data

def extract_features_from_preprocessed(data: pd.DataFrame, feature_order: List[str]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    特征矩阵构建
    """
    print("开始构建特征矩阵和目标变量...")
    
    try:
        X = data[feature_order]
        y = data['weight']
        print(f"特征提取完成 - 特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y
        
    except KeyError as e:
        print(f"特征提取失败: 缺少必要的特征列 {e}")
        return None, None

def apply_feature_transforms(X: pd.DataFrame, feature_order: List[str]) -> Tuple[pd.DataFrame, StandardScaler, RobustScaler]:
    """
    特征标准化处理
    """
    print("开始特征标准化处理...")
    X_transformed = X.copy()
    
    # 1. StandardScaler
    std_scaler = StandardScaler()
    std_features_available = [f for f in STD_SCALER_FEATURES if f in X_transformed.columns]
    
    if std_features_available:
        # 只对数据框中实际存在的列进行 fit_transform
        X_transformed[std_features_available] = std_scaler.fit_transform(X_transformed[std_features_available])
        print(f"StandardScaler 处理完成，影响特征数: {len(std_features_available)}")
    
    # 2. RobustScaler
    robust_scaler = RobustScaler()
    robust_features_available = [f for f in ROBUST_SCALER_FEATURES if f in X_transformed.columns]
    
    if robust_features_available:
        # 只对数据框中实际存在的列进行 fit_transform
        X_transformed[robust_features_available] = robust_scaler.fit_transform(X_transformed[robust_features_available])
        print(f"RobustScaler 处理完成，影响特征数: {len(robust_features_available)}")
    
    return X_transformed, std_scaler, robust_scaler

def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.Booster:
    """
    LightGBM 模型训练，返回原生 Booster 对象
    """
    print("开始 LightGBM 模型训练...")
    
    # 模型的训练部分应该使用 lgb.train 而不是 LGBMRegressor 
    # 因为 lgb.train 返回原生 Booster 对象，方便后续保存和 V3 编码
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    # 使用 Booster 对象评估 R2 得分
    from sklearn.metrics import r2_score
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"模型训练完成。最佳迭代次数: {model.best_iteration}")
    print(f"训练集R²得分: {train_r2:.4f}")
    print(f"测试集R²得分: {test_r2:.4f}")
    
    if test_r2 > 0.8:
        print("模型性能评估: 优秀")
    else:
        print("模型性能评估: 良好或需要改进")
    
    return model

def save_model_and_config(model: lgb.Booster, std_scaler: StandardScaler, robust_scaler: RobustScaler, feature_order: List[str], model_file: str) -> None:
    """
    模型序列化保存 (使用 V3 二进制编码)
    
    将 LightGBM 模型和 Scaler 配置一起打包为 Mihomo V3 要求的二进制格式。
    """
    print("--> 编码 LightGBM 模型到 Mihomo V3 二进制格式...")

    if not CREATOR_AVAILABLE:
        raise Exception("❌ smart_store_creator 库未导入，无法执行 V3 编码。请检查 Actions 依赖安装。")

    # 1. 临时保存 LightGBM 模型为文本，这是 V3 Creator 的输入要求
    temp_lgbm_model_path = "temp_lgbm_model.txt"
    model.save_model(temp_lgbm_model_path, num_iteration=model.best_iteration)
    
    # 2. 准备配置数据
    feature_to_index = {name: i for i, name in enumerate(feature_order)}
    
    # 准备 StandardScaler config
    std_indices = [feature_to_index[f] for f in STD_SCALER_FEATURES if f in feature_to_index]
    # 匹配 scaler 的内部特征顺序和 full_feature_list 的顺序
    std_data_map = {f: i for i, f in enumerate(STD_SCALER_FEATURES) if f in feature_order}
    std_mean = [std_scaler.mean_[std_data_map[f]] for f in STD_SCALER_FEATURES if f in std_data_map]
    std_scale = [std_scaler.scale_[std_data_map[f]] for f in STD_SCALER_FEATURES if f in std_data_map]

    # 准备 RobustScaler config
    robust_indices = [feature_to_index[f] for f in ROBUST_SCALER_FEATURES if f in feature_to_index]
    robust_data_map = {f: i for i, f in enumerate(ROBUST_SCALER_FEATURES) if f in feature_order}
    robust_center = [robust_scaler.center_[robust_data_map[f]] for f in ROBUST_SCALER_FEATURES if f in robust_data_map]
    robust_scale = [robust_scaler.scale_[robust_data_map[f]] for f in ROBUST_SCALER_FEATURES if f in robust_data_map]

    scaler_config_data: Dict[str, Any] = {
        'std': {
            'features': std_indices,
            'mean': std_mean,
            'scale': std_scale,
        },
        'robust': {
            'features': robust_indices,
            'center': robust_center,
            'scale': robust_scale,
        }
    }

    try:
        # 3. 初始化编码器
        creator = SmartStoreCreator(
            lgbm_model_path=temp_lgbm_model_path,
            feature_order=feature_order, 
            scaler_config=scaler_config_data,
            output_bin_path=model_file
        )
            
        # 4. 确保目标目录存在
        output_dir = os.path.dirname(os.path.abspath(model_file))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # ⚡ 执行 V3 编码
        creator.create_smartstore(version=3) 
        
        # 5. 检查文件大小并确认
        file_size = os.path.getsize(model_file)
        print(f"🎉 V3 二进制模型已成功保存到: {model_file} ({file_size} 字节)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"❌ V3 二进制模型编码失败: {e}")
    finally:
        # 6. 清理临时文件
        if os.path.exists(temp_lgbm_model_path):
            os.remove(temp_lgbm_model_path)


# ==============================================================================
# 4. 主程序流程控制
# ==============================================================================

def main() -> None:
    """
    主程序入口
    """
    print("=" * 60)
    print("Mihomo V3 智能权重模型训练")
    print("=" * 60)
    
    if not CREATOR_AVAILABLE:
        print("致命错误: V3 编码库 'smart_store_creator' 未安装。程序终止。")
        return
    
    # 步骤1: Go 源码解析
    print("\n[步骤1] Go 源码解析")
    try:
        parser = GoTransformParser(GO_FILE)
        feature_order = parser.get_feature_order()
        print(f"特征顺序解析完成，共 {len(feature_order)} 个特征")
    except Exception as e:
        print(f"Go 源码解析失败: {e}")
        print("程序终止")
        return
    
    # 步骤2: 数据加载与清洗
    print("\n[步骤2] 数据加载与清洗")
    dataset = load_and_clean_data(DATA_FILE)
    if dataset is None:
        print("数据加载失败，程序终止")
        return
    
    # 步骤3: 特征提取
    print("\n[步骤3] 特征提取")
    extraction_result = extract_features_from_preprocessed(dataset, feature_order)
    if extraction_result[0] is None:
        print("特征提取失败，程序终止")
        return
    
    X, y = extraction_result
    
    # 步骤4: 特征标准化
    print("\n[步骤4] 特征标准化")
    X_processed, std_scaler, robust_scaler = apply_feature_transforms(X, feature_order)
    
    # 步骤5: 数据集划分
    print("\n[步骤5] 训练测试集划分")
    # 注意：这里使用标准化后的数据 X_processed 进行划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=0.2,
        random_state=42
    )
    print(f"数据划分完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 步骤6: 模型训练 (返回 lgb.Booster)
    print("\n[步骤6] 模型训练")
    trained_model_booster = train_lgbm_model(X_train, y_train, X_test, y_test)
    
    # 步骤7: V3 编码保存 (使用 lgb.Booster)
    print("\n[步骤7] 模型 V3 编码与保存")
    try:
        # 将 LightGBM 的原生 Booster 对象、Scaler、特征顺序和目标文件路径传入
        save_model_and_config(trained_model_booster, std_scaler, robust_scaler, feature_order, MODEL_FILE)
    except Exception as e:
        print(f"模型保存失败: {e}")
        # 如果编码失败，将 Model.bin 删除，防止上传无效文件
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        return
        
    # 训练完成总结
    print("\n" + "=" * 60)
    print("模型训练流程完成")
    print(f"输出文件: {MODEL_FILE}")
    print("模型可进行生产环境部署")
    print("=" * 60)

# ==============================================================================
# 程序入口点
# ==============================================================================

if __name__ == "__main__":
    main()
