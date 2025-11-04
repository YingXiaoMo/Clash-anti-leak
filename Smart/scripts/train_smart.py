# combined_train.py
# ==============================================================================
# Mihomo 智能权重模型训练
# 出品：安格视界
# 功能：基于历史数据训练 LightGBM 回群模型，用于预测代理节点权重
# ==============================================================================

import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional

# ==============================================================================
# 1. Go 源码解析模块
# ==============================================================================

class GoTransformParser:
    """
    Go 源码解析器
    
    负责解析 Go 语言源文件中的特征顺序定义，提取 getDefaultFeatureOrder 函数中
    的特征映射关系。类似于编译器的词法分析器，将源码转换为可用的数据结构。
    """
    
    def __init__(self, go_file_path: str):
        """
        初始化解析器
        
        Args:
            go_file_path: Go 源文件路径
            
        Raises:
            FileNotFoundError: 当指定的 Go 文件不存在时抛出
        """
        try:
            with open(go_file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            print(f"成功加载 Go 源文件: {go_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Go 源文件 '{go_file_path}' 没找到。请确保文件存在于当前工作目录中。"
            )
        
        self.feature_order = self._parse_feature_order()
    
    def _parse_feature_order(self) -> List[str]:
        """
        解析特征顺序
        
        使用正则表达式解析 Go 函数中的 map 定义，提取特征名称和索引的映射关系。
        正则表达式在此处的作用类似于数据挖掘中的模式匹配算法。
        
        Returns:
            按索引排序的特征名称列表
        """
        print("开始解析 getDefaultFeatureOrder 函数...")
        
        # 使用正则表达式匹配Go函数定义
        function_pattern = r'func getDefaultFeatureOrder\(\) map\[int\]string \{\s*return map\[int\]string\{(.*?)\}\s*\}'
        match = re.search(function_pattern, self.content, re.DOTALL)
        
        if not match:
            print("警告: 没找到 getDefaultFeatureOrder 函数，使用预定义特征顺序")
            return self._get_fallback_feature_order()
        
        # 提取函数体中的键值对
        function_body = match.group(1)
        feature_pairs = re.findall(r'(\d+):\s*"([^"]+)"', function_body)
        
        if not feature_pairs:
            print("警告: 函数体中无有效特征定义，使用预定义特征顺序")
            return self._get_fallback_feature_order()
        
        # 构建特征字典并按索引排序
        feature_dict = {int(index): name for index, name in feature_pairs}
        sorted_features = [feature_dict[i] for i in sorted(feature_dict.keys())]
        
        print(f"成功解析 {len(sorted_features)} 个特征的顺序定义")
        return sorted_features
    
    def get_feature_order(self) -> List[str]:
        """
        获取特征顺序列表
        
        Returns:
            特征名称的有序列表
        """
        return self.feature_order
    
    def _get_fallback_feature_order(self) -> List[str]:
        """
        预定义特征顺序
        
        当无法从 Go 源码中解析特征顺序时，使用此预定义列表作为备选方案。
        这种设计模式称为"优雅降级"，确保系统在部分功能失效时仍能正常运行。
        
        Returns:
            预定义的特征名称列表
        """
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

# 文件路径配置
# 路径修正: Actions运行时在 Smart/scripts/ 中，数据在根目录的 data/ 文件夹中
DATA_FILE = '../../data/smart_weight_data.csv'   # 训练数据文件路径 (从 Smart/scripts/ 返回两级到根目录,再进入 data/)
# ⚡ 路径修正：指向 Smart/go_transform/transform.go
GO_FILE = '../go_transform/transform.go'         # Go 源码文件路径 (从 Smart/scripts/ 返回一级到 Smart/, 再进入 go_transform/)
# 路径修正: Actions运行时在 Smart/scripts/ 中，模型需要输出到根目录的 models/ 文件夹中
MODEL_FILE = '../../models/Model.bin'         # 输出模型文件路径

# 特征预处理配置
STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 
    'last_used_seconds', 'traffic_density'
]

ROBUST_SCALER_FEATURES = ['success', 'failure']

# LightGBM模型超参数配置
LGBM_PARAMS = {
    'objective': 'regression',     # 回归任务目标
    'metric': 'rmse',              # 均方根误差作为评估指标
    'n_estimators': 1000,          # 决策树数量
    'learning_rate': 0.03,         # 学习率
    'random_state': 42,            # 随机种子
    'n_jobs': -1                   # 使用所有可用 CPU 核心
}

EARLY_STOPPING_ROUNDS = 100           # 早停轮数

# ==============================================================================
# 3. 核心功能模块
# ==============================================================================

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    数据加载与预处理
    """
    print(f"开始加载数据文件: {file_path}")
    
    try:
        # 使用容错模式读取CSV文件，自动跳过格式异常的行
        data = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"数据加载完成，原始记录数: {len(data)}")
    except FileNotFoundError:
        print(f"错误: 数据文件 '{file_path}' 不存在")
        return None
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

    # 数据质量控制流程
    original_count = len(data)
    
    # 移除权重字段缺失的记录
    data.dropna(subset=['weight'], inplace=True)
    
    # 过滤非正权重值（权重必须为正数才有实际意义）
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
        # 按指定顺序提取特征
        X = data[feature_order]
        y = data['weight']
        
        print(f"特征提取完成 - 特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y
        
    except KeyError as e:
        print(f"特征提取失败: 缺少必要的特征列 {e}")
        available_columns = list(data.columns)
        print(f"数据中可用的列: {available_columns}")
        return None, None

def apply_feature_transforms(X: pd.DataFrame, feature_order: List[str]) -> Tuple[pd.DataFrame, StandardScaler, RobustScaler]:
    """
    特征标准化处理
    """
    print("开始特征标准化处理...")
    X_transformed = X.copy()
    
    # 应用 StandardScaler
    std_scaler = StandardScaler()
    std_features_available = [f for f in STD_SCALER_FEATURES if f in X_transformed.columns]
    
    if std_features_available:
        X_transformed[std_features_available] = std_scaler.fit_transform(X_transformed[std_features_available])
        print(f"StandardScaler 处理完成，影响特征数: {len(std_features_available)}")
    
    # 应用 RobustScaler
    robust_scaler = RobustScaler()
    robust_features_available = [f for f in ROBUST_SCALER_FEATURES if f in X_transformed.columns]
    
    if robust_features_available:
        X_transformed[robust_features_available] = robust_scaler.fit_transform(X_transformed[robust_features_available])
        print(f"RobustScaler 处理完成，影响特征数: {len(robust_features_available)}")
    
    # 注意: 在不需要 V3 编码时，我们不再需要返回 scaler 实例
    return X_transformed, std_scaler, robust_scaler

def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.LGBMRegressor:
    """
    LightGBM 模型训练
    """
    print("开始 LightGBM 模型训练...")
    
    # 初始化模型
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    # 执行训练过程
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    # 模型性能评估
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    print(f"模型训练完成")
    print(f"训练集R²得分: {train_r2:.4f}")
    print(f"测试集R²得分: {test_r2:.4f}")
    
    # R²解释：决定系数
    if test_r2 > 0.8:
        print("模型性能评估: 优秀")
    elif test_r2 > 0.6:
        print("模型性能评估: 良好")
    else:
        print("模型性能评估: 需要改进")
    
    return model

def save_model(model: lgb.LGBMRegressor, model_file: str) -> None:
    """
    模型序列化保存 - 使用 LightGBM 原生格式
    """
    print(f"开始保存模型至: {model_file}")
    
    try:
        # 使用 LightGBM 原生 save_model 方法
        model.booster_.save_model(model_file)
        print("模型保存成功，输出 LightGBM 原生格式 (Model.bin)")
    except Exception as e:
        print(f"模型保存失败: {e}")

# ==============================================================================
# 4. 主程序流程控制
# ==============================================================================

def main() -> None:
    """
    主程序入口
    """
    print("=" * 60)
    print("Mihomo 智能权重模型训练")
    print("出品：安格视界")
    print("=" * 60)
    
    # 步骤1: Go 源码解析
    print("\n[步骤1] Go 源码解析")
    try:
        # Go 文件假设位于当前脚本目录下
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
    # 注意：在原生保存模式下，scaler 实例不被使用，但仍需执行标准化以训练模型
    X_processed, _, _ = apply_feature_transforms(X, feature_order)
    
    # 步骤5: 数据集划分
    print("\n[步骤5] 训练测试集划分")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=0.2,      # 80/20 划分比例
        random_state=42     # 确保结果可复现
    )
    print(f"数据划分完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 步骤6: 模型训练
    print("\n[步骤6] 模型训练")
    trained_model = train_lgbm_model(X_train, y_train, X_test, y_test)
    
    # 步骤7: 模型保存
    print("\n[步骤7] 模型保存")
    save_model(trained_model, MODEL_FILE)
    
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

