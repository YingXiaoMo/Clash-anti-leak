# combined_train.py
# ==============================================================================
# Mihomo 智能权重模型训练
# 出品：安格视界
# 功能：基于历史数据训练 LightGBM 回归模型，用于预测代理节点权重
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

GO_FILE = os.path.join(BASE_DIR, "../go_transform/transform.go")
MODEL_FILE = os.path.join(BASE_DIR, "../../models/Model.bin")
DATA_DIR = os.path.join(BASE_DIR, "../../data") 

# 特征预处理配置
# StandardScaler 适用于正态分布或近似正态分布的特征
# 其工作原理类似于统计学中的 Z-score 标准化
STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 
    'last_used_seconds', 'traffic_density'
]

# RobustScaler 适用于包含异常值的特征
# 使用中位数和四分位距进行标准化，类似于箱线图的原理，对异常值不敏感
ROBUST_SCALER_FEATURES = ['success', 'failure']

# LightGBM模型超参数配置
# 这些参数的调优类似于精密仪器的校准过程
LGBM_PARAMS = {
    'objective': 'regression',     # 回归任务目标
    'metric': 'rmse',              # 均方根误差作为评估指标
    'n_estimators': 1000,          # 决策树数量，类似于民主投票中的选民数量
    'learning_rate': 0.03,         # 学习率，控制每次迭代的步长
    'random_state': 42,            # 随机种子，确保结果可复现
    'n_jobs': -1                   # 使用所有可用 CPU 核心
}

EARLY_STOPPING_ROUNDS = 100           # 早停轮数，防止过拟合

# ==============================================================================
# 3. 核心功能模块
# ==============================================================================

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    数据加载与预处理
    
    执行数据质量控制流程，包括格式验证、缺失值处理和异常值过滤。
    数据清洗过程类似于实验室的样品预处理，确保输入数据的质量。
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        清洗后的 DataFrame，失败时返回None
    """
    print(f"开始加载数据文件: {file_path}")
    
    try:
        # 使用容错模式读取CSV文件，自动跳过格式异常的行
        # on_bad_lines='skip'参数类似于质量检验中的不良品剔除机制
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
    
    从预处理数据中提取特征矩阵(X)和目标变量(y)。
    此过程类似于实验设计中的变量分离，将自变量和因变量明确区分。
    
    Args:
        data: 预处理后的数据框
        feature_order: 特征顺序列表
        
    Returns:
        特征矩阵和目标变量的元组，失败时返回(None, None)
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
    
    对不同类型的特征应用相应的标准化方法。StandardScaler 基于正态分布假设，
    而 RobustScaler 基于分位数，两者的选择类似于选择不同的测量标尺。
    
    Args:
        X: 原始特征矩阵
        feature_order: 特征顺序列表
        
    Returns:
        标准化后的特征矩阵和对应的标准化器实例
    """
    print("开始特征标准化处理...")
    X_transformed = X.copy()
    
    # 应用 StandardScaler
    # 工作原理：(x - mean) / std，类似于统计学中的标准化得分
    std_scaler = StandardScaler()
    std_features_available = [f for f in STD_SCALER_FEATURES if f in X_transformed.columns]
    
    if std_features_available:
        X_transformed[std_features_available] = std_scaler.fit_transform(X_transformed[std_features_available])
        print(f"StandardScaler 处理完成，影响特征数: {len(std_features_available)}")
    
    # 应用 RobustScaler
    # 工作原理：(x - median) / IQR，基于中位数和四分位距，对异常值更稳健
    robust_scaler = RobustScaler()
    robust_features_available = [f for f in ROBUST_SCALER_FEATURES if f in X_transformed.columns]
    
    if robust_features_available:
        X_transformed[robust_features_available] = robust_scaler.fit_transform(X_transformed[robust_features_available])
        print(f"RobustScaler 处理完成，影响特征数: {len(robust_features_available)}")
    
    return X_transformed, std_scaler, robust_scaler

def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.LGBMRegressor:
    """
    LightGBM 模型训练
    
    使用梯度提升决策树算法训练回归模型。LightGBM 的工作原理类似于
    专家委员会的集体决策：多个决策树各自给出预测，最终结果是所有预测的加权平均。
    
    Args:
        X_train, y_train: 训练集特征和目标
        X_test, y_test: 测试集特征和目标
        
    Returns:
        训练完成的 LightGBM 模型实例
    """
    print("开始 LightGBM 模型训练...")
    
    # 初始化模型
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    # 执行训练过程
    # eval_set 参数启用验证集监控，类似于考试中的模拟测试
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
    
    # R²解释：决定系数，表示模型解释方差的比例，1.0为完美预测
    if test_r2 > 0.8:
        print("模型性能评估: 优秀")
    elif test_r2 > 0.6:
        print("模型性能评估: 良好")
    else:
        print("模型性能评估: 需要改进")
    
    return model

def save_model(model: lgb.LGBMRegressor, model_file: str) -> None:
    """
    模型序列化保存
    
    将训练完成的模型保存为二进制文件，便于后续部署使用。
    保存过程类似于将程序编译为可执行文件。
    
    Args:
        model: 训练完成的模型实例
        model_file: 输出文件路径
    """
    print(f"开始保存模型至: {model_file}")
    
    try:
        # 保存 LightGBM 的原生格式，保持最佳兼容性
        model.booster_.save_model(model_file)
        print("模型保存成功，可以直接部署")
    except Exception as e:
        print(f"模型保存失败: {e}")

# ==============================================================================
# 4. 主程序流程控制
# ==============================================================================

def main() -> None:
    """
    主程序入口
    
    按照标准的机器学习工作流程执行模型训练，包括数据预处理、
    特征工程、模型训练和模型保存等关键步骤。
    """
    print("=" * 60)
    print("Mihomo 智能权重模型训练")
    print("出品：安格视界")
    print("=" * 60)
    
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
