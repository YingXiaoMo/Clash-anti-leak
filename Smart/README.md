# 🤖 Mihomo 智能权重模型训练与自动化部署

本项目旨在利用 LightGBM 回归模型，基于 Clash.Meta/Mihomo 产生的历史连接日志数据，训练出能够预测代理节点最佳权重的模型（`Model.bin`），并通过 GitHub Actions 实现自动化训练、构建和部署。

## ✨ 项目功能一览

1. **Go 源码特征解析：** Python 脚本自动解析 Mihomo Go 源码中定义的特征顺序，确保模型训练的特征与实际运行时使用的特征一致。

2. **数据清洗与加载：** 自动遍历数据文件，执行数据加载、缺失值处理和清洗。

3. **LightGBM 模型训练：** 使用 LightGBM 框架进行高效的模型训练。

4. **GitHub Actions 自动化：** 自动执行整个训练流程。

5. **最终产物发布（Release）：** 训练成功的模型文件（`Model.bin`）将通过 GitHub Release 发布，**不会**提交回仓库的代码或 `models/` 目录中。

## ⚠️ 关键操作警告

### 1. 数据来源与 Runner 清理机制

* **数据来源：** 本工作流**不存储数据**，假定历史数据是从**外部云存储**（例如 Google Cloud Storage 或 Google Drive）下载到 Runner 上的 `data/` 目录中。

* **Runner 清理机制：** 在工作流执行完毕后，托管它的 Runner 虚拟机就会被销毁。因此，`data/` 目录中的所有文件将随之**自动删除**。

* **重要提示：** 请勿将原始、不可替代的历史数据文件直接存储在您的 Git 仓库的 `data/` 目录中！您的数据应始终保存在外部可靠的存储服务上。

## 📂 文件结构与依赖

为了让项目正常运行，您需要确保以下文件和目录结构存在于您的仓库中：

| 路径 | 描述 | 状态 | 
| ----- | ----- | ----- | 
| `data/` | **必需。** 作为 CSV 数据下载的**临时目标目录**。 | **必填** | 
| `models/` | **必需。** 存放最终生成的模型文件 `Model.bin`。 | **必填** | 
| `Smart/go_transform/transform.go` | **必需。** Go 语言特征定义文件。`train_smart.py` 依赖此文件来获取特征顺序。 | **必填** | 
| `Smart/scripts/train_smart.py` | 训练主脚本。负责解析特征、数据处理、训练和编码。 | **必填** | 
| `required_dependencies.txt` | **必需。** 严格锁定的 Python 依赖列表文件。 | **必填** | 
| `.github/workflows/train.yml` | GitHub Actions 自动化工作流定义文件。 | **必填** | 

## 🛠️ 新用户使用指南：需要修改的关键位置

新用户使用此项目时，需要进行以下几个关键步骤的配置和文件修改。

### 1. 文件夹创建和数据准备

您必须在项目根目录下创建以下文件夹并准备相应文件：

| 文件夹/文件 | 内容要求 | 
| ----- | ----- | 
| `data/` | **创建空文件夹。** 这是 Actions 下载数据的目标位置。 | 
| `models/` | **创建空文件夹。** 训练脚本的输出位置。 | 
| `Smart/go_transform/transform.go` | **放入 Mihomo 源码中的 Go 特征定义文件。** | 

### 2. 训练脚本 (`Smart/scripts/train_smart.py`) 修改

| 变量/代码段 | 建议修改内容 | 
| ----- | ----- | 
| `DATA_FILE`, `GO_FILE`, `MODEL_FILE` | **修改这些路径变量**，以确保脚本能够找到 Go 文件、数据目录和正确的模型输出位置。注意：如果您的 `train_smart.py` 在 `Smart/scripts` 下，则需要使用相对路径。 | 
| `LGBM_PARAMS` | 如果模型性能不理想，可以调整 LightGBM 的**超参数**，例如 `learning_rate` (学习率)。 | 
| `STD_SCALER_FEATURES` / `ROBUST_SCALER_FEATURES` | **根据您 Go 源码中定义的特征类型**，调整哪些特征应该使用 `StandardScaler` 或 `RobustScaler` 进行标准化。 | 

### 3. 自动化工作流 (`.github/workflows/train.yml`) 修改

此 YAML 文件是自动化流程的核心。

| 配置项 | 建议修改内容 | 
| ----- | ----- | 
| `on:` 触发器 | 默认是 `push`。如果您希望定时训练（例如每天凌晨），请将配置改为 `schedule`：<br>`schedule: [ { cron: '0 0 * * *' } ]` | 
| **数据下载步骤** | **最关键！** 您需要添加步骤，以从您的**外部数据源**将历史数据 CSV 文件下载到 Runner 的 `./data/` 目录中。**请参考下面的 YAML 示例。** | 
| `python-version` | 检查并修改为您希望使用的 Python 版本（例如 `3.10` 或 `3.11`）。 | 
| **Python 依赖安装** | 确保在 `train` Job 中有以下步骤，以安装 `required_dependencies.txt` 中列出的所有依赖：<br>`pip install -r required_dependencies.txt` | 
| **Telegram Secrets** | 如果使用 Telegram 通知，请更新 `secrets.TG_BOT_TOKEN` 和 `secrets.TG_CHAT_ID`。 | 

---

🚀 关键步骤详解：数据下载配置示例（GCS / Rclone）  
由于数据需要从外部云存储下载，您需要在 `train.yml` 的 `train` Job 中添加相应的下载步骤。

---

### 选项 A：使用 Google Cloud Storage (GCS)

1. 准备 Secret：Google 服务账户密钥  
您必须在 GitHub Secrets 中配置一个用于认证 Google 服务的 Secret。

- **Secret 名称：** `GCP_SA_KEY`  
- **Secret 内容：** 您的 Google 服务账户密钥 JSON 文件内容。  
  > 注意：此密钥必须拥有访问您 GCS 存储桶中数据的权限。

2. YAML 示例（在 `train_and_deploy.yml` 的 `train` Job 中添加）  
请在 `steps:` 列表里，紧跟在 `actions/checkout` 步骤之后，添加如下步骤：

```yaml
# 1. 认证 Google Cloud (使用服务账户密钥 JSON)
- name: 认证 Google Cloud
  uses: google-github-actions/auth@v2 
  with:
    credentials_json: ${{ secrets.GCP_SA_KEY }}

# 2. 从 Google Cloud Storage 下载历史数据
- name: 从 Google Cloud Storage 下载历史数据 (gsutil)
  run: |
    echo "创建数据目录..."
    mkdir -p data
    
    # 🚨 替换 gs://your-gcs-bucket-name/smart_data/ 为您的 GCS 实际路径
    # *号用于匹配所有 CSV 文件
    gsutil cp gs://your-gcs-bucket-name/smart_data/*.csv ./data/
    
    echo "数据下载完成！"
```

---

### 选项 B：使用 Rclone (适用于 Google Drive / OneDrive / 其他云存储)

`rclone` 是一款命令行云存储同步工具，支持多种云服务。

1. 准备 Secret：Rclone 配置文件  
您需要先在本地配置好 rclone（例如配置 Google Drive 或其他远程），然后将生成的 `rclone.conf` 文件内容作为 Secret 存储。

- **Secret 名称：** `RCLONE_CONFIG`  
- **Secret 内容：** 您的 `rclone.conf` 文件中 **除了 `[remote_name]` 之外的所有配置内容**。

2. YAML 示例（在 `train.yml` 的 `train` Job 中添加）  
请在 `steps:` 列表里，紧跟在 `actions/checkout` 步骤之后，添加如下步骤：

```yaml
# 1. 安装 rclone 工具
- name: 安装 Rclone
  run: sudo apt-get install rclone -y

# 2. 配置 rclone (使用 Secret 写入配置文件)
- name: 配置 Rclone 认证
  run: |
    echo "创建数据目录..."
    mkdir -p data
    
    # 🚨 替换 [gdrive_remote_name] 为您在 rclone 中设置的远程名称
    mkdir -p ~/.config/rclone
    echo "[gdrive_remote_name]" > ~/.config/rclone/rclone.conf
    echo "${{ secrets.RCLONE_CONFIG }}" >> ~/.config/rclone/rclone.conf

# 3. 从云存储下载历史数据
- name: 从云存储下载历史数据 (rclone)
  run: |
    # 🚨 替换 gdrive_remote_name:path/to/data 为您的实际远程名称和路径
    rclone copy gdrive_remote_name:path/to/data ./data/ --include "*.csv"
    echo "数据下载完成！"
```

> 重要说明：  
> 在 Rclone 示例中，务必将 `[gdrive_remote_name]` 和 `gdrive_remote_name:path/to/data` 替换为您的实际远程名称和路径。  
> `~/.config/rclone/` 目录在 Runner 中默认可能不存在，但在 rclone 运行时会自动创建。


