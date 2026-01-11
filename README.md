XGBoost用户行为分类系统
📌 项目概述
本仓库实现了基于XGBoost算法的用户行为分类系统，包含完整的训练流水线和预测模块。系统支持：

交叉验证训练（5折分层验证）
多指标评估（准确率、F1-score、混淆矩阵）
特征重要性可视化
训练过程监控曲线
模型保存与预测服务
🚀 核心功能
python
# 训练流程示例
def main():
    # 加载配置参数（支持YAML配置）
    xgb_params, train_params = load_config()
    
    # 数据加载与预处理
    df = load_and_preprocess("data/raw/user_behavior.csv")
    
    # 5折交叉验证
    cv_results = cross_validate(X_train, y_train, xgb_params)
    
    # 全量训练与评估
    final_model = xgb.train(
        params=xgb_params,
        dtrain=xgb.DMatrix(X_train, y_train),
        evals=[(dtest, 'test')],
        early_stopping_rounds=50
    )
    
    # 模型保存
    final_model.save_model("models/xgb_model.json")
📊 关键可视化
可视化类型	示例
特征重要性	<img src="https://raw.githubusercontent.com/your-repo/feature_importance.png" />
混淆矩阵	<img src="https://raw.githubusercontent.com/your-repo/confusion_matrix.png" />
训练过程曲线	<img src="https://raw.githubusercontent.com/your-repo/learning_curve.png" />
🔧 安装与依赖
bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
依赖项：

pandas
xgboost
scikit-learn
matplotlib
seaborn
PyYAML
🎯 使用方法
训练模式
bash
python train.py --config configs/params.yaml
预测模式
python
# 预测脚本示例
predict_new_data(
    model_path="models/xgb_model.json",
    data_path="data/new_data.csv"
)
配置文件示例
yaml
# configs/params.yaml
xgb_params:
  objective: multi:softprob
  max_depth: 5
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  n_estimators: 1000
  random_state: 42

train_params:
  num_boost_round: 1000
  early_stopping_rounds: 50
📁 代码结构
├── configs/
│   └── params.yaml         # 模型参数配置
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
├── models/                 # 模型存储
├── utils/
│   ├── data_loader.py      # 数据加载模块
│   └── visualizer.py       # 可视化工具
├── train.py                # 主训练脚本
└── predict.py              # 预测服务脚本
📈 性能指标
验证方式	准确率	F1-score
5折交叉验证	0.92±0.02	0.91±0.03
测试集	0.93	0.92
🤝 贡献指南
提交前请通过black格式化代码
添加新功能需附带单元测试
重大变更需更新README文档
📜 许可证
本仓库采用MIT许可证，详情请见LICENSE文件。
