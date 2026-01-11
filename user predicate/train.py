import os
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_and_preprocess
from utils.visualizer import plot_feature_importance, plot_confusion_matrix

# 配置参数
CONFIG_PATH = "configs/params.yaml"
SEED = 42
plt.style.use('ggplot')


def load_config():
    """加载超参数配置"""
    with open(CONFIG_PATH) as f:
        params = yaml.safe_load(f)
    return params['xgb_params'], params['train_params']


def prepare_dmatrix(X_train, X_val, y_train, y_val):
    """创建XGBoost专用数据格式"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return dtrain, dval


def cross_validate(X, y, params, n_folds=5):
    """交叉验证训练"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    metrics = {'accuracy': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain, dval = prepare_dmatrix(X_train, X_val, y_train, y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            early_stopping_rounds=50,
            evals=[(dval, 'eval')],
            verbose_eval=False
        )

        preds = model.predict(dval)
        y_pred = [int(round(p)) for p in preds]

        fold_acc = accuracy_score(y_val, y_pred)
        fold_f1 = f1_score(y_val, y_pred, average='weighted')

        metrics['accuracy'].append(fold_acc)
        metrics['f1'].append(fold_f1)

        print(f"Fold {fold + 1} | Accuracy: {fold_acc:.4f} | F1: {fold_f1:.4f}")

    return {
        'mean_accuracy': sum(metrics['accuracy']) / n_folds,
        'mean_f1': sum(metrics['f1']) / n_folds
    }

def plot_learning_curve(results):
    """绘制模型训练过程指标变化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(results['train']['mlogloss'], label='Train')
    plt.plot(results['eval']['mlogloss'], label='Validation')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Multi-class Log Loss')
    plt.title('Training Process Monitoring')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.close()

# 修改训练代码
evals_result = {}
final_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=train_params['num_boost_round'],
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    evals_result=evals_result,
    early_stopping_rounds=50,
    verbose_eval=50
)
plot_learning_curve(evals_result)

def main():
    # 加载配置
    xgb_params, train_params = load_config()

    # 数据准备
    df = load_and_preprocess("data/raw/user_behavior.csv")
    X = df.drop('target_behavior', axis=1)
    y = df['target_behavior'].values

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    # 交叉验证
    print("=== Cross Validation ===")
    cv_results = cross_validate(X_train, y_train, xgb_params)
    print(f"\nCV Mean Accuracy: {cv_results['mean_accuracy']:.4f}")
    print(f"CV Mean F1: {cv_results['mean_f1']:.4f}")

    # 全量训练
    print("\n=== Full Training ===")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    final_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=train_params['num_boost_round'],
        evals=[(dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # 评估
    y_pred = final_model.predict(dtest)
    y_pred = [int(round(p)) for p in y_pred]

    print("\n=== Evaluation Report ===")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

    # 保存模型
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    final_model.save_model(os.path.join(model_dir, 'xgb_model.json'))

    # 特征重要性可视化
    plot_feature_importance(final_model, X.columns)


if __name__ == "__main__":
    main()