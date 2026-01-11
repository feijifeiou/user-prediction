import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_feature_importance(model, feature_names, top_n=15):
    """特征重要性可视化"""
    importance = model.get_score(importance_type='weight')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    features = [x[0] for x in importance]
    scores = [x[1] for x in importance]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=features, palette="viridis")
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


# 在visualizer.py中添加
def plot_prediction_distribution(predictions, class_names=None):
    """预测结果分布可视化"""
    plt.figure(figsize=(10, 6))
    counts = pd.Series(predictions).value_counts().sort_index()

    if class_names:
        counts.index = [class_names[i] for i in counts.index]

    sns.barplot(x=counts.index, y=counts.values, palette="rocket")
    plt.title('Prediction Distribution')
    plt.xlabel('Behavior Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('pred_distribution.png', bbox_inches='tight')
    plt.close()


def plot_actual_vs_pred(y_true, y_pred, sample_size=100):
    """实际值与预测值对比（抽样展示）"""
    plt.figure(figsize=(12, 6))
    df_compare = pd.DataFrame({'Actual': y_true[:sample_size],
                               'Predicted': y_pred[:sample_size]})
    sns.scatterplot(data=df_compare, s=100, alpha=0.7)
    plt.plot([0, 4], [0, 4], 'r--')  # 假设有5个类别
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.savefig('actual_vs_pred.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """混淆矩阵绘制"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()