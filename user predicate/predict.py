import pandas as pd
import xgboost as xgb
import joblib
from utils.data_loader import load_and_preprocess
from utils.visualizer import plot_prediction_distribution


def predict_new_data(model_path, data_path):
    # 加载模型
    model = xgb.Booster()
    model.load_model(model_path)

    # 处理新数据
    new_data = load_and_preprocess(data_path)
    dmatrix = xgb.DMatrix(new_data)

    # 执行预测
    predictions = model.predict(dmatrix)
    new_data['pred_prob'] = predictions.max(axis=1)  # 取最大概率
    new_data['pred_label'] = predictions.argmax(axis=1)

    # 保存结果
    new_data.to_csv('predictions.csv', index=False)

    # 可视化预测分布
    plot_prediction_distribution(new_data['pred_label'])

    return new_data


if __name__ == "__main__":
    predict_new_data("models/xgb_model.json", "data/new_data.csv")
