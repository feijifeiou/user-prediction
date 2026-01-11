import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(file_path):
    """数据加载与预处理"""
    df = pd.read_csv(file_path)

    # 处理时间特征
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    # 编码分类变量
    le = LabelEncoder()
    df['device'] = le.fit_transform(df['device'])
    df['target_behavior'] = le.fit_transform(df['target_behavior'])

    # 删除原始时间列
    df.drop('timestamp', axis=1, inplace=True)

    return df