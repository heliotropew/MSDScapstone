import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_dataset(data, time_steps, use_sentiment):
    dataX, dataY = [], []
    for i in range(len(data) - time_steps - 1):
        a = data[i:(i + time_steps), :]
        dataX.append(a)
        if use_sentiment:
            dataY.append(data[i + time_steps, 1])
        else:
            dataY.append(data[i + time_steps, 0])
    return np.array(dataX), np.array(dataY)


def dataloader(filename, use_sentiment = True):
    df = pd.read_csv(filename)

    if use_sentiment:
        sentiment = df[['Sentiment Value']].values
        features = df[['Credit Spread', 'Excess Spread', 'CFETS Valuation']].values
    else:
        features = df[['Credit Spread', 'Excess Spread', 'CFETS Valuation']].values

        # 只对三个指定的列进行规范化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # 如果使用 'sentiment'，则将其与规范化后的数据合并
    if use_sentiment:
        scaled_features = np.concatenate((sentiment, scaled_features), axis=1)

    split_frac = 0.8
    split_idx = int(len(scaled_features) * split_frac)
    train_data = scaled_features[:split_idx]
    test_data = scaled_features[split_idx:]

    time_steps = 1
    X_train, y_train = create_dataset(train_data, time_steps, use_sentiment)
    X_test, y_test = create_dataset(test_data, time_steps, use_sentiment)

    return X_train, X_test, y_train, y_test, scaler, df["Date"]
