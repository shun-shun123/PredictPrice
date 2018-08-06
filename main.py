import dataCreate
import numpy as np
import model
import matplotlib.pyplot as plt

market_info = dataCreate.createDataSet()

for coins in ['_x', '_y']:
    kwargs = {'close_off_high' + coins: lambda x: 2 * (x['High' + coins] - x['Close**' + coins]) / (x['High' + coins] - x['Open*' + coins]) - 1,
               'volatility' + coins: lambda x: (x['High' + coins] - x['Low' + coins]) / (x['Open*' + coins])}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['Date'] + [metric + coin for metric in ['Close**', 'Volume', 'close_off_high', 'volatility'] for coin in ['_x', '_y']]]
# 時間順になるようにソートする
model_data = model_data.sort_values(by='Date')
print("**********model_dataのヘッダ情報**********")
print(model_data.head())

# dateの列を削除
split_date = '2017-10-01'
training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

# データのまとまりを10日に設定
window_len = 10
norm_cols = [metric + coin for metric in ['Close**', 'Volume'] for coin in ['_x', '_y']]

# training_set, test_setをwindow_lenで分ける
LSTM_training_inputs = []
for i in range(len(training_set) - window_len):
    temp_set = training_set[i:(i + window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_trainint_outputs = (training_set['Close**_y'][window_len:].values / training_set['Close**_y'][:-window_len].values) - 1

LSTM_test_inputs = []
for i in range(len(test_set) - window_len):
    temp_set = test_set[i:(i + window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close**_y'][window_len:].values / test_set['Close**_y'][:-window_len].values) - 1

# trainint_dataのインプットデータを確認
print("**********LSTM_training_inputs確認**********")
print(LSTM_training_inputs[0])

# PandasのdataFrameからNumpy配列へ変換
LSTM_training_inputs = [np.array(LSTM_training_inputs) for LSTM_training_inputs in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

# ランダムシードの設定(randomで生成される数値の固定)
np.random.seed(202)

# モデルの構築
eth_model = model.createModel(LSTM_training_inputs, output_size=1, neurons = 20)

# モデルのアウトプットは次の窓の10番目の価格（正規化されている）
LSTM_trainint_outputs = (training_set['Close**_y'][window_len:].values / training_set['Close**_y'][:-window_len].values) - 1

# データを流してフィッティング
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_trainint_outputs, epochs=50, batch_size=1, verbose=2, shuffle=True)
# lossの変化をプロットして確認
fig, ax1 = plt.subplots(1, 1)
ax1.plot(eth_history.epoch, eth_history.history['loss'])
ax1.set_title("TrainingError")
if eth_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error(MAE)', fontsize=12)
else:
    ax1.set_ylabel("Model Loss", fontsize=12)
ax1.set_xlabel("# Epochs", fontsize=12)
plt.show()