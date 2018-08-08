import dataCreate
import numpy as np
import model
import matplotlib.pyplot as plt
import datetime

market_info = dataCreate.createDataSet()

for coins in ['_x', '_y']:
    kwargs = {'close_off_high' + coins: lambda x: 2 * (x['High' + coins] - x['Close**' + coins]) / (x['High' + coins] - x['Low' + coins]) - 1,
               'volatility' + coins: lambda x: (x['High' + coins] - x['Low' + coins]) / (x['Open*' + coins])}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['Date'] + [metric + coin for metric in ['Close**', 'Volume', 'close_off_high', 'volatility'] for coin in ['_x', '_y']]]
# 時間順になるようにソートする
model_data = model_data.sort_values(by='Date')
print("**********model_dataのヘッダ情報**********")
print(model_data.head())

# dateの列を削除
split_date = '2017-10-01'
# split_dateを定義して、それ未満の場合はtraining_set, それ以上の場合はtest_setに代入する
training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
# dropは指定の行or列を削除するdrop(labels, axis)
# axis=0は行, axis=1は列
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
LSTM_training_outputs = (training_set['Close**_y'][window_len:].values / training_set['Close**_y'][:-window_len].values) - 1

LSTM_test_inputs = []
for i in range(len(test_set) - window_len):
    temp_set = test_set[i:(i + window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close**_y'][window_len:].values / test_set['Close**_y'][:-window_len].values) - 1

# trainint_dataのインプットデータを確認
print("**********LSTM_training_inputs確認(データフレーム)**********")
print(LSTM_training_inputs[0])

# PandasのdataFrameからNumpy配列へ変換
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
LSTM_test_inputs = [np.array(LSTM_test_input) for LSTM_test_input in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

# ランダムシードの設定(randomで生成される数値の固定)
np.random.seed(202)

# モデルの構築
eth_model = model.createModel(LSTM_training_inputs, output_size=1, neurons = 20)

# モデルのアウトプットは次の窓の10番目の価格（正規化されている）
LSTM_training_outputs = (training_set['Close**_y'][window_len:].values / training_set['Close**_y'][:-window_len].values) - 1
# データを流してフィッティング
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs, epochs=50, batch_size=1, verbose=2, shuffle=True)

# lossの変化をプロットして確認
fig, ax1 = plt.subplots(1, 1)
print("eth_history.epoch\n" + str(eth_history.epoch))
print(type(eth_history))
print(eth_history.history)
ax1.plot(eth_history.epoch, eth_history.history['loss'])
ax1.set_title("TrainingError")
ax1.set_ylabel('Mean Absolute Error(MAE)', fontsize=12)
ax1.set_xlabel("# Epochs", fontsize=12)
plt.show()


# プロッティングと予測の処理を一緒に行う
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(1, 1)
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]])
ax1.set_xticklabels([datetime.date(i, j, 1).strftime('%d %y') for i in range(2013, 2019) for j in [1, 5, 9]])
ax1.plot(model_data[model_data['Date'] < split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['Close**_y'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set['Close**_y'].values[:-window_len])[0],
         label='Predicted')
ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_training_inputs))+1)-\
                                        (training_set['Close**_y'].values[window_len:])/(training_set['Close**_y'].values[:-window_len]))),
             xy=(0.75, 0.9),  xycoords='axes fraction',
             xytext=(0.75, 0.9), textcoords='axes fraction')
# 下記コードはこちらのURL参照しました：http://akuederle.com/matplotlib-zoomed-up-inset
axins = zoomed_inset_axes(ax1, 3.35, loc=10)
axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1, 5, 9]])
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['Close**_y'][window_len:], label='Actual')
axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set['Close**_y'].values[:-window_len])[0],
         label='Predicted')
axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
axins.set_ylim([10, 60])
axins.set_xticklabels('')
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()

# LSTMモデルがみたことの無い時期の予測
fig, ax1 = plt.subplots(1, 1)
ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
ax1.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y') for i in range(12)])
ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
         test_set['Close**_y'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(eth_model.predict(LSTM_test_inputs))+1) * test_set['Close**_y'].values[:-window_len])[0],
         label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_test_inputs))+1)-\
            (test_set['Close**_y'].values[window_len:])/(test_set['Close**_y'].values[:-window_len]))),
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction', fontsize=13)
ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.show()
