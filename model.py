from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

# モデル構築
def createModel(inputs, output_size, neurons, active_func='linear', dropout=0.25, loss='mean_squared_error', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units = output_size))
    model.add(Activation(active_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model