import keras
from keras.models import Sequential
from keras.regularizers import l1_l2 as l
from utils import (config, create, exists, init, listdir, names, np, os, read,
                   write)


def gen_lstm_model(input_shape):
       
    hyperparameter = config.demand_forecasting.hyperparameter
    no_layers, dropout, recurrent_dropout = hyperparameter.lstm_layer_details.split('_')
    _loss = hyperparameter.loss
    _optimizer = hyperparameter.optimizer

    model = Sequential()
    #current
    model.add(keras.layers.LSTM(units=int(no_layers),dropout=float(dropout),recurrent_dropout=float(recurrent_dropout), input_shape=input_shape,kernel_regularizer=l(0.001,0.001), return_sequences=False))
    #LSTM layer
    model.add(keras.layers.Dense(1))
    model.compile(loss=_loss, optimizer=_optimizer)
    return model

def predict_f(model, test_X, y_var):
    # if (config['demand_forecasting']["ModelParams"]["LAG_VARIABLES"] == "y_var") or (
    #     config['demand_forecasting']["ModelParams"]["LAG_VARIABLES"] == "y_var_current_and_last_year"):
    if (names.__dict__[config.fs.lstm.lag_column] == y_var):
        # some variables lag(rows,1,features(including the lag for selected features)) as traindata
        # model evaluation
        # make the predictions Test
        # Functionality: create y variable lags. For test data, looks back at the previous n weeks and creates lag by using previous rows' predicted y variables
        # For train data, uses actual y variable for lags
        yhat_test = []
        lag_weeks = int(config.fs.lstm.lag)

        for i in range(0, test_X.shape[0]):
            if i > 0:
                if i > lag_weeks:
                    to_replace = lag_weeks
                else:
                    to_replace = i
                for j in range(0, to_replace):
                    test_X[i, :, -lag_weeks + j] = yhat_test[-j - 1]
                    
            yhat_test.append(model.predict(test_X[i][:][:].reshape(1, 1, test_X.shape[-1]))[0][0])
        yhat_test = np.array(yhat_test).reshape(-1, 1)
    else:
        yhat_test = model.predict(test_X)
        yhat_test = yhat_test.reshape(len(yhat_test), 1)

    # if config['demand_forecasting']["ModelParams"]["LAG_VARIABLES"] == "all":
    if config.fs.lstm.lag_column == 'all':
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
    else:
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[-1]))
    
    # inv_yhat_test = np.concatenate((test_X, yhat_test), axis=1)
    
    # return inv_yhat_test
    no_lags = config.fs.lstm.lag
    lag_data = None
    if (names.__dict__[config.fs.lstm.lag_column] == y_var) and (no_lags > 0):
        lag_data = test_X[:,-no_lags:]
    return yhat_test, lag_data
