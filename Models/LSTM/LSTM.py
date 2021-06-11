import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import dirname, abspath
from keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils import read_data, read_raw_dataset, read_log_dataset, plot_roc_curve 

def data_preparation_lstm(dataset, dependent_variable, look_back=1):
    train_size = int(len(dataset) * 0.8)
    train = dataset.iloc[0:train_size]
    test = dataset.iloc[train_size:len(dataset)]

    X_train = train.drop(dependent_variable, axis=1)
    y_train = train[dependent_variable]
    X_test = test.drop(dependent_variable, axis=1)
    y_test = test[dependent_variable]

    dataX_train, dataY_train, dataX_test, dataY_test = [], [], [], []

    for i in range(len(X_train)-look_back-1):
        dataX_train.append(X_train[i:(i+look_back)])
        dataY_train.append(y_train[i + look_back])

    for i in range(len(X_test)-look_back-1):
        dataX_test.append(X_test[i:(i+look_back)])
        dataY_test.append(y_test[i + look_back])

    X_train = np.array(dataX_train) 
    y_train = np.array(dataY_train)
    X_test = np.array(dataX_test)
    y_test = np.array(dataY_test)  
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[2])

    return X_train, X_test, y_train, y_test

class LSTM():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.LAYERS = [8, 8, 8, 1]                # number of units in hidden and output layers
        self.EPOCH = 100                          # number of epochs
        self.LR = 5e-2                            # learning rate of the gradient descent
        self.LAMBD = 3e-2                         # lambda in L2 regularization
        self.DP = 0.0                             # dropout rate
        self.RDP = 0.0                            # recurrent dropout rate             
        self.model = None
    
    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
        
    def build_model(self):
        # Build the Model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(input_shape = [self.X_train.shape[1], self.X_train.shape[2]], units=self.LAYERS[0],
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(self.LAMBD), 
                    recurrent_regularizer=tf.keras.regularizers.l2(self.LAMBD),
                    dropout=self.DP, recurrent_dropout=self.RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LSTM(units=self.LAYERS[1],
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(self.LAMBD), 
                    recurrent_regularizer=tf.keras.regularizers.l2(self.LAMBD),
                    dropout=self.DP, recurrent_dropout=self.RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LSTM(units=self.LAYERS[2],
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(self.LAMBD), 
                    recurrent_regularizer=tf.keras.regularizers.l2(self.LAMBD),
                    dropout=self.DP, recurrent_dropout=self.RDP,
                    return_sequences=False, return_state=False,
                    stateful=False, unroll=False
                    ))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=self.LAYERS[3], activation='sigmoid'))

        # Compile the model with Adam optimizer
        self.model.compile(loss='binary_crossentropy',
                    metrics=['accuracy', self.f1_m, self.precision_m, self.recall_m],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.LR))
        #print(self.model.summary())

    def fitting_model(self):
        # Defining a learning rate decay method
        lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                    patience=30, verbose=0, 
                                    factor=0.5, min_lr=1e-8) 

        # Defining early stopping strategy
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                patience=30, verbose=1, mode='auto',
                                baseline=0, restore_best_weights=True)   

        # Training the model
        history = self.model.fit(self.X_train, self.y_train,
                        epochs=self.EPOCH,
                        batch_size= self.X_train.shape[0],
                        validation_split=0.2,
                        shuffle=False,
                        verbose=0,
                        callbacks = [lr_decay, early_stop])

        # Plotting validation and training loss curves
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        # Evaluate the model:
        test_evaluation = self.model.evaluate(self.X_test, self.y_test,
                                            batch_size=self.X_test.shape[0], verbose=0, return_dict=True)#[1]
        y_pred_proba = self.model.predict(self.X_test)
        #print(f'test error = {round((1 - test_evaluation) *  self.X_test.shape[0])} out of  {self.X_test.shape[0]} examples')
        print('\nAccuracy achieved with the test set: ', test_evaluation['accuracy'])
        print('Precision achieved with the test set: ', test_evaluation['precision_m'])
        print('Recall achieved with the test set: ', test_evaluation['recall_m'])
        print('F1 score achieved with the test set: ', test_evaluation['f1_m'])
        plot_roc_curve(self.y_test, y_pred_proba)

def main():
    np.random.seed(0)
    tf.random.set_seed(0)
    
    dataset_full = read_data()
    X_train, X_test, y_train, y_test = data_preparation_lstm(dataset_full, 'Bitcoin sign change')

    LSTM_classifier_full = LSTM(X_train, X_test, y_train, y_test)
    LSTM_classifier_full.build_model()
    LSTM_classifier_full.fitting_model()
    LSTM_classifier_full.evaluate_model()

    dataset_lag = read_log_dataset()
    X_train_lag, X_test_lag, y_train_lag, y_test_lag = data_preparation_lstm(dataset_lag, 'Bitcoin sign change')

    LSTM_classifier_lag = LSTM(X_train_lag, X_test_lag, y_train_lag, y_test_lag)
    LSTM_classifier_lag.build_model()
    LSTM_classifier_lag.fitting_model()
    LSTM_classifier_lag.evaluate_model()

    dataset_raw = read_raw_dataset()
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = data_preparation_lstm(dataset_raw, 'Bitcoin sign change')

    LSTM_classifier_raw = LSTM(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
    LSTM_classifier_raw.build_model()
    LSTM_classifier_raw.fitting_model()
    LSTM_classifier_raw.evaluate_model()


if __name__ == "__main__":
    main()