import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import read_data, split_train_test, plot_roc_curve


def split_dataset(dataset):
    train_size = int(len(dataset) * 0.8)
    train, test = dataset.iloc[0:train_size], dataset.iloc[train_size:len(dataset)]
    X_train = train.drop("Bitcoin sign change", axis=1)
    y_train = train["Bitcoin sign change"]
    X_test = test.drop("Bitcoin sign change", axis=1)
    y_test = test["Bitcoin sign change"]
    return X_train, X_test, y_train, y_test


class LSTM():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.LAYERS = [8, 8, 8, 1]                # number of units in hidden and output layers
        self.EPOCH = 300                           # number of epochs
        self.LR = 5e-2                            # learning rate of the gradient descent
        self.LAMBD = 3e-2                         # lambda in L2 regularization
        self.DP = 0.0                             # dropout rate
        self.RDP = 0.0                            # recurrent dropout rate

    def data_preparation_lstm(self, look_back=1):
        dataX_train, dataY_train, dataX_test, dataY_test = [], [], [], []

        for i in range(len(self.X_train)-look_back-1):
            dataX_train.append(self.X_train[i:(i+look_back)])
            dataY_train.append(self.y_train[i + look_back])

        for i in range(len(self.X_test)-look_back-1):
            dataX_test.append(self.X_test[i:(i+look_back)])
            dataY_test.append(self.y_test[i + look_back])

        self.X_train, self.y_train = np.array(dataX_train), np.array(dataY_train)
        self.X_test, self.y_test = np.array(dataX_test), np.array(dataY_test)  
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[2])
        # self.y_train, = self.y_train.reshape(self.y_train.shape[0], 1, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[2])
        # self.y_test = self.y_test.reshape(self.y_test.shape[0], 1, 1)


    def build_model(self):
        # Build the Model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(input_shape = [self.X_train.shape[1], self.X_train.shape[2]], units=self.LAYERS[0],
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(self.LAMBD), 
                    recurrent_regularizer=tf.keras.regularizers.l2(self.LAMBD),
                    dropout=self.DP, recurrent_dropout=self.RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(units=self.LAYERS[1],
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(self.LAMBD), 
                    recurrent_regularizer=tf.keras.regularizers.l2(self.LAMBD),
                    dropout=self.DP, recurrent_dropout=self.RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(units=self.LAYERS[2],
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(self.LAMBD), 
                    recurrent_regularizer=tf.keras.regularizers.l2(self.LAMBD),
                    dropout=self.DP, recurrent_dropout=self.RDP,
                    return_sequences=False, return_state=False,
                    stateful=False, unroll=False
                    ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=self.LAYERS[3], activation='sigmoid'))

        # Compile the model with Adam optimizer
        model.compile(loss='binary_crossentropy',
                    metrics=['accuracy'],
                    optimizer=tf.keras.optimizers.Adam(lr=self.LR))
        print(model.summary())

        # Define a learning rate decay method:
        lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                    patience=5, verbose=0, 
                                    factor=0.5, min_lr=1e-8)

        # Define Early Stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                patience=30, verbose=1, mode='auto',
                                baseline=0, restore_best_weights=True)
        # Training the model
        history = model.fit(self.X_train, self.y_train,
                        epochs=self.EPOCH,
                        batch_size= self.X_train.shape[0],
                        validation_split=0.2,
                        shuffle=False,
                        verbose=2,
                        callbacks = [lr_decay, early_stop])

        # Plotting validation and training loss curves
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

        # Evaluate the model:
        test_acc = model.evaluate(self.X_test, self.y_test,
                                            batch_size=self.X_test.shape[0], verbose=0)[1]
        print(f'test accuracy = {round(test_acc * 100, 4)}%')
        print(f'test error = {round((1 - test_acc) *  self.X_test.shape[0])} out of  {self.X_test.shape[0]} examples')


def main():
    np.random.seed(1)
    tf.random.set_seed(1)
    
    dataset = read_data()

    X_train, X_test, y_train, y_test = split_dataset(dataset)

    LSTM_classifier = LSTM(X_train, X_test, y_train, y_test)
    LSTM_classifier.data_preparation_lstm()
    LSTM_classifier.build_model()


if __name__ == "__main__":
    main()


"""y_hat = model.predict_classes(X_test, batch_size=M_TEST, verbose=1)
for i in range(y_hat.shape[0]):
    print(y_hat[i], y_test[i])"""