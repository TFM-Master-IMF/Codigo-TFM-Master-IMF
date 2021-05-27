import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import read_data, plot_roc_curve

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
        self.EPOCH = 300                          # number of epochs
        self.LR = 5e-2                            # learning rate of the gradient descent
        self.LAMBD = 3e-2                         # lambda in L2 regularization
        self.DP = 0.0                             # dropout rate
        self.RDP = 0.0                            # recurrent dropout rate             
        self.model = None
    
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
                    metrics=['accuracy'],
                    optimizer=tf.keras.optimizers.Adam(lr=self.LR))
        print(self.model.summary())

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
                        verbose=2,
                        callbacks = [lr_decay, early_stop])

        # Plotting validation and training loss curves
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        # Evaluate the model:
        test_acc = self.model.evaluate(self.X_test, self.y_test,
                                            batch_size=self.X_test.shape[0], verbose=0)[1]
        y_pred_proba = self.model.predict(self.X_test)
        print(f'test accuracy = {round(test_acc * 100, 4)}%')
        print(f'test error = {round((1 - test_acc) *  self.X_test.shape[0])} out of  {self.X_test.shape[0]} examples')
        plot_roc_curve(self.y_test, y_pred_proba)

def main():
    np.random.seed(1)
    tf.random.set_seed(1)
    
    dataset = read_data()
    X_train, X_test, y_train, y_test = data_preparation_lstm(dataset, 'Bitcoin sign change')

    LSTM_classifier = LSTM(X_train, X_test, y_train, y_test)
    LSTM_classifier.build_model()
    LSTM_classifier.fitting_model()
    LSTM_classifier.evaluate_model()

if __name__ == "__main__":
    main()