import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten

# convert all thing in order to fit the input shape of our model.
def Input_convert(x_train, x_test, y_train, y_test):
    # reshape the input to get a third demention
    xx_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    xx_test = np.reshape(x_test, (x_test.shape[0], 1, x_train.shape[1]))
    
    # convert the label "negative", "neutral", "positive" to on-hot vector.
    yy_train = np.zeros([y_train.shape[0],3])
    for n in range(y_train.shape[0]):
        if (y_train[n] == 0):yy_train[n] = [1,0,0]
        if (y_train[n] == 1):yy_train[n] = [0,1,0]
        if (y_train[n] == 2):yy_train[n] = [0,0,1]
    yy_test = np.zeros([y_test.shape[0],3])
    for n in range(y_test.shape[0]):
        if (y_test[n] == 0):yy_test[n] = [1,0,0]
        if (y_test[n] == 1):yy_test[n] = [0,1,0]
        if (y_test[n] == 2):yy_test[n] = [0,0,1]
            
    return xx_train, xx_test, yy_train, yy_test

# Built our model
def LSTM_modeling(x_train, x_test, y_train, y_test):
    
    # converting
    xx_train, xx_test, yy_train, yy_test = Input_convert(x_train, x_test, y_train, y_test)
    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
                   input_shape=(1, xx_train.shape[2])))  # returns a sequence of vectors of dimension 128
    model.add(LSTM(128))  # return a single vector of dimension 128
    model.add(Flatten())  # flatten layer
    model.add(Dropout(0.2))  # dropout layer
    model.add(Dense(32, activation='relu'))   #dense layer 
    model.add(Dropout(0.2))  # dropout layer
    model.add(Dense(3, activation='softmax'))  #dense (output) layer 

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


    r = model.fit(xx_train, yy_train,
              batch_size = 64, epochs = 10,
              validation_data=(xx_test, yy_test))
    
    return model

# ======================================================================================================================
# ===============================================Use only when Tuning===================================================
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return plt

# p_test = model1.predict(xx_test).argmax(axis=1)
# cm = confusion_matrix(y_test, p_test)
# plot_confusion_matrix(cm, list(range(3)))

# loss curve
def plot_loss_curve(r):
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    return r

# accuracy curve
def plot_accuracy_curve(r):
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    return plt