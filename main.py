import A.RF as RF_A
import A.LSTM as LSTM_A
import A.data_preprocessing as data_preprocessing_A
import B.RF as RF_B
import B.LSTM as LSTM_B
import B.data_preprocessing as data_preprocessing_B
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("The library is loaded successfully......")
# ======================================================================================================================
# ====================================================Loadind data======================================================
# Task A
Data_A = pd.read_csv("./Datasets/SemEval2017-task4-dev.subtask-A.english.INPUT.txt", header = None, \
                    delimiter = "\t", quoting = 3)     #load the text file from our datasets
# Task B
Data_B = pd.read_csv("./Datasets/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt", header = None, \
                    delimiter = "\t", quoting = 3)     #load the text file from our datasets
print("The data is loaded successfully......")
# ======================================================================================================================
# ====================================================Data preprocessing================================================
# Task A
label_A,Data_A,sentences_A = data_preprocessing_A.my_data_preprocessing(Data_A,1)         #clean our data
x_train_A, x_test_A, y_train_A, y_test_A = data_preprocessing_A.my_text_vectorization(label_A,Data_A, sentences_A,1)     #text vectorization
# Task B
label_B,Data_B,sentences_B = data_preprocessing_B.my_data_preprocessing(Data_B,1)         #clean our data
x_train_B, x_test_B, y_train_B, y_test_B = data_preprocessing_B.my_text_vectorization(label_B,Data_B, sentences_B,1)     #text vectorization
print("Data preprocessing successfully......")
# ======================================================================================================================
#====================================================Modeling===========================================================
# Task A
print("Now, the modeling starts......")

print("Task A: Woed2vec + Random forest modeling starts......")
model_A = RF_A.RF_modeling(x_train_A,y_train_A)                               # Build model object.
acc_A_train = RF_A.acc(x_train_A, y_train_A, model_A)                           # Train model based on the training set
acc_A_test = RF_A.acc(x_test_A, y_test_A, model_A)                             # Test model based on the test set.
print('The training accuracy is: {},and the test accuracy is: {};'.format(acc_A_train, acc_A_test))

print("Task A: Woed2vec + LSTM modeling starts......")
model_A = LSTM_A.LSTM_modeling(x_train_A, x_test_A, y_train_A, y_test_A)               # Build model object.

# Task B
print("Task B: Woed2vec + Random forest modeling starts......")
model_B = RF_B.RF_modeling(x_train_B,y_train_B)                 # Build model object.
acc_B_train = RF_B.acc(x_train_B, y_train_B, model_B) # Train model based on the training set
acc_B_test = RF_B.acc(x_test_B, y_test_B, model_B)   # Test model based on the test set.
print('The training accuracy is: {},and the test accuracy is: {};'.format(acc_B_train, acc_B_test))

print("Task B: Woed2vec + LSTM modeling starts......")
model_B = LSTM_B.LSTM_modeling(x_train_B, x_test_B, y_train_B, y_test_B)               # Build model object.

