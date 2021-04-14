AMLS_assignment20_21-SN20040326
 ======
 ## Background
Sentiment Analysis (SA) has gradually emerged in recent years for the analysis and classification of people’s opinions, sentiments, evaluations, appraisals, attitudes and emotions towards particular entities. SA is considered to be one of the challenging research areas in the branch of natural language processing (NLP). The typical tasks in SA are usually involved detecting the sentiment attached to the sentence such as positive or negative. The object of sentiment can be a person, an even, or a topic. Social media platforms such as Twitter has Tweet text have a huge text database, so it attracts a lot of attention in the field of SA. The characteristic of Tweet text is that its language is not formal (including Internet and oral language), there may be typos, punctuation and symbols are not standardized which make decisions of tweet sentiment tricky for computers.<br>
<br>
In this project, we downloaded the Twitter dataset from [SemEval2017-Task 4](https://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools). Two tasks are included in this project:<br>
* A: Message polarity classification: For a given message, classify whether it is positive, negative or neutral sentiment.<br>
* B: Topic-based message polarity classification: The topic of each message is given and the message is positive or negative.<br>

Based on the above two tasks, we propose two text vectorization methods: bag of words (BoW) and word2vec, and three ML models: RF, RNN and LSTM
## Install
### Requirement
* Python 3.6+<br>
* macOS, Linux or Windows
### Installation Options
Go check them out if you do not have them locally installed.
```python
import numpy
import pandas
import sklearn
import matplotlib
import tensorflow
import nltk
import gensim
```
Make sure your tensorflow version is higher than 2.0 <br>
```python
print(tf.__version__)
```
## File Description
### Base Directory
* **main.py** can be excuted to run the whole project.
* **Datasets folder**: Please use your own dataset.
* The file folder A, B contain code fles for each task. Basically, they cannot run separately but need to rely on the support of main.py. You should download the Twitter dataset from [SemEval2017-Task 4](https://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools). The names of the compressed packages you need to download are 4a-english.zip and 4b-english.zip. Finally, make sure that your Datasets file contains SemEval2017-task4-dev.subtask-A.english.INPUT.txt and SemEval2017-task4-dev.subtask-BD.english.INPUT.txt
### Folder A
* **data_preprocessing.py** contains defined function to preprocess our data. 
>>In this file, we shall only look at and call
>>```python
>>def my_data_preprocessing(Data,type)
>>```
>>This function is to clean up some useless data and unify the format of all data. *Data* is the dataset we used, *type* means the different text vectorization method. If type = 0, we use Bag of Word. Otherwise, we use Word2vec.
>>```python
>>def my_text_vectorization(label,Data, sentences,type)
>>```
>>This fuction can convert English text into a number vector to fit the input of the model. *label* is the ground true label. *Data* is the dataset we used. *sentences* is also our data but separated paragraphs into sentences. *type* means the different text vectorization method. If type = 0, we use Bag of Word. Otherwise, we use Word2vec.
* **RF.py** contains defined modeling function which can be called in **main.py**. 
>>Specifically, it includes hyper-parameter selection by using GridSearchCV `RF_ParameterTuning()`, model construction `RF_modeling()`, accuracy report `acc()`, and learning curve plotting `plot_learning_curve()`
* **LSTM.py** contains a lot of defined modeling function which can be called in **main.py**. 
>>Specifically, it includes `Input_convert` which converts all thing in order to fit the input shape of our model, model construction `LSTM_modeling()`, confusion matrix plotting `plot_confusion_matrix()`, loss curve plotting `plot_loss_curve()`, accuracy curve plotting `plot_accuracy_curve()`.
### Folder B 
>>It is almost the same as what is included in folder A. and the only difference is that label has changed from 3 to 2.

## Usage
Make sure your own datasets are in the same directory with **main.py** and have the following structure (subfiles)<br>
* Datasets
> * SemEval2017-task4-dev.subtask-A.english.INPUT.txt
> * SemEval2017-task4-dev.subtask-BD.english.INPUT.txt

Remember, if datasets are placed in the wrong path or missing subfiles, the main fuction will not run.
<br>
Next, just enter the following code in the command window
```
python main.py
```
Then, you can see the accuracy socre for training set and test set in each task<br>

***Hint: It will takes few minutes to executed.*** (The testing computer is configured as CPU: i7-10750H, RAM: 16GB, GPU: Nvidia GeForce RTX 2060, and it taks about 5 min to run this code)
## Contributors
This project exists thanks to all the people who contribute.<br>
[![Jingcheng Wang](https://avatars3.githubusercontent.com/u/72794136?s=60&v=4 "Jingcheng Wang")](https://github.com/Jingcheng-WANG)
