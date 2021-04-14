import pandas as pd       
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import logging

def review_to_words(raw_review, remove_stopwords = False):
    raw_review = re.sub("[^a-zA-Z]", " ", raw_review) # Remove non-letters
    raw_review = raw_review.lower()        # Convert to lower case
    meaningful_words = raw_review.split()        # Split into words
    # remove stopwords when needed
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in meaningful_words if not w in stops]
    return meaningful_words



text_processor = TextPreProcessor(
    
    # terms that will be omit
    omit=['url', 'user', 'hashtag'],
    # terms that will be normalized
    normalize=['email', 'percent', 'money', 'phone','time', 'date', 'number','url', 'user', 'hashtag'],
    
    annotate={None},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer.
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions.
    dicts=[emoticons]
)

# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_words( raw_sentence, \
              remove_stopwords ))
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def create_bag_of_centroids( wordlist, word_centroid_map ):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    # Return the "bag of centroids"
    return bag_of_centroids

def my_data_preprocessing(Data,type):
    Data = pd.DataFrame(Data)    # Dataframe
    Data = Data.iloc[:,[0,2,3]]  # remove useless columns
    num = Data.shape[0]
    Y = np.zeros(num)         # create a blank array to store the label
    for i in range(num):
        Data.iloc[:,2][i] = str(Data.iloc[:,2][i])
        if (Data.iloc[:,1][i] == "negative"):
            Y[i] = 1
        if (Data.iloc[:,1][i] == "positive"):
            Y[i] = 0
            
    sentences = []  # Initialize an empty list of sentences
    if (type == 0):        
        # BoW
        for i in range(num):
            for s in [Data.iloc[:,2][i]]:
                Data.iloc[:,2][i] = " ".join(text_processor.pre_process_doc(s))  # calling TextPreProcessor to clean the reviews
            Data.iloc[:,2][i] = review_to_words(Data.iloc[:,2][i], remove_stopwords = True)  # convert reviews to words
            Data.iloc[:,2][i] = str(Data.iloc[:,2][i])
    else:
        # Word2vec
        for i in range(num):
            for s in [Data.iloc[:,2][i]]:
                Data.iloc[:,2][i] = " ".join(text_processor.pre_process_doc(s))   # calling TextPreProcessor to clean the reviews
                
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []  # Initialize an empty list of sentences

        print("Parsing sentences from training set")
        for review in Data.iloc[:,2]:
            sentences += review_to_sentences(review, tokenizer)    # convert reviews to sentences
        
        for i in range(num):
            Data.iloc[:,2][i] = review_to_words(Data.iloc[:,2][i], remove_stopwords = False)   # convert reviews to words
    
        
    return Y,Data,sentences
    

def my_BoW(Data):
    print("Creating the bag of words...\n")

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    data_features = vectorizer.fit_transform(Data.iloc[:,2])

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    data_features = data_features.toarray()
    
    return data_features
    
def my_Word2vec(Data,sentences):
    # Import the built-in logging module and configure it so that Word2Vec 
    # creates nice output messages
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 20   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # calling init_sims will make the model much more memory-efficient.
    model.init_sims(replace = True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use.
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    
    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = word_vectors.shape[0] // 5

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((Data.iloc[0:,2].size, num_clusters), \
        dtype="float32" )

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in Data.iloc[:,2]:
        train_centroids[counter] = create_bag_of_centroids(review, \
            word_centroid_map )
        counter += 1
        
    return train_centroids
    
    
def my_text_vectorization(Y,Data,sentences,type):    
    if (type == 0):
        # BoW
        X = my_BoW(Data)
    else:
        #Wordvec
        X = my_Word2vec(Data,sentences)
    
    # split training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8)
    
    return x_train, x_test, y_train, y_test
