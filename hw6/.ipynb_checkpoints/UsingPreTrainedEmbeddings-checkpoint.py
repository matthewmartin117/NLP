import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2  # Import L2 regularization
import tensorflow as tf

import gensim
import gensim.downloader as api



# Download the movie_reviews dataset if not already downloaded
#nltk.download('movie_reviews')

# Load the English stop words from NLTK
stop_words = set(stopwords.words('english'))

file_path = 'IMDB_Dataset.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Assuming the Excel file has two columns named 'Column1' and 'Column2'
# You can access these columns as follows:
reviewWords = df['review']
reviewSentiment = df['sentiment']

# Load the movie_reviews dataset and labels
positiveDocuments = []
negativeDocuments = []
for i in range(len(reviewWords)):
    tokenizedWords = nltk.word_tokenize(reviewWords[i])
    if reviewSentiment[i] == 'positive':
        positiveDocuments.append( (tokenizedWords, 'pos'))
    else:
        negativeDocuments.append( (tokenizedWords, 'neg'))


print("There are", len(positiveDocuments), "movies with positive reviews")
print("There are", len(negativeDocuments), "movies with negative reviews")

allDocuments = positiveDocuments + negativeDocuments
print("There are", len(allDocuments), "documents.")

documents = allDocuments

# Shuffle the documents
# Set seeds for TensorFlow, NumPy, and Python random
random_seed = 113 # 59 #42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
#random.seed(random_seed)
np.random.shuffle(documents)
print("Documents are shuffled.")

# Separate the text and labels
texts = [' '.join(doc) for doc, label in documents]
labels = [label for _, label in documents]

# Encode labels as binary (0 for negative, 1 for positive)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print("Labels are encoded")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=random_seed)
# Preprocess the text to remove stop words
X_train_no_stopwords = [' '.join([word for word in doc.split() if word.lower() not in stop_words]) for doc in X_train]
X_test_no_stopwords = [' '.join([word for word in doc.split() if word.lower() not in stop_words]) for doc in X_test]


print("Training and test sets are complete.")
print("Training set size: ", len(X_train))
print("Test set size: ", len(X_test))

# Tokenize the text data and pad sequences
max_words = 4048
max_sequence_length = 512


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_no_stopwords)
print("Training set tokenized.")

X_train = tokenizer.texts_to_sequences(X_train_no_stopwords)
X_test = tokenizer.texts_to_sequences(X_test_no_stopwords)

X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)
print("Training and test sets tokenized and padded to length", len(X_train[0]))


# Load the pre-trained Word2Vec embeddings from Google
#word2vec_model = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
word2vec_model = (
    gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=50000))

# Create an embedding matrix using the pre-trained Word2Vec embeddings
embedding_matrix = np.zeros((max_words, 300))  # Assuming 300-dimensional Word2Vec embeddings

for word, i in tokenizer.word_index.items():
    if i < max_words:
        if word in word2vec_model:
            #print("Added", word)
            embedding_matrix[i] = word2vec_model[word]
        elif (word[0].upper() + word[1:]) in word2vec_model:
            #print("Added", word)
            embedding_matrix[i] = word2vec_model[word[0].upper() + word[1:]]
        else:
            #print("Did not add", word)
            pass

# Define and compile the neural network model
embedding_dim = 300  # Matching the dimension of the pre-trained Word2Vec embeddings
print("Embeddings created.")


model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))  # Use pre-trained embeddings, set trainable to True/False
model.add(Flatten())
model.add(Dense(200, activation='relu', kernel_regularizer=l2(0.001) ))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification, sigmoid activation

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model compiled.")

# Train the model
epochs = 3
batch_size = 128

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
print("Modeled trained.")

# Evaluate the model on the test data

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

correct_tags = ["pos" if correct == 1 else "neg" for correct in y_test]
test_tags = list(model.predict(X_test) )
test_tags = ["pos" if predicted >= 0.5 else "neg" for predicted in test_tags]

# how about its precision and recall per category
mtrx = nltk.ConfusionMatrix(correct_tags, test_tags)
print()
print(mtrx)
print()
print(mtrx.evaluate())
