import numpy as np
import nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from nltk.corpus import movie_reviews
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2, l1, l1_l2

# Download NLTK data (if not already downloaded)
#nltk.download("movie_reviews")
#nltk.download("stopwords")


# Load movie reviews and labels
documents = [list(movie_reviews.words(file_id)) for file_id in movie_reviews.fileids()]
labels = [1 if file_id.split("/")[0] == "pos" else 0 for file_id in movie_reviews.fileids()]


# Convert words into a bag-of-words representation
#vectorizer = CountVectorizer(binary=False, max_features=2000, stop_words=nltk.corpus.stopwords.words("english"))

vectorizer = TfidfVectorizer(max_features=5000, stop_words=nltk.corpus.stopwords.words("english"))
X = vectorizer.fit_transform([" ".join(doc) for doc in documents])
print()
X = X.toarray()


# # Iterate through the vocabulary and print the index, word, and count
# whatDoesVectorLookLike = []
# for word, index in vectorizer.vocabulary_.items():
#     count = X.sum(axis=0).A1[index]  # Count of the word in the corpus
#     #print(f"Index: {index}, Word: {word}, Count: {count}")
#     whatDoesVectorLookLike.append( (index, word, count))
# whatDoesVectorLookLike.sort()
# for (i, w, c) in whatDoesVectorLookLike:
#     print(i, w, c)


# firstRow = X[0]  # document 0
# print(firstRow.toarray())

# Set seeds for TensorFlow, NumPy, and Python random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


print(X_train.shape)


# Create a custom neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)), # Input layer for features with 2000 features
    keras.layers.Dense(200,activation='relu', kernel_regularizer=l2(0.001)), # Hidden layer with 200 nodes and ReLU activation
    keras.layers.Dense(200, activation='relu', kernel_regularizer=l1(0.001)),  # Hidden layer with 200 nodes and ReLU activation
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary sentiment classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

model.fit(X_train, y_train, epochs=3, batch_size=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

correct_tags = ["pos" if correct == 1 else "neg" for correct in y_test]
test_tags = list(model.predict(X_test) )
test_tags = ["pos" if predicted >= 0.5 else "neg" for predicted in test_tags]

# how about its precision and recall per category
mtrx = nltk.ConfusionMatrix(correct_tags, test_tags)
print()
print(mtrx)
print()
print(mtrx.evaluate())