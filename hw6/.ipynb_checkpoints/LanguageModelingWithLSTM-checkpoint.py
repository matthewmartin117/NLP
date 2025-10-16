import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import pandas as pd




file_path = 'IMDB_Dataset.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Assuming the Excel file has two columns named 'Column1' and 'Column2'
# You can access these columns as follows:
reviewWords = df['review']
reviewSentiment = df['sentiment']


# sentencesInPositiveReviews = []
# for i in range(len(reviewWords)):
#     tokenizedSentences = nltk.sent_tokenize(reviewWords[i])
#     if reviewSentiment[i] == 'positive':
#         sentencesInPositiveReviews += tokenizedSentences
#         # if len(sentencesInPositiveReviews) >= 10000:
#         #     break
# print("There are", len(sentencesInPositiveReviews), "sentences in the positive reviews.")
# print("Shuffling")
# np.random.shuffle(sentencesInPositiveReviews)
# sentencesInPositiveReviews = sentencesInPositiveReviews[:10000]
# print("There are now", len(sentencesInPositiveReviews), "sentences in the positive reviews.")

sentencesInNegativeReviews = []
for i in range(len(reviewWords)):
    tokenizedSentences = nltk.sent_tokenize(reviewWords[i])
    if reviewSentiment[i] == 'negative':
        sentencesInNegativeReviews += tokenizedSentences
        # if len(sentencesInPositiveReviews) >= 10000:
        #     break
print("There are", len(sentencesInNegativeReviews), "sentences in the negative reviews.")
print("Shuffling")
np.random.shuffle(sentencesInNegativeReviews)


# because of runtime issues, let's truncate to only 10,000 sentences
sentencesInNegativeReviews = sentencesInNegativeReviews[:10000]
print("There are now", len(sentencesInNegativeReviews), "sentences in the negative reviews.")

# out of curiosity, how many tokens?
numberOfTokens = sum(len(x) for x in sentencesInNegativeReviews)
print("There are now", numberOfTokens, "tokens in the negative reviews.")

# Sample dataset
corpus = [  # if you want to play with a really simple model
    "This is a simple language model",
    "Language models generate text",
    "Text generation is fascinating",
    "This is an example of a text model"
]

#corpus = sentencesInPositiveReviews
corpus = sentencesInNegativeReviews

# Tokenization
max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print("There are", total_words, "words in the vocabulary.")

max_words = min(max_words, total_words)  # only take 20,000 if there are more than 20,000 in the corpus
# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences and create predictors and labels
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
print("Number of input sequences:", len(input_sequences))
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=max_words)
print("Input sequences built.")

# Build the model
embedding_size = 100
model = Sequential([
    Embedding(max_words, embedding_size, input_length=max_sequence_len-1),
    LSTM(200),
    Dense(max_words, activation='softmax')
])
print("Model created")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model compiled")



# Function to generate text
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


def generate_text2(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        # Convert seed text to a sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        # Get the model's predicted probabilities
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Sample a word based on cumulative probabilities
        predicted_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)

        # Find the word corresponding to the index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Append the chosen word to the seed text
        seed_text += " " + output_word

    return seed_text


# Train the model
epoch_size = 2
for i in range(1, 51):
    model.fit(X, y, epochs=epoch_size)
    # Generate text

    next_words = 50
    print("After", i*epoch_size, "epochs:")

    seed_text = "This adventure movie is"
    print("Adventure text 1: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    print("Adventure Text 2: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    seed_text = "This science fiction movie is"
    print("Science Fiction text 1: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    print("Science Fiction Text 2: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    seed_text = "This horror movie is"
    print("Horror text 1: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    print("Horror Text 2: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    seed_text = "This romantic comedy movie is"
    print("Romantic Comedy text 1: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    print("Romantic Comedy Text 2: ")
    print(generate_text2(seed_text, next_words, max_sequence_len))
    print("\n")
