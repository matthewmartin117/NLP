import gensim.downloader as api
import nltk
from sklearn.metrics.pairwise import cosine_similarity

listOfExamples = [
    ("apple", "banana", "pear", "brocolli"),
    ("dog", "cat", "bird", "chair"),
    ("red", "green", "blue", "circle"),
    ("hammer", "screwdriver", "nail", "pen"),
    ("mountain", "river", "ocean", "book"),
    ("Monday", "Tuesday", "July", "Thursday"),
    ("lion", "tiger", "bear", "shark"),
    ("table", "chair", "bed", "computer"),
    ("football", "basketball", "chess", "baseball"),
    ("Paris", "London", "New_York", "Africa"),
    ("gold", "silver", "water", "bronze"),
    ("run", "jump", "dance", "chair"),
    ("spring", "summer", "hot", "winter"),
    ("triangle", "circle", "square", "potato"),
    ("car", "train", "plane", "dog"),
    ("math", "science", "history", "pizza"),
    ("eagle", "sparrow", "crow", "dolphin"),
    ("pencil", "eraser", "notebook", "shoes"),
    ("Earth", "Mars", "Saturn", "Sun"),
    ("ice", "fire", "snow", "water"),
]



# Load pre-trained Word2Vec model (Google's Word2Vec)
model = api.load('word2vec-google-news-300')
# Made around 2013 from Google News Dataset.  About 100 billion words.
# The size of the vector is 300.
# The date range of the news articles is mid-2000s to 2013.

def average_similarity(model, w1, w2, w3):
    '''
    Computes the average cosine similarity between
    w1 and w2, w1 and w3, and w2 and w3.
    :param model: The word2vec model passed in.
    :param w1: Word 1.  A string.
    :param w2: Word 2.  A string.
    :param w3: Word 3.  A string.
    :return: A floating point number. The average cosine similarity.
    '''
    # needs to be completed

    # here's an example of how to compute the cosine similarity betwee w1 and w2
    # the model at the specifc word contains a dense embedding with the word vector resulting
    vector1 = model[w1]
    vector2 = model[w2]
    vector3 = model[w3]
    # sklearn cosine similarity reurns a matrix where [i][j] represents the similairtu between ith sample in x and jth sample in y
    # so in this case it would just be 1 entry for each since it is a singular vector for each 
    # 1 and 2 
    sim_1_2 = cosine_similarity([vector1], [vector2])[0][0]
    # 1 and 3 
    sim_1_3 = cosine_similarity([vector1], [vector3])[0][0]
    #2 and 3 
    sim_1_2 = cosine_similarity([vector2], [vector3])[0][0]
    
    
    # divide the sum of cosine similarity by the num of words 
    # returns the highest avg cosine similarity between word groups
    average = ( sim_1_2 + sim_1_3 + sim_1_2 ) / 3
    return average
    
def find_odd_one_out(example):
    '''
    Finds the odd one out of a given example.
    :param example: A tuple of four words (strings)
    :return: The odd one out.  A string.
    '''
    (a, b, c, d) = example  # get the four words
    # needs to be completed
    # c (n,r) = n! / r!(n-r)! 
    # 24 / 6 = 4 combinations of words
    # initliaze a lowest similarity score - set it to first combination a b c
    highest_score = float('-inf')
    # for each combination of words calculate the similarity and update it 
    combinations = [
        (a,b,c,d),
        (a,b,d,c),
        (a,c,d,b),
        (b,c,d,a),
    ]
    for w in combinations:
        print(w)
        sim_score = average_similarity(model, w[0], w[1], w[2])
        print("sim score by removing word:",w[3],sim_score)
        if sim_score > highest_score:
          highest_score = sim_score
          odd_word_out = w[3]
          print("odd word out is now",odd_word_out)
    oddOneOut = odd_word_out
    return oddOneOut


for example in listOfExamples:
    oddOneOut = find_odd_one_out(example)
    print(oddOneOut, "is the odd one out or group", example )


