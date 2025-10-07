
import gensim
import gensim.downloader as api

def chooseBestAnswer(possibleAnswers, a, b, c):
    '''
    Choose the first answer in the list of possible answers that is not
    a, b, or c, nor are they substrings.
    :param possibleAnswers: A list of words.
    :param a: A word.
    :param b: A word.
    :param c: A word.
    :return: A word
    '''
    a = a.lower()
    b = b.lower()
    c = c.lower()
    for x in possibleAnswers:
        x = x.lower()
        if x != a and x != b and x != c and \
            x not in a and a not in x and \
            x not in b and b not in x and \
            x not in c and c not in x:
            return x
    return possibleAnswers[0] # by default


# Load pre-trained Word2Vec model (Google's Word2Vec)
model = api.load('word2vec-google-news-300')
# Made around 2013 from Google News Dataset.  About 100 billion words.
# The size of the vector is 300.
# The date range of the news articles is mid-2000s to 2013.

listOfExamples = [
    ('man', 'king', 'woman'),
    ('audio', 'video', 'radio'),
    ('cherry', 'red', 'banana'),
    ("chemistry", "molecules", "biology"),
    ("Australia", "Pacific", "England"),
    ("hate", "love", "up"),
    ("walk", "run", "jog"),
    ("actor", "actress", "boy"),
    ("fast", "faster", "happy"),
    ("elbow", "arm", "knee"),
    ("eye", "seeing", "ear"),
    ("Herman_Melville", "Moby_Dick", "Shakespeare"),
    ("quick", "quickly", "slow"),
    ("America", "American", "France"),
    ("work", "worked", "run"),
    ("cat", "cats", "woman"),
    ("goose", "geese", "mouse"),
    ("infant", "adult", "puppy"),
    ("ice", "water", "snow"),
    ("father", "doctor", "mother"),
    ("man", "computer_programmer", "woman"),
    ("music", "lady_gaga","film")
]

for example in listOfExamples:
    # Calculating a is to a* as b is to b*
    a = example[0]
    aStar = example[1]
    b = example[2]

    a_vector = model[a]
    aStar_vector = model[aStar]
    b_vector = model[b]

    # Calculate the target vector: king - man + woman
    bStar_vector = aStar_vector - a_vector + b_vector

    # Use most_similar with just the bStar vector
    result = model.most_similar(positive=[bStar_vector], topn=10)
    possible_answers = [x[0] for x in result]
    # Output the result
    #print(f"{a} is to {aStar} as {b} is to {possible_answers}.")
    bestAnswer = chooseBestAnswer(possible_answers, a, aStar, b)
    print(f"{a} is to {aStar} as {b} is to {bestAnswer}.")
    #break