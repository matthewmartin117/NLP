
# In the chapter on logistic regression, the book suggests a very small number
# of features for classifying movie reviews
# x1: count of positive words in the document
# x2: count of negative words in the document
# x3: 1 if "no" is in document, 0 otherwise
# x4 count of first and second person pronouns
# x5 1 if "!" is in document, 0 otherwise
# x6 log(word count of document)

# let's see if it works with Stochastic Gradient Descent
import string

from nltk.corpus import movie_reviews
import random
import nltk
import math
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


#the following block of code creates a dictionary of sentiment
# words whose value is 1 if it is a positive word
# and 0 if it is a negative word.
sentimentWordDictionary = {}
f = open('positive-words.txt', 'r', encoding="latin-1")
for line in f:
    line = line.strip()
    if len(line) == 0: # ignore this line
        continue
    if line[0] == ';': # ignore this line
        continue
    sentimentWordDictionary[line.lower()] = 1
f.close()
f = open('negative-words.txt', 'r', encoding="latin-1")
for line in f:
    line = line.strip()
    if len(line) == 0: # ignore this line
        continue
    if line[0] == ';': # ignore this line
        continue
    sentimentWordDictionary[line.lower()] = 0
f.close()

# for debugging purposes
print("There are", len(sentimentWordDictionary), "sentiment words.")

# Grab all the documents and shuffle them
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
print("There are", len(documents), "documents.") # good for debugging


# this function builds feature x1 and x2 by finding the num of positive and negative words in a certain document
def countPositiveAndNegativeWords(d):
    '''
    Counts the number of positive and negative words in the document
    :param d: A list containing the words in the document
    :return: A tuple (positive, negative) with two integer values representing
            the number of positive and negative words
    '''
    countPositive = 0
    countNegative = 0
    # iterate through every word in the document (bag of words)
    for word in d:
        # reference the dictionary containing the seniment for each word
        # if the word is associated with a postive sentiment increment the count by 1
        if word in sentimentWordDictionary:
            if sentimentWordDictionary[word]== 1:
                countPositive+=1
            else:
                countNegative+=1
        
    # you have to do this
    return countPositive, countNegative


#this function builds feature x3 for each document, checks if no is in the document 
def noInDocument(d):
    '''
    Returns 1 if the word "no" is in the document.   You may
    want to contemplate whether you need to make this case sensitive or not.
    :param d: A list of words in the document.
    :return: 1 if no is in document; 0, otherwise.
    '''
    # convert all words to lowercase in the document first , will catch No and no
    words = [word.lower() for word in d]

    # check if the word no is in the list 
    if "no" in words:
        return 1
    # otherwise return 0 
   
    return 0
    
def countFirstSecondPersonPronouns(d):
    '''
    Returns a count of the number of first and second person pronouns
    within the document.  You might want to look up what is a first or second
    person pronoun.
    :param d: A list of words in the document.
    :return: The count of personal pronouns
    '''
    # create a list of first and second pronouns
    count = 0
    firstandsecondpronouns = ["i","we","you","me","us","my","our","your","mine","ours","yours","myself","ourselves","yourself","yourself"]
     # convert all words to lowercase in the document first
    words = [word.lower() for word in d]
    for word in words:
        if word in  firstandsecondpronouns:
            count+=1
    # complete this
    return count
    
def exclamationInDocument(d):
    '''
    Returns 1 if the word "!" is in the document.
    :param d: A list of words in the document.
    :return: 1 if ! is in document; 0, otherwise.
    '''
    for word in d:
        if word == "!":
            return 1

    # complete this
    return 0

def logOfLength(d):
    '''
    Computes and returns the log of the number of tokens in the document.
    :param d: A list of words in the document.
    :return: log(number of words)
    '''
    # complete this
    # find the num of tokens 
    numtokens = len(d)
    
    return math.log(numtokens)


def document_features(document):
    '''
    Builds the set of features for each document.
    You don't need to modify this unless
    you want to add another feature.
    :param document: A list of words in the document.
    :return: A dictionary containing the features for that document.
    '''
    document_words = list(document) # do not turn into a set!!
    features = {}
    positive, negative = countPositiveAndNegativeWords(document_words)

    features['positiveCount']  = positive
    features['negativeCount'] = negative
    features['noInDoc'] = noInDocument(document_words)
    features['personalPronounCount'] = countFirstSecondPersonPronouns(document_words)
    features['exclamation'] = exclamationInDocument(document_words)
    features['logLength'] = logOfLength(document_words)

    return features

# for each document, extract its features
featuresets = [(document_features(d), c) for (d,c) in documents]

# build the training and test sets
trainingSize = int(0.8*len(featuresets))
train_set, test_set = featuresets[0:trainingSize], featuresets[trainingSize:]

# use stochastic gradient descent with log loss function
classifier = LogisticRegression(max_iter=1000, verbose=0)
x_Train = [list(a.values()) for (a,b) in train_set]
y_Train = [b for (a,b) in train_set]
classifier.fit(x_Train, y_Train)

# print(classifier.coef_)  # if you want to see the coefficients, unsorted

x_Test = [list(a.values()) for (a,b) in test_set]
y_Test = [b for (a,b) in test_set]

print("LR Fit", classifier.score(x_Test, y_Test))

# here is a block of code that sorts the features by absolute value
# and prints them out
featureNames = ['positiveCount', 'negativeCount', 'noInDoc', 'personalPronounCount',  'exclamation', 'logLength']
featuresPlusImportance = [ (featureNames[i], classifier.coef_[0][i]) for i in range(len(classifier.coef_[0]))]
featuresPlusImportance.sort(key = lambda x: abs(x[1]), reverse=True)
for x in range(len(featuresPlusImportance)):
    print(featuresPlusImportance[x])


correct_tags = [c for (w, c) in test_set]
test_tags = list(classifier.predict(x_Test))

# how about its precision and recall per category
mtrx = nltk.ConfusionMatrix(correct_tags, test_tags)
print()
print(mtrx)
print()
print(mtrx.evaluate())


print("""The logistic regression classifier returned an accuracy score of 72%, meaning it used the fearures extracted to predict the sentiment correctly about 72% of the time. The classifier returned a precision score of about 72% for both negative and postive sentiment movies. 

The classifier labeled 137 movies correctly as negative, but misclassified 58 as positive (false positives).

Similarly, it labeled 151 movies correctly as positive, but misclassified 54 as negative (false negatives).”

This is not ideal but also it does not seem like the model favor negative or positive classifactions over the other, meaning it is a relativley "fair" classifier. 

The recall score was 70% for negative senitment and ~74% for positive sentiment. so of all postivley sentimented movie reviews, the classifer caught 74% percent of them, and for all negative sentiment, the classifer caught 70% of them.


The weights seem to make sense, I would have expected the positive and negative weights to be more important, but it could be that the words themselves individual may not be as important as the overall meaning, for example not and bad seperatley vs not bad, or not good. It is also interesting that lengthier reviews tend to be weighted more positivley by the model.""")


