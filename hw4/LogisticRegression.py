import random

import nltk

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
#nltk.download('movie_reviews')
#nltk.download('stopwords')

# grab all movie reviews with pos and neg, know because of file idds 
positiveReviews = [fileid for fileid in movie_reviews.fileids("pos")]
negativeReviews = [fileid for fileid in movie_reviews.fileids("neg")]

# a document will be a list of words and its category
positiveDocuments = [ (list(movie_reviews.words(id)), "pos") for id in positiveReviews ]
negativeDocuments = [ (list(movie_reviews.words(id)), "neg") for id in negativeReviews ]
allDocuments = positiveDocuments + negativeDocuments

#
# why are we shuffling them?
random.seed(42) # comment out if you want "true randomness"
random.shuffle(allDocuments)

stops = list(stopwords.words('english'))
# Decide whether you want all words,
# only words built from a-z, or
# only words built from a-z and that are not stop words

allWordsFreqDist = nltk.FreqDist(w.lower() for w in movie_reviews.words())
#allWordsFreqDist = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w.isalpha())
#allWordsFreqDist = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w.isalpha() and w not in stops)

#
def document_features(document, word_features):
    document_words = set(document)  # gets rid of duplicates
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
#
# You could play around with using more or less words
word_features = list(allWordsFreqDist)[:2000]  # 2000 most common words


#
featureSets = [(document_features(d, word_features), c) for (d,c) in allDocuments]
sizeOfTrainingSet = int(0.8*len(featureSets))
train_set, test_set = featureSets[0:sizeOfTrainingSet], featureSets[sizeOfTrainingSet:]

# You could toy with changing how regularization is done.
# Be sure to read the documentation so that you have some idea
# of reasonable values for "C".
classifier = LogisticRegression(max_iter=1000, random_state=42)
#classifier = LogisticRegression(max_iter=1000, penalty="l2", C=0.1, random_state=42)
#classifier = LogisticRegression(max_iter=1000, solver="liblinear", penalty="l1", C=0.05, random_state=42)

x_Train = [list(a.values()) for (a,b) in train_set]
y_Train = [b for (a,b) in train_set]
# notice the call to "fit" rather than "train"
classifier.fit(x_Train, y_Train)

correct_tags = [c for (w, c) in test_set]
y_Test = [list(a.values()) for (a,b) in test_set]
test_tags = list(classifier.predict(y_Test) )


# how about its precision and recall per category
mtrx = nltk.ConfusionMatrix(correct_tags, test_tags)
print()
print(mtrx)
print()
print(mtrx.evaluate())

x_Test = [list(a.values()) for (a,b) in test_set]
y_Test = [b for (a,b) in test_set]

print("LR Fit", classifier.score(x_Test, y_Test))

featuresPlusImportance = [ (word_features[i], classifier.coef_[0][i]) for i in range(len(classifier.coef_[0]))]
featuresPlusImportance.sort(key = lambda x: abs(x[1]), reverse=True)
print("\nMost Important Features")
for x in range(20):
    print(featuresPlusImportance[x])


