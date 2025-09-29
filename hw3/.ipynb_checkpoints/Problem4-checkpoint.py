import nltk
import random
from nltk.corpus import movie_reviews, stopwords
from nltk.metrics import precision, recall
from nltk.classify import accuracy as nltk_classify_accuracy

# construct a list of documents labeled with the appropriate category , use this for training and testing , gold standard
# documents will be a list of tuples where the first term is all the words for that specific file, and the second is the category 
documents = [(list(movie_reviews.words(fileid)), category)
             # go thru each category for movies
             for category in movie_reviews.categories()
             # for each file in that category , retrive the words^, and category
              for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)


""" define the feature extractor 
- each word will be a feature,
so whether or not the document contains that word is a feature. 
take 2000 most common words in the corpus"""

# get the freqDist the words in the movie_review
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# get the 2000 most freq words 
word_features = list(all_words)[:2000]


# build the feature extractor 
# take the document (words, category)
def document_features(document): 
    # get all the unique words in the document , vocabulary |V|
    document_words = set(document) 
    # init a dict
    features = {}
    # go through each freq word (2000 most occuring)
    for word in word_features:
        # set boolean for each unique word, whether it occurs or not
        features['contains({})'.format(word)] = (word in document_words)
    return features


# train the feature classifier
# list of tuples where d is the dictionary containing whether a word is present or not and c is category
featuresets = [(document_features(d), c) for (d,c) in documents]
# split the set into training and test 
train_set, test_set = featuresets[100:], featuresets[:100]

# run naive bayes on training
bayesclassifier = nltk.NaiveBayesClassifier.train(train_set)


# run a decision tree 
decisionclassifier = nltk.DecisionTreeClassifier.train(train_set)

# run a maximum entropy
maxentropyclassifier =  nltk.MaxentClassifier.train(train_set, 
    algorithm='GIS',  
    max_iter=100,      
    min_lldelta=0.01 
                                                   )


# calculate precision and recall for bayes classifer , decision tree, and maximum entropy 
from nltk.metrics import precision, recall, f_measure
from nltk.classify import accuracy as nltk_classify_accuracy  

# note: used genAI to figure out the best method for getting precision and recall since I could not find it in the NLTK book
# write a function that that evaluates a given classifier, that way can encapsulate precision recall and accuracy 
def evaluate_classifier(classifier,test_set):
    # get the predictions , i.e what the model thinks the classification of the document is , in this case sentiment based on words in movie review
    refsets = {'pos': set(), 'neg': set()}  # Actual labels
    testsets = {'pos': set(), 'neg': set()} # Predicted label

    # populate the refsets and testsets
    for i, (features,label) in enumerate(test_set):
        refsets[label].add(i)
        predicted = classifier.classify(features)
        testsets[predicted].add(i)

    # Calculate metrics for each class
    print(f"\n{'Class':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 45)

    # for each classification calculate precision and recall 
    for label in ['pos', 'neg']:
        prec = precision(refsets[label],testsets[label])
        rec = recall(refsets[label], testsets[label])

        print(f"{label:<10} {prec:<10.3f} {rec:<10.3f}")


              # Overall accuracy
    acc = nltk_classify_accuracy (classifier, test_set)
    print(f"\nOverall Accuracy: {acc:.3f}")


        
    return acc


# Evaluate all classifiers

print("Naive Bayes Classifier:")
evaluate_classifier(bayesclassifier, test_set)

print("\nDecision Tree Classifier:")
evaluate_classifier(decisionclassifier, test_set)

print("\nMaximum Entropy Classifier:")
evaluate_classifier(maxentropyclassifier, test_set)

# recreate dataset without stop words
from nltk.corpus import stopwords
nltk.download('stopwords')  
stop_words = set(stopwords.words('english'))
# now have a set of common english stop words 

# now just need to modify document feature extractor tyo only extract words that are not in the stopwoirds
# stop words setup
stop_words = set(stopwords.words('english'))
word_features_noSW = [w for w in word_features if w not in stop_words]

def document_features_nostopwords(document): 
    document_words = set(document)
    features = {}
    for word in word_features_noSW:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresetsnoSW = [(document_features_nostopwords(d), c) for (d,c) in documents]
train_set, test_set = featuresetsnoSW[100:], featuresetsnoSW[:100]


# run classifiers on training set with stop words removed

# run naive bayes on training
bayesclassifier = nltk.NaiveBayesClassifier.train(train_set)
# run a decision tree 
decisionclassifier = nltk.DecisionTreeClassifier.train(train_set)
# max ent
maxentropyclassifier =  nltk.MaxentClassifier.train(train_set, 
    algorithm='GIS',  
    max_iter=100,      
    min_lldelta=0.01 
                                                   )


# evaluate classifiers with new feature aspect: stop words removed

print("Naive Bayes Classifier:")
evaluate_classifier(bayesclassifier, test_set)

print("\nDecision Tree Classifier:")
evaluate_classifier(decisionclassifier, test_set)

print("\nMaximum Entropy Classifier:")
evaluate_classifier(maxentropyclassifier, test_set)


negation_words = {'not', "n't"}
punctuation = {'.', '!', '?'}
# create a function to mark negation in the words in each document
def mark_negation(document):
    # create a new document
    new_doc = []
    # create a negative boolean flag that will be set to true when a negation word is encountered 
    negative = False
    # loop thru each word in the document
    for word in document:
        # conver tto lower case
        lw = word.lower()
        # word is not or n't 
        if lw in negation_words:
            # set the flag to true and add the word to the document 
            negative = True
            new_doc.append(lw)  
            # if the flag is up then modify the word with the negaation 
        elif negative:
            new_doc.append(f"not_{lw}") 
        else:
            new_doc.append(lw)
        # if puncuation is encountered now the flag set to false and regular words are added
        if lw in punctuation:
            negative = False 
    # returns an array of words for each document with negations
    return new_doc

# create documents negated 

# create documents negated , returns a list of tuples containing the negation cotaining list and classifcation pair for each document
documents_negated = [(mark_negation(d), c) for (d, c) in documents]
# extract features as normal just from documents_negated
featuresets = [(document_features(d), c) for (d,c) in documents_negated]
# split the set into training and test 
train_set, test_set = featuresets[100:], featuresets[:100]


# train and test classifers as normal 
bayesclassifier = nltk.NaiveBayesClassifier.train(train_set)
# run a decision tree 
decisionclassifier = nltk.DecisionTreeClassifier.train(train_set)
# max ent
maxentropyclassifier =  nltk.MaxentClassifier.train(train_set, 
    algorithm='GIS',  
    max_iter=100,      
    min_lldelta=0.01 )


# evaluate classifiers with new feature aspect: negation words considered

print("Naive Bayes Classifier:")
evaluate_classifier(bayesclassifier, test_set)

print("\nDecision Tree Classifier:")
evaluate_classifier(decisionclassifier, test_set)

print("\nMaximum Entropy Classifier:")
evaluate_classifier(maxentropyclassifier, test_set)

