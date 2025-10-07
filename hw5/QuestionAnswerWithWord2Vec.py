import random

import gensim
import gensim.downloader as api

import nltk
from sklearn.metrics.pairwise import cosine_similarity


# this block reads in the questions from the tab-delimited file
questions = []
filename = "myQuestions.txt"   # tab separated
f = open(filename, 'r')
for line in f:
    line = line.strip()
    if len(line) == 0: # ignore blank lines
        continue
    tokens = line.split('\t') # tab delimited
    questions.append(tokens)
f.close()
print("there are", len(questions), "questions.")  # for debugging

# pretty print the questions, for debugging mostly, but nice
for q in questions:
    print("Question:\t", q[0])
    for i in range(1, len(q)):
        print("\t" + chr(ord('A') + i - 1) + "." + q[i])


# uncomment the next line when you are ready to work with word2vec
#model = api.load("word2vec-google-news-300")

def averageVector(s):
    '''
    Compute the average word2Vec vector for each word in s
    For instance, s might be the string "The dog ran after the cat."
    :param s: A string of words
    :return: the average vector
    '''

    # this method needs to be fixed!!
    averageVector = [10*random.random() - 5 for i in range(300)]   # a placeholder.  These values aren't random!
    # Remember that the google news
    # database is a vector of size 300.

    # tokenize the sentence using nltk word_tokenize.
    words = nltk.tokenize(s)
    # code needed here

    # write a loop that will sum up all of the word2vec vectors for each word
    # you will need a try-except block to handle the case where the word is not
    # in the word2vec "dictionary".   Just ignore words that aren't in the dictionary.

    sumVectors = 300*[0]
    countVectors = 0   # how many vectors get added together?  You'll need that to
            # compute the average

    # iterate through each word in the string
        # find its word2vec vector (if it exists)
        # add it to the running sum (sumVectors)
        # if you add something, increase the counter

    # once you have the sum of the vectors, compute the average vector
    # by taking each vector value and dividing by how many vectors were
    # added together

    return averageVector


# for each question
# no changes needed below
for q in questions:
    theQuestion = q[0]   # the question is the first item in the list
    questionV = averageVector(theQuestion)  # the average vector for the question
    closest = ("No answer", float('-inf'))
    print(theQuestion)

    # the values in indices 1 to 4 are the possible answers
    # find the one with the maximum similarity
    for i in range(1, len(q)):
        answerV = averageVector(q[i])  # the average vector for answer "i"
        similarity = cosine_similarity([questionV], [answerV])[0][0]
        print('\t', i, similarity, q[i])  # for debugging purposes
        if similarity > closest[1]:
            closest = (q[i], similarity)

    if closest[0] == q[1]:  # the first choice is always the correct one in my database
        print("The computed answer is ", closest[0], "which is correct.")
    else:
        print("The computed answer is ", closest[0], "which is incorrect.")


# questions 
# what is the meaning of love - baby dont hurt me , attachment, infatuation , loyalty
# main character of the movie The Matrix ? 
