import nltk
# take the raw text and tokenize it 
# import the dorian gray file first
f = open('DorianGray.txt')
# create a string that stores all the text
raw = f.read()
tokens = nltk.word_tokenize(raw)
# compute all the bigrams and convert to list 
all_bigrams = list(nltk.bigrams(tokens))
# claculate the freq dist and take top five 
dist = nltk.FreqDist(all_bigrams)
topfive = dist.most_common(5)
print("top 5 most common bigrams",topfive)
# most common words that follow the or The ? 
the = [words for words in all_bigrams if words[0].lower() == "the"]
# create a freq dist for these bigrams to see which ioccur most freq
thedist = nltk.FreqDist(the)
# select the top 5 miost freq 
thetopfive = thedist.most_common(5)
secondwords = [bigram[0][1] for bigram in thetopfive]
print("5 most common words that follow the word the ", secondwords)