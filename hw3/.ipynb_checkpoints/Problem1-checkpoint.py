import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

# polysemy the number of senses a word has
print("number of synsets the word dog has", len(wn.synsets('dog', 'n')))


nounsysnsets = list(wn.all_synsets(pos='n'))
verbsynsets = list(wn.all_synsets(pos='v'))
adjsynsets = list(wn.all_synsets(pos='a'))
adverbsynsets = list(wn.all_synsets(pos='r'))


# for nouns find the average 
# loop through nouns 
# init empty dict to keep the names
nouncount = {}
for synset in nounsysnsets:
    for lemma in synset.lemmas():
        word = lemma.name()
        nouncount[word]= nouncount.get(word,0)+ 1
total = sum(nouncount.values())
avg = total/ len(nouncount)
print("polysemy for nouns", avg)


# for verbs find the average 
# loop through verbs 
# init empty dict to keep the names
verbcount = {}
for synset in verbsynsets:
    for lemma in synset.lemmas():
        word = lemma.name()
        verbcount[word]= verbcount.get(word,0)+ 1
total = sum(verbcount.values())
avg = total/ len(verbcount)
print("polysemy for verbs", avg)

# for adj find the average 
# loop through adj 
# init empty dict to keep the names
adjcount = {}
for synset in adjsynsets:
    for lemma in synset.lemmas():
        word = lemma.name()
        adjcount[word]= adjcount.get(word,0)+ 1
total = sum(adjcount.values())
avg = total/ len(adjcount)
print("polysemy for adjectives", avg)


# for adverbs find the average 
# loop through adverbs 
# init empty dict to keep the names
adverbcount = {}
for synset in adverbsynsets:
    for lemma in synset.lemmas():
        word = lemma.name()
        adverbcount[word]= adverbcount.get(word,0)+ 1
total = sum(adverbcount.values())
avg = total/ len(adverbcount)
print("polysemy for adverbs", avg)