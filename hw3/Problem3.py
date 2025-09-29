# import the brown corpus 
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
from nltk.corpus import brown
taggedWords = nltk.corpus.brown.tagged_words()

# find all the singular nouns and plural nouns in the brown corpus
singularNouns = [w for (w, p) in taggedWords if p == 'NN']
pluralNouns = [w for (w, p) in taggedWords if p == 'NNS']


# calc a freq distribution for each word
sN = nltk.FreqDist(singularNouns)
pN = nltk.FreqDist(pluralNouns)


all_words = set(sN.keys()) | set(pN.keys())
# compare singular vs plural
more_common_plural = [
    w for w in set(sN.keys()) | set(pN.keys())
    if pN[w] > sN[w] and w.endswith("s")
]

    
    
print("words where the plural version is more common", more_common_plural )


# note used nltk solution 
tags_dict = {}
for word, tag in taggedWords:
    if word not in tags_dict:
        tags_dict[word] = set()
    tags_dict[word].add(tag) 
 
        
sorted_items_desc = sorted(tags_dict.items(), key=lambda item: len(item[1]),reverse= True)
print("word with the most distinct uses",sorted_items_desc[0])



# nltk 2.4 code
# create bigrams from the tagged words
word_tag_pairs = nltk.bigrams(taggedWords)
# get the first word where the second word is a noun
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NN']
# calculatie the freq for the first word
fdist = nltk.FreqDist(noun_preceders)
# get the tag for the most common word
print("tags that are commonly found after nouns")
print([tag for (tag, _) in fdist.most_common()])

