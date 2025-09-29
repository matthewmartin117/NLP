import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')


word_pairs = [
    ("car", "automobile"),
    ("gem", "jewel"),
    ("journey", "voyage"),
    ("boy", "lad"),
    ("coast", "shore"),
    ("asylum", "madhouse"),
    ("magician", "wizard"),
    ("midday", "noon"),
    ("furnace", "stove"),
    ("food", "fruit"),
    ("bird", "cock"),
    ("bird", "crane"),
    ("tool", "implement"),
    ("brother", "monk"),
    ("lad", "brother"),
    ("crane", "implement"),
    ("journey", "car"),
    ("monk", "oracle"),
    ("cemetery", "woodland"),
    ("food", "rooster"),
    ("coast", "hill"),
    ("forest", "graveyard"),
    ("shore", "woodland"),
    ("monk", "slave"),
    ("coast", "forest"),
    ("lad", "wizard"),
    ("chord", "smile"),
    ("glass", "magician"),
    ("rooster", "voyage"),
    ("noon", "string"),
]
# for each pair we want to make a dictionary with the pair and similarity score 
def CalcWordSimilarity(pairs):
    dictionary = {}
    # loop through each
    for pair in pairs:
        word1 = pair[0]
        word2 = pair[1]
        # get the synset of each word 
        syn1 = wn.synsets(word1,pos='n')
        syn2 = wn.synsets(word2,pos='n')
        # init a var for max_sim 
        max_sim = 0
        # nested loop to calculate similarity for each synset to the each synset in the other word
        for s1 in syn1:
            for s2 in syn2:
                sim = s1.path_similarity(s2)
                # if the similarity is higher than the current similarity , can replace it 
                if sim > max_sim:
                    max_sim = sim
        # after the max similarity between the         
        dictionary[pair] = max_sim


    # take the words and print them in decreasing order
    sorted_pairs = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    for pair in sorted_pairs:
        print(pair)


print("similarity of word pairs, shown in decreasing order")
CalcWordSimilarity(word_pairs)
