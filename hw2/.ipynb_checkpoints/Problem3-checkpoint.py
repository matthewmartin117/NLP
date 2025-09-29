import nltk
# char by char distance , part A
from nltk.metrics.distance import edit_distance
intention = "INTENTION"
execution = "EXECUTION"
distance = edit_distance(intention, execution, substitution_cost=2)
print("distance between INTENTION and EXECUTION",distance)
# word by word , Part B
s1 = "The girl hit the ball"
s2 = "The girl danced at the ball"
# convert strings into a list of words using nltk , turn it into text 
s1 = nltk.word_tokenize(s1)
s2 = nltk.word_tokenize(s2)
# calculate the distance should be 3-  one subsitution and one insertion 
distance = edit_distance(s1, s2, substitution_cost=2)
print("distance between the words in the sentence",distance)