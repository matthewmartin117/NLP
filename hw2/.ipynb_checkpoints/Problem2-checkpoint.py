import nltk
import re
import sentencepiece as spm

# import the dorian gray file first
f = open('DorianGray.txt')
# create a string that stores all the text
raw = f.read()
# tokenize the texts into tokens( basic unit you get when you split text into smaller pieces) 
tokens = nltk.word_tokenize(raw)
# calculate how many tokens are in the dorian gray text
print("amount of tokens in Dorain Gray: ",len(tokens))
print("amount of unique tokens in Dorain Gray: ",len(set(tokens)))
# train the BPE model on the text
spm.SentencePieceTrainer.train(
    input="DorianGray.txt", # this is the text it uses to build the BPE encoding, could be any file of text
    model_prefix="bpe",  # this will create a model called "bpe.model"
    vocab_size=2000, # this needs to be adjusted somewhat depending on the size of the input text and the number of unique characters
    model_type="bpe", # use Byte-Pair Encoding
    user_defined_symbols=["<eos>"], # marker for end-of-sentence
)
# Load and use the model
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")
# we want to know how many non unique bpe encodings are needed 
#encode dorian gray , reassign tokens variable
tokens = sp.encode(raw,out_type = str)
print("number of tokens", len(tokens))
# how many unique BPE encodings are needed 
print("number of unique tokens",len(set(tokens)))