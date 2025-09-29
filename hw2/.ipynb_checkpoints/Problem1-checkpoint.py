# declare the iterable 
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']

# use list comprehension to iterate trhough 
result = [(word,len(word)) for word in sent]
print(result)


    