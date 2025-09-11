# import regex for python
import re 

# define the subsitution rules as a list of tuples, the pattern to search for , and repsonse to give 
subsitution_rules = [
    (r'.*My (order|package) is late.*', r'I am sorry to hear about your \1, Can you provide me with your order details?'),
    (r'.*order number.*(\d+)', r'I looked up order number \1 and email you the details'),
    (r'.*all.*', r'In what way?'),
    (r'.*refund.*',r'I understand you are asking about a refund, I will log it with customer service'),
    (r'.*broken.*',r'I am sorry to hear the product is broken. Can you describe the details?'),
    (r'.*[Tt]hanks.*',r'You are welcome. is there anything else I can help you with?'),
    (r'.*[Nn]o.*',r'alright, thank you for contacting customer service'),
]

#func to get responses based on the user input 
def eliza_response(user_input):
    # loop through all the subsitutuion rules:
    for pattern, response in subsitution_rules:
        # check if the current pattern matches the user input 
        # use re match syntaxt is re.match(pattern, string, flags =0 default) use ignorecase for case insensitive matching 
        if re.match(pattern,user_input,re.IGNORECASE):
            # if the match occurs subsitute for the response and return it , sub replaces a match with a diff string 
            return re.sub(pattern,response,user_input, flags=re.IGNORECASE)
            
    return "Please tell me more."
    


# Main program to interact with the user 
def eliza_chat():
    print("Hello I am ELIZA. How can I help you today?")
    while True:
        user_input = input("You:")
        # end the conversation if the user says bye 
        if user_input.lower() == 'bye':
            print("ELIZA:Goodbye!Take Care.")
            # end the loop and end the program
            break 
        # otherwise generate an ELIZA response 
        response = eliza_response(user_input)
        print(f"ELIZA:{response}")


# start the chatbot 
if __name__ == "__main__":
    eliza_chat()