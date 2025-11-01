from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
# be sure to pip install langchain-openai and langchain-community

# Set up OpenAI LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_KEY")
)
# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])

# Create a Runnable sequence (recommended in new LangChain versions)
chain = prompt | llm

# Example usage
question = "What is the capital of France?"
response = chain.invoke({"question": question})
print("Response:", response.content)
