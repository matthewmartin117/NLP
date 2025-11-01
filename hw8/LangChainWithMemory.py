from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langgraph.prebuilt import create_react_agent
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os

# Set up OpenAI Chat Model
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_KEY")
)

# ✅ Use ChatMessageHistory to store messages
message_history = ChatMessageHistory()

# ✅ Define structured conversation history
initial_messages = [SystemMessage(content="You are a helpful AI assistant.")]

# ✅ Define the calculator tool
calculator_tool = Tool(
    name="Calculator",
    func=PythonAstREPLTool().invoke,  # Properly formatted callable
    description="Use this tool to perform mathematical calculations. Input should be a valid Python math expression."
)

# ✅ Create the LangGraph ReAct agent
agent_executor = create_react_agent(
    tools=[calculator_tool],
    model=llm
)

def chat():
    print("Start chatting with the bot (type 'exit' to stop):")

    # Copy system message into history
    conversation_history = initial_messages[:]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        try:
            # ✅ Convert user input to a HumanMessage and store it
            user_message = HumanMessage(content=user_input)
            conversation_history.append(user_message)
            message_history.add_message(user_message)  # Save in history

            # ✅ Invoke agent with structured conversation history
            response = agent_executor.invoke({"messages": conversation_history})

            # ✅ Extract the last AI message from the response
            if "messages" in response:
                bot_reply = response["messages"][-1].content  # Get last AI response
                ai_message = AIMessage(content=bot_reply)
                conversation_history.append(ai_message)  # Save to local history
                message_history.add_message(ai_message)  # Save in ChatMessageHistory
            else:
                bot_reply = "Error: Unexpected response format."

            print(f"Chatbot: {bot_reply}")

        except Exception as e:
            print(f"Chatbot encountered an error: {e}")

# Run the chatbot
chat()
