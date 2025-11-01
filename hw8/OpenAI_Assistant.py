import openai
import os
import time

# Set up OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

# ✅ Step 1: Create an Assistant (only needs to be done once)
assistant = client.beta.assistants.create(
    name="Helpful Assistant",
    instructions="You are a helpful AI assistant.",
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}]  # Add tools like file_search, Code Interpreter, etc.
)

# ✅ Step 2: Start a New Thread for Conversation
thread = client.beta.threads.create()

def chat():
    """ Continuously chats with the Assistant until 'exit' is typed. """
    print("Start chatting with the bot (type 'exit' to stop):")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        try:
            # ✅ Step 3: Add User Message to the Thread
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_input
            )

            # ✅ Step 4: Run the Assistant on the Thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )

            # ✅ Step 5: Poll for Completion
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id, run_id=run.id
                )
                if run_status.status == "completed":
                    break
                time.sleep(1)  # Wait a bit before checking again

            # ✅ Step 6: Retrieve and Print Latest Assistant Response
            messages = client.beta.threads.messages.list(thread_id=thread.id)

            # Get last assistant message
            bot_reply = next(
                (msg.content[0].text.value for msg in messages.data if msg.role == "assistant"),
                "Error: No response received."
            )

            print(f"Chatbot: {bot_reply}")

        except Exception as e:
            print(f"Chatbot encountered an error: {e}")

# Run the chatbot
chat()
