from openai import OpenAI
import gradio as gr
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_KEY"),  # This is the default and can be omitted
)

# Function to interact with the OpenAI model
def chatbot(user_input):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0.7,
    )
    # Get the bot's response from the API call
    bot_response = response.choices[0].message.content.strip()
    return bot_response


# Create Gradio Interface
with gr.Blocks() as demo:
    # Define the chatbot interface with input box and output area
    gr.Markdown("## Chat with OpenAI!")

    chatbot_input = gr.Textbox(label="User Input")
    chatbot_output = gr.Textbox(label="Chatbot Response", interactive=False)

    # Define the button and connect the chatbot function
    submit_button = gr.Button("Send")
    submit_button.click(fn=chatbot, inputs=chatbot_input, outputs=chatbot_output)

# Launch Gradio interface
demo.launch(share=True)
