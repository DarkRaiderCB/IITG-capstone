import click
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('data.csv')


training_data = []
for index, row in data.iterrows():
    training_data.append({'role': 'user', 'content': row['input']})
    training_data.append({'role': 'assistant', 'content': row['response']})

print("Training data prepared.")

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def display_chatbox(messages):
    chat_history = ""
    for message in messages:
        if message['role'] == 'user':
            chat_history += f"User: {message['content']}\n"
        else:
            chat_history += f"Assistant: {message['content']}\n"
    print(chat_history)


def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs, max_length=150, num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7, top_k=50, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


pre_prompt = "Assume you are a customer service executive, you are professional, humble, and fun to talk to. You can clearly explain complex things in simple language. It's effortless for customers to get their queries sorted as you provide precise on-point responses. Customer is saying: "

messages = []

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting the chat.")
        break
    else:
        final_prompt = pre_prompt + user_input

        messages.append({'role': 'user', 'content': user_input})

        response = generate_response(final_prompt)

        messages.append({'role': 'assistant', 'content': response})

        click.clear()
        display_chatbox(messages)
