import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re
import os

# Model and tokenizer configurations
base_model = 'google/gemma-2b-it'
saved_model_name = './backend/Therapy_Gemma_2bi_QLoRA_v1'
tokenizerid = 'philschmid/gemma-tokenizer-chatml'

def _setHFToken():
    with open("./hf_token.txt", "r") as file:
        token = file.read()      
    return token
os.environ['HF_TOKEN'] = _setHFToken()

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Set page configuration
st.set_page_config(page_title="Therapy Chat-Bot")

# Load tokenizer and model
@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(tokenizerid, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(base_model, token=os.environ['HF_TOKEN'])
    model = PeftModel.from_pretrained(model, saved_model_name)
    return tokenizer, model

tokenizer, model = get_model()
if torch.cuda.is_available():
    device = torch.device(type='cuda', index=torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    # print("Current CUDA device:", device)
    # print("Total memory available:", properties.total_memory / (1024 * 1024), "MB")
    # print("Memory allocated:", torch.cuda.memory_allocated(device) / (1024 * 1024), "MB")
else:
    print("CUDA is not available. Using CPU.")

model = model.to(device)

# def clear_text():
#     st.session_state.my_text = ""

# Define UI layout
def main():
    st.header("Therapy ChatBot")
    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Get user input
    user_input = st.text_input(USER_AVATAR+"You:")

    # Get conversation history from session state
    history = st.session_state.get("history", [])

    # Process user input and display conversation history
    if user_input:
        # Add user input to the history
        history.append((USER_AVATAR+"You:", user_input))

        # Generate bot response
        chat = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
        bot_response = tokenizer.decode(outputs[0])

        # print(bot_response)
        temp = bot_response.split('<|im_start|>assistant')
        bot_response = temp[-1].replace('<eos>', '').strip()
        # Add bot response to the history
        history.append((BOT_AVATAR+"Bot:", bot_response))

        # Store updated conversation history in session state
        st.session_state["history"] = history
        st.session_state.user_input = ""

    # Display conversation history
    for role, message in history:
        st.write(role, message)

if __name__ == '__main__':
    main()