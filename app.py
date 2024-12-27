import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the fine-tuned model
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

finetuned_model_path = "local_models/finetuned_agent"
finetuned_pipe = load_model(finetuned_model_path)

# Function to generate responses
def generate_response(pipe, prompt):
    response = pipe(prompt, max_length=100, num_return_sequences=1, truncation=True)[0]["generated_text"]
    return response

# Initialize session state for conversation history
if "finetuned_history" not in st.session_state:
    st.session_state.finetuned_history = []

# Streamlit page layout
st.title("Fine-Tuned Model Chat")

st.header("Fine-Tuned Model")
finetuned_prompt = st.text_input("Enter prompt for fine-tuned model:", key="finetuned")
if st.button("Send", key="send_finetuned"):
    if finetuned_prompt:
        finetuned_response = generate_response(finetuned_pipe, finetuned_prompt)
        st.session_state.finetuned_history.append((finetuned_prompt, finetuned_response))

# Display conversation history
for user_msg, bot_msg in st.session_state.finetuned_history:
    st.write(f"**User:** {user_msg}")
    st.write(f"**Agent:** {bot_msg}")
