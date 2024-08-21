import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sqlite3
import torch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("path_to_your_fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("path_to_your_fine_tuned_model")

# Database setup
conn = sqlite3.connect('reports.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS reports
             (id INTEGER PRIMARY KEY, content TEXT)''')

# Streamlit UI
st.title("Radiology Report Auto-Complete")

# Configuration inputs
n_words = st.sidebar.number_input("Number of words to predict", min_value=1, max_value=50, value=10)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
prob_threshold = st.sidebar.slider("Probability Threshold", min_value=0.0, max_value=1.0, value=0.9)

# Text input
input_text = st.text_area("Enter report text:", height=200)

if st.button("Generate"):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate prediction
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + n_words,
            num_return_sequences=1,
            temperature=temperature if temperature > 0 else 1e-7,
            do_sample=temperature > 0,
            top_p=prob_threshold
        )
    
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Display prediction
    st.markdown(predicted_text)

# Save report
if st.button("Save Report"):
    c.execute("INSERT INTO reports (content) VALUES (?)", (input_text,))
    conn.commit()
    st.success("Report saved successfully!")

# Close database connection
conn.close()