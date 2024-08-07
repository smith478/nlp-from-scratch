# Reuters article topic classification

We will compare a few different modeling techniques on the Reuters dataset to identify topics using the free text from the article. In particular we will compare:
- Fine tune LLM as a multi-class, multi-label classification problem
- Use retrieval augmented generation (RAG)
- Train LLM from scratch on Reuters data
    - Next token prediction like GPT
    - Masked language predication like BERT
- PEFT - Use LORA (or similar variant) to train efficiently