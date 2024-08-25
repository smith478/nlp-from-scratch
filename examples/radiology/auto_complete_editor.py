import tiktoken
import tkinter as tk
from tkinter import filedialog
import torch
from gpt2 import GPTModel

CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class AutoCompleteEditor:
    def __init__(self, master):
        self.master = master
        master.title("GPT-2 Auto-Complete Editor")

        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPTModel(CONFIG)
        self.model.load_state_dict(torch.load("../../models/model_124M.pth", map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Create text widget
        self.text = tk.Text(master, wrap='word')
        self.text.pack(expand=True, fill='both')

        # Bind events
        self.text.bind('<KeyRelease>', self.on_key_release)
        self.text.bind('<Tab>', self.accept_suggestion)

        # Create save button
        self.save_button = tk.Button(master, text="Save", command=self.save_file)
        self.save_button.pack()

        # Suggestion variables
        self.suggestion = ""
        self.suggestion_start = 0

    def on_key_release(self, event):
        if event.keysym != 'Tab':
            self.generate_suggestion()

    def generate_suggestion(self):
        # Get the current text
        text = self.text.get("1.0", tk.END).strip()
        
        # Tokenize the text
        inputs = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=inputs.shape[1] + 20, num_return_sequences=1, 
                                          do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        
        predicted_text = self.tokenizer.decode(outputs[0])
        
        # Get the suggestion (new text only)
        self.suggestion = predicted_text[len(text):].strip()
        self.suggestion_start = self.text.index(tk.INSERT)
        
        # Show suggestion in gray
        self.text.insert(tk.INSERT, self.suggestion, 'suggestion')
        self.text.tag_config('suggestion', foreground='gray')

    def accept_suggestion(self, event):
        if self.suggestion:
            self.text.delete(self.suggestion_start, tk.END)
            self.text.insert(tk.INSERT, self.suggestion)
            self.suggestion = ""
        return 'break'  # Prevents default Tab behavior

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.text.get("1.0", tk.END))

if __name__ == "__main__":
    root = tk.Tk()
    editor = AutoCompleteEditor(root)
    root.mainloop()