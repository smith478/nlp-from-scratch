import tiktoken
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import torch
from gpt2 import GPTModel
import os
from datetime import datetime
import uuid

MODEL_CONFIGS = {
    "124M": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "model_path": "../../models/model_124M.pth"
    },
    "355M": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "model_path": "../../models/model_355M.pth"
    },
    "774M": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "model_path": "../../models/model_774M.pth"
    }
}

MODEL = "774M"
CONFIG = MODEL_CONFIGS[MODEL]
model_path = CONFIG.pop("model_path")


# Global variable for wait time
INFERENCE_WAIT_TIME = 2000  # milliseconds
ORIGINAL_INFERENCE_WAIT_TIME = INFERENCE_WAIT_TIME
ESC_INFERENCE_WAIT_TIME = 8000  # milliseconds
DICTATION_PAUSE_TIME = 2000  # milliseconds

def generate(model, tokenizer, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    idx = idx.to(torch.long)  # Ensure idx is of type long
    generated_tokens = 0
    period_id = tokenizer.encode('.')[0]  # Get the token ID for the period character

    for _ in range(max_new_tokens):
        if generated_tokens >= 40:
            break

        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            logits = torch.where(logits < torch.topk(logits, top_k)[0][:, -1, None], torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        generated_tokens += 1

        if idx_next.item() == period_id:
            break

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

class AutoCompleteEditor:
    def __init__(self, master):
        self.master = master
        master.title("GPT-2 Auto-Complete Editor")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPTModel(CONFIG)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.text = tk.Text(master, wrap='word')
        self.text.pack(expand=True, fill='both')

        self.text.bind('<<Modified>>', self.on_text_modified)
        self.text.bind('<Tab>', self.accept_suggestion)
        self.text.bind('<Escape>', self.cancel_suggestion)
        self.text.bind('<KeyRelease>', self.on_cursor_move)
        self.text.bind('<Command-z>', self.undo_last_addition)
        self.text.bind('<Key>', self.handle_key_press)

        self.save_button = tk.Button(master, text="Save", command=self.save_file)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.current_id = self.generate_id()

        self.suggestion = ""
        self.suggestion_start = 0
        self.timer = None
        self.waiting_for_action = False
        self.inference_allowed = True
        self.last_content = ""
        self.last_action_was_escape = False
        self.action_since_escape = False
        self.last_added_text = ""

    def handle_key_press(self, event):
        if event.keysym != 'Tab' and self.suggestion:
            self.cancel_suggestion()

    def is_cursor_at_end(self):
        cursor_position = self.text.index(tk.INSERT)
        end_position = self.text.index(tk.END)
        return cursor_position == self.text.index(f"{end_position} - 1c")

    def on_text_modified(self, event):
        if self.text.edit_modified():
            self.action_since_escape = True
            global INFERENCE_WAIT_TIME
            INFERENCE_WAIT_TIME = ORIGINAL_INFERENCE_WAIT_TIME
            current_content = self.text.get("1.0", tk.END)
            if current_content != self.last_content:
                self.last_content = current_content
                if self.is_cursor_at_end():
                    self.schedule_inference(DICTATION_PAUSE_TIME)
                else:
                    self.cancel_suggestion(None)
            self.text.edit_modified(False)

    def on_cursor_move(self, event):
        self.action_since_escape = True
        global INFERENCE_WAIT_TIME
        INFERENCE_WAIT_TIME = ORIGINAL_INFERENCE_WAIT_TIME
        if self.is_cursor_at_end() and not self.waiting_for_action:
            if self.action_since_escape or not self.last_action_was_escape:
                self.schedule_inference()
                self.last_action_was_escape = False

    def schedule_inference(self, delay=INFERENCE_WAIT_TIME):
        self.inference_allowed = True
        if self.timer:
            self.master.after_cancel(self.timer)
        self.timer = self.master.after(delay, self.generate_suggestion)

    def generate_suggestion(self):
        if self.waiting_for_action or not self.is_cursor_at_end() or self.last_action_was_escape:
            return

        text = self.text.get("1.0", tk.END).strip()
        inputs = text_to_token_ids(text, self.tokenizer).to(self.device)

        outputs = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            idx=inputs,
            max_new_tokens=40,
            context_size=CONFIG["context_length"],
            top_k=40,
            temperature=0.7
        )

        predicted_text = token_ids_to_text(outputs, self.tokenizer)
        
        self.suggestion = predicted_text[len(text):].strip()
        
        if self.suggestion:
            cursor_position = self.text.index(tk.INSERT)
            char_before_cursor = self.text.get(f"{cursor_position} - 1c")
            if not self.suggestion.startswith(' ') and char_before_cursor != ' ':
                self.suggestion = ' ' + self.suggestion

            self.suggestion_start = self.text.index(tk.INSERT)
            
            self.text.insert(tk.INSERT, self.suggestion, 'suggestion')
            self.text.tag_config('suggestion', foreground='gray')

            self.waiting_for_action = True

    def accept_suggestion(self, event):
        if self.suggestion:
            self.text.delete(self.suggestion_start, tk.END)
            self.text.insert(tk.INSERT, self.suggestion)
            self.last_added_text = self.suggestion
            self.suggestion = ""
            self.waiting_for_action = False
            self.last_content = self.text.get("1.0", tk.END)
            self.last_action_was_escape = False
            self.action_since_escape = False
            global INFERENCE_WAIT_TIME
            INFERENCE_WAIT_TIME = ORIGINAL_INFERENCE_WAIT_TIME
            self.schedule_inference()
        return 'break'

    def cancel_suggestion(self):
        if self.suggestion:
            self.text.delete(self.suggestion_start, tk.END)
            self.suggestion = ""
        self.waiting_for_action = False
        self.last_action_was_escape = True
        self.action_since_escape = False
        global INFERENCE_WAIT_TIME
        INFERENCE_WAIT_TIME = ESC_INFERENCE_WAIT_TIME

    def undo_last_addition(self, event):
        if self.last_added_text:
            current_text = self.text.get("1.0", tk.END)
            if current_text.endswith(self.last_added_text):
                self.text.delete(f"end-{len(self.last_added_text)}c", tk.END)
                self.last_added_text = ""
                self.last_content = self.text.get("1.0", tk.END)
        return 'break'

    def generate_id(self):
        return str(uuid.uuid4())[:8]

    def save_file(self):
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"{current_date}_{self.current_id}.txt"
        file_path = os.path.join("../../data/radiology_reports", filename)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            file.write(self.text.get("1.0", tk.END))
        
        messagebox.showinfo("Save", f"File saved as {filename}")

    def clear_text(self):
        self.text.delete("1.0", tk.END)
        self.current_id = self.generate_id()
        self.last_content = ""
        self.suggestion = ""
        self.waiting_for_action = False
        self.last_action_was_escape = False
        self.action_since_escape = True
        self.schedule_inference()

if __name__ == "__main__":
    root = tk.Tk()
    editor = AutoCompleteEditor(root)
    root.mainloop()