import json
import os
import re
import sys
import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime

# this is for gui basesd chatbot, the main logic is in main.py, this file is for the interface and interaction with the user. It uses the rules defined in rules.json to generate responses based on user input. The GUI is built using tkinter, providing a simple chat interface with a display area for messages and an input field for user queries.
script_dir = os.path.dirname(os.path.abspath(__file__))
rules_path = os.path.join(script_dir, "rules.json")


def normalize_text(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    rule_entries = []
    default_response = "Sorry, I don't understand that. Can you rephrase?"

    if isinstance(loaded, dict):
        default_response = loaded.pop("default", default_response)
        for trigger, response in loaded.items():
            rule_entries.append((trigger, response))
    elif isinstance(loaded, list):
        for item in loaded:
            if item.get("default"):
                default_response = item.get("response", default_response)
                continue
            triggers = item.get("triggers", [])
            if isinstance(triggers, str):
                triggers = [triggers]
            response = item.get("response", "")
            for trigger in triggers:
                rule_entries.append((trigger, response))
    else:
        raise ValueError("Unsupported rules format in rules.json")

    rule_entries.sort(key=lambda pair: len(pair[0]), reverse=True)
    return rule_entries, default_response


try:
    rule_entries, default_response = load_rules(rules_path)
except FileNotFoundError:
    print(f"Error: rules.json not found at {rules_path}")
    sys.exit(1)
except json.JSONDecodeError as exc:
    print(f"Error: failed to parse rules.json: {exc}")
    sys.exit(1)
except ValueError as exc:
    print(f"Error: {exc}")
    sys.exit(1)


def get_response(user_input):
    normalized_input = normalize_text(user_input)
    for trigger, response in rule_entries:
        if re.search(rf"\b{re.escape(trigger)}\b", normalized_input):
            return response
    return default_response


class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Rule-Based Chatbot")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # Title frame
        title_frame = tk.Frame(root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)

        title_label = tk.Label(
            title_frame,
            text="🤖 Chatbot Assistant",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=10)

        # Chat frame
        chat_frame = tk.Frame(root, bg="white")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            height=15,
            width=70,
            font=("Arial", 10),
            bg="white",
            fg="#333",
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for styling
        self.chat_display.tag_config("user", foreground="#0066cc", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("bot", foreground="#00aa00", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("time", foreground="#888888", font=("Arial", 8))

        # Input frame
        input_frame = tk.Frame(root, bg="#f0f0f0")
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Input field
        self.input_field = tk.Entry(
            input_frame,
            font=("Arial", 11),
            bg="white",
            fg="#333"
        )
        self.input_field.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        self.input_field.bind("<Return>", lambda e: self.send_message())

        # Send button
        send_btn = tk.Button(
            input_frame,
            text="Send ➤",
            command=self.send_message,
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10, "bold"),
            width=10,
            cursor="hand2"
        )
        send_btn.pack(side=tk.LEFT, padx=2)

        # Clear button
        clear_btn = tk.Button(
            input_frame,
            text="Clear",
            command=self.clear_chat,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10),
            width=8,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=2)

        # Welcome message
        self.add_bot_message("Hello! I'm your chatbot assistant. How can I help you today?")
        self.input_field.focus()

    def add_user_message(self, message):
        """Add user message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "time")
        self.chat_display.insert(tk.END, f"You: ", "user")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def add_bot_message(self, message):
        """Add bot message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "time")
        self.chat_display.insert(tk.END, f"Bot: ", "bot")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message(self):
        """Handle sending a message"""
        user_input = self.input_field.get().strip()

        if not user_input:
            return

        if user_input.lower() == "exit":
            if messagebox.askyesno("Exit", "Do you want to quit?"):
                self.root.quit()
            return

        self.add_user_message(user_input)

        response = get_response(user_input)
        self.add_bot_message(response)

        self.input_field.delete(0, tk.END)
        self.input_field.focus()

    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.add_bot_message("Chat cleared. How can I help you?")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
