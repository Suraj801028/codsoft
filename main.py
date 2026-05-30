import json
import os
import sys

# Load rules from the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
rules_path = os.path.join(script_dir, "rules.json")
try:
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
except FileNotFoundError:
    print(f"Error: rules.json not found at {rules_path}")
    sys.exit(1)
except json.JSONDecodeError as exc:
    print(f"Error: failed to parse rules.json: {exc}")
    sys.exit(1)

def chatbot_response(user_input):
    user_input = user_input.lower()
    for key in rules:
        if key in user_input:
            return rules[key]
    return rules["default"]

def main():
    print("🤖 Rule-Based Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
