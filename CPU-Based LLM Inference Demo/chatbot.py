import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model on CPU...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu"
    )

    return tokenizer, model


def generate_response(tokenizer, model, user_input):
    prompt = (
        "Define the following technical concept in simple terms.\n"
        f"Concept: {user_input}\n"
        "Definition:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=40,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)




def chatbot():
    tokenizer, model = load_model()

    print("\n🤖 LLM Chatbot (type 'exit' to quit)")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! 👋")
            break

        bot_response = generate_response(tokenizer, model, user_input)
        print("Bot:", bot_response)


if __name__ == "__main__":
    chatbot()
