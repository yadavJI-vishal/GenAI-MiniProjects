from transformers import pipeline

def text_generation(prompt: str):
    """
    Generates text based on a given prompt using a transformer model.
    """
    generator = pipeline(
        "text-generation",
        model="gpt2"
    )

    output = generator(
        prompt,
        max_length=100,
        num_return_sequences=1
    )

    return output[0]["generated_text"]


def sentiment_analysis(text: str):
    """
    Performs sentiment classification on the given text.
    """
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    result = classifier(text)
    return result[0]


def main():
    print("=== AI Text Generator & Sentiment Analyzer ===\n")

    prompt = input("Enter a prompt for text generation:\n> ")

    print("\nGenerating text...\n")
    generated_text = text_generation(prompt)

    print("Generated Text:\n")
    print(generated_text)

    print("\nAnalyzing sentiment of generated text...\n")
    sentiment = sentiment_analysis(generated_text)

    print("Sentiment Result:")
    print(f"Label: {sentiment['label']}")
    print(f"Confidence: {sentiment['score']:.4f}")


if __name__ == "__main__":
    main()
