from time import sleep, time

from transformers import pipeline


# Load pre-trained model and tokenizer
qa_pipeline = pipeline("question-answering", model='deepset/xlm-roberta-large-squad2', framework='pt')

with open("context.txt", "r", encoding="utf-8") as file:
    context = file.read()


def get_response(question):
    start = time()
    result = qa_pipeline(question=question, context=context)
    end = time()
    return f"{result['answer']} --- [score: {round(result['score'], 4)} | time: {round(end - start, 4)}]"


if __name__ == "__main__":
    sleep(2)
    print("Bot: Cześć, jak Ci mogę pomóc?")
    while True:
        try:
            user_input = input("Ty: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Na razie!")
                break
            response = get_response(user_input)
            print(f"Bot: {response}")
        except (KeyboardInterrupt, EOFError, SystemExit):
            break
