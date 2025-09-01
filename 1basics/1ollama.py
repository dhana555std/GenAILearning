from ollama import Client


def main():
    client = Client()
    prompt = "Write a short poem with the Theme `God is great`."
    model = "tinyllama"

    res = client.generate(model=model, prompt=prompt)
    print(f"Poem on `God is great`:\n\n{res.response}")


if __name__ == "__main__":
    main()
