from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from warnings import warn


def get_llm():
    llm = None
    try:
        load_dotenv()
        model = os.getenv("LLM_MODEL")
        model_provider = os.getenv("LLM_MODEL_PROVIDER")
        print(f"LLM_MODEL={model} and LLM_MODEL_PROVIDER={model_provider}")

        if model_provider == "google_genai":
            print("Considering google genai as the LLM.")
            llm = ChatGoogleGenerativeAI(model=model)

        if model_provider == "openai":
            print("Considering Open AI as the LLM.")
            llm = ChatOpenAI(model_name=model)
    except Exception as e:
        print(f"Something went wrong {e}.")
        warn("You are by default falling back to Local LLM.", RuntimeWarning)
        llm = OllamaLLM(model="llama3.2")
    finally:
        return llm


def get_ollama():
    return OllamaLLM(model="llama3.2")


def get_google_genai():
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_GENAI_LLM_MODEL"))
