import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()


def _require(var):
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Variável {var} não definida no .env")
    return value


def get_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=_require("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        google_api_key=_require("GOOGLE_API_KEY"),
        transport="rest",
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
    )


def get_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=_require("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        google_api_key=_require("GOOGLE_API_KEY"),
        transport="rest",
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite"),
        temperature=0,
    )
