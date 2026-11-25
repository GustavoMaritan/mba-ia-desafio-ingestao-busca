import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import get_embeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "20"))


def _add_with_retry(store, docs, ids, tentativas=5):
    for tentativa in range(1, tentativas + 1):
        try:
            store.add_documents(docs, ids=ids)
            return
        except Exception as e:
            limite = "429" in str(e) or "quota" in str(e).lower() or "ResourceExhausted" in type(e).__name__
            if not limite or tentativa == tentativas:
                raise
            print(f"  limite de requisições atingido, aguardando 60s (tentativa {tentativa}/{tentativas})...")
            time.sleep(60)


def ingest_pdf():
    for var, value in [("PDF_PATH", PDF_PATH), ("DATABASE_URL", DATABASE_URL), ("PG_VECTOR_COLLECTION_NAME", COLLECTION_NAME)]:
        if not value:
            raise RuntimeError(f"Variável {var} não definida no .env")

    docs = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # tira metadados vazios pra não poluir o banco
    for chunk in chunks:
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v not in ("", None)}

    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        # recria a collection a cada ingestão, assim rodar de novo não duplica os chunks
        pre_delete_collection=True,
    )

    # manda em lotes pequenos: o free tier do Gemini tem limite de tokens por minuto
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [f"doc-{i}" for i in range(start, start + len(batch))]
        _add_with_retry(store, batch, ids)
        print(f"  {start + len(batch)}/{len(chunks)} chunks enviados")

    print(f"{len(docs)} páginas lidas, {len(chunks)} chunks salvos na collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()
