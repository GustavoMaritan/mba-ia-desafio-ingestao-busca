# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio de ingestão e busca do MBA em Engenharia de Software com IA (Full Cycle).

O projeto lê um PDF, quebra o conteúdo em chunks, gera embeddings e salva tudo no PostgreSQL com pgVector. Depois, um chat no terminal responde perguntas usando **apenas** o conteúdo do PDF. Se a resposta não estiver lá, ele avisa que não tem a informação.

```
PERGUNTA: Qual o faturamento da empresa Alfa Energia Holding?
RESPOSTA: O faturamento da empresa Alfa Energia Holding é R$ 858.537,02.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Stack

- Python 3.12 + LangChain
- PostgreSQL 17 + pgVector (Docker Compose)
- Gemini (padrão) ou OpenAI para embeddings e LLM

## Como funciona

**Ingestão (`src/ingest.py`)**
1. Carrega o `document.pdf` com `PyPDFLoader`.
2. Divide em chunks de 1000 caracteres com overlap de 150 (`RecursiveCharacterTextSplitter`).
3. Gera os embeddings e grava no pgVector via `PGVector`.

A collection é recriada a cada execução, então dá pra rodar a ingestão de novo sem duplicar dados. O envio é feito em lotes de 20 chunks, com espera automática quando o limite de requisições do plano gratuito do Gemini é atingido.

**Busca (`src/search.py`) e chat (`src/chat.py`)**
1. A pergunta é vetorizada.
2. `similarity_search_with_score(pergunta, k=10)` traz os 10 trechos mais relevantes.
3. Os trechos entram no prompt definido no desafio junto com a pergunta.
4. A LLM responde e o chat mostra a resposta.

A escolha do provedor fica em `src/providers.py`, controlada pela variável `PROVIDER`.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Uma API key do [Google AI Studio](https://aistudio.google.com/app/apikey) (Gemini) ou da [OpenAI](https://platform.openai.com/api-keys)

## Configuração

Copie o arquivo de exemplo e preencha a sua chave:

```bash
cp .env.example .env
```

| Variável | Descrição |
|---|---|
| `PROVIDER` | `gemini` (padrão) ou `openai` |
| `GOOGLE_API_KEY` | chave do Gemini |
| `GOOGLE_EMBEDDING_MODEL` | modelo de embeddings (padrão `models/gemini-embedding-001`) |
| `GOOGLE_LLM_MODEL` | modelo de chat (padrão `gemini-3.5-flash-lite`) |
| `OPENAI_API_KEY` | chave da OpenAI (se `PROVIDER=openai`) |
| `OPENAI_EMBEDDING_MODEL` / `OPENAI_LLM_MODEL` | modelos da OpenAI |
| `DATABASE_URL` | conexão com o Postgres |
| `PG_VECTOR_COLLECTION_NAME` | nome da collection no pgVector |
| `PDF_PATH` | caminho do PDF (padrão `document.pdf`) |
| `PG_PORT` | opcional, porta do Postgres na sua máquina (padrão `5432`) |

> Se a porta 5432 já estiver em uso, defina `PG_PORT` no `.env` (ex.: `PG_PORT=5436`) e ajuste a porta no `DATABASE_URL`.

> Trocou o modelo de embeddings? Os vetores mudam de dimensão, então apague a collection (ou rode `docker compose down -v`) e refaça a ingestão.

## Execução

Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

1. Suba o banco:

```bash
docker compose up -d
```

2. Rode a ingestão do PDF:

```bash
python src/ingest.py
```

3. Inicie o chat:

```bash
python src/chat.py
```

Digite `sair` (ou `Ctrl+C`) para encerrar.

## Estrutura

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py      # ingestão do PDF
│   ├── search.py      # busca vetorial + prompt + LLM
│   ├── chat.py        # CLI
│   └── providers.py   # embeddings/LLM do Gemini ou OpenAI
├── document.pdf
└── README.md
```
