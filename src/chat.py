from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAté mais!")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Até mais!")
            break

        try:
            resposta = chain.invoke(pergunta)
        except Exception as e:
            print(f"Erro ao consultar: {e}\n")
            continue

        print(f"RESPOSTA: {resposta.strip()}\n")


if __name__ == "__main__":
    main()
