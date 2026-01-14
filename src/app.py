from rag_chain import get_rag_chain

def main():
    chain = get_rag_chain()
    print("Company FAQ Chatbot (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        response = chain.invoke(query)
        print("Bot:", response)
        print()

if __name__ == "__main__":
    main()
