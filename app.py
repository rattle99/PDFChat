import requests
import chromadb
import json
from dotenv import dotenv_values
from utils import get_embedding

config = dotenv_values(".env")
client = chromadb.PersistentClient(path=config["PERSIST_DIRECTORY"])
collection = client.get_or_create_collection(name=config["COLLECTION_NAME"])
messages = [
    {
        "role": "system",
        "content": "I am a helpful assistant.",
    }
]
initial_query = True


def searchVectorStore(query, K=10):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=embedding,
        n_results=K,  # how many results to return
    )

    return results


def createQueryPrompt(query):
    results = searchVectorStore(query)
    relevantChunks = [chunk for chunk in results["documents"][0]]

    queryPrompt = (
        f"Use the context provided below to answer the following question: {query}\n\n"
    )
    for idx, chunk in enumerate(relevantChunks):
        item = f"{idx+1}. {chunk} \n"
        queryPrompt += item

    queryPrompt += f"\nYou have all the context you need provided above."

    return queryPrompt


def prompt_llm(messages):
    params = {
        "model": config["CHAT_MODEL"],
        "messages": messages,
        "options": {
            "num_predict": 512,
        },
        "stream": True,
    }

    # Making POST request to LLM
    response = requests.post(config["LLM_URL"], json=params, stream=True)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        results = []
        # Iterate over the streamed response
        for line in response.iter_lines():
            if line:
                # Parse JSON response
                # print(line)
                result = json.loads(line)
                results.append(result)
                # Yield each result as it is received
                yield result
        # After streaming, yield the complete results list
        yield results
    else:
        # Request was not successful
        print("Error:", response.status_code)
        yield None


while True:
    userInput = input("\n\nUser : ")
    if userInput.lower() == "exit":
        break
    prompt = userInput
    if initial_query:
        prompt = createQueryPrompt(userInput)

    message = {"role": "user", "content": prompt}
    messages.append(message)

    results = prompt_llm(messages)

    assistant_message = ""
    print("\nAssistant: ", end="")
    for result in results:
        if result:
            if isinstance(result, list):
                # Final results list, not doing anything with it as we already processed all results
                # print(type(result))
                # print(result)
                messages.append({"role": "assistant", "content": assistant_message})
            else:
                # Individual streamed result
                if not result["done"]:
                    predicted_token = result["message"]["content"]
                    assistant_message += predicted_token
                    print(predicted_token, end="", flush=True)
        else:
            print("Error communicating with the language model.")
            break

    initial_query = False
