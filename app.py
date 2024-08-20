import json

import requests
import weaviate
from dotenv import dotenv_values
from weaviate.classes.query import MetadataQuery

from utils import get_embedding

config = dotenv_values(".env")
client = weaviate.connect_to_local()
collection = client.collections.get(name=config["COLLECTION_NAME"])
messages = [
    {
        "role": "system",
        "content": "I am a helpful assistant.",
    }
]
initial_query = True


def keyword_search(query, K=10):
    response = collection.query.bm25(
        query=query,
        limit=K,
        return_metadata=MetadataQuery(score=True),
    )

    return [obj.properties["content"] for obj in response.objects]


def vector_search(query, K=10):
    embedding = get_embedding(query)
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=K,
        return_metadata=MetadataQuery(distance=True),
    )

    return [obj.properties["content"] for obj in response.objects]


def hybrid_search(query, K=10):
    embedding = get_embedding(query)
    response = collection.query.hybrid(
        query=query,
        vector=embedding,
        limit=K,
        return_metadata=MetadataQuery(score=True),
    )

    return [obj.properties["content"] for obj in response.objects]


def getRelevantChunks(query, strategy=config["RETRIEVAL"]):
    if strategy == "keyword":
        return keyword_search(query)
    if strategy == "vector":
        return vector_search(query)
    if strategy == "hybrid":
        return hybrid_search(query)

    raise ValueError(
        f'Invalid argument: {strategy}. Expected strategy one of "keyword", "vector" or "hybrid".'
    )


def createQueryPrompt(query):
    relevantChunks = getRelevantChunks(query)

    queryPrompt = (
        f"Use the context provided below to answer the following question: {query}\n\n"
    )

    for idx, chunk in enumerate(relevantChunks):
        item = f"{idx+1}. {chunk} \n"
        queryPrompt += item

    queryPrompt += "\nYou have all the context you need provided above."

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
    if userInput.lower() == "new":
        initial_query = True
        messages = messages[:1]
        continue
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

client.close()
