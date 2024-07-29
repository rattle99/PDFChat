import requests
import chromadb
from dotenv import dotenv_values
from utils import get_embedding

config = dotenv_values(".env")
client = chromadb.PersistentClient(path=config["PERSIST_DIRECTORY"])
collection = client.get_or_create_collection(name=config["COLLECTION_NAME"])

messages = [{"role":"system", "content":"I am going to ask you a question, which I would like you to answer based only on the provided context, and not any other information. If there is not enough information in the context to answer the question, say 'I am not sure', then try to make a guess. Make sure to keep all your responses under 3 lines. This is very important."}]
# messages = [{"role":"system", "content":"You are a good chatbot."}]

def searchVectorStore(query, K=5):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=embedding,
        n_results=K # how many results to return
    )

    return results

def createQueryPrompt(query):
    results = searchVectorStore(query)
    relevantChunks = [chunk for chunk in results['documents'][0]]

    queryPrompt = 'Here is all the context you have:\n'
    for idx, chunk in enumerate(relevantChunks):
        item = f'{idx+1}. {chunk} \n'
        queryPrompt += item

    queryPrompt += f'Based on the above information, answer the following question: {query}' 

    return queryPrompt


def prompt_llm(messages):
    params = {
        "model": config["CHAT_MODEL"],
        "messages": messages,
        "options": {
            "num_predict": 512
        },
        "stream": False
    }

    
    # Making POST request to LLM
    response = requests.post(config["LLM_URL"], json=params)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Parse JSON response
        results = response.json()
        # Do something with the results
        return results
    else:
        # Request was not successful
        print("Error:", response.status_code)
        return None


query = input("User : ")
prompt = createQueryPrompt(query)

message = {"role":"user", "content":query}
messages.append(message)
result = prompt_llm(messages=messages)
print("\nAssistant :", result["message"]["content"])
message = {"role":"assistant", "content":result["message"]["content"]}
messages.append(message)

initial_query = True

while True:
    content = input("\nUser : ")
    if content.lower() == 'exit':
        break

    
    message = {"role":"user", "content":content}
    messages.append(message)
    result = prompt_llm(messages)
    if result:
        assistant_message = result["message"]["content"]
        messages.append({"role":"assistant", "content":assistant_message})
        print("\nAssistant :", assistant_message)
    else:
        print("Error communicating with the language model.")
        break

    initial_query = False
