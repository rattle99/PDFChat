import json

import gradio as gr
import requests
import weaviate
from dotenv import dotenv_values
from weaviate.classes.query import MetadataQuery

from utils import get_embedding

CONFIG = dotenv_values(".env")
client = weaviate.connect_to_local()
collection = client.collections.get(name=CONFIG["COLLECTION_NAME"])
SYSTEM_PROMPT_BASE = "I am a helpful assistant.\n"


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


def getRelevantChunks(query, strategy=CONFIG["RETRIEVAL"]):
    if strategy == "keyword":
        return keyword_search(query)
    if strategy == "vector":
        return vector_search(query)
    if strategy == "hybrid":
        return hybrid_search(query)

    raise ValueError(
        f'Invalid argument: {strategy}. Expected strategy one of "keyword", "vector" or "hybrid".'
    )


def createRetrievedContext(query):
    relevantChunks = getRelevantChunks(query)

    retrievedContext = (
        SYSTEM_PROMPT_BASE
        + "Use the context provided below to answer any questions asked after this:\n\n"
    )
    for idx, chunk in enumerate(relevantChunks):
        item = f"{idx+1}. {chunk} \n"
        retrievedContext += item

    retrievedContext += "\nYou have all the context you need provided above."

    return retrievedContext


def prompt_llm(messages):
    params = {
        "model": CONFIG["CHAT_MODEL"],
        "messages": messages,
        "options": {
            "num_predict": 512,
        },
        "stream": True,
    }

    # Making POST request to LLM
    response = requests.post(CONFIG["LLM_URL"], json=params, stream=True)

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


def chat_fn(userInput, messages, sessionParams):
    prompt = userInput
    if sessionParams[0]["initialQuery"]:
        retrievedContext = createRetrievedContext(userInput)
        messages[0]["content"] = retrievedContext
        sessionParams[0]["initialQuery"] = False
        sessionParams[0]["queryString"] = prompt
    sessionParams[0]["submissionCount"] += 1

    # Add the user's message to the history
    messages.append({"role": "user", "content": prompt})
    assistant_message = ""

    # Stream the assistant's response
    for result in prompt_llm(messages):
        if result:
            if isinstance(result, list):
                # Final results list, not doing anything with it as we already processed all results
                continue
            else:
                # Individual streamed result
                if not result["done"]:
                    predicted_token = result["message"]["content"]
                    assistant_message += predicted_token

                    # Update the assistant's response in the last message
                    if len(messages) > 1 and messages[-1]["role"] == "assistant":
                        messages[-1]["content"] = assistant_message
                    else:
                        # If there's no assistant message yet, append one
                        messages.append(
                            {"role": "assistant", "content": assistant_message}
                        )

                    yield (
                        messages,
                        messages,
                        sessionParams,
                    )  # Yield the updated conversation history
        else:
            # Handle any error in communication
            if len(messages) > 1 and messages[-1]["role"] == "assistant":
                messages[-1]["content"] = "Error communicating with the language model."
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Error communicating with the language model.",
                    }
                )
            yield messages, messages, sessionParams

    # Final yield after streaming is complete
    if len(messages) > 1 and messages[-1]["role"] == "assistant":
        messages[-1]["content"] = assistant_message
    else:
        messages.append({"role": "assistant", "content": assistant_message})
    yield messages, messages, sessionParams


# Define the function to be called on submit
def retrievedResults(query, sessionParams):
    query = sessionParams[0]["queryString"]
    submissionCount = sessionParams[0]["submissionCount"]
    print(sessionParams)

    if submissionCount == 1:
        relevantChunks = getRelevantChunks(query)[:5]

        markdown_content = "### Please rate the following options:\n\n"
        options_list = ["A", "B", "C", "D", "E"]
        for idx, chunk in enumerate(relevantChunks):
            item = f"#### Option {options_list[idx]}\n\n{chunk}\n\n"
            markdown_content += item

        return (
            gr.update(
                value=markdown_content, visible=True
            ),  # Update Markdown component
            gr.update(
                visible=True
            ),  # Show container with all radio buttons and submit button
            gr.update(visible=False),  # Hide thank you textbox initially
        )
    else:
        return (
            gr.update(),  # No update needed for Markdown component
            gr.update(
                visible=False
            ),  # Hide container with all radio buttons and submit button
            gr.update(),  # No update needed for thank you textbox
        )


# Function to handle the radio button selection
def handle_option_selection(option_a, option_b, option_c, option_d, option_e):
    # Display a thank you message summarizing the selections
    thank_you_message = (
        f"Thank you! Here are your selections:\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n"
        f"Option C: {option_c}\n"
        f"Option D: {option_d}\n"
        f"Option E: {option_e}"
    )
    return (
        gr.update(
            visible=False
        ),  # Hide container with all radio buttons and submit button
        gr.update(value=thank_you_message, visible=True),  # Show thank you message
    )


# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Markdown component to display the retrieved options
            markdown = gr.Markdown(visible=False)

            # Container for radio buttons and submit button
            with gr.Column(visible=False) as options_container:
                # Define each radio button
                option_a = gr.Radio(
                    choices=["Highly Relevant", "Relevant", "Irrelevant"],
                    label="Option A",
                )
                option_b = gr.Radio(
                    choices=["Highly Relevant", "Relevant", "Irrelevant"],
                    label="Option B",
                )
                option_c = gr.Radio(
                    choices=["Highly Relevant", "Relevant", "Irrelevant"],
                    label="Option C",
                )
                option_d = gr.Radio(
                    choices=["Highly Relevant", "Relevant", "Irrelevant"],
                    label="Option D",
                )
                option_e = gr.Radio(
                    choices=["Highly Relevant", "Relevant", "Irrelevant"],
                    label="Option E",
                )

                # Define submit button
                submit_button = gr.Button("Submit")

            # Textbox for displaying the thank you message
            thank_you_textbox = gr.Textbox(interactive=False, visible=False)

        with gr.Column(scale=2):
            messageHistory = gr.State(
                [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
            )
            sessionParams = gr.State(
                [{"initialQuery": True, "queryString": "", "submissionCount": 0}]
            )

            # Chatbot interface
            chatbot = gr.Chatbot(type="messages")
            textbox = gr.Textbox()

        # Handle the first submit to update the markdown, show the container with radio buttons and submit button
        textbox.submit(
            fn=chat_fn,
            inputs=[textbox, messageHistory, sessionParams],
            outputs=[chatbot, messageHistory, sessionParams],
        ).then(
            fn=retrievedResults,
            inputs=[textbox, sessionParams],
            outputs=[markdown, options_container, thank_you_textbox, sessionParams],
        )

        # Handle the final submit to process the selections and show the thank you message
        submit_button.click(
            fn=handle_option_selection,
            inputs=[option_a, option_b, option_c, option_d, option_e],
            outputs=[options_container, thank_you_textbox],
        )

# Launch the interface
demo.launch()
