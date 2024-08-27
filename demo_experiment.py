import json
import logging

import gradio as gr
import requests
import weaviate
from dotenv import dotenv_values
from weaviate.classes.query import MetadataQuery

from utils import get_embedding, setup_logger

CONFIG = dotenv_values(".env")
client = weaviate.connect_to_local()
collection = client.collections.get(name=CONFIG["COLLECTION_NAME"])
SYSTEM_PROMPT_BASE = "I am a helpful assistant.\n"

# Setup the logger for the current module
logger = setup_logger(__name__)


def keyword_search(query, K=10):
    """
    Perform a keyword-based search using the BM25 algorithm.

    Args:
        query (str): The search query string.
        K (int, optional): The number of results to return. Defaults to 10.

    Returns:
        list: A list of content strings that match the search query.
    """
    response = collection.query.bm25(
        query=query,
        query_properties=["content"],
        limit=K,
        return_metadata=MetadataQuery(score=True),
    )

    return [obj.properties["content"] for obj in response.objects]


def vector_search(query, K=10):
    """
    Perform a vector-based search using an embedding of the query.

    Args:
        query (str): The search query string.
        K (int, optional): The number of results to return. Defaults to 10.

    Returns:
        list: A list of content strings that are most similar to the query vector.
    """
    embedding = get_embedding(query)
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=K,
        return_metadata=MetadataQuery(distance=True),
    )

    return [obj.properties["content"] for obj in response.objects]


def hybrid_search(query, K=10):
    """
    Perform a hybrid search combining keyword-based and vector-based approaches.

    Args:
        query (str): The search query string.
        K (int, optional): The number of results to return. Defaults to 10.

    Returns:
        list: A list of content strings that match both the query text and query vector.
    """
    embedding = get_embedding(query)
    response = collection.query.hybrid(
        query=query,
        query_properties=["content"],
        vector=embedding,
        limit=K,
        return_metadata=MetadataQuery(score=True),
    )

    return [obj.properties["content"] for obj in response.objects]


def getRelevantChunks(query, strategy=CONFIG["RETRIEVAL"]):
    """
    Retrieve relevant chunks of content based on the specified retrieval strategy.

    Args:
        query (str): The search query string.
        strategy (str, optional): The retrieval strategy to use ("keyword", "vector", or "hybrid").
                                  Defaults to the strategy specified in CONFIG["RETRIEVAL"].

    Returns:
        list: A list of relevant content strings based on the specified strategy.

    Raises:
        ValueError: If an invalid strategy is provided.
    """

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
    """
    Create a context string from relevant content chunks for use in an LLM prompt.

    Args:
        query (str): The search query string.

    Returns:
        str: A context string built from relevant content chunks.
    """
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
    """
    Send a prompt to the language model and stream the response as a generator.

    This function makes a POST request to the language model and yields the response
    incrementally, allowing for real-time processing of the results.

    Args:
        messages (list): A list of message dictionaries to send to the language model.

    Yields:
        dict or None: The response from the language model, streamed in real-time as a dictionary.
                      If there is an error, None is yielded.
    """
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
    """
    Manage a chat session with an LLM, including context retrieval and message streaming, as a generator.

    This function handles the conversation flow with a language model by updating the
    chat history and streaming the assistant's response incrementally.

    Args:
        userInput (str): The user's input message.
        messages (list): The chat history containing message dictionaries.
        sessionParams (list): A list of session parameters, including flags for query handling.

    Yields:
        tuple: A tuple containing updated messages, the full chat history, and session parameters
               after processing each step.
    """
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
    # yield messages, messages, sessionParams

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
    # query = sessionParams[0]["queryString"]
    submissionCount = sessionParams[0]["submissionCount"]

    if submissionCount == 0:
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
def handle_option_selection(
    option_a, option_b, option_c, option_d, option_e, sessionParams
):
    if logger.isEnabledFor(logging.DEBUG):
        # parse query, responses
        # create dict to save for dataset

        relevantChunks = getRelevantChunks(sessionParams[0]["queryString"])[:5]
        chunkLabels = [
            {
                "chunk": relevantChunks[0],
                "label": option_a,
            },
            {
                "chunk": relevantChunks[1],
                "label": option_b,
            },
            {
                "chunk": relevantChunks[2],
                "label": option_c,
            },
            {
                "chunk": relevantChunks[3],
                "label": option_d,
            },
            {
                "chunk": relevantChunks[4],
                "label": option_e,
            },
        ]
        userFeedback = {
            "query": sessionParams[0]["queryString"],
            "userLabels": chunkLabels,
        }
        logger.debug(f"options submitted {json.dumps(userFeedback)}")

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
        fn=retrievedResults,
        inputs=[textbox, sessionParams],
        outputs=[markdown, options_container, thank_you_textbox],
    )

    # Might be possible to have two events for textbox instead of chaining
    textbox.submit(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=textbox,
    ).then(
        fn=chat_fn,
        inputs=[textbox, messageHistory, sessionParams],
        outputs=[chatbot, messageHistory, sessionParams],
    ).then(
        fn=lambda: gr.update(value="", interactive=True),
        inputs=None,
        outputs=textbox,
    )

    # Handle the final submit to process the selections and show the thank you message
    submit_button.click(
        fn=handle_option_selection,
        inputs=[option_a, option_b, option_c, option_d, option_e, sessionParams],
        outputs=[options_container, thank_you_textbox],
    )

    demo.unload(client.close)

# Launch the interface
demo.launch()
