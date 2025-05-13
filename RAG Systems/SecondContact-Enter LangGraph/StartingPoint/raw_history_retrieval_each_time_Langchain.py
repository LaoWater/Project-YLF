Cannot
find
reference
'MemoryCheckpoint' in 'init.py'

also, let
's switch to using Groq llms for both memory and response.
Example
with groq api set up to take logic from and update beginning of our script:

# ────────────────────────────── Groq helper ───────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def groq_generate(prompt: str,
                  model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                  system: str | None = None) -> str:
    """Call Groq Chat Completion and return the assistant message text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    completion = groq_client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return completion.choices[0].message.content


(but
 we'll need to choose appropriate groq models for each) - of course we need a fast, yet more powerful one for response.
 import os
import json
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from copy import deepcopy

# Ensure your OpenAI API key is set in the environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="ft:gpt-4o-mini-2024-07-18:personal::AioG35Tv"
)

# Define the prompt template with MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# Add a debug function to print the full context window before sending to LLM
def format_context_window(messages):
    context_str = "\n" + "=" * 50 + "\n"
    context_str += "FULL CONTEXT WINDOW SENT TO LLM:\n"
    context_str += "=" * 50 + "\n\n"

    for msg in messages:
        if isinstance(msg, SystemMessage):
            context_str += f"[SYSTEM]: {msg.content}\n\n"
        elif isinstance(msg, HumanMessage):
            context_str += f"[HUMAN]: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            context_str += f"[ASSISTANT]: {msg.content}\n\n"
        else:
            context_str += f"[{type(msg).__name__}]: {msg.content}\n\n"

    context_str += "=" * 50 + "\n"
    return context_str


# In-memory store for chat histories
chat_histories: Dict[str, BaseChatMessageHistory] = {}


# Function to save history to a file
def save_history_to_file(session_id: str):
    file_name = f"{session_id}_history.json"
    history = chat_histories[session_id]

    # Print messages to debug their structure
    print("\n--- DEBUG: Current Messages in History ---")
    for message in history.messages:
        print(f"Type: {type(message).__name__}, Content: {message.content}")
    print("--- END DEBUG ---\n")

    # Convert messages to JSON-serializable format
    serialized_messages = []
    for message in history.messages:
        if hasattr(message, "content"):
            serialized_messages.append({
                "role": "human" if "HumanMessage" in type(message).__name__ else "assistant",
                "content": message.content
            })

    with open(file_name, "w") as file:
        json.dump(serialized_messages, file, indent=4)


# Function to load history from a file with enhanced debugging
def load_history_from_file(session_id: str) -> BaseChatMessageHistory:
    file_name = f"{session_id}_history.json"
    print(f"\n--- DEBUG: Attempting to load history from {file_name} ---")

    if os.path.exists(file_name):
        # Check if the file is empty
        if os.stat(file_name).st_size == 0:
            print(f"File {file_name} is empty. Starting a new session.")
            return ChatMessageHistory()

        try:
            with open(file_name, "r") as file:
                messages = json.load(file)

            print(f"--- DEBUG: Successfully loaded {len(messages)} messages from file ---")
            print("Message structure from file:")
            for i, msg in enumerate(messages):
                print(
                    f"  Message {i + 1}: role={msg['role']}, content={msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']}")

            history = ChatMessageHistory()
            for message in messages:
                # Use the appropriate method based on role
                if message["role"] == "human":
                    history.add_user_message(message["content"])
                else:  # assistant
                    history.add_ai_message(message["content"])

            print("--- DEBUG: Converted file messages to ChatMessageHistory ---")
            print(f"Number of messages in history object: {len(history.messages)}")
            for i, msg in enumerate(history.messages):
                print(
                    f"  Message {i + 1}: type={type(msg).__name__}, content={msg.content[:50] + '...' if len(msg.content) > 50 else msg.content}")

            return history
        except json.JSONDecodeError as e:
            print(f"Error loading history from {file_name}: {e}")
            print("The file is corrupted. Starting a new session.")
            return ChatMessageHistory()
    else:
        print(f"History file {file_name} not found. Starting a new session.")
        return ChatMessageHistory()


# Enhanced function to retrieve or create session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    print(f"\n--- DEBUG: get_session_history called for session '{session_id}' ---")

    if session_id not in chat_histories:
        print(f"Session '{session_id}' not found in memory. Loading from file...")
        chat_histories[session_id] = load_history_from_file(session_id)
    else:
        print(f"Session '{session_id}' found in memory with {len(chat_histories[session_id].messages)} messages")

    history = chat_histories[session_id]

    # Print the content of the history that will be fed to the LLM
    print("\n--- DEBUG: Messages being fed to the LLM from history ---")
    for i, msg in enumerate(history.messages):
        print(
            f"  Message {i + 1}: type={type(msg).__name__}, content={msg.content[:50] + '...' if len(msg.content) > 50 else msg.content}")

    return history


# Create a custom ChatOpenAI wrapper that logs prompts
class LoggingChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        # Print the input context
        if isinstance(input, list):  # This will be the messages
            print(format_context_window(input))
        return super().invoke(input, config, **kwargs)


# Replace the standard LLM with our logging version
llm = LoggingChatOpenAI(
    openai_api_key=api_key,
    model="ft:gpt-4o-mini-2024-07-18:personal::AioG35Tv"
)

# Create the runnable with message history
conversation = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


# Continuous conversation loop
def start_conversation():
    session_id = input("Enter a session ID (or press Enter for default): ").strip() or "default_session"

    print("Conversation started. Type 'exit' or 'quit' to end the conversation gracefully.")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Ending conversation. Saving your session history...")
                save_history_to_file(session_id)
                print("Session history saved. Goodbye!")
                break

            print(f"\n--- Starting processing for user input: '{user_input}' ---")

            # Invoke the conversation with the user's input
            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # Print the assistant's response
            print(f"Assistant: {response.content}")

            # Save history after each interaction
            save_history_to_file(session_id)

    except KeyboardInterrupt:
        print("\nDetected interruption. Saving your session history...")
        save_history_to_file(session_id)
        print("Session history saved. Goodbye!")


# Start the conversation
if __name__ == "__main__":
    start_conversation()