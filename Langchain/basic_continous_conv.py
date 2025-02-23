import os
import json
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

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



# Function to load history from a file
def load_history_from_file(session_id: str) -> BaseChatMessageHistory:
    file_name = f"{session_id}_history.json"
    if os.path.exists(file_name):
        # Check if the file is empty
        if os.stat(file_name).st_size == 0:
            print(f"File {file_name} is empty. Starting a new session.")
            return ChatMessageHistory()

        try:
            with open(file_name, "r") as file:
                messages = json.load(file)
            history = ChatMessageHistory()
            for message in messages:
                history.add_message(message["role"], message["content"])
            return history
        except json.JSONDecodeError as e:
            print(f"Error loading history from {file_name}: {e}")
            print("The file is corrupted. Starting a new session.")
            return ChatMessageHistory()
    else:
        return ChatMessageHistory()


# Function to retrieve or create session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = load_history_from_file(session_id)
    return chat_histories[session_id]


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
