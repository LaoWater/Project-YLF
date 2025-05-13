import os
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


# Function to retrieve or create session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]


# Create the runnable with message history
conversation = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Example conversation
session_id = "session_1"
response = conversation.invoke(
    {"input": "What is LangChain?"},
    config={"configurable": {"session_id": session_id}}
)
print(response.content)
