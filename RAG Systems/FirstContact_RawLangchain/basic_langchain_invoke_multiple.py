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
    ("system", "You are a helpful assistant specialized in healing & philosophy. Always a humble student of Life."
               "You have traveler the world and time dimensions through portal of books, experiences and wisdom - and have returned Home to aid other people - "
               "Aid them in finding their balance, peace and personal Legend. For a better, more connected - World. "
               "Use the context of the conversation to improve your responses."),

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

# Simulate a conversation with multiple prompts
session_id = "session_1"

# User's first question
response = conversation.invoke(
    {"input": "Hi, I need help with a Diet. There is a deep programming of the body and as evening comes - it awaits enourmous amounts of food - due to faulty 7 years programming."},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response.content)

# User's follow-up question
response = conversation.invoke(
    {"input": "Why do these patterns occur ? Its fascinating how the mind finds complicated reliefs of pain through others"},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response.content)

# User asks about local cuisine
response = conversation.invoke(
    {"input": "What would be a valuable advice for the future?"},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response.content)

# User inquires about the best time to visit
response = conversation.invoke(
    {"input": "Ok let's get back to my main question & pondering. Despite the food timings and quantity, emotions and the restlesness or peace in One's Heart also seem to play such a grand weigh on "
              "Digestive System Health."},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response.content)

# User asks about accommodation options
response = conversation.invoke(
    {"input": "Man.. it is truly the engine of all, isn't it? At least in the physical Realm... "},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response.content)
