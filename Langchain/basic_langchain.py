import os

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from openai import OpenAI


client = OpenAI()

api_key = os.getenv("OPENAI_API_KEY")

# Set up your fine-tuned OpenAI model
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="ft:gpt-4o-mini-2024-07-18:personal::AioG35Tv"
)

# Memory for context
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # Enable for debug info
)

# Example conversation
response = conversation.run("What is LangChain?")
print(response)
