import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Set up your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the fine-tuned LLM
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="ft:gpt-4o-mini-2024-07-18:personal::AioG35Tv"  # Replace with your fine-tuned model
)

# Memory to store conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a chat prompt template with a system message and placeholders
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent customer support assistant for an e-commerce website. "
               "Your goal is to provide helpful and accurate answers to user questions about products, shipping, returns, and policies. "
               "Use the context of the conversation to improve your responses."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True  # Enable for debug info
)

# Example conversation with the bot
response_1 = conversation.run("Hi, I need help with a Diet. There is a deep programming of the body and as evening comes - it awaits "
                              "enourmous amounts of food - due to faulty 7 years programming. ")
print(response_1)

response_2 = conversation.run("Why do these patterns occur ? Its fascinating how the mind finds complicated reliefs of pain through others")
print(response_2)

response_3 = conversation.run("What would be a valuable advice for the future?")
print(response_3)
