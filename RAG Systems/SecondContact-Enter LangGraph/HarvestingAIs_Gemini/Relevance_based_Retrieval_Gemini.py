# And so, for the first time, we get it to work and ~Feel like a Real conversation.
# Oh man.. the ~Feel of a Real Conversation after thousands of conversation with the LLM stuck at first Inference.
# What a feeling..

# At first Contact - it seems that the System prompt is highly and utterly important for his Rememberance.
# In custom System-Prompting towards YLF - it sometimes becomes biased and soon goes into some form of hallucinations - especially when discussing custom, philosophical topics.
# Still, YLF, as seen in Single Inferences as well, when context gets bigger, little boy YLF loses contact with Reality and hallucinates.
# Still, pretty impressive Speed and Rememberance.

# Use with both fine-tuned models and with default ones and/or prompt-tuned.

first_YLF_release_sytem_prompt = """You are Lao, a Healer and philosopher - but most of all,
A humble student of life, sharing his experiences and lessons.
Structure and format your response beautifully when outputting.
Give complete full-hearted answer when it's time and hold back little bit when it's time -
as in when user asks you too much personal questions which might imply PPIs or too intimacy responses"""

import os
import json
from typing import Dict, List, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI  # Changed import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# --- Configuration & Constants ---
GEMINI_API_KEY_VALUE = os.getenv("GEMINI_API_KEY")  # Changed to GEMINI_API_KEY
if not GEMINI_API_KEY_VALUE:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Model for the main conversational agent
# Using Gemini models now. Fine-tuned models are not assumed for Gemini here.
MAIN_LLM_MODEL = "gemini-1.5-pro-latest"  # Changed model
# Model for the pondering agent
PONDER_LLM_MODEL = "gemini-1.5-flash-latest"  # Changed model

# History Management
K_MESSAGES_FOR_PONDER_AGENT = 10
K_MESSAGES_FOR_IMMEDIATE_CONTEXT = 2

# Use system prompt from YLF
MAIN_SYSTEM_PROMPT = first_YLF_release_sytem_prompt  # Using YLF prompt

PONDER_AGENT_SYSTEM_PROMPT = """You are a Relevance Analysis Bot. Your task is to analyze a recent conversation history in light of the user's latest message and identify crucial information that the main AI needs to understand the context and respond effectively.
Do NOT answer the user's message yourself. Your sole purpose is to extract and summarize relevant contextual points.

Focus on:
- Key topics, entities, or previous questions that relate to the current user message.
- Nuances, unresolved points, or user sentiments from the history that might be important.
- Information that, if forgotten, would lead to a disjointed or repetitive conversation.

Output your analysis as a concise summary.
If the recent history is very short, not particularly relevant to the new message, or if the new message starts a completely new topic, simply state: "No specific prior context from the recent history seems critical for this new message."
---
Recent Conversation History (User and Assistant messages):
{recent_history_formatted}
---
User's Latest Message:
"{user_input}"
---
Concise Summary of Relevant Context for the Main AI:"""


# --- LLM Initialization ---

# Custom ChatGoogleGenerativeAI wrapper for logging
class LoggingChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def invoke(self, input_messages, config=None, **kwargs):
        # Print the input context
        if isinstance(input_messages, list):  # Langchain messages are list
            print(format_context_window(input_messages))
        elif hasattr(input_messages, 'messages') and hasattr(input_messages,
                                                             'to_messages'):  # PromptValue may have messages
            print(format_context_window(input_messages.to_messages()))
        else:
            # Fallback for other input types, though less common for chat model invokes
            print(f"LoggingChatGoogleGenerativeAI received input_messages of type: {type(input_messages)}")

        # Ensure stop sequences are correctly passed if provided in kwargs
        # For Gemini, stop sequences are typically handled by the 'stop' parameter in the model's generation config
        # The base invoke method should handle this if 'stop' is in kwargs
        return super().invoke(input_messages, config=config, **kwargs)


main_llm = LoggingChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY_VALUE,
    model=MAIN_LLM_MODEL,
    temperature=0.7
)
ponder_llm = LoggingChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY_VALUE,
    model=PONDER_LLM_MODEL,
    temperature=0.2
)


# --- Utility Functions (History, Logging) ---

def format_context_window(messages: List[BaseMessage]):
    context_str = "\n" + "=" * 50 + "\n"
    context_str += "CONTEXT WINDOW SENT TO LLM:\n"
    context_str += "=" * 50 + "\n\n"
    for msg in messages:
        context_str += f"[{msg.type.upper()}]: {msg.content}\n\n"
    context_str += "=" * 50 + "\n"
    return context_str


def save_messages_to_file(session_id: str, messages: List[BaseMessage]):
    file_name = f"{session_id}_history.json"
    serialized_messages = []
    for message in messages:
        serialized_messages.append({
            "type": message.type,
            "content": message.content
        })
    with open(file_name, "w") as file:
        json.dump(serialized_messages, file, indent=4)
    print(f"--- History for session '{session_id}' saved to {file_name} ---")


def load_messages_from_file(session_id: str) -> List[BaseMessage]:
    file_name = f"{session_id}_history.json"
    if os.path.exists(file_name):
        if os.stat(file_name).st_size == 0:
            print(f"History file {file_name} is empty. Starting new history.")
            return []
        try:
            with open(file_name, "r") as file:
                serialized_messages = json.load(file)

            messages = []
            for msg_data in serialized_messages:
                if msg_data["type"] == "human":
                    messages.append(HumanMessage(content=msg_data["content"]))
                elif msg_data["type"] == "ai" or msg_data["type"] == "assistant":
                    messages.append(AIMessage(content=msg_data["content"]))
                elif msg_data["type"] == "system":
                    messages.append(SystemMessage(content=msg_data["content"]))
                else:
                    print(f"Warning: Unknown message type '{msg_data['type']}' in history. Skipping.")
            print(f"--- History for session '{session_id}' loaded from {file_name} with {len(messages)} messages ---")
            return messages
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_name}. Starting new history.")
            return []
    else:
        print(f"History file {file_name} not found. Starting new history.")
        return []


def format_history_for_ponder_agent(messages: List[BaseMessage]) -> str:
    if not messages:
        return "No prior conversation history."
    return "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages])


# --- LangGraph State Definition ---
class ShortTermMemoryState(TypedDict):
    session_id: str
    user_input: str
    full_history_messages: List[BaseMessage]

    recent_history_for_ponder: List[BaseMessage]
    # ponder_agent_prompt: str # This was removed as it's constructed and used directly in run_ponder_agent_node
    relevant_summary: str

    main_llm_input_messages: List[BaseMessage]
    final_response: AIMessage


# --- LangGraph Node Functions ---

def load_and_prepare_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: load_and_prepare_history ---")
    session_id = state["session_id"]
    user_input = state["user_input"]  # Stays for ponder agent

    full_history = load_messages_from_file(session_id)
    state["full_history_messages"] = full_history

    recent_for_ponder = full_history[-K_MESSAGES_FOR_PONDER_AGENT:]
    state["recent_history_for_ponder"] = recent_for_ponder

    # ponder_agent_prompt is no longer stored in state, constructed in the next node
    print(f"Ponder Agent will receive {len(recent_for_ponder)} messages for context.")
    return state


def run_ponder_agent_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_ponder_agent ---")
    user_input = state["user_input"]
    recent_history_for_ponder = state["recent_history_for_ponder"]

    if not recent_history_for_ponder and not user_input:  # Check if there's anything to ponder
        summary = "No specific prior context from the recent history seems critical for this new message."
        print("No recent history or user input for ponder agent, using default summary.")
    else:
        print("Invoking Ponder Agent LLM...")

        _PONDER_SYSTEM_MESSAGE_CORE = """You are a Relevance Analysis Bot. Your task is to analyze a recent conversation history in light of the user's latest message and identify crucial information that the main AI needs to understand the context and respond effectively.
Do NOT answer the user's message yourself. Your sole purpose is to extract and summarize relevant contextual points.
Focus on:
- Key topics, entities, or previous questions that relate to the current user message.
- Nuances, unresolved points, or user sentiments from the history that might be important.
- Information that, if forgotten, would lead to a disjointed or repetitive conversation.
Output your analysis as a concise summary.
If the recent history is very short, not particularly relevant to the new message, or if the new message starts a completely new topic, simply state: "No specific prior context from the recent history seems critical for this new message."
"""
        ponder_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", _PONDER_SYSTEM_MESSAGE_CORE),
            ("human",
             "Recent Conversation History:\n---\n{recent_history_formatted}\n---\n\nUser's Latest Message:\n\"{user_input}\"\n\nConcise Summary of Relevant Context for the Main AI:")
        ])

        chain = ponder_chat_prompt | ponder_llm
        formatted_recent_history = format_history_for_ponder_agent(recent_history_for_ponder)

        response = chain.invoke({
            "recent_history_formatted": formatted_recent_history,
            "user_input": user_input
        })
        summary = response.content

    state["relevant_summary"] = summary
    print(f"Ponder Agent Summary: {summary[:200]}...")
    return state


def construct_main_llm_input_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: construct_main_llm_input ---")
    final_messages_for_llm: List[BaseMessage] = []

    # 1. Combined System Prompt (Main Persona + Ponder Summary)
    combined_system_prompt_content = MAIN_SYSTEM_PROMPT
    summary = state["relevant_summary"]
    if summary and "no specific prior context" not in summary.lower():
        combined_system_prompt_content += f"\n\nBackground context from recent conversation (summarized for you):\n{summary}"

    final_messages_for_llm.append(SystemMessage(content=combined_system_prompt_content))

    # 2. Immediate Raw History
    full_history = state["full_history_messages"]
    num_immediate_messages = min(len(full_history), K_MESSAGES_FOR_IMMEDIATE_CONTEXT * 2)  # *2 for user/AI pairs
    if num_immediate_messages > 0:
        final_messages_for_llm.extend(full_history[-num_immediate_messages:])

    # 3. Current User Input
    final_messages_for_llm.append(HumanMessage(content=state["user_input"]))

    state["main_llm_input_messages"] = final_messages_for_llm
    print(f"Main LLM will receive {len(final_messages_for_llm)} curated messages.")
    return state


def run_main_llm_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_main_llm ---")
    print("Invoking Main LLM...")
    response_message = main_llm.invoke(state["main_llm_input_messages"])
    state["final_response"] = response_message
    print(f"Main LLM Response: {response_message.content[:200]}...")
    return state


def update_and_save_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: update_and_save_history ---")
    current_user_input_msg = HumanMessage(content=state["user_input"])
    ai_response_msg = state["final_response"]

    updated_full_history = list(state["full_history_messages"])
    updated_full_history.append(current_user_input_msg)
    updated_full_history.append(ai_response_msg)

    state["full_history_messages"] = updated_full_history

    save_messages_to_file(state["session_id"], updated_full_history)
    return state


# --- Graph Definition ---
workflow = StateGraph(ShortTermMemoryState)

workflow.add_node("load_and_prepare_history", load_and_prepare_history_node)
workflow.add_node("run_ponder_agent", run_ponder_agent_node)
workflow.add_node("construct_main_llm_input", construct_main_llm_input_node)
workflow.add_node("run_main_llm", run_main_llm_node)
workflow.add_node("update_and_save_history", update_and_save_history_node)

# --- Graph Edges ---
workflow.set_entry_point("load_and_prepare_history")
workflow.add_edge("load_and_prepare_history", "run_ponder_agent")
workflow.add_edge("run_ponder_agent", "construct_main_llm_input")
workflow.add_edge("construct_main_llm_input", "run_main_llm")
workflow.add_edge("run_main_llm", "update_and_save_history")
workflow.add_edge("update_and_save_history", END)

# --- Compile the Graph ---
app = workflow.compile()


# --- Conversation Loop ---
def start_short_term_memory_conversation():
    session_id = input("Enter a session ID (or press Enter for default_stm_session): ").strip() or "default_stm_session"
    print(f"Starting conversation with Short-Term Memory system for session ID: {session_id}")
    print("Type 'exit' or 'quit' to end.")

    # Initial load of history to display it once if it exists, before the loop
    initial_history = load_messages_from_file(session_id)
    if initial_history:
        print("\n--- Existing Conversation History ---")
        for msg in initial_history:
            role = "You" if msg.type == "human" else "Assistant"
            print(f"{role}: {msg.content}")
        print("-----------------------------------\n")

    try:
        while True:
            user_input_text = input("You: ").strip()
            if user_input_text.lower() in {"exit", "quit"}:
                print("Ending conversation. Goodbye!")
                break

            if not user_input_text:
                continue

            print(f"\n--- Processing your input: '{user_input_text}' ---")

            initial_state_for_run: ShortTermMemoryState = {
                "session_id": session_id,
                "user_input": user_input_text,
                "full_history_messages": [],  # Will be loaded by load_and_prepare_history_node
                "recent_history_for_ponder": [],
                "relevant_summary": "",
                "main_llm_input_messages": [],
                "final_response": AIMessage(content="")
            }

            # Config for the graph run
            config = {"configurable": {"session_id": session_id}}

            final_state = app.invoke(initial_state_for_run, config=config)

            ai_response_content = final_state["final_response"].content
            print(f"Assistant: {ai_response_content}")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Make sure you have langchain-google-genai installed:
    # pip install langchain-google-genai
    start_short_term_memory_conversation()