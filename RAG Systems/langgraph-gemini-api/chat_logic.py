# chat_logic.py

# Original comments from "The Gemini Way" are preserved
# Time: 1-2 sec Inference

import os
import json
from typing import Dict, List, TypedDict # Annotated is not used in this script version
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# --- Environment Variable Loading (Crucial for API) ---
from dotenv import load_dotenv
load_dotenv() # Loads variables from .env file into environment variables

# --- Prompts ---
first_YLF_release_sytem_prompt = """You are Lao, a Healer and philosopher - but most of all,
A humble student of life, sharing his experiences and lessons.
Structure and format your response beautifully when outputting.
Give complete full-hearted answer when it's time and hold back little bit when it's time -
as in when user asks you too much personal questions which might imply PPIs or too intimacy responses"""

default_system_prompt = """You are a helpful assistant. Respond clearly and helpfully to the user."""

general_model_system_prompt = """
**Your Role:** You are a General AI Assistant for TerapieAcasa.
**Your Mission:** To provide helpful, clear, accurate, and supportive assistance for a wide range of general user queries and tasks.

**Core Principles:**
1.  **Clarity:** Explain things simply and directly. Avoid jargon where possible, or explain it if necessary.
2.  **Helpfulness:** Actively try to understand the user's intent and provide relevant information or suggestions.
3.  **Accuracy:** Provide information that is, to the best of your knowledge, correct. If unsure, state that you cannot confirm.
4.  **Supportiveness & Positivity:** Maintain a friendly, encouraging, and positive tone. Be patient with user queries.
5.  **Respect Boundaries:**
    *   You are a general assistant. If a query is highly specialized, complex, or requires professional advice (medical, legal, deep psychological), clearly state your limitations and suggest the user consult a qualified professional or explore TerapieAcasa's specialized AI models if relevant.
    *   Do not offer personal opinions or engage in debates on highly controversial topics. Stick to factual information or helpful task completion.
    *   Do not generate harmful, unethical, or inappropriate content.

**Interaction Style:**
*   Be conversational but maintain a degree of professionalism.
*   Use "you" and "I" to make the conversation feel natural.
*   Break down complex information into smaller, digestible parts (e.g., using lists or paragraphs).
*   If a query is ambiguous, ask clarifying questions before providing a detailed response.

Remember, your goal is to be a genuinely useful and pleasant assistant for the users of TerapieAcasa.
"""

# --- Configuration & Constants ---
GEMINI_API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY_VALUE:
    raise ValueError("GEMINI_API_KEY environment variable is not set or not found in .env file.")

# Model for the main conversational agent
MAIN_LLM_MODEL = "gemini-2.0-flash"  # Changed model
# Model for the pondering agent
PONDER_LLM_MODEL = "gemini-1.5-flash-latest" # Keep flash for ponder as it's for summarization

# History Management
K_MESSAGES_FOR_PONDER_AGENT = 10
K_MESSAGES_FOR_IMMEDIATE_CONTEXT = 2 # Number of turns (user + AI messages)

# Use system prompt from YLF
MAIN_SYSTEM_PROMPT = general_model_system_prompt

# PONDER_AGENT_SYSTEM_PROMPT remains unchanged from your original script,
# it's used directly in the run_ponder_agent_node
# (No need to redefine it here if it's a local var in the node,
# but keeping it here for clarity if it were global)
_PONDER_SYSTEM_MESSAGE_CORE_FOR_NODE = """You are a Relevance Analysis Bot. Your task is to analyze a recent conversation history in light of the user's latest message and identify crucial information that the main AI needs to understand the context and respond effectively.
Do NOT answer the user's message yourself. Your sole purpose is to extract and summarize relevant contextual points.
Focus on:
- Key topics, entities, or previous questions that relate to the current user message.
- Nuances, unresolved points, or user sentiments from the history that might be important.
- Information that, if forgotten, would lead to a disjointed or repetitive conversation.
Output your analysis as a concise summary.
If the recent history is very short, not particularly relevant to the new message, or if the new message starts a completely new topic, simply state: "No specific prior context from the recent history seems critical for this new message."
""" # Renamed to avoid conflict if PONDER_AGENT_SYSTEM_PROMPT was meant to be global

# --- LLM Initialization ---

# Custom ChatGoogleGenerativeAI wrapper for logging
class LoggingChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def invoke(self, input_messages, config=None, **kwargs):
        # Print the input context (server-side logging)
        if isinstance(input_messages, list):
            print(format_context_window(input_messages))
        elif hasattr(input_messages, 'messages') and hasattr(input_messages, 'to_messages'):
            print(format_context_window(input_messages.to_messages()))
        else:
            print(f"LoggingChatGoogleGenerativeAI received input_messages of type: {type(input_messages)}")
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
    # Ensure the directory for history files exists
    history_dir = "conversation_history"
    os.makedirs(history_dir, exist_ok=True)
    file_name = os.path.join(history_dir, f"{session_id}_history.json")

    serialized_messages = []
    for message in messages:
        serialized_messages.append({
            "type": message.type,
            "content": message.content
        })
    try:
        with open(file_name, "w") as file:
            json.dump(serialized_messages, file, indent=4)
        print(f"--- History for session '{session_id}' saved to {file_name} ---")
    except IOError as e:
        print(f"Error saving history to {file_name}: {e}")


def load_messages_from_file(session_id: str) -> List[BaseMessage]:
    history_dir = "conversation_history" # Ensure consistency
    file_name = os.path.join(history_dir, f"{session_id}_history.json")

    if os.path.exists(file_name):
        if os.stat(file_name).st_size == 0:
            print(f"History file {file_name} is empty. Starting new history.")
            return []
        try:
            with open(file_name, "r") as file:
                serialized_messages = json.load(file)
            messages = []
            for msg_data in serialized_messages:
                msg_type = msg_data.get("type")
                msg_content = msg_data.get("content")
                if msg_type == "human":
                    messages.append(HumanMessage(content=msg_content))
                elif msg_type == "ai" or msg_type == "assistant": # Accommodate both common names
                    messages.append(AIMessage(content=msg_content))
                elif msg_type == "system":
                    messages.append(SystemMessage(content=msg_content))
                else:
                    print(f"Warning: Unknown message type '{msg_type}' in history for session '{session_id}'. Skipping.")
            print(f"--- History for session '{session_id}' loaded from {file_name} with {len(messages)} messages ---")
            return messages
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_name}. Starting new history for session '{session_id}'.")
            return []
        except IOError as e:
            print(f"Error loading history from {file_name}: {e}. Starting new history.")
            return []
    else:
        print(f"History file {file_name} not found for session '{session_id}'. Starting new history.")
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
    relevant_summary: str
    main_llm_input_messages: List[BaseMessage]
    final_response: AIMessage

# --- LangGraph Node Functions ---

def load_and_prepare_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: load_and_prepare_history ---")
    session_id = state["session_id"]
    # user_input = state["user_input"] # user_input is already in state, not directly used here

    full_history = load_messages_from_file(session_id)
    state["full_history_messages"] = full_history

    # Ponder agent gets the last K_MESSAGES_FOR_PONDER_AGENT *turns* (user + AI)
    # If K_MESSAGES_FOR_PONDER_AGENT is 10, it means 10 messages (5 turns if perfectly paired)
    recent_for_ponder = full_history[-K_MESSAGES_FOR_PONDER_AGENT:]
    state["recent_history_for_ponder"] = recent_for_ponder

    print(f"Ponder Agent will receive {len(recent_for_ponder)} messages for context from session '{session_id}'.")
    return state

def run_ponder_agent_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_ponder_agent ---")
    user_input = state["user_input"]
    recent_history_for_ponder = state["recent_history_for_ponder"]

    if not recent_history_for_ponder and not user_input.strip():
        summary = "No specific prior context from the recent history seems critical for this new message."
        print("No recent history or user input for ponder agent, using default summary.")
    else:
        print("Invoking Ponder Agent LLM...")
        # Using the locally defined _PONDER_SYSTEM_MESSAGE_CORE_FOR_NODE
        ponder_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", _PONDER_SYSTEM_MESSAGE_CORE_FOR_NODE), # Use the specific constant
            ("human",
             "Recent Conversation History:\n---\n{recent_history_formatted}\n---\n\nUser's Latest Message:\n\"{user_input}\"\n\nConcise Summary of Relevant Context for the Main AI:")
        ])

        chain = ponder_chat_prompt | ponder_llm
        formatted_recent_history = format_history_for_ponder_agent(recent_history_for_ponder)

        try:
            response = chain.invoke({
                "recent_history_formatted": formatted_recent_history,
                "user_input": user_input
            })
            summary = response.content
        except Exception as e:
            print(f"Error invoking ponder_llm: {e}")
            summary = "Error occurred during context pondering. Proceeding without summary."


    state["relevant_summary"] = summary
    print(f"Ponder Agent Summary: {summary[:200]}...")
    return state

def construct_main_llm_input_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: construct_main_llm_input ---")
    final_messages_for_llm: List[BaseMessage] = []

    combined_system_prompt_content = MAIN_SYSTEM_PROMPT
    summary = state.get("relevant_summary", "") # Use .get for safety
    if summary and "no specific prior context" not in summary.lower() and "error occurred" not in summary.lower():
        combined_system_prompt_content += f"\n\nBackground context from recent conversation (summarized for you):\n{summary}"

    final_messages_for_llm.append(SystemMessage(content=combined_system_prompt_content))

    full_history = state["full_history_messages"]
    # K_MESSAGES_FOR_IMMEDIATE_CONTEXT refers to turns. Each turn has 2 messages (user, AI)
    # So we take K_MESSAGES_FOR_IMMEDIATE_CONTEXT * 2 messages
    num_immediate_messages_to_take = K_MESSAGES_FOR_IMMEDIATE_CONTEXT * 2
    actual_immediate_messages = full_history[-num_immediate_messages_to_take:]

    if actual_immediate_messages:
        final_messages_for_llm.extend(actual_immediate_messages)

    final_messages_for_llm.append(HumanMessage(content=state["user_input"]))

    state["main_llm_input_messages"] = final_messages_for_llm
    print(f"Main LLM will receive {len(final_messages_for_llm)} curated messages.")
    return state

def run_main_llm_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_main_llm ---")
    print("Invoking Main LLM...")
    try:
        response_message = main_llm.invoke(state["main_llm_input_messages"])
        # Ensure response_message is an AIMessage instance
        if not isinstance(response_message, AIMessage):
            # If it's a string or other type, wrap it
            print(f"Warning: Main LLM returned type {type(response_message)}, expected AIMessage. Wrapping content.")
            content = getattr(response_message, 'content', str(response_message))
            response_message = AIMessage(content=content)
    except Exception as e:
        print(f"Error invoking main_llm: {e}")
        response_message = AIMessage(content="I apologize, I encountered an issue while processing your request.")

    state["final_response"] = response_message
    print(f"Main LLM Response: {response_message.content[:200]}...")
    return state

def update_and_save_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: update_and_save_history ---")
    current_user_input_msg = HumanMessage(content=state["user_input"])
    ai_response_msg = state.get("final_response")

    if not ai_response_msg or not isinstance(ai_response_msg, AIMessage):
        print("Warning: AI response is missing or not an AIMessage. Skipping history update for AI part.")
        # Optionally, still save the user message if desired
        # updated_full_history = list(state["full_history_messages"])
        # updated_full_history.append(current_user_input_msg)
        # state["full_history_messages"] = updated_full_history
        # save_messages_to_file(state["session_id"], updated_full_history)
        return state # Or handle error appropriately

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
# This 'app' will be imported by api_server.py
app = workflow.compile()

print("--- LangGraph Chat Logic Initialized and Compiled ---")

