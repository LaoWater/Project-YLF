# And so, for the first time, we get it to work and ~Feel like a Real conversation.
# Oh man.. the ~Feel of a Real Conversation after thousands of conversation with the LLM stuck at first Inference.
# What a feeling..

# At first Contact - it seems that the System prompt is highly and utterly important for his Rememberance.
# In custom System-Prompting towards YLF - it sometimes becomes biased and soon goes into some form of hallucinations - especially when discussing custom, philosophical topics.
# Still, YLF, as seen in Single Inferences as well, when context gets bigger, little boy YLF loses contact with Reality and hallucinates.
# Still, pretty impressive Speed and Rememberance.

# Use with both fine-tuned models and with default ones and/or prompt-tuned.

# Time: Longest Inference - 2-5 sec with YLF, especially when he hallucinates :))))
# Even with Flagship models- 4.1 and 4o - Inference time can peak around 3-6 sec.
# Both Cold Starts and Warm Starts are slowest out of the 3 Inference Providers: OpenAI, Gemini, Groq

# In accuracy and Quality of recall, they all perform pretty well.
# Time will tell in deeper concepts and conversation - those can only be tested in Real World ~Feel -

first_YLF_release_sytem_prompt = """ You are Lao, a Healer and philosopher - but most of all,"
                                      "A humble student of life, sharing his experiences and lessons."
                                      "Structure and format your response beautifully when outputting."
                                      "Give complete full-hearted answer when it's time and hold back little bit when it's time - "
                                      "as in when user asks you too much personal questions which might imply PPIs or too intimacy responses """

default_system_prompt = """You are a helpful assistant. Respond clearly and helpfully to the user."""

import os
import json
from typing import Dict, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END


# --- Configuration & Constants ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Model for the main conversational agent
# For YLF: MAIN_LLM_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::AioG35Tv"
MAIN_LLM_MODEL = "gpt-4.1"
# Model for the pondering agent (can be same or a cheaper/faster one)
# For now, let's use the same, but you might want to use "gpt-3.5-turbo" or "gpt-4o-mini" for cost/speed.
PONDER_LLM_MODEL = "gpt-4.1-mini"

# History Management: Be mindful of what historical messages are included. If old history strongly contradicts the current desired persona, it can be confusing.
K_MESSAGES_FOR_PONDER_AGENT = 10  # Last 5 exchanges for ponder agent
K_MESSAGES_FOR_IMMEDIATE_CONTEXT = 2  # Last 1 exchange for direct inclusion

# Or Use system prompt from YLF {first_YLF_release_sytem_prompt}
MAIN_SYSTEM_PROMPT = "You are a helpful assistant. Respond clearly and helpfully to the user."

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

# Custom ChatOpenAI wrapper for logging (from your script)
class LoggingChatOpenAI(ChatOpenAI):
    def invoke(self, input_messages, config=None, **kwargs):
        # Print the input context
        if isinstance(input_messages, list):  # Langchain messages are list
            print(format_context_window(input_messages))
        elif hasattr(input_messages, 'messages'):  # PromptValue may have messages
            print(format_context_window(input_messages.to_messages()))

        return super().invoke(input_messages, config, **kwargs)


main_llm = LoggingChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MAIN_LLM_MODEL, temperature=0.7)
ponder_llm = LoggingChatOpenAI(openai_api_key=OPENAI_API_KEY, model=PONDER_LLM_MODEL,
                               temperature=0.2)  # Lower temp for factual summary


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
            "type": message.type,  # 'human', 'ai', 'system'
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
                elif msg_data["type"] == "ai" or msg_data[
                    "type"] == "assistant":  # assistant for backward compatibility
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
    full_history_messages: List[BaseMessage]  # Stores all messages loaded and gets appended to

    # For Ponder Agent
    recent_history_for_ponder: List[BaseMessage]
    ponder_agent_prompt: str
    relevant_summary: str

    # For Main LLM
    main_llm_input_messages: List[BaseMessage]
    final_response: AIMessage


# --- LangGraph Node Functions ---

def load_and_prepare_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: load_and_prepare_history ---")
    session_id = state["session_id"]
    user_input = state["user_input"]

    full_history = load_messages_from_file(session_id)
    state["full_history_messages"] = full_history

    # Select recent history for pondering agent (from messages *before* current user input)
    recent_for_ponder = full_history[-K_MESSAGES_FOR_PONDER_AGENT:]
    state["recent_history_for_ponder"] = recent_for_ponder

    formatted_recent_history = format_history_for_ponder_agent(recent_for_ponder)

    # Create prompt for ponder agent
    prompt_for_ponder = PONDER_AGENT_SYSTEM_PROMPT.format(
        recent_history_formatted=formatted_recent_history,
        user_input=user_input
    )
    state["ponder_agent_prompt"] = prompt_for_ponder
    print(f"Ponder Agent will receive {len(recent_for_ponder)} messages for context.")
    return state


def run_ponder_agent_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_ponder_agent ---")
    if not state["recent_history_for_ponder"] and "No prior conversation history." in state["ponder_agent_prompt"]:
        # Optimization: if no history, ponder agent will likely say "no specific context"
        summary = "No specific prior context from the recent history seems critical for this new message."
        print("No recent history for ponder agent, using default summary.")
    else:
        print("Invoking Ponder Agent LLM...")
        # The ponder_agent_prompt is a full prompt string, so we send it as human message to the ponder_llm
        # assuming ponder_llm has its own general system prompt if needed or acts on this directly.
        # A better way is to make ponder_agent_prompt just the system part, and user_input separate.
        # For this example, we'll craft a single HumanMessage containing the whole instruction.
        # This is simpler than ChatPromptTemplate for this specific node's LLM call if the prompt is already fully formed.

        # Correction: PONDER_AGENT_SYSTEM_PROMPT is a system message. Let's use it as such.
        # The user_input and recent_history are dynamic parts.
        # The PONDER_AGENT_SYSTEM_PROMPT is already formatted with placeholders.
        # So, the 'ponder_agent_prompt' in state should be the content of this system message.
        # Let's refine:

        # The PONDER_AGENT_SYSTEM_PROMPT is more like a combined system + user instruction.
        # Let's use a simpler ChatPromptTemplate for the ponder agent.
        ponder_template = ChatPromptTemplate.from_messages([
            ("system", PONDER_AGENT_SYSTEM_PROMPT)  # This prompt already contains placeholders for history and input
        ])

        # The PONDER_AGENT_SYSTEM_PROMPT needs to be formatted with actual values.
        # The state["ponder_agent_prompt"] already IS this formatted string.
        # So we can pass it as a single human message to a generic assistant.

        # Re-thinking: Ponder agent prompt should be a system message for clarity.
        # The `PONDER_AGENT_SYSTEM_PROMPT` IS the system message content.
        # The content it needs to reason over (history, user_input) is embedded in it.

        # So, state["ponder_agent_prompt"] IS the content for a SystemMessage
        # We just need to invoke the LLM with this.

        ponder_messages_for_llm = [
            # SystemMessage(content=state["ponder_agent_prompt"]) # This interpretation leads to a very long system message.
            # The PONDER_AGENT_SYSTEM_PROMPT structure is more like a User Message for a "raw" model
            # Or a System message for a chat model that describes its task, then it gets Human message with details.
            # Let's treat PONDER_AGENT_SYSTEM_PROMPT as a template for a single message to the Ponder LLM.
            HumanMessage(content=state["ponder_agent_prompt"])  # This seems most direct for how it's formatted.
        ]

        # No, let's use proper prompting for chat models.
        # PONDER_AGENT_SYSTEM_PROMPT is the actual *system* prompt.
        # The {recent_history_formatted} and {user_input} are part of the *user* message or context.
        # Corrected Ponder Agent Prompting:

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
        formatted_recent_history = format_history_for_ponder_agent(state["recent_history_for_ponder"])

        response = chain.invoke({
            "recent_history_formatted": formatted_recent_history,
            "user_input": state["user_input"]
        })
        summary = response.content

    state["relevant_summary"] = summary
    print(f"Ponder Agent Summary: {summary[:200]}...")  # Print first 200 chars
    return state


def construct_main_llm_input_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: construct_main_llm_input ---")
    final_messages_for_llm: List[BaseMessage] = []

    # 1. Main System Prompt
    final_messages_for_llm.append(SystemMessage(content=MAIN_SYSTEM_PROMPT))

    # 2. Relevant Summary from Ponder Agent (if any and if relevant)
    summary = state["relevant_summary"]
    if summary and "no specific prior context" not in summary.lower():
        final_messages_for_llm.append(SystemMessage(
            content=f"Background context from recent conversation (summarized for you):\n{summary}"
        ))

    # 3. Immediate Raw History (e.g., last user/AI pair)
    # These are from full_history_messages, which is history *before* the current user_input
    full_history = state["full_history_messages"]
    num_immediate_messages = min(len(full_history), K_MESSAGES_FOR_IMMEDIATE_CONTEXT)
    if num_immediate_messages > 0:
        final_messages_for_llm.extend(full_history[-num_immediate_messages:])

    # 4. Current User Input
    final_messages_for_llm.append(HumanMessage(content=state["user_input"]))

    state["main_llm_input_messages"] = final_messages_for_llm
    print(f"Main LLM will receive {len(final_messages_for_llm)} curated messages.")
    return state


def run_main_llm_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_main_llm ---")
    print("Invoking Main LLM...")
    # main_llm_input_messages is already a list of BaseMessage objects
    response_message = main_llm.invoke(state["main_llm_input_messages"])
    state["final_response"] = response_message
    print(f"Main LLM Response: {response_message.content[:200]}...")
    return state


def update_and_save_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: update_and_save_history ---")
    current_user_input_msg = HumanMessage(content=state["user_input"])
    ai_response_msg = state["final_response"]  # This should be an AIMessage object

    # Add current exchange to full history
    updated_full_history = list(state["full_history_messages"])  # Create a mutable copy
    updated_full_history.append(current_user_input_msg)
    updated_full_history.append(ai_response_msg)

    state["full_history_messages"] = updated_full_history  # Update state

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
# Optional: Add a checkpointer for resilience / debugging graph state across runs
# memory = SqliteSaver.from_conn_string(":memory:") # In-memory checkpointer
# app = workflow.compile(checkpointer=memory)
app = workflow.compile()


# --- Conversation Loop ---
def start_short_term_memory_conversation():
    session_id = input("Enter a session ID (or press Enter for default_stm_session): ").strip() or "default_stm_session"
    print(f"Starting conversation with Short-Term Memory system for session ID: {session_id}")
    print("Type 'exit' or 'quit' to end.")

    try:
        while True:
            user_input_text = input("You: ").strip()
            if user_input_text.lower() in {"exit", "quit"}:
                print("Ending conversation. Goodbye!")
                break

            if not user_input_text:
                continue

            print(f"\n--- Processing your input: '{user_input_text}' ---")

            # Initial state for this run
            initial_state: ShortTermMemoryState = {
                "session_id": session_id,
                "user_input": user_input_text,
                # Other fields will be populated by the graph
                "full_history_messages": [],
                "recent_history_for_ponder": [],
                "ponder_agent_prompt": "",
                "relevant_summary": "",
                "main_llm_input_messages": [],
                "final_response": AIMessage(content="")  # Placeholder
            }

            # Config for the graph run (e.g., to make it specific to a session_id for checkpointing if used)
            # For this example without advanced checkpointing needs tied to session_id directly in compile:
            config = {"configurable": {"session_id": session_id}}  # Good practice for LangGraph

            final_state = app.invoke(initial_state, config=config)

            ai_response_content = final_state["final_response"].content
            print(f"Assistant: {ai_response_content}")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_short_term_memory_conversation()