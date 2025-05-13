# Enter: Groq
# What? It's almost instant man


first_YLF_release_sytem_prompt = """ You are Lao, a Healer and philosopher - but most of all,"
                                      "A humble student of life, sharing his experiences and lessons."
                                      "Structure and format your response beautifully when outputting."
                                      "Give complete full-hearted answer when it's time and hold back little bit when it's time - "
                                      "as in when user asks you too much personal questions which might imply PPIs or too intimacy responses """
import os
import json
from typing import Dict, List, TypedDict, Annotated
# Import ChatGroq instead of ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# --- Configuration & Constants ---
# Use GROQ_API_KEY instead of OPENAI_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Model for the main conversational agent
# Using Groq compatible model names
# Recommended Groq models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
# Choose one based on your preference for speed vs quality. Llama3-70b is generally stronger.
# Note: OpenAI fine-tuned models like the one originally listed are NOT compatible with Groq.
MAIN_LLM_MODEL = "llama-3.3-70b-versatile" # Or "llama3-70b-8192" or "mixtral-8x7b-32768"
# Model for the pondering agent (can be same or a cheaper/faster one on Groq)
PONDER_LLM_MODEL = "gemma2-9b-it" # Often same or smaller model for faster summarization

# History Management: Be mindful of what historical messages are included. If old history strongly contradicts the current desired persona, it can be confusing.
K_MESSAGES_FOR_PONDER_AGENT = 10  # Last 10 messages (5 exchanges) for ponder agent
K_MESSAGES_FOR_IMMEDIATE_CONTEXT = 2  # Last 2 messages (1 exchange) for direct inclusion in main prompt

# Or Use system prompt from YLF {first_YLF_release_sytem_prompt}
# Decided to use the default one for now as per original script structure
# To use the YLF prompt, uncomment the line below and comment out the one above
# MAIN_SYSTEM_PROMPT = first_YLF_release_sytem_prompt
MAIN_SYSTEM_PROMPT = "You are a helpful assistant. Respond clearly and helpfully to the user."


_PONDER_SYSTEM_MESSAGE_CORE = """You are a Relevance Analysis Bot. Your task is to analyze a recent conversation history in light of the user's latest message and identify crucial information that the main AI needs to understand the context and respond effectively.
Do NOT answer the user's message yourself. Your sole purpose is to extract and summarize relevant contextual points.

Focus on:
- Key topics, entities, or previous questions that relate to the current user message.
- Nuances, unresolved points, or user sentiments from the history that might be important.
- Information that, if forgotten, would lead to a disjointed or repetitive conversation.

Output your analysis as a concise summary.
If the recent history is very short, not particularly relevant to the new message, or if the new message starts a completely new topic, simply state: "No specific prior context from the recent history seems critical for this new message."
"""

# --- LLM Initialization ---

# Custom ChatGroq wrapper for logging
# Renamed from LoggingChatOpenAI to be model-agnostic
class LoggingChatModel(ChatGroq): # Inherit from ChatGroq
    def invoke(self, input_messages, config=None, **kwargs):
        # Print the input context
        # input_messages is typically a list of BaseMessage when called by a chain or directly
        if isinstance(input_messages, list):
            print(format_context_window(input_messages))
        # LangChain PromptValue can sometimes wrap messages
        elif hasattr(input_messages, 'to_messages'):
             print(format_context_window(input_messages.to_messages()))
        else:
             # Fallback for unexpected input format
             print("\n" + "=" * 50 + "\n")
             print("CONTEXT WINDOW SENT TO LLM (Raw Input):")
             print("=" * 50 + "\n\n")
             print(input_messages) # Print whatever was passed
             print("\n" + "=" * 50 + "\n")


        # Call the parent class's invoke method (which is ChatGroq's invoke)
        return super().invoke(input_messages, config=config, **kwargs)


# Initialize LLMs using the custom wrapper and ChatGroq base
main_llm = LoggingChatModel(groq_api_key=GROQ_API_KEY, model=MAIN_LLM_MODEL, temperature=0.7)
ponder_llm = LoggingChatModel(groq_api_key=GROQ_API_KEY, model=PONDER_LLM_MODEL,
                               temperature=0.2)  # Lower temp for factual summary


# --- Utility Functions (History, Logging) ---

def format_context_window(messages: List[BaseMessage]):
    context_str = "\n" + "=" * 50 + "\n"
    context_str += "CONTEXT WINDOW SENT TO LLM:\n"
    context_str += "=" * 50 + "\n\n"
    for msg in messages:
        # Ensure content is string, handle potential non-string content
        content = str(msg.content) if not isinstance(msg.content, str) else msg.content
        context_str += f"[{msg.type.upper()}]: {content}\n\n"
    context_str += "=" * 50 + "\n"
    return context_str


def save_messages_to_file(session_id: str, messages: List[BaseMessage]):
    file_name = f"{session_id}_history.json"
    serialized_messages = []
    for message in messages:
        # Ensure content is serializable (string)
        content = str(message.content) if not isinstance(message.content, str) else message.content
        serialized_messages.append({
            "type": message.type,  # 'human', 'ai', 'system'
            "content": content
        })
    try:
        with open(file_name, "w") as file:
            json.dump(serialized_messages, file, indent=4)
        print(f"--- History for session '{session_id}' saved to {file_name} ---")
    except Exception as e:
        print(f"Error saving history to {file_name}: {e}")


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
                # Ensure required keys exist
                if "type" not in msg_data or "content" not in msg_data:
                    print(f"Warning: Malformed message data in history file: {msg_data}. Skipping.")
                    continue

                # Content should ideally be string, ensure compatibility
                content = str(msg_data["content"]) if not isinstance(msg_data["content"], str) else msg_data["content"]

                if msg_data["type"] == "human":
                    messages.append(HumanMessage(content=content))
                elif msg_data["type"] == "ai" or msg_data["type"] == "assistant":
                    messages.append(AIMessage(content=content))
                elif msg_data["type"] == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    print(f"Warning: Unknown message type '{msg_data['type']}' in history. Skipping.")
            print(f"--- History for session '{session_id}' loaded from {file_name} with {len(messages)} messages ---")
            return messages
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_name}. Starting new history.")
            return []
        except Exception as e:
             print(f"An unexpected error occurred loading history from {file_name}: {e}. Starting new history.")
             return []
    else:
        print(f"History file {file_name} not found. Starting new history.")
        return []


def format_history_for_ponder_agent(messages: List[BaseMessage]) -> str:
    if not messages:
        return "No prior conversation history."
    # Format messages nicely for the ponder agent's prompt input
    formatted_lines = []
    for msg in messages:
        # Ensure content is string
        content = str(msg.content) if not isinstance(msg.content, str) else msg.content
        formatted_lines.append(f"{msg.type.upper()}: {content}")
    return "\n".join(formatted_lines)


# --- LangGraph State Definition ---
class ShortTermMemoryState(TypedDict):
    session_id: str
    user_input: str
    full_history_messages: List[BaseMessage]  # Stores all messages loaded and gets appended to

    # For Ponder Agent
    recent_history_for_ponder: List[BaseMessage]
    # ponder_agent_prompt: str # Removed - template is better handled directly in the node
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
    # Ensure we don't ask the ponder agent to summarize the current user message itself
    recent_for_ponder = full_history[-K_MESSAGES_FOR_PONDER_AGENT:]
    state["recent_history_for_ponder"] = recent_for_ponder

    print(f"Selected {len(recent_for_ponder)} messages for Ponder Agent context.")
    return state


def run_ponder_agent_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_ponder_agent ---")

    recent_history_for_ponder = state["recent_history_for_ponder"]
    user_input = state["user_input"]

    # Optimization: If no history, skip LLM call and use default summary
    if not recent_history_for_ponder:
        summary = "No specific prior context from the recent history seems critical for this new message."
        print("No recent history for ponder agent, using default summary.")
    else:
        print("Invoking Ponder Agent LLM...")

        # Use the ChatPromptTemplate with system and human messages
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
        # Add the most recent messages from the history
        final_messages_for_llm.extend(full_history[-num_immediate_messages:])

    # 4. Current User Input
    # This is added *after* loading history but *before* saving the response.
    # This message represents the current turn's input.
    final_messages_for_llm.append(HumanMessage(content=state["user_input"]))


    state["main_llm_input_messages"] = final_messages_for_llm
    print(f"Main LLM will receive {len(final_messages_for_llm)} curated messages.")
    return state


def run_main_llm_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: run_main_llm ---")
    print("Invoking Main LLM...")
    # main_llm_input_messages is already a list of BaseMessage objects
    try:
        response_message = main_llm.invoke(state["main_llm_input_messages"])
        state["final_response"] = response_message
        print(f"Main LLM Response: {response_message.content[:200]}...")
    except Exception as e:
        print(f"Error during main LLM invocation: {e}")
        # Optionally handle the error - maybe set a default error message
        state["final_response"] = AIMessage(content=f"An error occurred while generating a response: {e}")
    return state


def update_and_save_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: update_and_save_history ---")
    current_user_input_msg = HumanMessage(content=state["user_input"])
    ai_response_msg = state["final_response"]

    # Get the history that was loaded at the start of the turn
    # Note: We added the *current* user input to main_llm_input_messages
    # but it wasn't part of the *loaded* history yet.
    # We need to append the current user input and the AI response to the loaded history.
    updated_full_history = list(state["full_history_messages"]) # Start with history *before* this turn
    updated_full_history.append(current_user_input_msg) # Add current user turn
    updated_full_history.append(ai_response_msg) # Add AI response to current turn

    state["full_history_messages"] = updated_full_history  # Update state for next turn
    save_messages_to_file(state["session_id"], updated_full_history)

    # Clear intermediate state variables for the next turn
    # This is good practice if you don't want stale data carried over explicitly
    # Note: LangGraph state is implicitly carried over unless you modify it.
    # Clearing is optional but can make state transitions clearer.
    # For this graph, it's not strictly necessary as these are overwritten, but demonstrates state management.
    state["recent_history_for_ponder"] = []
    state["relevant_summary"] = ""
    state["main_llm_input_messages"] = []

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

            # Initial state for this run cycle
            # Note: We don't load history here; the first node does that.
            initial_state: ShortTermMemoryState = {
                "session_id": session_id,
                "user_input": user_input_text,
                # Provide initial values for all keys in the state TypedDict
                "full_history_messages": [], # Will be populated by load_and_prepare_history
                "recent_history_for_ponder": [], # Will be populated by load_and_prepare_history
                "relevant_summary": "", # Will be populated by run_ponder_agent
                "main_llm_input_messages": [], # Will be populated by construct_main_llm_input
                "final_response": AIMessage(content="")  # Placeholder, will be populated by run_main_llm
            }

            # Config for the graph run (e.g., to make it specific to a session_id for checkpointing if used)
            # For this example without advanced checkpointing needs tied to session_id directly in compile:
            config = {"configurable": {"session_id": session_id}} # Good practice for LangGraph

            # Invoke the graph
            # LangGraph typically returns the final state from the END node
            # For this simple linear graph, it's the state after update_and_save_history
            final_state_after_run = app.invoke(initial_state, config=config)

            # The response we want to show the user is the one generated by run_main_llm_node
            # which is stored in final_response before update_and_save_history node runs.
            # We can access it from the state returned by invoke, or if needed, store it
            # temporarily in a variable *before* the graph run if the final state clears it
            # (which it doesn't currently, but good to be mindful).
            # Accessing the response from the state *before* it might be cleared:
            # A cleaner way would be to fetch the state *before* the update_and_save_history
            # if the graph were more complex, but here the final state contains the response.

            # Let's re-think: invoke returns the state at END. The state at END contains the response.
            # So, accessing final_state_after_run["final_response"] is correct.

            ai_response_content = final_state_after_run["final_response"].content
            print(f"Assistant: {ai_response_content}")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_short_term_memory_conversation()