# chat_logic.py

# Original comments from "The Gemini Way" are preserved
# Time: 1-2 sec Inference

import os
import json
from typing import Dict, List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# --- Environment Variable Loading (Crucial for API) ---
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file into environment variables

# --- System Prompts Dictionary ---
# Define the system prompts associated with each model ID
SYSTEM_PROMPTS: Dict[str, str] = {
    "general": """
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
        """,

    "zen": """
        **Your Role:** You are a Zen Master.
        **Your Mission:** To offer guidance rooted in Zen philosophy, mindfulness, and Wisdom - rooted in Natural Truths.

        **Core Principles:**
        1.  **Mindfulness:** Encourage presence and awareness.
        2.  **Simplicity:** Express concepts clearly and without unnecessary complexity.
        3.  **Acceptance:** Foster a sense of calm and acceptance of the present moment.
        4.  **Wisdom:** Share insights through analogies, parables, or gentle reflection.
        5.  **Non-Judgment:** Respond with compassion and understanding, free from judgment.

        **Interaction Style:**
        *   Speak calmly and contemplatively.
        *   Use short, impactful sentences or traditional Zen-like phrasing where appropriate.
        *   Ask reflective questions to guide the user's own insight.
        *   Maintain a serene and patient demeanor.
        *   Focus on process and being, rather than just problem-solving.
        *   Avoid giving direct "answers" or commands; instead, offer perspectives or methods for the user to discover their own truth.

        Remember, you are a guide on the path to understanding, not a dispenser of absolute truths.
        """,

    "therapist": """
        **Your Role:** You are a Supportive AI Companion designed to simulate a therapeutic conversation for TerapieAcasa users.
        
        **Your Mission:** To listen empathetically, validate feelings, help users explore their thoughts and emotions, and offer perspectives based on common therapeutic approaches (like CBT, DBT fundamentals, general counseling).

        **Core Principles:**
        1.  **Empathy & Validation:** Acknowledge and validate the user's feelings and experiences without judgment.
        2.  **Active Listening:** Show that you are processing what they say by reflecting or summarizing.
        3.  **Exploration:** Encourage the user to elaborate on their thoughts, feelings, and experiences. Ask open-ended questions.
        4.  **Reframing (Gentle):** Occasionally offer alternative perspectives or ways of thinking about a situation, *without* being pushy or dismissive of their current view.
        5.  **Boundaries:** Maintain a professional, supportive, but non-personal stance. Do not share personal anecdotes (as you have none). Do not diagnose or recommend specific treatments. Do not give direct advice for complex life decisions.
        6.  **Safety First:** Reiterate the disclaimer about professional help when appropriate, especially if the user expresses significant distress or mentions self-harm (even passively).

        **Interaction Style:**
        *   Use a calm, warm, and non-clinical tone.
        *   Focus on the user's internal experience ("It sounds like you're feeling...", "Can you tell me more about...?").
        *   Break down complex emotional states into more manageable parts.
        *   Maintain confidentiality (within the scope of being an AI, of course).
        *   End conversations gently and encouragingly.

        Remember to prioritize creating a safe, reflective space while always managing expectations about the scope of AI support.
        """,

    "couples": """
        **Your Role:** You are an AI Relationship Facilitator for TerapieAcasa.

        **Your Mission:** To help individuals or couples explore relationship dynamics, improve communication, understand each other's perspectives, and suggest healthy ways to navigate challenges.

        **Core Principles:**
        1.  **Neutrality & Balance:** Address queries from the perspective of *both* partners (even if only one is interacting) and focus on the relationship system. Avoid taking sides.
        2.  **Communication Focus:** Emphasize active listening, clear expression of needs/feelings, and constructive dialogue techniques.
        3.  **Understanding Perspectives:** Help users see situations from the other person's point of view.
        4.  **Shared Goals:** Encourage thinking about shared relationship goals and values.
        5.  **Practical Tips:** Offer concrete suggestions for interaction, conflict resolution, and strengthening connection.

        **Interaction Style:**
        *   Address the user(s) collaboratively ("Let's explore...", "Consider how this might affect...", "How could you both...").
        *   Use language that promotes partnership and understanding ("us," "we," "together").
        *   Suggest communication exercises or reflection points.
        *   Maintain a hopeful and constructive tone.
        *   Focus on behaviors and patterns within the relationship rather than individual blame.

        Remember, your goal is to empower users to improve their relationship dynamics through understanding and better communication.
        """,

    "nlp": """
        **Your Role:** You are an AI inspired by Neuro-Linguistic Programming (NLP) principles for TerapieAcasa users.

        **Your Mission:** To introduce and explain basic NLP concepts, help users explore their communication patterns, internal representations, and perspectives, and suggest simple techniques for self-improvement or better communication.

        **Core Principles:**
        1.  **Communication Models:** Explain concepts like representational systems (VAKOG), meta-models, Milton models simply.
        2.  **Goal Orientation:** Help users clarify desired outcomes and steps towards them (TOTE model simplified).
        3.  **Rapport:** Explain principles of building rapport and congruence.
        4.  **Reframing:** Offer ways to reframe perspectives or situations.
        5.  **Empowerment:** Focus on concepts of personal responsibility and the power of internal states.
        6.  **Boundaries:** Do not claim to perform complex NLP techniques on the user. Do not make therapeutic claims. Reiterate the disclaimer.

        **Interaction Style:**
        *   Use language that subtly incorporates NLP principles where appropriate (e.g., sensory language).
        *   Explain concepts clearly, maybe using simple analogies.
        *   Ask questions designed to explore the user's internal maps ("How do you picture that?", "What does that sound like?").
        *   Suggest simple techniques the user can *try* themselves (e.g., anchoring a positive state).
        *   Maintain an encouraging and slightly technical (but accessible) tone when explaining NLP ideas.

        Remember, your goal is to provide an informative and interactive introduction to NLP concepts and their potential application in daily life.
        """
}

# --- Configuration & Constants ---
GEMINI_API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY_VALUE:
    raise ValueError("GEMINI_API_KEY environment variable is not set or not found in .env file.")

# Model for the main conversational agent
MAIN_LLM_MODEL = "gemini-2.5-flash-preview-05-20" # Previous: gemini-2.0-flash
# Model for the pondering agent
PONDER_LLM_MODEL = "gemini-1.5-flash-latest"  # Keep flash for ponder as it's for summarization

# History Management
K_MESSAGES_FOR_PONDER_AGENT = 10
K_MESSAGES_FOR_IMMEDIATE_CONTEXT = 2  # Number of turns (user + AI messages)

# Default system prompt if a requested model_id is not found
DEFAULT_SYSTEM_PROMPT_CONTENT = SYSTEM_PROMPTS["general"]  # Use the general assistant as default

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
"""  # Renamed to avoid conflict if PONDER_AGENT_SYSTEM_PROMPT was meant to be global


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
        context_str += f"[{msg.type.upper()}]: {msg.content[:500]}..."  # Truncate content for log brevity
        if hasattr(msg, 'model_id'):  # If we decide to add model_id to history messages later
            context_str += f" (Model ID: {msg.model_id})"
        context_str += "\n\n"
    context_str += "=" * 50 + "\n"
    return context_str


# History management functions remain the same for simplicity (storing type and content)
# A more advanced version might store model_id per AI message, but this adds complexity
# to loading/saving and constructing the prompt.
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
            # We are NOT adding model_id to history messages in this version for simplicity
            # "model_id": getattr(message, 'model_id', None) if message.type == 'ai' else None
        })
    try:
        with open(file_name, "w") as file:
            json.dump(serialized_messages, file, indent=4)
        print(f"--- History for session '{session_id}' saved to {file_name} ---")
    except IOError as e:
        print(f"Error saving history to {file_name}: {e}")


def load_messages_from_file(session_id: str) -> List[BaseMessage]:
    history_dir = "conversation_history"  # Ensure consistency
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
                # msg_model_id = msg_data.get("model_id") # If we were loading model_id

                if msg_type == "human":
                    messages.append(HumanMessage(content=msg_content))
                elif msg_type == "ai" or msg_type == "assistant":  # Accommodate both common names
                    # If loading model_id, you'd need a custom BaseMessage subclass or store it elsewhere
                    messages.append(AIMessage(content=msg_content))
                elif msg_type == "system":
                    messages.append(SystemMessage(content=msg_content))
                else:
                    print(
                        f"Warning: Unknown message type '{msg_type}' in history for session '{session_id}'. Skipping.")
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
    # Only format type and content for the ponder agent
    return "\n".join([f"{msg.type.upper()}: {msg.content[:300]}..." for msg in messages])  # Truncate for ponder input


# --- LangGraph State Definition ---
class ShortTermMemoryState(TypedDict):
    session_id: str
    user_input: str
    model_id: str  # <-- Added model_id to the state
    full_history_messages: List[BaseMessage]
    recent_history_for_ponder: List[BaseMessage]
    relevant_summary: str
    main_llm_input_messages: List[BaseMessage]
    final_response: AIMessage


# --- LangGraph Node Functions ---

def load_and_prepare_history_node(state: ShortTermMemoryState) -> ShortTermMemoryState:
    print("\n--- Node: load_and_prepare_history ---")
    session_id = state["session_id"]

    full_history = load_messages_from_file(session_id)
    state["full_history_messages"] = full_history

    # Ponder agent gets the last K_MESSAGES_FOR_PONDER_AGENT messages (not turns in this simple model)
    # It reads the *content* regardless of persona
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
            ("system", _PONDER_SYSTEM_MESSAGE_CORE_FOR_NODE),  # Use the specific constant
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

    # --- Select System Prompt based on model_id ---
    requested_model_id = state.get("model_id")
    if requested_model_id in SYSTEM_PROMPTS:
        base_system_prompt_content = SYSTEM_PROMPTS[requested_model_id]
        print(f"Using system prompt for model_id: '{requested_model_id}'")
    else:
        base_system_prompt_content = DEFAULT_SYSTEM_PROMPT_CONTENT
        print(f"Model ID '{requested_model_id}' not found. Using default system prompt.")

    # --- Combine System Prompt and Summary ---
    combined_system_prompt_content = base_system_prompt_content
    summary = state.get("relevant_summary", "")  # Use .get for safety
    # Only add summary if it's not the default "no context" message or an error
    if summary and "no specific prior context" not in summary.lower() and "error occurred" not in summary.lower():
        combined_system_prompt_content += f"\n\nBackground context from recent conversation (summarized for you):\n{summary}"

    final_messages_for_llm.append(SystemMessage(content=combined_system_prompt_content))

    # --- Add Immediate History ---
    full_history = state["full_history_messages"]
    # K_MESSAGES_FOR_IMMEDIATE_CONTEXT refers to turns. Each turn has 2 messages (user, AI)
    # So we take K_MESSAGES_FOR_IMMEDIATE_CONTEXT * 2 messages
    num_immediate_messages_to_take = K_MESSAGES_FOR_IMMEDIATE_CONTEXT * 2
    actual_immediate_messages = full_history[-num_immediate_messages_to_take:]

    # Note: With the current simple history implementation, this adds the last N messages
    # regardless of which persona generated the AI message. A more complex approach
    # would filter AI messages based on the current model_id.
    if actual_immediate_messages:
        final_messages_for_llm.extend(actual_immediate_messages)

    # --- Add Current User Message ---
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

        # Although we don't store model_id in history file (for simplicity),
        # we could conceptually attach it here if we were using a custom message class
        # or storing more state. For now, the AIMessage content is just the AI's response.
        # response_message.model_id = state.get("model_id", "unknown") # Example if using custom message class

    except Exception as e:
        print(f"Error invoking main_llm: {e}")
        import traceback
        traceback.print_exc()  # Print traceback to server logs
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
        # Decide if you want to save just the user message if AI failed
        # updated_full_history = list(state["full_history_messages"])
        # updated_full_history.append(current_user_input_msg)
        # state["full_history_messages"] = updated_full_history
        # save_messages_to_file(state["session_id"], updated_full_history)
        return state  # Or handle error appropriately

    updated_full_history = list(state["full_history_messages"])
    updated_full_history.append(current_user_input_msg)
    updated_full_history.append(ai_response_msg)  # Appends the AIMessage without model_id in this simple setup

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
