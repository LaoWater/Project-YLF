import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Replace MemoryCheckpoint with the correct memory implementation
from langgraph.checkpoint import MemorySaver

# ────────────────────────────── Groq Integration ───────────────
from groq import Groq
from langchain_groq import ChatGroq

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

groq_client = Groq(api_key=groq_api_key)


def groq_generate(prompt: str,
                  model: str = "llama-4-scout-17b-16e-instruct",
                  system: str | None = None) -> str:
    """Call Groq Chat Completion and return the assistant message text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    completion = groq_client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return completion.choices[0].message.content


# Initialize LangChain Groq models for both memory and response
# For memory operations: Using a smaller, fast model
memory_llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Fast and efficient for memory operations
    groq_api_key=groq_api_key,
    temperature=0.0  # Keep it deterministic for memory operations
)

# For generating responses: Using a more powerful model
response_llm = ChatGroq(
    model="llama-4-scout-17b-16e-instruct",  # More powerful for main responses
    groq_api_key=groq_api_key,
    temperature=0.7  # Allow for some creativity in responses
)

# Initialize the embedding model for semantic similarity
try:
    # Try to use OpenAI embeddings if available
    from langchain_openai import OpenAIEmbeddings

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Fall back to keyword-based similarity
        print("Warning: OpenAI API key not available. Falling back to keyword-based similarity.")
        embeddings = None
except ImportError:
    print("Warning: OpenAIEmbeddings not available. Falling back to keyword-based similarity.")
    embeddings = None

# Maximum tokens to send in context (adjust based on your model's context window)
MAX_CONTEXT_TOKENS = 8000  # Increased for Mixtral's larger context window
# Number of most recent messages to always include
ALWAYS_INCLUDE_RECENT = 3
# Number of messages for determining conversation topic
TOPIC_DETECTION_WINDOW = 5


# ------------------- Memory Management System -------------------

class SmartMemoryManager:
    """Manages conversation memory with intelligent context retrieval"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.full_history = ChatMessageHistory()
        self.memory_summaries = []  # Store periodic summaries of conversation
        self.conversation_topics = []  # Track evolving topics
        self.last_summarized_idx = 0  # Index of last summarized message

    def add_message(self, message: BaseMessage) -> None:
        """Add a new message to the history"""
        self.full_history.add_message(message)

        # Update topics when we have enough new messages
        if len(self.full_history.messages) - self.last_summarized_idx >= TOPIC_DETECTION_WINDOW:
            self._update_conversation_topics()
            self._create_summary_if_needed()

    def _update_conversation_topics(self) -> None:
        """Extract topics from recent conversation using Groq"""
        recent_messages = self.full_history.messages[-TOPIC_DETECTION_WINDOW:]

        # Combine messages into a single text for topic extraction
        combined_text = "\n".join([msg.content for msg in recent_messages])

        # Use groq_generate instead of the memory_llm directly
        system_prompt = "Extract 3-5 key topics from this conversation segment. Return only the topics as a comma-separated list, no explanation."
        topics_text = groq_generate(combined_text, model="mixtral-8x7b-32768", system=system_prompt)

        # Add to our topics list
        self.conversation_topics.append({
            "topics": topics_text.split(","),
            "start_idx": len(self.full_history.messages) - TOPIC_DETECTION_WINDOW,
            "end_idx": len(self.full_history.messages) - 1
        })

    def _create_summary_if_needed(self) -> None:
        """Create a summary of older conversation segments using Groq"""
        # Only summarize if we have enough new messages
        if len(self.full_history.messages) - self.last_summarized_idx < 10:
            return

        # Get messages to summarize
        messages_to_summarize = self.full_history.messages[self.last_summarized_idx:len(self.full_history.messages) - 5]

        if not messages_to_summarize:
            return

        # Combine messages for summarization
        combined_text = "\n".join([
            f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages_to_summarize
        ])

        # Create summary using groq_generate
        system_prompt = "Create a concise summary of this conversation segment that preserves key information, decisions, and context. Focus on essential details that would be needed for continuing this conversation naturally."
        summary_text = groq_generate(combined_text, model="mixtral-8x7b-32768", system=system_prompt)

        # Store the summary
        self.memory_summaries.append({
            "summary": summary_text,
            "start_idx": self.last_summarized_idx,
            "end_idx": len(self.full_history.messages) - 6,
            "timestamp": datetime.now().isoformat()
        })

        # Update the last summarized index
        self.last_summarized_idx = len(self.full_history.messages) - 5

    def get_relevant_context(self, current_query: str, token_limit: int = MAX_CONTEXT_TOKENS) -> List[BaseMessage]:
        """Retrieve the most relevant context for the current query"""

        # We'll build our context with these steps:
        # 1. Always include the most recent messages
        # 2. Include relevant prior messages based on similarity
        # 3. If needed, include relevant summaries

        messages = self.full_history.messages
        if not messages:
            return []

        # Always include the system message if it exists
        final_context = []
        if messages and isinstance(messages[0], SystemMessage):
            final_context.append(messages[0])
            messages = messages[1:]

        # Always include most recent messages
        recent_messages = messages[-ALWAYS_INCLUDE_RECENT:]

        # Find relevant earlier messages
        if len(messages) > ALWAYS_INCLUDE_RECENT:
            earlier_messages = messages[:-ALWAYS_INCLUDE_RECENT]

            # Get relevant messages using semantic similarity or keyword matching
            relevant_indices = self._get_relevant_message_indices(current_query, earlier_messages)
            relevant_earlier = [earlier_messages[i] for i in relevant_indices]

            # If we have few relevant messages, also include conversation summaries
            if len(relevant_earlier) < 3 and self.memory_summaries:
                # Find the most relevant summary
                relevant_summary = self._get_relevant_summary(current_query)
                if relevant_summary:
                    summary_msg = SystemMessage(content=f"Previous conversation summary: {relevant_summary['summary']}")
                    final_context.append(summary_msg)

            # Add the relevant earlier messages
            final_context.extend(relevant_earlier)

        # Add the recent messages
        final_context.extend(recent_messages)

        # Ensure we don't exceed the token limit (simple estimation)
        return self._trim_to_token_limit(final_context, token_limit)

    def _get_relevant_message_indices(self, query: str, messages: List[BaseMessage]) -> List[int]:
        """Find indices of messages most relevant to the query"""
        if not messages:
            return []

        if embeddings is not None:
            # Use embeddings for semantic similarity
            try:
                query_embedding = embeddings.embed_query(query)
                message_embeddings = [
                    embeddings.embed_query(msg.content)
                    for msg in messages
                ]

                # Calculate cosine similarity
                similarities = [
                    np.dot(query_embedding, msg_emb) /
                    (np.linalg.norm(query_embedding) * np.linalg.norm(msg_emb))
                    for msg_emb in message_embeddings
                ]

                # Get indices of top 5 most similar messages
                top_indices = np.argsort(similarities)[-5:]
                return top_indices.tolist()

            except Exception as e:
                print(f"Error using embeddings: {e}. Falling back to keyword matching.")
                # Fall back to keyword matching on error

        # Simple keyword matching as fallback
        query_keywords = set(query.lower().split())
        scores = []

        for i, msg in enumerate(messages):
            msg_text = msg.content.lower()
            # Count keywords that appear in the message
            match_score = sum(1 for keyword in query_keywords if keyword in msg_text)
            # Add recency bias
            recency_score = i / len(messages)  # 0 for first, 1 for last
            combined_score = match_score + (recency_score * 0.5)  # Weight recency less than relevance
            scores.append(combined_score)

        # Get indices of top 5 messages
        top_indices = np.argsort(scores)[-5:]
        return top_indices.tolist()

    def _get_relevant_summary(self, query: str) -> Optional[dict]:
        """Find the most relevant conversation summary using Groq"""
        if not self.memory_summaries:
            return None

        # Format summaries for the LLM
        summaries_text = "\n\n".join([
            f"Summary {i + 1}: {summary['summary']}"
            for i, summary in enumerate(self.memory_summaries)
        ])

        # Use groq_generate to select the most relevant summary
        system_prompt = "Given a user query and several conversation summaries, return ONLY the number of the most relevant summary. Return just the number, nothing else."
        prompt = f"Query: {query}\n\n{summaries_text}"

        try:
            selection = groq_generate(prompt, model="mixtral-8x7b-32768", system=system_prompt).strip()
            summary_idx = int(selection) - 1
            if 0 <= summary_idx < len(self.memory_summaries):
                return self.memory_summaries[summary_idx]
        except:
            # If there's any error, just return the most recent summary
            return self.memory_summaries[-1]

        return self.memory_summaries[-1]  # Default to most recent

    def _trim_to_token_limit(self, messages: List[BaseMessage], token_limit: int) -> List[BaseMessage]:
        """Ensure the context doesn't exceed the token limit"""
        # Simple token estimation (4 chars ≈ 1 token)
        estimated_tokens = sum(len(msg.content) // 4 + 5 for msg in messages)

        if estimated_tokens <= token_limit:
            return messages

        # If we're over the limit, keep removing older non-system messages until we're under the limit
        # But always preserve the most recent messages
        preserved_count = min(ALWAYS_INCLUDE_RECENT, len(messages))
        removable_messages = messages[:-preserved_count]
        must_keep_messages = messages[-preserved_count:]

        # Keep removing until we're under the limit
        while removable_messages and estimated_tokens > token_limit:
            # Remove a message but never remove the system message at position 0
            if len(removable_messages) > 1 or not isinstance(removable_messages[0], SystemMessage):
                removed_msg = removable_messages.pop(1 if isinstance(removable_messages[0], SystemMessage) else 0)
                estimated_tokens -= (len(removed_msg.content) // 4 + 5)
            else:
                break

        return removable_messages + must_keep_messages

    def save_to_file(self) -> None:
        """Save the memory state to a file"""
        file_name = f"{self.session_id}_memory.json"

        # Serialize the messages
        serialized_data = {
            "messages": [
                {
                    "role": "system" if isinstance(msg, SystemMessage) else
                    "human" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                }
                for msg in self.full_history.messages
            ],
            "summaries": self.memory_summaries,
            "topics": self.conversation_topics,
            "last_summarized_idx": self.last_summarized_idx
        }

        with open(file_name, "w") as file:
            json.dump(serialized_data, file, indent=4)

    def load_from_file(self) -> bool:
        """Load memory state from a file"""
        file_name = f"{self.session_id}_memory.json"

        if not os.path.exists(file_name):
            return False

        try:
            with open(file_name, "r") as file:
                data = json.load(file)

            # Clear existing history
            self.full_history = ChatMessageHistory()

            # Load messages
            for msg in data["messages"]:
                if msg["role"] == "system":
                    self.full_history.add_message(SystemMessage(content=msg["content"]))
                elif msg["role"] == "human":
                    self.full_history.add_message(HumanMessage(content=msg["content"]))
                else:  # assistant
                    self.full_history.add_message(AIMessage(content=msg["content"]))

            # Load other data
            self.memory_summaries = data["summaries"]
            self.conversation_topics = data["topics"]
            self.last_summarized_idx = data["last_summarized_idx"]

            return True
        except Exception as e:
            print(f"Error loading memory from file: {e}")
            return False


# ------------------- Memory Manager Integration -------------------

# Dictionary to store memory managers for different sessions
memory_managers: Dict[str, SmartMemoryManager] = {}


def get_or_create_memory_manager(session_id: str) -> SmartMemoryManager:
    """Get existing memory manager or create a new one"""
    if session_id not in memory_managers:
        memory_managers[session_id] = SmartMemoryManager(session_id)
        # Try to load existing memory
        memory_managers[session_id].load_from_file()
    return memory_managers[session_id]


# Function to provide relevant context for the current query
def get_relevant_context(session_id: str, query: str) -> BaseChatMessageHistory:
    """Get a chat history with relevant context for the given query"""
    manager = get_or_create_memory_manager(session_id)
    relevant_messages = manager.get_relevant_context(query)

    # Create a new chat history with just the relevant messages
    context_history = ChatMessageHistory()
    for msg in relevant_messages:
        context_history.add_message(msg)

    return context_history


# Custom get_session_history function for the runnable
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get session history for the LangChain runnable"""
    # We'll always retrieve the full history here, then filter it in the graph
    manager = get_or_create_memory_manager(session_id)
    return manager.full_history


# Update chat history after a response
def update_chat_history(session_id: str, user_input: str, ai_response: str) -> None:
    """Add the user input and AI response to the chat history"""
    manager = get_or_create_memory_manager(session_id)
    manager.add_message(HumanMessage(content=user_input))
    manager.add_message(AIMessage(content=ai_response))
    manager.save_to_file()


# ------------------- LangGraph Implementation -------------------

# Define our state
class ConversationState(dict):
    """State for our conversation graph"""

    def __init__(self, session_id: str, user_input: str):
        self.update({
            "session_id": session_id,
            "user_input": user_input,
            "relevant_context": [],
            "ai_response": "",
        })


# Define the nodes in our graph
def retrieve_context(state: ConversationState) -> ConversationState:
    """Retrieve relevant context for the current query"""
    session_id = state["session_id"]
    user_input = state["user_input"]

    manager = get_or_create_memory_manager(session_id)
    relevant_messages = manager.get_relevant_context(user_input)

    # Debug info
    print(f"\n--- Retrieved {len(relevant_messages)} relevant messages for context ---")

    # Update state
    state["relevant_context"] = relevant_messages
    return state


def generate_response(state: ConversationState) -> ConversationState:
    """Generate a response using the LLM"""
    # Prepare the messages for the LLM
    messages = [
        SystemMessage(content="You are a helpful assistant. Respond based on the conversation context provided."),
    ]

    # Add relevant context
    messages.extend(state["relevant_context"])

    # Add the current user input
    messages.append(HumanMessage(content=state["user_input"]))

    # Print the full context window being sent to the LLM (for debugging)
    print(format_context_window(messages))

    # Generate the response
    response = response_llm.invoke(messages)

    # Update state
    state["ai_response"] = response.content
    return state


def update_memory(state: ConversationState) -> ConversationState:
    """Update the memory with the new interaction"""
    update_chat_history(
        state["session_id"],
        state["user_input"],
        state["ai_response"]
    )
    return state


# Format context window for debugging
def format_context_window(messages):
    context_str = "\n" + "=" * 50 + "\n"
    context_str += "CONTEXT WINDOW SENT TO LLM:\n"
    context_str += "=" * 50 + "\n\n"

    for msg in messages:
        if isinstance(msg, SystemMessage):
            context_str += f"[SYSTEM]: {msg.content}\n\n"
        elif isinstance(msg, HumanMessage):
            context_str += f"[HUMAN]: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            context_str += f"[ASSISTANT]: {msg.content}\n\n"
        else:
            context_str += f"[{type(msg).__name__}]: {msg.content}\n\n"

    context_str += "=" * 50 + "\n"
    return context_str


# Build the graph
def build_conversation_graph():
    """Build the LangGraph for conversation"""
    # Create the graph
    graph_builder = StateGraph(ConversationState)

    # Add nodes
    graph_builder.add_node("retrieve_context", retrieve_context)
    graph_builder.add_node("generate_response", generate_response)
    graph_builder.add_node("update_memory", update_memory)

    # Define the edges
    graph_builder.add_edge("retrieve_context", "generate_response")
    graph_builder.add_edge("generate_response", "update_memory")
    graph_builder.add_edge("update_memory", END)

    # Set the entry point
    graph_builder.set_entry_point("retrieve_context")

    # Create the graph
    graph = graph_builder.compile()

    # Replace MemoryCheckpoint with MemorySaver
    checkpoint = MemorySaver()
    graph_with_memory = graph.with_checkpoint(checkpoint)

    return graph_with_memory


# Create the conversation graph
conversation_graph = build_conversation_graph()


# ------------------- Main Application Logic -------------------

def start_conversation():
    """Start a conversation with the AI"""
    session_id = input("Enter a session ID (or press Enter for default): ").strip() or "default_session"

    print("Conversation started. Type 'exit' or 'quit' to end the conversation gracefully.")
    try:
        # Try to load existing session if available
        memory_manager = get_or_create_memory_manager(session_id)

        # Add system message if this is a new session
        if len(memory_manager.full_history.messages) == 0:
            memory_manager.add_message(SystemMessage(content="You are a helpful assistant."))
            memory_manager.save_to_file()

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Ending conversation. Saving your session...")
                memory_managers[session_id].save_to_file()
                print("Session saved. Goodbye!")
                break

            print(f"\n--- Processing user input: '{user_input}' ---")

            # Initialize the state and invoke the graph
            initial_state = ConversationState(session_id=session_id, user_input=user_input)
            final_state = conversation_graph.invoke(initial_state)

            # Print the assistant's response
            print(f"Assistant: {final_state['ai_response']}")

    except KeyboardInterrupt:
        print("\nDetected interruption. Saving your session...")
        if session_id in memory_managers:
            memory_managers[session_id].save_to_file()
        print("Session saved. Goodbye!")


# ------------------- Entry Point -------------------

if __name__ == "__main__":
    start_conversation()