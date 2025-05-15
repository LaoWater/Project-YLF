# Start with: uvicorn api_server:api --reload
# In ragvenv environment to host API server from local network.


import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List # We might need these later if we expand responses

# Import the compiled LangGraph app and necessary message types from your chat_logic
from chat_logic import app as langgraph_app # Rename to avoid conflict
from chat_logic import ShortTermMemoryState, AIMessage # Import your state and AIMessage

# --- Pydantic Models for API Request and Response ---
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    ai_response: str
    session_id: str # Good to send back for confirmation

# --- FastAPI Application ---
api = FastAPI(
    title="LangGraph Gemini Chat API",
    description="An API for interacting with a LangGraph-powered Gemini chat model with short-term memory.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows your frontend (running on a different port/domain) to talk to this API
# Adjust origins if your frontend will run on a different port during development
origins = [
    "http://localhost",         # Common default
    "http://localhost:3000",    # Common for React dev
    "http://localhost:5173",    # Common for Vite (Vue/React) dev
    "http://localhost:8080",    # Common for Vue CLI dev
    # Add any other origins your frontend might use
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---
@api.get("/", summary="Root endpoint", description="A simple health check endpoint.")
async def read_root():
    return {"status": "LangGraph Gemini Chat API is running!"}

@api.post("/chat", response_model=ChatResponse, summary="Process a chat message",
          description="Sends user input to the LangGraph agent and returns the AI's response.")
async def handle_chat_message(request: ChatRequest):
    """
    Handles a chat message from the user.

    - **session_id**: A unique identifier for the conversation session.
    - **user_input**: The text message from the user.
    """
    print(f"\n--- API Request Received ---")
    print(f"Session ID: {request.session_id}")
    print(f"User Input: '{request.user_input}'")

    if not request.user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    try:
        # Prepare the initial state for the LangGraph run
        # This matches the 'ShortTermMemoryState' TypedDict from your chat_logic.py
        initial_state_for_run: ShortTermMemoryState = {
            "session_id": request.session_id,
            "user_input": request.user_input,
            "full_history_messages": [],  # Loaded by load_and_prepare_history_node
            "recent_history_for_ponder": [],
            "relevant_summary": "",
            "main_llm_input_messages": [],
            "final_response": AIMessage(content="") # Ensure AIMessage is imported from chat_logic or langchain_core.messages
        }

        # Configuration for the LangGraph invocation
        # (Not strictly needed for your current graph if session_id is part of the state,
        # but good practice if LangGraph uses it for checkpointing or other features)
        config = {"configurable": {"session_id": request.session_id}}

        print("Invoking LangGraph application...")
        # Invoke the LangGraph application
        # 'langgraph_app' is the compiled graph from chat_logic.py
        final_state = langgraph_app.invoke(initial_state_for_run, config=config)

        if not final_state or "final_response" not in final_state or not final_state["final_response"]:
            print("Error: LangGraph did not return a final_response or it was empty.")
            raise HTTPException(status_code=500, detail="AI agent failed to produce a response.")

        ai_response_content = final_state["final_response"].content
        print(f"AI Response: '{ai_response_content[:200]}...'") # Log a snippet

        return ChatResponse(
            ai_response=ai_response_content,
            session_id=request.session_id
        )

    except Exception as e:
        print(f"!!! An error occurred during LangGraph execution: {str(e)}")
        import traceback
        traceback.print_exc() # This will print the full traceback to your server console
        # For production, you might want to log this to a file or monitoring service
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- To run this FastAPI application (from your terminal, in the project root): ---
# Make sure your virtual environment is activated: source venv/bin/activate (or .\venv\Scripts\activate on Windows)
# Then run: uvicorn api_server:api --reload
#
# Open your browser to http://127.0.0.1:8000/docs to see the interactive API documentation.