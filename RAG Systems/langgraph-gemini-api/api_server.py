# Start with: uvicorn api_server:api --reload
# In ragvenv environment to host API server from local network.


import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

# Import the compiled LangGraph app and necessary message types from your chat_logic
from chat_logic import app as langgraph_app # Rename to avoid conflict
# Also import the state definition and AIMessage
from chat_logic import ShortTermMemoryState, AIMessage, SYSTEM_PROMPTS # Import SYSTEM_PROMPTS for validation

# --- Pydantic Models for API Request and Response ---
class ChatRequest(BaseModel):
    session_id: str
    user_input: str
    model_id: str = "general" # <-- Added model_id with a default value

class ChatResponse(BaseModel):
    ai_response: str
    session_id: str
    model_id: str # Good to send back the model_id used

# --- FastAPI Application ---
api = FastAPI(
    title="LangGraph Gemini Chat API",
    description="An API for interacting with a LangGraph-powered Gemini chat model with short-term memory, supporting multiple personas.",
    version="1.1.0" # Updated version
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "https://terapie-acasa.ro",  # PRODUCTION FRONTEND ORIGIN

    # Add any other origins your frontend might use
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@api.get("/", summary="Root endpoint", description="A simple health check endpoint.")
async def read_root():
    return {"status": "LangGraph Gemini Chat API is running!"}

@api.get("/models", summary="List available models", description="Returns a list of available AI model IDs and descriptions.")
async def get_available_models():
     # This is a simple mapping, you might want a more detailed structure
     # including descriptions if not already in the system prompt keys
     model_list = [
         {"id": key, "description": SYSTEM_PROMPTS[key][:100] + "..." if len(SYSTEM_PROMPTS[key]) > 100 else SYSTEM_PROMPTS[key]}
         for key in SYSTEM_PROMPTS.keys()
     ]
     # Add friendly names based on your Vue app's definition (optional, could be hardcoded or configured)
     friendly_names = {
         'general': 'Asistent General',
         'zen': 'Maestru Zen',
         'therapist': 'Psihoterapeut',
         'couples': 'Terapeut de Cuplu',
         'nlp': 'Practician NLP'
     }
     for model in model_list:
          model['name'] = friendly_names.get(model['id'], model['id']) # Add friendly name

     return model_list


@api.post("/chat", response_model=ChatResponse, summary="Process a chat message",
          description="Sends user input to the LangGraph agent and returns the AI's response based on the selected model.")
async def handle_chat_message(request: ChatRequest):
    """
    Handles a chat message from the user.

    - **session_id**: A unique identifier for the conversation session.
    - **user_input**: The text message from the user.
    - **model_id**: The identifier for the desired AI model/persona (e.g., 'general', 'zen'). Defaults to 'general'.
    """
    print(f"\n--- API Request Received ---")
    print(f"Session ID: {request.session_id}")
    print(f"User Input: '{request.user_input}'")
    print(f"Requested Model ID: '{request.model_id}'")


    if not request.user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")
    # Optional: Validate model_id against available prompts
    if request.model_id not in SYSTEM_PROMPTS:
         print(f"Warning: Requested model_id '{request.model_id}' not found. Using default 'general'.")
         # Could raise HTTPException or just fallback
         # raise HTTPException(status_code=400, detail=f"Invalid model_id: {request.model_id}")
         effective_model_id = "general" # Fallback to default
    else:
        effective_model_id = request.model_id


    try:
        # Prepare the initial state for the LangGraph run
        initial_state_for_run: ShortTermMemoryState = {
            "session_id": request.session_id,
            "user_input": request.user_input,
            "model_id": effective_model_id, # <-- Pass the validated model_id
            "full_history_messages": [],  # Loaded by load_and_prepare_history_node
            "recent_history_for_ponder": [],
            "relevant_summary": "",
            "main_llm_input_messages": [],
            "final_response": AIMessage(content="")
        }

        # Configuration for the LangGraph invocation
        config = {"configurable": {"session_id": request.session_id}}

        print("Invoking LangGraph application...")
        # Invoke the LangGraph application
        final_state = langgraph_app.invoke(initial_state_for_run, config=config)

        if not final_state or "final_response" not in final_state or not final_state["final_response"]:
            print("Error: LangGraph did not return a final_response or it was empty.")
            # Check if an error occurred within a node and potentially surfaced in the state
            error_detail = "AI agent failed to produce a response."
            if "error" in final_state: # If you added error handling to the state
                 error_detail += f" Internal graph error: {final_state['error']}"
            raise HTTPException(status_code=500, detail=error_detail)

        ai_response_content = final_state["final_response"].content
        print(f"AI Response: '{ai_response_content[:200]}...'") # Log a snippet

        return ChatResponse(
            ai_response=ai_response_content,
            session_id=request.session_id,
            model_id=effective_model_id # Return the model_id that was actually used
        )

    except Exception as e:
        print(f"!!! An error occurred during LangGraph execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- To run this FastAPI application (from your terminal, in the project root): ---
# Make sure your virtual environment is activated: source venv/bin/activate (or .\venv\Scripts\activate on Windows)
# Then run: uvicorn api_server:api --reload
#
# Open your browser to http://127.0.0.1:8000/docs to see the interactive API documentation.