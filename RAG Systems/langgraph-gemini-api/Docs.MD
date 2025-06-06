# RAG System Integration Plan: Web App with Vue Frontend, Python LangGraph Backend, and Supabase

## Current System Overview

### Python Backend (LangGraph API)
- **Deployment**: Hosted on Google Cloud Run at `https://langgraph-api-<project-id>.<region>.run.app/` (e.g., `europe-central2`).
- **Input**: Expects `session_id` and `user_input` via `POST` to `/chat`.
- **Context Loading**: 
  - `load_and_prepare_history_node` loads history from local JSON files (`<session_id>_history.json`).
- **Context Updating**:
  - `update_and_save_history_node` appends new conversation turns and saves to local JSON.
- **Context Usage**:
  - Used for LangGraph’s pondering logic and context generation.
- **API Endpoints**:
  - `GET /`: Returns `{"status": "LangGraph Gemini Chat API is running!"}` to confirm API health.
  - `POST /chat`: Accepts JSON payload (e.g., `{"session_id": "123", "message": "Hello"}`) and returns AI response.
- **CORS Configuration**:
  - FastAPI middleware allows cross-origin requests (e.g., `allow_origins=["*"]` for development).
- **Port**: Runs on port `8000` (configured in Cloud Run).

### Vue Frontend
- **Session Management**: Maintains `session_id` or user-linked identifier.
- **API Integration**: Sends `POST` requests to `https://langgraph-api-<project-id>.<region>.run.app/chat`.
- **Data Storage**: Stores conversation turns (user, AI, timestamps, etc.) in Supabase.
- **Data Usage**: Displays history and conversation UI.

---

## Deployment on Google Cloud Run

### Prerequisites
- Google Cloud project (e.g., `terapie-acasa`).
- `gcloud` CLI installed and authenticated.
- Docker installed for building the container image.
- FastAPI app with LangGraph logic in a `Dockerfile`.

### Deployment Steps
1. **Build and Push Docker Image**:
   ```bash
   gcloud builds submit --tag gcr.io/terapie-acasa/langgraph-api
   ```
   - Builds the Docker image and pushes it to Google Container Registry (GCR).
   - Ensure the `Dockerfile` includes FastAPI, LangGraph dependencies, and exposes port `8000`.

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy langgraph-api --image gcr.io/terapie-acasa/langgraph-api --platform managed --region europe-central2 --allow-unauthenticated --port 8000
   ```
   - Deploys the image as a managed Cloud Run service.
   - `--allow-unauthenticated`: Allows public access.
   - `--port 8000`: Matches the FastAPI app’s port.
   - Output: Service URL (e.g., `https://langgraph-api-39620233431.europe-central2.run.app/`).

3. **Grant Logging Permissions (Optional)**:
   - If Cloud Build warns about missing logging permissions, grant the `Logs Writer` role:
   ```bash
   gcloud projects add-iam-policy-binding terapie-acasa --member="serviceAccount:<project-number>-compute@developer.gserviceaccount.com" --role="roles/logging.logWriter"
   ```

### FastAPI Backend Code
Example `main.py` for the LangGraph API:
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "LangGraph Gemini Chat API is running!"}

@app.post("/chat")
async def chat(request: dict):
    session_id = request.get("session_id")
    message = request.get("message")
    if not session_id or not message:
        raise HTTPException(status_code=400, detail="session_id and message are required")
    # LangGraph logic here (load history, process, save history)
    return {"response": "AI response"}  # Replace with actual LangGraph output
```

### Testing the Deployment
- **Health Check**: `curl https://langgraph-api-<project-id>.<region>.run.app/`
  - Expected: `{"status": "LangGraph Gemini Chat API is running!"}`
- **Chat Endpoint**: 
  ```bash
  curl -X POST https://langgraph-api-<project-id>.<region>.run.app/chat -H "Content-Type: application/json" -d '{"session_id": "123", "message": "Hello"}'
  ```
  - Expected: `{"response": "<AI response>"}`

---

## Integration Options for Conversation History

### Option 1: Backend as Source of Truth (Simpler Change)

#### How it Works
1. **Message Flow**:
   - Vue app sends `session_id` + `user_input` to `POST /chat`.
   - Python API processes via LangGraph using local JSON history.
   - Python API returns `ai_response`.
2. **Frontend Storage**:
   - Vue saves user message + AI response to Supabase for UI display and persistence.

#### Pros
- Minimal changes to existing backend.
- Encapsulates LangGraph logic within Python.
- Leverages existing JSON-based memory model.

#### Cons
- **Data Duplication**: History stored in local JSON (temporary) and Supabase (persistent).
- **Session Sync Issue**: New device or server restart loses local `_history.json`.

#### Addressing Session Sync Limitation
- Vue fetches history from Supabase on session load.
- Add `/prime_session_history` endpoint to rehydrate backend context.

##### `/prime_session_history` Endpoint
- **Accepts**: `session_id`, `history` (last N turns from Supabase).
- **Action**: Overwrites/creates local `<session_id>_history.json`.
- **Usage**: Called after server restart or new session.

---

### Option 2: Supabase as Source of Truth (More Robust)

#### How it Works
1. **Python Backend Updates**:
   - Use `supabase-py` client.
   - `load_and_prepare_history_node` queries Supabase instead of JSON files.
   - `update_and_save_history_node` writes to Supabase.
2. **Message Flow**:
   - Vue sends `session_id` + `user_input` to `/chat`.
   - Backend pulls history from Supabase.
   - Backend processes LangGraph flow.
   - Backend saves messages to Supabase.
   - Vue displays `ai_response`.

#### Pros
- **Single Source of Truth**: No duplication.
- **Device Persistence**: History available across devices.
- **Scalability**: Suitable for stateless cloud deployments.

#### Cons
- **Backend Refactor**: Requires Supabase integration and DB operations.
- **Latency**: DB queries may be slower than file I/O (mitigated with indexing).
- **Security**: Must secure Supabase API keys.

---

## Recommended Path Forward

### Phase 1: Basic Working Integration (Option 1)
- **Backend**: Use local JSON files.
- **Vue**:
  - Sends `session_id` + `user_input` to `/chat`.
  - Stores user and AI messages in Supabase.
- **Limitation**: Backend lacks history on new devices or after restarts.

### Phase 2: Add Session Priming Endpoint
- **Backend**:
  - Add `/prime_session_history` route.
  - Accepts `{ session_id, messages: [...] }`.
  - Saves messages as `_history.json`.
- **Frontend**:
  - On session load, fetch history from Supabase.
  - If `_history.json` missing, send history to `/prime_session_history`.

### Phase 3: Migrate to Supabase-Backed History (Option 2)
- **Backend**:
  - Install `supabase-py`.
  - Replace file-based functions:
    - `load_messages_from_file` → `load_messages_from_supabase`
    - `save_messages_to_file` → `save_messages_to_supabase`
  - Manage DB connection and message mapping.
- **Security**: Use environment variables for Supabase credentials.

---

---

## Summary of Integration Stages

| Stage         | Backend Memory | Frontend Role           | Backend Role                            |
|---------------|----------------|--------------------------|------------------------------------------|
| Phase 1       | Local JSON     | Display + Save to DB     | Use local JSON only                      |
| Phase 2       | JSON + Priming | Prime session when needed| Accept external history                  |
| Phase 3       | Supabase DB    | Optional Save (UI only)  | Full fetch/write from/to Supabase        |

---