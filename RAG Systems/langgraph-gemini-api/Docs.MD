# RAG System Integration Plan: Web App with Vue Frontend, Python LangGraph Backend, and Supabase

## Current System Overview

### Python Backend (LangGraph API)
- **Input:** Expects `session_id` and `user_input`.
- **Context Loading:** 
  - `load_and_prepare_history_node` loads history from local JSON files (`<session_id>_history.json`).
- **Context Updating:**
  - `update_and_save_history_node` appends new conversation turns and saves to local JSON.
- **Context Usage:**
  - Used for LangGraph’s pondering logic and context generation.
  
### Vue Frontend
- **Session Management:** Likely maintains its own `session_id` or user-linked identifier.
- **Data Storage:** Stores conversation turns (user, AI, timestamps, etc.) in Supabase.
- **Data Usage:** Displays history and conversation UI.

---

## Integration Options for Conversation History

### Option 1: Backend as Source of Truth (Simpler Change)

#### How it Works
1. **Message Flow:**
   - Vue app sends `session_id` + `user_input` → Python API (`/chat`).
   - Python API processes via LangGraph using local JSON history.
   - Python API returns `ai_response`.

2. **Frontend Storage:**
   - Vue saves user message + AI response to Supabase (UI display, long-term persistence).

#### Pros
- Minimal changes to existing backend.
- Encapsulates LangGraph logic and context within the Python layer.
- Leverages existing working memory model (JSON).

#### Cons
- **Data Duplication:** History is stored both in local JSON (temporary) and Supabase (persistent).
- **Session Sync Issue:** New device/browser = no local `_history.json`, so backend lacks context.

#### Addressing the Session Sync Limitation
- Vue fetches history from Supabase on session load.
- Add a new backend endpoint: `/prime_session_history`.

##### `/prime_session_history` Endpoint
- Accepts: `session_id`, `history` (last N turns from Supabase).
- Overwrites/creates local `<session_id>_history.json`.
- Used to rehydrate backend context after server restart or new instance.

---

### Option 2: Supabase as Source of Truth (More Robust, More Change)

#### How it Works
1. **Python Backend Updates:**
   - Use `supabase-py` client.
   - `load_and_prepare_history_node` queries Supabase instead of local files.
   - `update_and_save_history_node` writes new turns to Supabase.

2. **Message Flow:**
   - Vue sends `session_id` + `user_input` → Python API.
   - Backend pulls conversation history from Supabase.
   - Backend processes LangGraph flow.
   - Backend saves both messages to Supabase.
   - Vue receives and displays `ai_response`.

#### Pros
- **Single Source of Truth:** No duplication.
- **Device Persistence:** History is always available, regardless of device/session.
- **Scalability:** Works well with stateless/cloud deployments.

#### Cons
- **Backend Refactor Required:**
  - Integrate Supabase client.
  - Replace file-based history functions with DB operations.
- **Latency:** DB I/O instead of file I/O (minor if indexed well).
- **Security:** Backend requires secure handling of Supabase API keys.

---

## Recommended Path Forward

### Phase 1: Basic Working Integration (Option 1)
- Keep using JSON files in Python backend.
- Vue sends:
  - `session_id`
  - `user_input`
- Python:
  - Uses local `_history.json`.
  - Returns AI response.
- Vue:
  - Displays AI response.
  - Stores both user and AI messages to Supabase.

**Limitation:** Backend will not have history on new device/server reboot.

---

### Phase 2: Add Session Priming Endpoint

#### Backend
- Add `/prime_session_history` route.
- Accepts JSON: `{ session_id, messages: [...] }`.
- Converts messages into LangGraph-compatible format.
- Saves them as `_history.json`.

#### Frontend
- When user loads a session:
  - Vue fetches full history from Supabase.
  - If local `_history.json` doesn’t exist:
    - Vue sends last N messages to `/prime_session_history`.

---

### Phase 3: Migrate to Supabase-Backed History (Option 2)

#### Python Backend Changes
- Install `supabase-py`.
- Modify:
  - `load_messages_from_file` → `load_messages_from_supabase`
  - `save_messages_to_file` → `save_messages_to_supabase`
- Handle:
  - DB connection
  - Message mapping (Supabase schema ↔ LangGraph `BaseMessage` list)

#### Security
- Store Supabase credentials as environment variables.

---

## Supabase Table Schema Suggestions

| Column         | Type        | Description                          |
|----------------|-------------|--------------------------------------|
| `id`           | UUID        | Unique ID                            |
| `session_id`   | Text        | Identifier per conversation session  |
| `role`         | Enum/Text   | 'user' or 'assistant'                |
| `content`      | Text        | Message text                         |
| `timestamp`    | Timestamp   | Message creation time                |

---

## Summary of Integration Stages

| Stage         | Backend Memory | Frontend Role           | Backend Role                            |
|---------------|----------------|--------------------------|------------------------------------------|
| Phase 1       | Local JSON     | Display + Save to DB     | Use local JSON only                      |
| Phase 2       | JSON + Priming | Prime session when needed| Accept external history                  |
| Phase 3       | Supabase DB    | Optional Save (UI only)  | Full fetch/write from/to Supabase        |

