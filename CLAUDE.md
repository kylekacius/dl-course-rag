# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

**Prerequisites:**
- Python 3.13+
- uv package manager (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Anthropic API key

**Initial Setup:**
```bash
uv sync                           # Install dependencies
echo "ANTHROPIC_API_KEY=your_key_here" > .env  # Add API key
```

**Running the Application:**
```bash
./run.sh                          # Quick start script
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Application URLs:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Query API: POST http://localhost:8000/api/query
- Course Stats: GET http://localhost:8000/api/courses

## System Architecture

This is a **Retrieval-Augmented Generation (RAG)** system that processes course materials and answers questions using semantic search + Claude AI.

### Core Data Flow

**Document Ingestion:**
`docs/*.txt` → `DocumentProcessor` → `VectorStore` → `ChromaDB`

**Query Processing:**
`User Query` → `RAGSystem` → `AIGenerator` + `SearchTools` → `Claude API` → `Response + Sources`

### Key Components

**Backend Structure (`/backend/`):**
- `app.py` - FastAPI web server with CORS, static file serving, startup document loading
- `rag_system.py` - Main orchestrator coordinating all components, handles document ingestion and query processing
- `document_processor.py` - Parses structured course documents, extracts metadata, creates sentence-based chunks (800 chars, 100 overlap)
- `vector_store.py` - ChromaDB interface with two collections: `course_catalog` (metadata) and `course_content` (searchable chunks)
- `ai_generator.py` - Claude API integration with tool-based search, conversation history management
- `search_tools.py` - Semantic search functionality using vector similarity, course/lesson filtering
- `session_manager.py` - Conversation tracking with configurable history retention
- `models.py` - Pydantic data models: `Course`, `Lesson`, `CourseChunk`
- `config.py` - Environment-based configuration (API keys, chunk sizes, limits)

**Frontend (`/frontend/`):**
- Static HTML/CSS/JS served by FastAPI
- Interactive chat interface for querying course materials

### Document Format

Course documents in `/docs/` follow this structured format:
```
Course Title: [Course Name]
Course Link: [URL] 
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[lesson content...]

Lesson 1: [Next Lesson]
[more content...]
```

### Vector Storage Architecture

**ChromaDB Collections:**
- `course_catalog`: Course metadata, instructor info, lesson structure (JSON serialized)  
- `course_content`: Text chunks with course/lesson context for semantic search

**Embeddings:** Uses `all-MiniLM-L6-v2` sentence transformer model

**Search Strategy:** 
1. Semantic search across content chunks
2. Optional filtering by course name/lesson number  
3. Course name resolution via vector similarity
4. Returns top 5 results by default

### AI Integration

**Tool-Based Approach:**
- Claude receives `CourseSearchTool` for dynamic content retrieval
- AI can search, filter, and cite sources automatically
- Conversation history maintained per session (max 2 exchanges)

**Model Configuration:**
- Claude: `claude-sonnet-4-20250514`
- Embeddings: `all-MiniLM-L6-v2`
- Chunk size: 800 characters with 100 character overlap

## Configuration

**Environment Variables (.env):**
```
ANTHROPIC_API_KEY=your_api_key_here
```

**Key Settings (config.py):**
- `CHUNK_SIZE=800` - Text chunk size for vector storage
- `CHUNK_OVERLAP=100` - Character overlap between chunks  
- `MAX_RESULTS=5` - Maximum search results returned
- `MAX_HISTORY=2` - Conversation messages to retain
- `CHROMA_PATH="./chroma_db"` - Vector database storage location

## Development Patterns

**Adding New Documents:**
1. Place `.txt`/`.pdf`/`.docx` files in `/docs/` directory
2. Follow the structured course format above
3. Restart the server to trigger document ingestion
4. System automatically parses, chunks, and indexes new content

**Database Management:**
- ChromaDB persists in `backend/chroma_db/`
- System avoids re-processing existing courses by title
- Clear database: delete `backend/chroma_db/` directory and restart

**API Extension:**
- Add endpoints in `app.py` 
- Extend search functionality in `search_tools.py`
- Modify document processing logic in `document_processor.py`

**Session Management:**
- Sessions auto-created if not provided in API calls
- Conversation history enables contextual follow-up questions
- Session data stored in memory (not persisted)

## Project Structure Context

```
├── backend/           # Core Python application
├── frontend/          # Static web interface (HTML/CSS/JS)
├── docs/             # Course documents for RAG system
├── pyproject.toml    # uv dependency management
├── run.sh            # Application startup script
└── .env              # API keys and configuration
```

This RAG system demonstrates production-ready patterns for document ingestion, semantic search, and AI-powered question answering with proper separation of concerns across specialized components.