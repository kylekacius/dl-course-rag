#!/usr/bin/env python3
"""
RAG System Flow Visualization
Text-based diagram showing how the Course Materials RAG system works
"""

def print_section(title, content="", separator="="):
    print(f"\n{separator * 60}")
    print(f" {title}")
    print(f"{separator * 60}")
    if content:
        print(content)

def main():
    print("🎓 COURSE MATERIALS RAG SYSTEM - ARCHITECTURE OVERVIEW")
    
    print_section("📄 DOCUMENT INGESTION FLOW", """
    1. Raw Documents (.txt, .pdf, .docx)
       ↓
    2. DocumentProcessor (document_processor.py)
       • Parses course metadata (title, instructor, links)
       • Identifies lessons using regex: "Lesson \\d+:"
       • Chunks content into 800-char pieces with 100-char overlap
       ↓
    3. VectorStore (vector_store.py)
       • Creates embeddings using 'all-MiniLM-L6-v2'
       • Stores in ChromaDB with 2 collections:
         - course_catalog: Course/lesson metadata
         - course_content: Searchable text chunks
    """)
    
    print_section("🔍 QUERY PROCESSING FLOW", """
    User Query: "How does retrieval work in RAG systems?"
       ↓
    1. FastAPI Endpoint (/api/query)
       ↓
    2. RAG System Orchestrator (rag_system.py)
       ↓
    3. AI Generator (ai_generator.py)
       • Sends query + tools to Claude API
       • Uses CourseSearchTool for semantic search
       ↓
    4. Search Tools (search_tools.py)
       • Performs vector search in ChromaDB
       • Filters by course/lesson if specified
       • Returns relevant text chunks
       ↓
    5. Claude AI Response
       • Generates answer using retrieved context
       • Cites sources from search results
       ↓
    6. Return to User (JSON response with answer + sources)
    """)
    
    print_section("🏗️ CORE COMPONENTS", """
    ┌─────────────────┬──────────────────┬─────────────────────┐
    │ FILE            │ PURPOSE          │ KEY FEATURES        │
    ├─────────────────┼──────────────────┼─────────────────────┤
    │ app.py          │ Web Server       │ FastAPI, CORS, APIs │
    │ rag_system.py   │ Main Orchestr.   │ Coordinates all     │
    │ document_proc.  │ Text Processing  │ Parse, chunk, meta  │
    │ vector_store.py │ Vector Database  │ ChromaDB interface  │
    │ ai_generator.py │ AI Integration   │ Claude API calls    │
    │ search_tools.py │ Search Logic     │ Vector similarity   │
    │ models.py       │ Data Structures  │ Course, Lesson, etc │
    │ config.py       │ Settings         │ API keys, params    │
    └─────────────────┴──────────────────┴─────────────────────┘
    """)
    
    print_section("💾 DATA STRUCTURE", """
    Course Document Format:
    ┌────────────────────────────────────────┐
    │ Course Title: Advanced AI Course      │
    │ Course Link: https://...              │
    │ Course Instructor: Dr. Smith          │
    │                                        │
    │ Lesson 1: Introduction to AI          │
    │ Lesson Link: https://...              │
    │ [lesson content here...]               │
    │                                        │
    │ Lesson 2: Machine Learning Basics     │
    │ [more lesson content...]               │
    └────────────────────────────────────────┘
    
    Vector Storage:
    • course_catalog: {title, instructor, lessons_json}
    • course_content: {content, course_title, lesson_number, chunk_index}
    """)
    
    print_section("⚙️ TECHNICAL STACK", """
    Backend:
    • FastAPI - Web framework and REST APIs
    • ChromaDB - Vector database for embeddings
    • Pydantic - Data validation and models
    • python-dotenv - Environment variables
    
    AI & ML:
    • Anthropic Claude - Language model (claude-sonnet-4-20250514)
    • Sentence Transformers - Text embeddings (all-MiniLM-L6-v2)
    
    Configuration:
    • Chunk Size: 800 characters
    • Chunk Overlap: 100 characters  
    • Max Results: 5 per search
    • Max History: 2 conversation turns
    """)
    
    print_section("🔄 LIVE SYSTEM STATUS", """
    Currently Running:
    • Server: http://localhost:8000
    • Loaded Courses: 4
    • Total Chunks: 528
    
    Courses:
    1. Advanced Retrieval for AI with Chroma (90 chunks)
    2. Prompt Compression and Query Optimization (121 chunks)  
    3. Building Towards Computer Use with Anthropic (153 chunks)
    4. MCP: Build Rich-Context AI Apps with Anthropic (164 chunks)
    """)
    
    print_section("🌊 REQUEST FLOW EXAMPLE", """
    [User] "What is retrieval augmented generation?"
        ↓
    [FastAPI] POST /api/query {"query": "...", "session_id": "abc123"}
        ↓
    [RAGSystem] Creates tools, gets conversation history
        ↓
    [AIGenerator] Sends to Claude: query + CourseSearchTool
        ↓
    [Claude] Calls search_course_content("retrieval augmented generation")
        ↓
    [SearchTool] Vector search in ChromaDB → relevant chunks
        ↓
    [Claude] Generates response using chunks as context
        ↓
    [RAGSystem] Returns {"answer": "...", "sources": [...]}
        ↓
    [User] Receives AI response with source citations
    """)

if __name__ == "__main__":
    main()