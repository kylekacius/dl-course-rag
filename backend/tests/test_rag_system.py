"""
End-to-end tests for RAG system functionality
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import tempfile

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "test-model"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    MAX_HISTORY = 2


class TestRAGSystem:
    """Test RAG system end-to-end functionality"""
    
    def setup_method(self):
        """Set up test fixtures with mocked components"""
        self.mock_config = MockConfig()
        
        # Mock all the component dependencies
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr, \
             patch('rag_system.ToolManager') as mock_tool_mgr, \
             patch('rag_system.CourseSearchTool') as mock_search_tool, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool:
            
            # Set up mock instances
            self.mock_document_processor = Mock()
            self.mock_vector_store = Mock()
            self.mock_ai_generator = Mock()
            self.mock_session_manager = Mock()
            self.mock_tool_manager = Mock()
            self.mock_search_tool = Mock()
            self.mock_outline_tool = Mock()
            
            # Configure mock classes to return mock instances
            mock_doc_proc.return_value = self.mock_document_processor
            mock_vector_store.return_value = self.mock_vector_store
            mock_ai_gen.return_value = self.mock_ai_generator
            mock_session_mgr.return_value = self.mock_session_manager
            mock_tool_mgr.return_value = self.mock_tool_manager
            mock_search_tool.return_value = self.mock_search_tool
            mock_outline_tool.return_value = self.mock_outline_tool
            
            # Initialize RAG system
            self.rag_system = RAGSystem(self.mock_config)
    
    def test_initialization(self):
        """Test RAG system initialization"""
        assert self.rag_system.config == self.mock_config
        assert self.rag_system.document_processor == self.mock_document_processor
        assert self.rag_system.vector_store == self.mock_vector_store
        assert self.rag_system.ai_generator == self.mock_ai_generator
        assert self.rag_system.session_manager == self.mock_session_manager
        assert self.rag_system.tool_manager == self.mock_tool_manager
        
        # Verify tools were registered
        self.mock_tool_manager.register_tool.assert_any_call(self.mock_search_tool)
        self.mock_tool_manager.register_tool.assert_any_call(self.mock_outline_tool)
        assert self.mock_tool_manager.register_tool.call_count == 2
    
    def test_add_course_document_success(self):
        """Test adding a course document successfully"""
        # Mock document processing
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://test.com",
            lessons=[Lesson(1, "Lesson 1", "http://test.com/lesson1")]
        )
        mock_chunks = [
            CourseChunk("Test Course", 1, 0, "Test content chunk")
        ]
        
        self.mock_document_processor.process_course_document.return_value = (mock_course, mock_chunks)
        
        course, chunk_count = self.rag_system.add_course_document("test.txt")
        
        # Verify processing chain
        self.mock_document_processor.process_course_document.assert_called_once_with("test.txt")
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)
        
        assert course == mock_course
        assert chunk_count == 1
    
    def test_add_course_document_failure(self):
        """Test handling of document processing failure"""
        self.mock_document_processor.process_course_document.side_effect = Exception("Processing error")
        
        course, chunk_count = self.rag_system.add_course_document("invalid.txt")
        
        assert course is None
        assert chunk_count == 0
        
        # Verify vector store was not called
        self.mock_vector_store.add_course_metadata.assert_not_called()
        self.mock_vector_store.add_course_content.assert_not_called()
    
    def test_query_without_session(self):
        """Test query processing without session context"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "AI response about course content"
        
        # Mock tool manager sources
        mock_sources = [{"text": "Test Course - Lesson 1", "url": "http://test.com"}]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        response, sources = self.rag_system.query("What is RAG?")
        
        # Verify AI generator call
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        
        assert "Answer this question about course materials: What is RAG?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] == self.mock_tool_manager.get_tool_definitions.return_value
        assert call_args[1]["tool_manager"] == self.mock_tool_manager
        
        # Verify response
        assert response == "AI response about course content"
        assert sources == mock_sources
        
        # Verify sources were reset
        self.mock_tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session(self):
        """Test query processing with session context"""
        session_id = "test-session-123"
        mock_history = "Previous: What is AI?\nAssistant: AI stands for..."
        
        self.mock_session_manager.get_conversation_history.return_value = mock_history
        self.mock_ai_generator.generate_response.return_value = "Contextual AI response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query(
            "Tell me more about machine learning",
            session_id=session_id
        )
        
        # Verify session management
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator received history
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == mock_history
        
        # Verify conversation was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id,
            "Tell me more about machine learning",
            "Contextual AI response"
        )
        
        assert response == "Contextual AI response"
    
    def test_query_complete_flow_with_tools(self):
        """Test complete query flow including tool usage"""
        # Set up mocks for tool-based query
        self.mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        
        # Mock AI response that would use tools
        self.mock_ai_generator.generate_response.return_value = "Based on the search results, RAG systems..."
        
        # Mock sources from tool execution
        mock_sources = [
            {"text": "AI Fundamentals - Lesson 3", "url": "http://example.com/lesson3"},
            {"text": "Advanced RAG - Lesson 1", "url": "http://example.com/rag1"}
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        response, sources = self.rag_system.query("How do RAG systems work?")
        
        # Verify full tool chain
        self.mock_tool_manager.get_tool_definitions.assert_called_once()
        
        # Verify AI generator received tools
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["tools"] == [{"name": "search_course_content", "description": "Search course content"}]
        assert call_args[1]["tool_manager"] == self.mock_tool_manager
        
        # Verify sources management
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
        
        assert response == "Based on the search results, RAG systems..."
        assert sources == mock_sources
    
    def test_add_course_folder_with_existing_courses(self):
        """Test adding course folder with some existing courses"""
        # Mock existing courses
        self.mock_vector_store.get_existing_course_titles.return_value = ["Existing Course"]
        
        # Mock document processing for new courses
        new_course = Course(
            title="New Course",
            instructor="New Instructor", 
            course_link="http://new.com",
            lessons=[]
        )
        existing_course = Course(
            title="Existing Course",
            instructor="Existing Instructor",
            course_link="http://existing.com", 
            lessons=[]
        )
        new_chunks = [CourseChunk("New Course", 1, 0, "New content")]
        existing_chunks = [CourseChunk("Existing Course", 1, 0, "Existing content")]
        
        # Mock file processing
        self.mock_document_processor.process_course_document.side_effect = [
            (new_course, new_chunks),
            (existing_course, existing_chunks)
        ]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['new_course.txt', 'existing_course.txt']), \
             patch('os.path.isfile', return_value=True):
            
            total_courses, total_chunks = self.rag_system.add_course_folder("/test/docs")
        
        # Should only add the new course
        assert total_courses == 1
        assert total_chunks == 1
        
        # Verify only new course was added to vector store
        self.mock_vector_store.add_course_metadata.assert_called_once_with(new_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(new_chunks)
    
    def test_add_course_folder_clear_existing(self):
        """Test adding course folder with clear_existing flag"""
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=[]):
            
            self.rag_system.add_course_folder("/test/docs", clear_existing=True)
        
        # Verify data was cleared
        self.mock_vector_store.clear_all_data.assert_called_once()
    
    def test_add_course_folder_nonexistent_path(self):
        """Test adding course folder with non-existent path"""
        with patch('os.path.exists', return_value=False):
            total_courses, total_chunks = self.rag_system.add_course_folder("/nonexistent")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    def test_get_course_analytics(self):
        """Test getting course analytics"""
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        analytics = self.rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
    
    def test_tool_integration(self):
        """Test that tools are properly integrated with vector store"""
        # Verify search tool was initialized with vector store
        from rag_system import CourseSearchTool
        CourseSearchTool.assert_called_once_with(self.mock_vector_store)
        
        # Verify outline tool was initialized with vector store  
        from rag_system import CourseOutlineTool
        CourseOutlineTool.assert_called_once_with(self.mock_vector_store)


class TestRAGSystemErrorScenarios:
    """Test RAG system error handling scenarios"""
    
    def setup_method(self):
        """Set up test fixtures for error scenarios"""
        self.mock_config = MockConfig()
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            self.mock_ai_generator = Mock()
            mock_ai_gen.return_value = self.mock_ai_generator
            
            self.rag_system = RAGSystem(self.mock_config)
    
    def test_query_with_ai_generator_failure(self):
        """Test query handling when AI generator fails"""
        # This test documents current behavior - no exception handling
        self.mock_ai_generator.generate_response.side_effect = Exception("API Error")
        
        # Currently would raise exception - shows need for error handling
        with pytest.raises(Exception, match="API Error"):
            self.rag_system.query("Test query")
    
    def test_query_with_tool_manager_failure(self):
        """Test query with tool manager failures"""
        # Mock AI generator to succeed but tool manager to have issues
        self.mock_ai_generator.generate_response.return_value = "Response despite tool issues"
        
        # Mock tool manager with method failures
        mock_tool_manager = self.rag_system.tool_manager
        mock_tool_manager.get_last_sources.side_effect = Exception("Tool error")
        
        # Should handle gracefully or raise - documents current behavior
        with pytest.raises(Exception, match="Tool error"):
            self.rag_system.query("Test query")


if __name__ == "__main__":
    pytest.main([__file__])