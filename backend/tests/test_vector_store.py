"""
Integration tests for vector_store module
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import tempfile
import json

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults data class"""
    
    def test_from_chroma(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
    
    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc'], [{}], [0.1])
        
        assert empty_results.is_empty() is True
        assert non_empty_results.is_empty() is False


class TestVectorStore:
    """Test VectorStore functionality"""
    
    def setup_method(self):
        """Set up test fixtures with mocked ChromaDB"""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock ChromaDB components
        self.mock_client = Mock()
        self.mock_catalog_collection = Mock()
        self.mock_content_collection = Mock()
        
        # Mock the collections
        self.mock_client.get_or_create_collection.side_effect = [
            self.mock_catalog_collection,
            self.mock_content_collection
        ]
        
        with patch('vector_store.chromadb.PersistentClient', return_value=self.mock_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            self.vector_store = VectorStore(
                chroma_path=self.temp_dir,
                embedding_model="test-model",
                max_results=5
            )
    
    def test_initialization(self):
        """Test VectorStore initialization"""
        assert self.vector_store.max_results == 5
        assert self.vector_store.course_catalog == self.mock_catalog_collection
        assert self.vector_store.course_content == self.mock_content_collection
    
    def test_search_without_filters(self):
        """Test search without course or lesson filters"""
        # Mock ChromaDB query response
        self.mock_content_collection.query.return_value = {
            'documents': [['Sample content']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        results = self.vector_store.search("test query")
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
        assert results.documents == ['Sample content']
        assert results.error is None
    
    def test_search_with_max_results_zero(self):
        """Test search behavior when max_results is 0"""
        # Create vector store with max_results = 0 (the bug we're testing)
        with patch('vector_store.chromadb.PersistentClient', return_value=self.mock_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            zero_results_store = VectorStore(
                chroma_path=self.temp_dir,
                embedding_model="test-model",
                max_results=0
            )
        
        # Mock empty response
        self.mock_content_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = zero_results_store.search("test query")
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=0,
            where=None
        )
        assert results.is_empty() is True
    
    def test_search_with_course_name(self):
        """Test search with course name filter"""
        # Mock course name resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        # Mock content search
        self.mock_content_collection.query.return_value = {
            'documents': [['Course content']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        results = self.vector_store.search("test query", course_name="Test")
        
        # Should call course resolution first
        self.mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Test"],
            n_results=1
        )
        
        # Then call content search with filter
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Test Course"}
        )
    
    def test_search_with_course_not_found(self):
        """Test search when course name cannot be resolved"""
        # Mock empty course resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = self.vector_store.search("test query", course_name="NonExistent")
        
        assert results.error == "No course found matching 'NonExistent'"
    
    def test_search_with_lesson_number(self):
        """Test search with lesson number filter"""
        self.mock_content_collection.query.return_value = {
            'documents': [['Lesson content']],
            'metadatas': [[{'lesson_number': 2}]],
            'distances': [[0.1]]
        }
        
        results = self.vector_store.search("test query", lesson_number=2)
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"lesson_number": 2}
        )
    
    def test_search_with_both_filters(self):
        """Test search with both course name and lesson number"""
        # Mock course name resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        # Mock content search
        self.mock_content_collection.query.return_value = {
            'documents': [['Specific content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        
        results = self.vector_store.search(
            "test query",
            course_name="Test",
            lesson_number=3
        )
        
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_with_custom_limit(self):
        """Test search with custom result limit"""
        self.mock_content_collection.query.return_value = {
            'documents': [['Limited content']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        results = self.vector_store.search("test query", limit=3)
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where=None
        )
    
    def test_search_exception_handling(self):
        """Test search exception handling"""
        self.mock_content_collection.query.side_effect = Exception("Connection error")
        
        results = self.vector_store.search("test query")
        
        assert results.error == "Search error: Connection error"
    
    def test_add_course_metadata(self):
        """Test adding course metadata"""
        course = Course(
            title="Test Course",
            instructor="John Doe",
            course_link="http://example.com",
            lessons=[
                Lesson(0, "Introduction", "http://example.com/lesson0"),
                Lesson(1, "Advanced", "http://example.com/lesson1")
            ]
        )
        
        self.vector_store.add_course_metadata(course)
        
        # Verify the call to ChromaDB
        self.mock_catalog_collection.add.assert_called_once()
        call_args = self.mock_catalog_collection.add.call_args
        
        assert call_args[1]["documents"] == ["Test Course"]
        assert call_args[1]["ids"] == ["Test Course"]
        
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "John Doe"
        assert metadata["course_link"] == "http://example.com"
        assert metadata["lesson_count"] == 2
        
        # Check lessons JSON
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == 2
        assert lessons_data[0]["lesson_number"] == 0
        assert lessons_data[0]["lesson_title"] == "Introduction"
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        chunks = [
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
                content="First chunk content"
            ),
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
                content="Second chunk content"
            )
        ]
        
        self.vector_store.add_course_content(chunks)
        
        # Verify the call to ChromaDB
        self.mock_content_collection.add.assert_called_once()
        call_args = self.mock_content_collection.add.call_args
        
        assert call_args[1]["documents"] == ["First chunk content", "Second chunk content"]
        assert call_args[1]["ids"] == ["Test_Course_0", "Test_Course_1"]
        
        metadatas = call_args[1]["metadatas"]
        assert len(metadatas) == 2
        assert metadatas[0]["course_title"] == "Test Course"
        assert metadatas[0]["lesson_number"] == 1
        assert metadatas[0]["chunk_index"] == 0
    
    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        self.mock_catalog_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == ['Course 1', 'Course 2']
    
    def test_get_course_count(self):
        """Test getting course count"""
        self.mock_catalog_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        count = self.vector_store.get_course_count()
        
        assert count == 3
    
    def test_get_course_metadata_by_title(self):
        """Test getting course metadata by title"""
        expected_metadata = {
            'title': 'Test Course',
            'instructor': 'John Doe',
            'course_link': 'http://example.com',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Intro"}]'
        }
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [expected_metadata]
        }
        
        metadata = self.vector_store.get_course_metadata_by_title("Test Course")
        
        assert metadata == expected_metadata
        self.mock_catalog_collection.get.assert_called_once_with(ids=["Test Course"])
    
    def test_get_course_metadata_by_title_not_found(self):
        """Test getting metadata for non-existent course"""
        self.mock_catalog_collection.get.return_value = {
            'metadatas': []
        }
        
        metadata = self.vector_store.get_course_metadata_by_title("NonExistent")
        
        assert metadata is None


if __name__ == "__main__":
    pytest.main([__file__])