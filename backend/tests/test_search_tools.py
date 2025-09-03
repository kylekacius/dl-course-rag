"""
Unit tests for search_tools module
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_get_tool_definition(self):
        """Test that tool definition is properly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
    
    def test_execute_with_results(self):
        """Test execute method with successful search results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is sample course content", "More course content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert "[Test Course - Lesson 1]" in result
        assert "[Test Course - Lesson 2]" in result
        assert "This is sample course content" in result
        assert "More course content" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
    
    def test_execute_with_course_filter(self):
        """Test execute method with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered course content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 1}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query", course_name="Specific Course")
        
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course",
            lesson_number=None
        )
        assert "[Specific Course - Lesson 1]" in result
    
    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson-specific content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query", lesson_number=3)
        
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=3
        )
        assert "[Test Course - Lesson 3]" in result
    
    def test_execute_with_both_filters(self):
        """Test execute method with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Highly filtered content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 2}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            "test query", 
            course_name="Specific Course",
            lesson_number=2
        )
        
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course",
            lesson_number=2
        )
        assert "[Specific Course - Lesson 2]" in result
    
    def test_execute_with_error(self):
        """Test execute method handles search errors"""
        mock_results = SearchResults.empty("Search failed due to connection error")
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "Search failed due to connection error"
    
    def test_execute_with_no_results(self):
        """Test execute method handles empty results"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "No relevant content found."
    
    def test_execute_with_no_results_and_filters(self):
        """Test execute method handles empty results with filters"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            "test query",
            course_name="NonExistent Course",
            lesson_number=5
        )
        
        assert result == "No relevant content found in course 'NonExistent Course' in lesson 5."
    
    def test_sources_tracking(self):
        """Test that sources are properly tracked for UI"""
        mock_results = SearchResults(
            documents=["Sample content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        result = self.search_tool.execute("test query")
        
        # Check that sources were tracked
        assert len(self.search_tool.last_sources) == 1
        source = self.search_tool.last_sources[0]
        assert source["text"] == "Test Course - Lesson 1"
        assert source["url"] == "http://example.com/lesson1"


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.outline_tool = CourseOutlineTool(self.mock_vector_store)
    
    def test_get_tool_definition(self):
        """Test that tool definition is properly formatted"""
        definition = self.outline_tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_name"]
        assert "course_name" in definition["input_schema"]["properties"]
    
    def test_execute_with_valid_course(self):
        """Test execute method with valid course"""
        # Mock course resolution and metadata
        self.mock_vector_store._resolve_course_name.return_value = "Test Course"
        self.mock_vector_store.get_course_metadata_by_title.return_value = {
            "title": "Test Course",
            "instructor": "John Doe",
            "course_link": "http://example.com/course",
            "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Advanced Topics"}]'
        }
        
        result = self.outline_tool.execute("Test")
        
        assert "**Course:** Test Course" in result
        assert "**Instructor:** John Doe" in result
        assert "**Course Link:** http://example.com/course" in result
        assert "- Lesson 1: Introduction" in result
        assert "- Lesson 2: Advanced Topics" in result
    
    def test_execute_with_course_not_found(self):
        """Test execute method when course is not found"""
        self.mock_vector_store._resolve_course_name.return_value = None
        
        result = self.outline_tool.execute("NonExistent Course")
        
        assert result == "No course found matching 'NonExistent Course'"
    
    def test_execute_with_metadata_error(self):
        """Test execute method when metadata retrieval fails"""
        self.mock_vector_store._resolve_course_name.return_value = "Test Course"
        self.mock_vector_store.get_course_metadata_by_title.return_value = None
        
        result = self.outline_tool.execute("Test Course")
        
        assert result == "Could not retrieve metadata for course 'Test Course'"
    
    def test_execute_with_malformed_lessons_json(self):
        """Test execute method with malformed lessons JSON"""
        self.mock_vector_store._resolve_course_name.return_value = "Test Course"
        self.mock_vector_store.get_course_metadata_by_title.return_value = {
            "title": "Test Course",
            "instructor": "John Doe",
            "course_link": "http://example.com/course",
            "lessons_json": "invalid json"
        }
        
        result = self.outline_tool.execute("Test Course")
        
        assert "**Course:** Test Course" in result
        assert "- Error parsing lesson data" in result


class TestToolManager:
    """Test ToolManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool"
        }
        self.mock_tool.execute.return_value = "Test result"
    
    def test_register_tool(self):
        """Test tool registration"""
        self.tool_manager.register_tool(self.mock_tool)
        
        assert "test_tool" in self.tool_manager.tools
        assert self.tool_manager.tools["test_tool"] == self.mock_tool
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        self.tool_manager.register_tool(self.mock_tool)
        
        definitions = self.tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
    
    def test_execute_tool(self):
        """Test tool execution"""
        self.tool_manager.register_tool(self.mock_tool)
        
        result = self.tool_manager.execute_tool("test_tool", param1="value1")
        
        assert result == "Test result"
        self.mock_tool.execute.assert_called_once_with(param1="value1")
    
    def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool"""
        result = self.tool_manager.execute_tool("nonexistent_tool")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self):
        """Test getting last sources from tools"""
        mock_search_tool = Mock()
        mock_search_tool.last_sources = [{"text": "source1", "url": "url1"}]
        mock_search_tool.get_tool_definition.return_value = {"name": "search_tool"}
        
        self.tool_manager.register_tool(mock_search_tool)
        
        sources = self.tool_manager.get_last_sources()
        
        assert sources == [{"text": "source1", "url": "url1"}]
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        mock_search_tool = Mock()
        mock_search_tool.last_sources = [{"text": "source1", "url": "url1"}]
        mock_search_tool.get_tool_definition.return_value = {"name": "search_tool"}
        
        self.tool_manager.register_tool(mock_search_tool)
        self.tool_manager.reset_sources()
        
        assert mock_search_tool.last_sources == []


if __name__ == "__main__":
    pytest.main([__file__])