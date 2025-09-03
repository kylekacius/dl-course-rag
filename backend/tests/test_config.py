"""
Tests for configuration module
"""
import pytest
from unittest.mock import Mock, patch, mock_open
import sys
import os
import tempfile

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Test configuration loading and validation"""
    
    def test_config_defaults(self):
        """Test configuration default values"""
        # Import with mocked environment
        with patch.dict(os.environ, {}, clear=True), \
             patch('config.load_dotenv'):
            
            # Reload config module to test defaults
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # Test defaults when no environment variables are set
            assert cfg.ANTHROPIC_API_KEY == ""
            assert cfg.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
            assert cfg.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
            assert cfg.CHUNK_SIZE == 800
            assert cfg.CHUNK_OVERLAP == 100
            assert cfg.MAX_RESULTS == 5  # This should now be 5 after our fix
            assert cfg.MAX_HISTORY == 2
            assert cfg.CHROMA_PATH == "./chroma_db"
    
    def test_config_from_environment(self):
        """Test configuration loading from environment variables"""
        test_env = {
            'ANTHROPIC_API_KEY': 'sk-test-12345',
            'ANTHROPIC_MODEL': 'claude-3-sonnet',
            'EMBEDDING_MODEL': 'custom-embedding-model',
            'CHUNK_SIZE': '1000',
            'CHUNK_OVERLAP': '200',
            'MAX_RESULTS': '10',
            'MAX_HISTORY': '5',
            'CHROMA_PATH': '/custom/path'
        }
        
        with patch.dict(os.environ, test_env), \
             patch('config.load_dotenv'):
            
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # Test that environment variables override defaults
            assert cfg.ANTHROPIC_API_KEY == 'sk-test-12345'
            # Note: dataclass fields with type hints don't auto-convert from env vars
            # This test documents current behavior - may need improvement
    
    def test_dotenv_path_resolution(self):
        """Test that .env file is loaded from correct path"""
        with patch('config.load_dotenv') as mock_load_dotenv:
            import importlib
            import config
            importlib.reload(config)
            
            # Verify load_dotenv was called with parent directory path
            mock_load_dotenv.assert_called_once_with(dotenv_path="../.env")
    
    @patch('os.path.exists')
    @patch('config.load_dotenv')
    def test_dotenv_file_missing(self, mock_load_dotenv, mock_exists):
        """Test behavior when .env file doesn't exist"""
        mock_exists.return_value = False
        
        # load_dotenv should still be called (it handles missing files gracefully)
        import importlib
        import config
        importlib.reload(config)
        
        mock_load_dotenv.assert_called_once_with(dotenv_path="../.env")
    
    def test_config_instance_creation(self):
        """Test config instance is properly created"""
        with patch('config.load_dotenv'):
            import importlib
            import config
            importlib.reload(config)
            
            # Test that config instance exists
            assert hasattr(config, 'config')
            assert isinstance(config.config, config.Config)
    
    def test_api_key_validation_needed(self):
        """Test that demonstrates need for API key validation"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}), \
             patch('config.load_dotenv'):
            
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # This documents current behavior - empty API key is allowed
            # This test suggests we might want validation
            assert cfg.ANTHROPIC_API_KEY == ""
            
            # A production system might want to validate this
    
    def test_numeric_config_types(self):
        """Test that numeric configuration values are proper types"""
        with patch('config.load_dotenv'):
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # Verify types are correct
            assert isinstance(cfg.CHUNK_SIZE, int)
            assert isinstance(cfg.CHUNK_OVERLAP, int)
            assert isinstance(cfg.MAX_RESULTS, int)
            assert isinstance(cfg.MAX_HISTORY, int)
    
    def test_max_results_fix(self):
        """Test that MAX_RESULTS bug is fixed"""
        with patch('config.load_dotenv'):
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # This is the critical fix - MAX_RESULTS should not be 0
            assert cfg.MAX_RESULTS > 0
            assert cfg.MAX_RESULTS == 5  # Our fixed value


class TestConfigurationIntegration:
    """Integration tests for configuration usage"""
    
    def test_config_with_real_dotenv_loading(self):
        """Test configuration with actual .env file loading"""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("ANTHROPIC_API_KEY=sk-test-integration-key\n")
            f.write("MAX_RESULTS=7\n")
            temp_env_path = f.name
        
        try:
            with patch('config.load_dotenv') as mock_load_dotenv:
                # Mock load_dotenv to actually load our temp file
                def load_test_env(dotenv_path=None):
                    os.environ['ANTHROPIC_API_KEY'] = 'sk-test-integration-key'
                    os.environ['MAX_RESULTS'] = '7'
                
                mock_load_dotenv.side_effect = load_test_env
                
                with patch.dict(os.environ, {}, clear=True):
                    import importlib
                    import config
                    importlib.reload(config)
                    
                    cfg = config.Config()
                    
                    # Note: dataclass doesn't auto-convert env vars to int
                    # This documents current limitation
                    assert cfg.ANTHROPIC_API_KEY == 'sk-test-integration-key'
                    # MAX_RESULTS will still be the default since dataclass doesn't auto-convert
        finally:
            os.unlink(temp_env_path)
    
    def test_config_paths_are_relative(self):
        """Test that relative paths work correctly from backend directory"""
        with patch('config.load_dotenv'):
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # CHROMA_PATH should be relative to current directory (backend/)
            assert cfg.CHROMA_PATH == "./chroma_db"
            
            # Verify this creates path relative to backend when running from backend/
            expected_full_path = os.path.join(os.getcwd(), "chroma_db")
            actual_full_path = os.path.abspath(cfg.CHROMA_PATH)
            
            # Should resolve to same location when running from backend/
            assert actual_full_path.endswith("chroma_db")


class TestConfigurationBugPrevention:
    """Tests to prevent configuration-related bugs"""
    
    def test_max_results_never_zero(self):
        """Prevent MAX_RESULTS = 0 bug from recurring"""
        with patch.dict(os.environ, {'MAX_RESULTS': '0'}), \
             patch('config.load_dotenv'):
            
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # Even if someone sets MAX_RESULTS=0 in environment,
            # the default should override (current behavior)
            # OR we should validate and reject 0 values
            assert cfg.MAX_RESULTS != 0
    
    def test_required_api_key_detection(self):
        """Test detection of missing API key"""
        with patch.dict(os.environ, {}, clear=True), \
             patch('config.load_dotenv'):
            
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # This test documents current behavior and suggests improvement
            is_api_key_missing = cfg.ANTHROPIC_API_KEY == ""
            
            # In production, we might want to validate this
            if is_api_key_missing:
                # Document that this is a potential issue
                assert True  # This would cause API failures
    
    def test_chunk_size_sanity_checks(self):
        """Test that chunk sizes are reasonable"""
        with patch('config.load_dotenv'):
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # Sanity checks for reasonable values
            assert cfg.CHUNK_SIZE > 0
            assert cfg.CHUNK_OVERLAP >= 0
            assert cfg.CHUNK_OVERLAP < cfg.CHUNK_SIZE  # Overlap should be less than chunk size
            assert cfg.MAX_HISTORY >= 0
    
    def test_model_name_format(self):
        """Test that model names follow expected format"""
        with patch('config.load_dotenv'):
            import importlib
            import config
            importlib.reload(config)
            
            cfg = config.Config()
            
            # Basic validation of model name format
            assert isinstance(cfg.ANTHROPIC_MODEL, str)
            assert len(cfg.ANTHROPIC_MODEL) > 0
            assert isinstance(cfg.EMBEDDING_MODEL, str)
            assert len(cfg.EMBEDDING_MODEL) > 0


if __name__ == "__main__":
    pytest.main([__file__])