"""
Unit tests for the HuggingFace LLM integration.

Tests model loading, text generation, error handling, and performance features.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from openagent.core.llm import HuggingFaceLLM, ModelConfig, get_llm
from openagent.core.exceptions import AgentError


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_model_categories(self):
        """Test that model categories are properly defined."""
        assert len(ModelConfig.CODE_MODELS) > 0
        assert len(ModelConfig.CHAT_MODELS) > 0
        assert len(ModelConfig.LIGHTWEIGHT_MODELS) > 0
        
        # Test specific models exist
        assert "codellama-7b" in ModelConfig.CODE_MODELS
        assert "tiny-llama" in ModelConfig.LIGHTWEIGHT_MODELS
        assert "mistral-7b" in ModelConfig.CHAT_MODELS
        
    def test_model_paths_are_valid(self):
        """Test that model paths follow Hugging Face format."""
        all_models = {
            **ModelConfig.CODE_MODELS,
            **ModelConfig.CHAT_MODELS,
            **ModelConfig.LIGHTWEIGHT_MODELS
        }
        
        for name, path in all_models.items():
            assert isinstance(path, str)
            assert "/" in path  # Should be in org/model format
            assert len(path.split("/")) >= 2


class TestHuggingFaceLLM:
    """Test HuggingFaceLLM class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "tiny-llama"
        
    def test_llm_initialization_basic(self):
        """Test basic LLM initialization."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(
                model_name=self.model_name,
                device="cpu",
                temperature=0.5
            )
            
            assert llm.model_name == self.model_name
            assert llm.temperature == 0.5
            assert llm.device == "cpu"
            assert llm.model is None  # Lazy loading
            assert llm.tokenizer is None
            
    def test_llm_initialization_with_config(self):
        """Test LLM initialization with custom configuration."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(
                model_name="codellama-7b",
                device="cuda",
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                load_in_4bit=True,
                hf_token="test_token"
            )
            
            assert llm.model_name == "codellama-7b"
            assert llm.temperature == 0.8
            assert llm.top_p == 0.9
            assert llm.top_k == 40
            assert llm.load_in_4bit is True
            
    def test_device_selection_auto(self):
        """Test automatic device selection."""
        with patch('openagent.core.llm.torch') as mock_torch, \
             patch('openagent.core.llm.psutil'):
            
            # Test CUDA available
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            
            llm = HuggingFaceLLM(model_name=self.model_name, device="auto")
            assert llm.device == "cuda"
            
            # Test CUDA not available
            mock_torch.cuda.is_available.return_value = False
            
            llm = HuggingFaceLLM(model_name=self.model_name, device="auto")
            assert llm.device == "cpu"
            
    def test_model_path_resolution(self):
        """Test model path resolution from names."""
        with patch('openagent.core.llm.psutil'):
            # Test predefined model
            llm = HuggingFaceLLM(model_name="tiny-llama")
            assert llm.model_path == ModelConfig.LIGHTWEIGHT_MODELS["tiny-llama"]
            
            # Test custom model path
            custom_path = "custom/model-path"
            llm = HuggingFaceLLM(model_name=custom_path)
            assert llm.model_path == custom_path
            
    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model loading."""
        with patch('openagent.core.llm.AutoTokenizer') as mock_tokenizer, \
             patch('openagent.core.llm.AutoModelForCausalLM') as mock_model, \
             patch('openagent.core.llm.GenerationConfig') as mock_gen_config, \
             patch('openagent.core.llm.psutil'):
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = "[PAD]"
            mock_tokenizer_instance.eos_token_id = 2
            mock_tokenizer_instance.pad_token_id = 0
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Mock generation config
            mock_gen_config_instance = Mock()
            mock_gen_config.return_value = mock_gen_config_instance
            
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            await llm.load_model()
            
            assert llm.tokenizer is not None
            assert llm.model is not None
            assert llm.generation_config is not None
            mock_tokenizer.from_pretrained.assert_called_once()
            mock_model.from_pretrained.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_load_model_fallback_seq2seq(self):
        """Test model loading with seq2seq fallback."""
        with patch('openagent.core.llm.AutoTokenizer') as mock_tokenizer, \
             patch('openagent.core.llm.AutoModelForCausalLM') as mock_causal_model, \
             patch('openagent.core.llm.AutoModelForSeq2SeqLM') as mock_seq2seq_model, \
             patch('openagent.core.llm.GenerationConfig') as mock_gen_config, \
             patch('openagent.core.llm.psutil'):
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "[EOS]"
            mock_tokenizer_instance.eos_token_id = 2
            mock_tokenizer_instance.pad_token_id = 0
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Make causal model fail, seq2seq succeed
            mock_causal_model.from_pretrained.side_effect = Exception("Not a causal model")
            mock_seq2seq_instance = Mock()
            mock_seq2seq_model.from_pretrained.return_value = mock_seq2seq_instance
            
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            await llm.load_model()
            
            assert llm.model is not None
            assert llm.model_type == "seq2seq"
            assert llm.tokenizer.pad_token == "[EOS]"  # Should set pad_token to eos_token
            
    @pytest.mark.asyncio
    async def test_load_model_cuda_oom_fallback(self):
        """Test CUDA OOM fallback to CPU."""
        with patch('openagent.core.llm.AutoTokenizer') as mock_tokenizer, \
             patch('openagent.core.llm.AutoModelForCausalLM') as mock_model, \
             patch('openagent.core.llm.torch') as mock_torch, \
             patch('openagent.core.llm.asyncio') as mock_asyncio, \
             patch('openagent.core.llm.psutil'):
            
            # Mock CUDA OOM error
            mock_torch.cuda.OutOfMemoryError = Exception  # Create the exception class
            mock_model.from_pretrained.side_effect = [
                mock_torch.cuda.OutOfMemoryError("CUDA OOM"),
                Mock()  # Second call succeeds
            ]
            
            # Mock sleep
            mock_asyncio.sleep = AsyncMock()
            
            llm = HuggingFaceLLM(model_name=self.model_name, device="cuda")
            
            # Mock unload_model method
            llm.unload_model = AsyncMock()
            
            await llm.load_model()
            
            assert llm.device == "cpu"  # Should fallback to CPU
            llm.unload_model.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful response generation."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            
            # Mock loaded model and tokenizer
            mock_tokenizer = Mock()
            mock_model = Mock()
            mock_generation_config = Mock()
            
            llm.tokenizer = mock_tokenizer
            llm.model = mock_model
            llm.generation_config = mock_generation_config
            
            # Mock tokenization
            mock_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
            mock_tokenizer.return_value = mock_inputs
            mock_inputs["input_ids"].shape = [1, 10]
            
            # Mock generation
            mock_outputs = [Mock()]
            mock_outputs[0].__getitem__ = lambda self, key: Mock()
            mock_model.generate.return_value = mock_outputs
            
            # Mock decoding
            mock_tokenizer.decode.return_value = "Test response"
            
            response = await llm.generate_response("Test prompt")
            
            assert response == "Test response"
            mock_model.generate.assert_called_once()
            mock_tokenizer.decode.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_generate_response_lazy_loading(self):
        """Test that generate_response triggers model loading."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            
            # Mock load_model
            llm.load_model = AsyncMock()
            
            # Mock model components after loading
            async def mock_load():
                llm.tokenizer = Mock()
                llm.model = Mock()
                llm.generation_config = Mock()
                
                # Setup mocks for generation
                mock_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
                mock_inputs["input_ids"].shape = [1, 10]
                llm.tokenizer.return_value = mock_inputs
                
                mock_outputs = [Mock()]
                mock_outputs[0].__getitem__ = lambda self, key: Mock()
                llm.model.generate.return_value = mock_outputs
                
                llm.tokenizer.decode.return_value = "Response"
                
            llm.load_model.side_effect = mock_load
            
            response = await llm.generate_response("Test prompt")
            
            llm.load_model.assert_called_once()
            assert response == "Response"
            
    def test_format_prompt_llama(self):
        """Test prompt formatting for Llama models."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name="tiny-llama", device="cpu")
            llm.model_path = "meta-llama/Llama-2-7b"
            
            prompt = "Hello"
            system_prompt = "You are helpful"
            
            formatted = llm._format_prompt(prompt, system_prompt)
            
            assert "[INST]" in formatted
            assert "[/INST]" in formatted
            assert "<<SYS>>" in formatted
            assert "You are helpful" in formatted
            assert "Hello" in formatted
            
    def test_format_prompt_mistral(self):
        """Test prompt formatting for Mistral models."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name="mistral-7b", device="cpu")
            llm.model_path = "mistralai/Mistral-7B"
            
            formatted = llm._format_prompt("Hello", "You are helpful")
            
            assert "<s>[INST]" in formatted
            assert "[/INST]" in formatted
            
    def test_format_prompt_generic(self):
        """Test generic prompt formatting."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name="custom-model", device="cpu")
            llm.model_path = "custom/model"
            
            formatted = llm._format_prompt("Hello", "You are helpful")
            
            assert "System: You are helpful" in formatted
            assert "User: Hello" in formatted
            assert "Assistant:" in formatted
            
    @pytest.mark.asyncio
    async def test_analyze_code(self):
        """Test code analysis functionality."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            llm.generate_response = AsyncMock(return_value="Code analysis result")
            
            result = await llm.analyze_code("def test(): pass", "python")
            
            assert result == "Code analysis result"
            llm.generate_response.assert_called_once()
            call_args = llm.generate_response.call_args
            assert "python" in call_args[0][0].lower()
            assert "def test(): pass" in call_args[0][0]
            
    @pytest.mark.asyncio
    async def test_generate_code(self):
        """Test code generation functionality."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            llm.generate_response = AsyncMock(return_value="Generated code")
            
            result = await llm.generate_code("Create a hello world function", "python")
            
            assert result == "Generated code"
            llm.generate_response.assert_called_once()
            call_args = llm.generate_response.call_args
            assert "python" in call_args[0][0].lower()
            assert "hello world" in call_args[0][0].lower()
            
    @pytest.mark.asyncio
    async def test_explain_command(self):
        """Test command explanation functionality."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(model_name=self.model_name, device="cpu")
            llm.generate_response = AsyncMock(return_value="Command explanation")
            
            result = await llm.explain_command("ls -la")
            
            assert result == "Command explanation"
            llm.generate_response.assert_called_once()
            call_args = llm.generate_response.call_args
            assert "ls -la" in call_args[0][0]
            
    def test_get_model_info(self):
        """Test model information retrieval."""
        with patch('openagent.core.llm.psutil'):
            llm = HuggingFaceLLM(
                model_name=self.model_name,
                device="cpu",
                temperature=0.7,
                top_p=0.9
            )
            
            info = llm.get_model_info()
            
            assert info["model_name"] == self.model_name
            assert info["device"] == "cpu"
            assert info["temperature"] == 0.7
            assert info["top_p"] == 0.9
            assert info["loaded"] is False  # Model not loaded yet
            
    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test model unloading."""
        with patch('openagent.core.llm.torch') as mock_torch, \
             patch('openagent.core.llm.psutil'):
            
            llm = HuggingFaceLLM(model_name=self.model_name, device="cuda")
            llm.model = Mock()
            llm.tokenizer = Mock()
            
            # Mock CUDA operations
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = Mock()
            
            await llm.unload_model()
            
            assert llm.model is None
            assert llm.tokenizer is None
            mock_torch.cuda.empty_cache.assert_called_once()


class TestGetLLM:
    """Test the get_llm function."""
    
    def test_get_llm_creates_instance(self):
        """Test that get_llm creates LLM instances."""
        with patch('openagent.core.llm.HuggingFaceLLM') as mock_llm_class, \
             patch('openagent.core.llm.llm', None):  # Reset global
            
            mock_instance = Mock()
            mock_llm_class.return_value = mock_instance
            
            result = get_llm("test-model", temperature=0.5)
            
            assert result == mock_instance
            mock_llm_class.assert_called_once_with(model_name="test-model", temperature=0.5)
            
    def test_get_llm_reuses_instance(self):
        """Test that get_llm reuses instances for same model."""
        with patch('openagent.core.llm.HuggingFaceLLM') as mock_llm_class:
            
            mock_instance = Mock()
            mock_instance.model_name = "test-model"
            mock_llm_class.return_value = mock_instance
            
            # Set global llm
            import openagent.core.llm
            openagent.core.llm.llm = mock_instance
            
            result = get_llm("test-model")
            
            assert result == mock_instance
            # Should not create new instance
            mock_llm_class.assert_not_called()
            
    def test_get_llm_creates_new_for_different_model(self):
        """Test that get_llm creates new instance for different model."""
        with patch('openagent.core.llm.HuggingFaceLLM') as mock_llm_class:
            
            # Existing instance with different model
            old_instance = Mock()
            old_instance.model_name = "old-model"
            
            # New instance
            new_instance = Mock()
            mock_llm_class.return_value = new_instance
            
            # Set global llm to old instance
            import openagent.core.llm
            openagent.core.llm.llm = old_instance
            
            result = get_llm("new-model")
            
            assert result == new_instance
            mock_llm_class.assert_called_once_with(model_name="new-model")


if __name__ == "__main__":
    pytest.main([__file__])
