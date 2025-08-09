"""
Hugging Face LLM integration for OpenAgent.

This module provides integration with Hugging Face models for natural language
processing, code generation, and intelligent agent responses.
"""

import asyncio
import logging
import torch
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    pipeline,
    Pipeline
)
from huggingface_hub import login, HfApi
import psutil
from openagent.core.exceptions import AgentError, ConfigError

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for Hugging Face models."""
    
    # Code-focused models (like Warp AI)
    CODE_MODELS = {
        "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
        "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf", 
        "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
        "starcoder": "bigcode/starcoder",
        "starcoder2": "bigcode/starcoder2-7b",
        "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "phind-codellama": "Phind/Phind-CodeLlama-34B-v2",
    }
    
    # General conversation models
    CHAT_MODELS = {
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "openchat": "openchat/openchat-3.5-0106",
        "vicuna-7b": "lmsys/vicuna-7b-v1.5",
        "wizard-coder": "WizardLM/WizardCoder-15B-V1.0",
    }
    
    # Lightweight models for resource-constrained environments
    LIGHTWEIGHT_MODELS = {
        "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "phi-2": "microsoft/phi-2",
        "stable-code": "stabilityai/stable-code-3b",
    }


class HuggingFaceLLM:
    """
    Hugging Face LLM integration with advanced features for AI agents.
    
    This class provides a comprehensive interface to Hugging Face models
    with optimizations for agent workflows, code generation, and terminal assistance.
    """
    
    def __init__(
        self,
        model_name: str = "codellama-7b",
        device: Optional[str] = None,
        max_length: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        hf_token: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Initialize the Hugging Face LLM.
        
        Args:
            model_name: Name of the model to use (from ModelConfig or full HF path)
            device: Device to run the model on ('cuda', 'cpu', 'auto')
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            hf_token: Hugging Face API token
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.kwargs = kwargs
        
        # Authenticate with Hugging Face if token provided
        if hf_token:
            login(token=hf_token)
        
        # Device selection
        self.device = self._select_device(device)
        
        # Model and tokenizer will be loaded lazily
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Model path resolution
        self.model_path = self._resolve_model_path(model_name)
        
        # System information
        self._log_system_info()
        
        logger.info(f"HuggingFaceLLM initialized with model: {self.model_path}")
    
    def _select_device(self, device: Optional[str] = None) -> str:
        """Select the appropriate device for model execution."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        
        return device
    
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to full Hugging Face model path."""
        # Check if it's a predefined model
        all_models = {
            **ModelConfig.CODE_MODELS,
            **ModelConfig.CHAT_MODELS, 
            **ModelConfig.LIGHTWEIGHT_MODELS
        }
        
        if model_name in all_models:
            return all_models[model_name]
        
        # Assume it's a full Hugging Face path
        return model_name
    
    def _log_system_info(self):
        """Log system information for debugging."""
        memory = psutil.virtual_memory()
        logger.info(f"System RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {gpu_memory / (1024**3):.1f} GB")
    
    async def load_model(self) -> None:
        """Load the model and tokenizer asynchronously."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model loading configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Quantization options
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            # Try to load as causal LM first, then seq2seq
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
                self.model_type = "causal"
            except:
                logger.info("Trying to load as seq2seq model")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
                self.model_type = "seq2seq"
            
            # Move to device if not using device_map
            if not model_kwargs.get("device_map") and self.device != "cpu":
                self.model = self.model.to(self.device)
            
            # Create generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Model loaded successfully")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM while loading model: {e}. Falling back to CPU.")
            # Fallback to CPU
            self.device = "cpu"
            await self.unload_model()
            await asyncio.sleep(0)
            await self.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise AgentError(f"Failed to load model {self.model_path}: {e}")
    
    async def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum new tokens to generate
            system_prompt: Optional system prompt for instruction
            context: Optional conversation context
            
        Returns:
            Generated response string
        """
        if self.model is None:
            await self.load_model()
        
        try:
            # Format the prompt based on model type
            formatted_prompt = self._format_prompt(prompt, system_prompt, context)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Update generation config if needed
            generation_config = self.generation_config
            if max_new_tokens:
                generation_config.max_new_tokens = max_new_tokens
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during generation: {e}. Retrying on CPU.")
            # Retry on CPU once
            if self.device != "cpu":
                self.device = "cpu"
                await self.unload_model()
                await self.load_model()
                return await self.generate_response(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    system_prompt=system_prompt,
                    context=context,
                )
            raise AgentError("Out of memory while generating response. Try a smaller model or CPU mode.")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise AgentError(f"Failed to generate response: {e}")
    
    async def stream_response(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response to the given prompt.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            context: Optional conversation context
            
        Yields:
            Streaming response tokens
        """
        # Note: True streaming requires more complex implementation
        # For now, we'll simulate streaming by yielding the full response
        response = await self.generate_response(prompt, system_prompt=system_prompt, context=context)
        
        # Simulate streaming by yielding words
        words = response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Small delay for streaming effect
    
    def _format_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format the prompt based on the model's expected format."""
        
        # Default system prompt for code-focused models
        if system_prompt is None:
            if any(code_model in self.model_path.lower() for code_model in ["code", "coder", "star"]):
                system_prompt = (
                    "You are an expert programming assistant. You help users with coding tasks, "
                    "debugging, explanations, and command-line operations. Provide clear, "
                    "accurate, and helpful responses."
                )
            else:
                system_prompt = (
                    "You are a helpful AI assistant. Provide accurate, helpful, and "
                    "informative responses to user questions."
                )
        
        # Format based on common patterns
        if "llama" in self.model_path.lower():
            # Llama format
            if context:
                conversation = ""
                for msg in context:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        conversation += f"[INST] {content} [/INST]\n"
                    else:
                        conversation += f"{content}\n"
                formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{conversation}{prompt} [/INST]"
            else:
                formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        
        elif "mistral" in self.model_path.lower():
            # Mistral format
            formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        
        elif any(model in self.model_path.lower() for model in ["vicuna", "wizard", "openchat"]):
            # Vicuna/WizardCoder format
            formatted_prompt = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        
        elif "zephyr" in self.model_path.lower():
            # Zephyr format
            formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        else:
            # Generic format
            formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        return formatted_prompt
    
    async def analyze_code(self, code: str, language: str = "python") -> str:
        """
        Analyze code and provide insights.
        
        Args:
            code: The code to analyze
            language: Programming language
            
        Returns:
            Code analysis and suggestions
        """
        prompt = f"""
Analyze the following {language} code and provide:
1. Code review and quality assessment
2. Potential bugs or issues
3. Performance improvements
4. Best practices recommendations
5. Security considerations (if applicable)

```{language}
{code}
```

Please provide a detailed analysis:
"""
        
        return await self.generate_response(
            prompt,
            system_prompt="You are an expert code reviewer and software engineer."
        )
    
    async def generate_code(self, description: str, language: str = "python") -> str:
        """
        Generate code based on description.
        
        Args:
            description: Description of what the code should do
            language: Target programming language
            
        Returns:
            Generated code
        """
        prompt = f"""
Generate {language} code that accomplishes the following:

{description}

Requirements:
- Write clean, well-documented code
- Include error handling where appropriate
- Follow best practices for {language}
- Add comments explaining key parts

Please provide the complete code:
"""
        
        return await self.generate_response(
            prompt,
            system_prompt=f"You are an expert {language} developer."
        )
    
    async def explain_command(self, command: str) -> str:
        """
        Explain a command-line command.
        
        Args:
            command: The command to explain
            
        Returns:
            Detailed explanation of the command
        """
        prompt = f"""
Explain the following command in detail:

`{command}`

Please provide:
1. What the command does
2. Breakdown of each part/flag
3. Common use cases
4. Potential risks or considerations
5. Alternative commands if applicable
"""
        
        return await self.generate_response(
            prompt,
            system_prompt="You are an expert system administrator and command-line expert."
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "model_type": getattr(self, "model_type", None),
            "loaded": self.model is not None,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
    
    async def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded successfully")


# Global LLM instance
llm = None

def get_llm(model_name: str = "codellama-7b", **kwargs) -> HuggingFaceLLM:
    """Get or create global LLM instance."""
    global llm
    if llm is None or llm.model_name != model_name:
        llm = HuggingFaceLLM(model_name=model_name, **kwargs)
    return llm
