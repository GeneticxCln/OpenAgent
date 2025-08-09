LLM API Reference
==================

This section documents the Large Language Model integration components.

Hugging Face LLM
-----------------

Main LLM integration class for Hugging Face models.

.. autoclass:: openagent.core.llm.HuggingFaceLLM
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Model Configuration
-------------------

Configuration classes and utilities for model management.

.. autoclass:: openagent.core.llm.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Helper functions for LLM operations.

.. autofunction:: openagent.core.llm.get_llm

Examples
--------

Basic LLM Usage
~~~~~~~~~~~~~~~

.. code-block:: python

    from openagent.core.llm import HuggingFaceLLM
    
    # Initialize LLM with a lightweight model
    llm = HuggingFaceLLM(
        model_name="tiny-llama",
        device="cpu",
        temperature=0.7
    )
    
    # Load the model
    await llm.load_model()
    
    # Generate a response
    response = await llm.generate_response(
        "Explain how to use Git for version control"
    )
    print(response)

Code Generation
~~~~~~~~~~~~~~~

.. code-block:: python

    # Generate code with specific model
    code_llm = HuggingFaceLLM(
        model_name="codellama-7b",
        device="cuda",
        temperature=0.3  # Lower temperature for more deterministic code
    )
    
    await code_llm.load_model()
    
    # Generate Python function
    code = await code_llm.generate_code(
        "Create a function to calculate Fibonacci numbers",
        language="python"
    )
    print(code)

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from openagent.core.llm import ModelConfig
    
    # List available models
    print("Code Models:", ModelConfig.CODE_MODELS)
    print("Chat Models:", ModelConfig.CHAT_MODELS)
    print("Lightweight Models:", ModelConfig.LIGHTWEIGHT_MODELS)
    
    # Custom model configuration
    custom_llm = HuggingFaceLLM(
        model_name="custom/my-model",
        device="auto",
        load_in_4bit=True,
        temperature=0.8,
        top_p=0.9,
        max_length=2048
    )

Advanced Features
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Stream responses for long generations
    async for token in llm.stream_response("Write a detailed explanation..."):
        print(token, end="", flush=True)
    
    # Analyze existing code
    analysis = await llm.analyze_code(
        code="def hello(): print('world')",
        language="python"
    )
    print(analysis)
    
    # Explain shell commands
    explanation = await llm.explain_command("docker run -it ubuntu bash")
    print(explanation)
    
    # Get model information
    info = llm.get_model_info()
    print(f"Model: {info['model_name']}, Device: {info['device']}")
    
    # Unload model to free memory
    await llm.unload_model()

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Optimize for memory usage
    memory_efficient_llm = HuggingFaceLLM(
        model_name="tiny-llama",
        device="cuda",
        load_in_4bit=True,  # Use 4-bit quantization
        max_length=1024     # Reduce context length
    )
    
    # Optimize for speed
    fast_llm = HuggingFaceLLM(
        model_name="tiny-llama",
        device="cuda",
        temperature=0.1,  # Lower temperature = faster sampling
        top_k=10         # Limit vocabulary for faster generation
    )
