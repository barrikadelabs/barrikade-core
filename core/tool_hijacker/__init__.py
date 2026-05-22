"""
ToolHijacker - Prompt Injection Attack Generator for LLM Tool Selection

This package implements the ToolHijacker attack framework for systematically
crafting malicious tool documents that target tool selection in LLM agents.

The attack is designed for a no-box scenario where the attacker cannot access:
- The target retriever
- The target LLM
- Details of the tool library or task descriptions

Main components:
- Shadow Framework: Simulates the tool selection pipeline
- Retrieval Optimizer: Optimizes R to improve retrieval
- Selection Optimizer: Optimizes S to improve selection
- Attack Generator: Coordinates the complete attack process

Usage:
    from core.tool_hijacker import ToolHijacker
    
    # Create attack generator
    hijacker = ToolHijacker()
    
    # Generate attack
    result = hijacker.generate_attack(
        target_task="Analyze customer sentiment from reviews",
        malicious_tool_name="DataExfiltrator",
        optimization_method="gradient_free"
    )
    
    # Access malicious tool
    malicious_tool = result.malicious_tool
    print(f"Tool: {malicious_tool.name}")
    print(f"Description: {malicious_tool.description}")
    print(f"Success rate: {result.overall_success_rate:.2%}")
"""

from .tool_document import ToolDocument, MaliciousToolDocument
from .shadow_framework import (
    ShadowFramework,
    ShadowRetriever,
    ShadowLLM,
    ShadowTask
)
from .task_generator import ShadowTaskGenerator
from .tool_library import ShadowToolLibrary
from .retrieval_optimizer import (
    RetrievalOptimizer,
    GradientFreeRetrievalOptimizer,
    GradientBasedRetrievalOptimizer
)
from .selection_optimizer import (
    SelectionOptimizer,
    GradientFreeSelectionOptimizer,
    GradientBasedSelectionOptimizer
)
from .attack_generator import ToolHijacker, AttackResult

__all__ = [
    # Main classes
    'ToolHijacker',
    'AttackResult',
    
    # Tool documents
    'ToolDocument',
    'MaliciousToolDocument',
    
    # Shadow framework
    'ShadowFramework',
    'ShadowRetriever',
    'ShadowLLM',
    'ShadowTask',
    
    # Utilities
    'ShadowTaskGenerator',
    'ShadowToolLibrary',
    
    # Optimizers
    'RetrievalOptimizer',
    'GradientFreeRetrievalOptimizer',
    'GradientBasedRetrievalOptimizer',
    'SelectionOptimizer',
    'GradientFreeSelectionOptimizer',
    'GradientBasedSelectionOptimizer',
]

from core.__version__ import __version__
