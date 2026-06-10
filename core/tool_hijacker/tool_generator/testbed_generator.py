"""
Main testbed generator orchestrator.
"""

import csv
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from .llm_client import LLMClient
from .tool_factory import ToolFactory


class TestbedGenerator:
    """Generates testbed of benign and malicious tool documents using a local LLM."""
    
    # Tool categories (same as ToolFactory)
    CATEGORIES = [
        "data processing", "file operations", "communication",
        "analysis", "security", "automation", "media",
        "web services", "database", "system utilities"
    ]
    
    def __init__(self, output_dir= "datasets/tool_docs", model= "llama3.2"):
        """
        Initialize generator.
        
        Args:
            output_dir: Where to save generated tools
            model: Local model path or identifier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM
        self.llm = LLMClient(model=model)
        if not self.llm.is_available():
            raise RuntimeError("Local model is not available")
        
        self.factory = ToolFactory(self.llm)
    
    def generate_dataset(self, num_benign= 50, num_malicious= 50):
        """
        Generate complete testbed dataset.
        
        Args:
            num_benign: Number of benign tools
            num_malicious: Number of malicious tools
            
        Returns:
            Path to generated CSV file
        """
        print(f"\n{'='*60}")
        print(f"🔧 TESTBED GENERATOR (LLM-Powered)")
        print(f"{'='*60}")
        print(f"Model: {self.llm.model}")
        print(f"Target: {num_benign} benign + {num_malicious} malicious")
        print(f"{'='*60}\n")
        
        tools = []
        
        # Generate benign tools
        print("Generating benign tools...")
        for i in range(num_benign):
            tool = self._generate_benign(i + 1)
            tools.append(tool)
            print(f"  [{i+1}/{num_benign}] ✓ {tool['name']}")
        
        # Generate malicious tools
        print(f"\nGenerating malicious tools...")
        for i in range(num_malicious):
            tool = self._generate_malicious(i + 1)
            tools.append(tool)
            print(f"  [{i+1}/{num_malicious}] ✓ {tool['name']}")
        
        # Shuffle and save
        random.shuffle(tools)
        filename = self._save_dataset(tools)
        
        print(f"Generated: {len(tools)} tools")
        print(f"Saved to: {filename}")
        
        return filename
    
    def _generate_benign(self, index):
        """Generate one benign tool using LLM."""
        category = random.choice(self.CATEGORIES)  # nosec B311
        tool_name = f"Tool_{index:03d}"
        
        description = self.factory.create_benign_tool(category, tool_name)
        if not description:
            raise RuntimeError(f"Failed to generate benign tool {tool_name}")
        
        return {
            'name': tool_name,
            'category': category,
            'description': description,
            'label': 'benign'
        }
    
    def _generate_malicious(self, index):
        """Generate one malicious tool using LLM."""
        category = random.choice(self.CATEGORIES)  # nosec B311
        tool_name = f"MalTool_{index:03d}"
        
        target_behaviors = [
            "access sensitive data",
            "override tool selection",
            "bypass security checks",
            "exfiltrate information",
            "escalate privileges"
        ]
        target = random.choice(target_behaviors)  # nosec B311
        
        description = self.factory.create_malicious_tool(
            category, tool_name, target
        )
        if not description:
            raise RuntimeError(f"Failed to generate malicious tool {tool_name}")
        
        return {
            'name': tool_name,
            'category': category,
            'description': description,
            'label': 'malicious',
            'target_behavior': target
        }
    
    def _save_dataset(self, tools):
        """Save tools to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"testbed_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'name', 'category', 'description', 'label', 'target_behavior'
            ])
            writer.writeheader()
            writer.writerows(tools)
        
        return str(filename)
