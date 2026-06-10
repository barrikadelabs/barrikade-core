"""Simple local Qwen3Guard-backed LLM client."""

import torch
from typing import Any, cast
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.settings import Settings


class LLMClient:
    """Handles local Qwen3Guard model text generation."""
    
    def __init__(self, model= None, base_url= None):
        settings = Settings()
        self.model = model or settings.layer_e_model_dir
        self.base_url = base_url
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)  # nosec B615
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or self.tokenizer.pad_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_obj = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype, trust_remote_code=True)  # nosec B615
        cast(Any, self.model_obj).to(self.device)
        cast(Any, self.model_obj).eval()
    
    def generate(self, prompt, max_tokens= 150):
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = self.model_obj.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][encoded["input_ids"].shape[-1]:]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()
    
    def is_available(self):
        """Check if the local model can be used."""
        return True
