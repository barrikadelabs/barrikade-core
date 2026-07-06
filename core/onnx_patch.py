import sys
import os
import types
from pathlib import Path
import torch
import onnxruntime as ort

class ORTModelMock:
    def __init__(self, model_path, config=None, provider="CPUExecutionProvider"):
        providers = [provider] if isinstance(provider, str) else provider
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.config = config
        self.device = torch.device("cpu")

    def _save_pretrained(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class ORTModelForFeatureExtraction(ORTModelMock):
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        session_inputs = self.session.get_inputs()
        input_names = [i.name for i in session_inputs]
        onnx_inputs = {}
        if "input_ids" in input_names and input_ids is not None:
            onnx_inputs["input_ids"] = input_ids.cpu().numpy()
        if "attention_mask" in input_names and attention_mask is not None:
            onnx_inputs["attention_mask"] = attention_mask.cpu().numpy()
        if "token_type_ids" in input_names and token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().numpy()
        
        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = torch.from_numpy(outputs[0])
        
        class ModelOutput(tuple):
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, args)
            def __init__(self, *args, **kwargs):
                super().__init__()
                for k, v in kwargs.items():
                    setattr(self, k, v)
        return ModelOutput(last_hidden_state, last_hidden_state=last_hidden_state)

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, provider="CPUExecutionProvider", **kwargs):
        p = Path(model_name_or_path)
        model_path = p / "onnx" / "model.onnx"
        if not model_path.exists():
            model_path = p / "model.onnx"
        if not model_path.exists() and "file_name" in kwargs:
            model_path = p / kwargs["file_name"]
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found in {model_name_or_path}")
        return cls(model_path, config=config, provider=provider)

class ORTModelForSequenceClassification(ORTModelMock):
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        session_inputs = self.session.get_inputs()
        input_names = [i.name for i in session_inputs]
        onnx_inputs = {}
        if "input_ids" in input_names and input_ids is not None:
            onnx_inputs["input_ids"] = input_ids.cpu().numpy()
        if "attention_mask" in input_names and attention_mask is not None:
            onnx_inputs["attention_mask"] = attention_mask.cpu().numpy()
        if "token_type_ids" in input_names and token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().numpy()
            
        outputs = self.session.run(None, onnx_inputs)
        logits = torch.from_numpy(outputs[0])
        
        class ModelOutput(tuple):
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, args)
            def __init__(self, *args, **kwargs):
                super().__init__()
                for k, v in kwargs.items():
                    setattr(self, k, v)
        return ModelOutput(logits, logits=logits)

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, provider="CPUExecutionProvider", **kwargs):
        p = Path(model_name_or_path)
        model_path = p / "onnx" / "model.onnx"
        if not model_path.exists():
            model_path = p / "model.onnx"
        if not model_path.exists() and "file_name" in kwargs:
            model_path = p / kwargs["file_name"]
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found in {model_name_or_path}")
        return cls(model_path, config=config, provider=provider)

def apply_patch():
    optimum = types.ModuleType("optimum")
    onnxruntime = types.ModuleType("optimum.onnxruntime")
    optimum.onnxruntime = onnxruntime
    onnxruntime.ONNX_WEIGHTS_NAME = "model.onnx"
    onnxruntime.ORTModelForFeatureExtraction = ORTModelForFeatureExtraction
    onnxruntime.ORTModelForSequenceClassification = ORTModelForSequenceClassification
    onnxruntime.ORTModelForMaskedLM = None
    
    sys.modules["optimum"] = optimum
    sys.modules["optimum.onnxruntime"] = onnxruntime

# Apply the patch immediately upon import
apply_patch()
