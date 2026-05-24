import unittest
import pytest
from unittest.mock import MagicMock, patch
from core.orchestrator import PIPipeline
from models.verdicts import DecisionLayer, FinalVerdict
from models.LayerEResult import LayerEResult

def dummy_init(self):
    self.layer_a_analyze = MagicMock()
    self.layer_b_engine = MagicMock()
    self.layer_c_classifier = MagicMock()
    self.layer_d_classifier = MagicMock()
    self.layer_e_judge = MagicMock()

@pytest.mark.telemetry
@patch("core.orchestrator.PIPipeline.__init__", dummy_init)
@patch("core.orchestrator.ensure_runtime_artifacts")
@patch("core.layer_a.pipeline.analyze_text")
@patch("core.layer_b.signature_engine.SignatureEngine")
@patch("core.layer_c.classifier.Classifier")
@patch("core.layer_d.classifier.LayerDClassifier")
@patch("core.layer_e.llm_judge.LLMJudge")
class TestOrchestratorTelemetry(unittest.TestCase):
    @patch("core.orchestrator.telemetry")
    def test_detect_telemetry_emit_layer_a(self, mock_telemetry, mock_llm_judge, mock_d, mock_c, mock_b, mock_a, mock_ensure):
        pipeline = PIPipeline()
        
        # Mock Layer A to return a block result
        mock_layer_a_res = MagicMock()
        mock_layer_a_res.get_verdict.return_value = "block"
        mock_layer_a_res.verdict = "block"
        mock_layer_a_res.processed_text = "ignored text"
        mock_layer_a_res.confidence_score = 0.95
        mock_layer_a_res.processing_time_ms = 12.5
        mock_layer_a_res.get_risk_score.return_value = 85.0
        
        pipeline.layer_a_analyze = MagicMock(return_value=mock_layer_a_res)
        
        # Call detect
        res = pipeline.detect(
            "dangerous prompt",
            workload_id="work-123",
            trace_id="trace-456",
            span_id="span-789"
        )
        
        # Verify result is a PipelineResult
        self.assertEqual(res.final_verdict, FinalVerdict.BLOCK)
        self.assertEqual(res.decision_layer, DecisionLayer.LAYER_A)
        
        # Check telemetry was emitted with expected arguments
        mock_telemetry.emit_sampled.assert_called_once()
        call_kwargs = mock_telemetry.emit_sampled.call_args[1]
        
        self.assertEqual(call_kwargs["event_type"], "pipeline_run")
        self.assertEqual(call_kwargs["workload_id"], "work-123")
        self.assertEqual(call_kwargs["trace_id"], "trace-456")
        self.assertEqual(call_kwargs["span_id"], "span-789")
        
        payload = call_kwargs["payload"]
        self.assertEqual(payload["input_hash"], res.input_hash)
        self.assertEqual(payload["final_verdict"], "block")
        self.assertEqual(payload["decision_layer"], "A")
        self.assertEqual(payload["layer_a_verdict"], "block")
        self.assertIsNone(payload["layer_b_verdict"])
        
        metrics = call_kwargs["metrics"]
        self.assertEqual(metrics["risk_score"], 85.0)
        self.assertEqual(metrics["layer_a_time_ms"], 12.5)
        self.assertNotIn("layer_b_time_ms", metrics)

    @patch("core.orchestrator.telemetry")
    def test_detect_telemetry_emit_layer_e(self, mock_telemetry, mock_llm_judge, mock_d, mock_c, mock_b, mock_a, mock_ensure):
        pipeline = PIPipeline()
        
        # Mock Layer A to allow
        mock_layer_a_res = MagicMock()
        mock_layer_a_res.get_verdict.return_value = "allow"
        mock_layer_a_res.verdict = "allow"
        mock_layer_a_res.processed_text = "some text"
        mock_layer_a_res.confidence_score = 0.1
        mock_layer_a_res.processing_time_ms = 5.0
        mock_layer_a_res.get_risk_score.return_value = 0.0
        pipeline.layer_a_analyze = MagicMock(return_value=mock_layer_a_res)
        
        # Mock Layer B to pass / "none"
        mock_layer_b_res = MagicMock()
        mock_layer_b_res.verdict = "none"
        mock_layer_b_res.confidence_score = 0.0
        mock_layer_b_res.processing_time_ms = 2.0
        pipeline.layer_b_engine.detect = MagicMock(return_value=mock_layer_b_res)
        
        # Mock Layer C to pass / "none"
        mock_layer_c_res = MagicMock()
        mock_layer_c_res.verdict = "none"
        mock_layer_c_res.confidence_score = 0.0
        mock_layer_c_res.processing_time_ms = 3.0
        pipeline.layer_c_classifier.predict = MagicMock(return_value=mock_layer_c_res)
        
        # Mock Layer D to pass / "none"
        mock_layer_d_res = MagicMock()
        mock_layer_d_res.verdict = "none"
        mock_layer_d_res.confidence_score = 0.0
        mock_layer_d_res.processing_time_ms = 4.0
        pipeline.layer_d_classifier.predict = MagicMock(return_value=mock_layer_d_res)
        
        # Mock Layer E LLM judge
        mock_judge_out = MagicMock()
        mock_judge_out.decision = "allow"
        mock_judge_out.rationale = "Safe content"
        mock_judge_out.model = "qwen3.5:2b"
        mock_judge_out.no_think = True
        mock_judge_out.raw_response = "VERDICT: ALLOW\nRATIONALE: Safe content"
        mock_judge_out.reasoning_trace = "trace..."
        mock_judge_out.prompt_tokens = 100
        mock_judge_out.completion_tokens = 20
        mock_judge_out.total_tokens = 120
        
        pipeline.layer_e_judge.call_judge = MagicMock(return_value=mock_judge_out)
        
        # Run detect
        res = pipeline.detect("benign prompt", workload_id="w1")
        
        # Assertions on pipeline result
        self.assertEqual(res.final_verdict, FinalVerdict.ALLOW)
        self.assertEqual(res.decision_layer, DecisionLayer.LAYER_E)
        
        # Verify Layer E result structure
        self.assertIsInstance(res.layer_e_result, LayerEResult)
        self.assertEqual(res.layer_e_result.verdict, "allow")
        self.assertEqual(res.layer_e_result.rationale, "Safe content")
        self.assertEqual(res.layer_e_result.model, "qwen3.5:2b")
        self.assertEqual(res.layer_e_result.no_think, True)
        self.assertEqual(res.layer_e_result.raw_response, "VERDICT: ALLOW\nRATIONALE: Safe content")
        self.assertEqual(res.layer_e_result.reasoning_trace, "trace...")
        self.assertEqual(res.layer_e_result.prompt_tokens, 100)
        self.assertEqual(res.layer_e_result.completion_tokens, 20)
        self.assertEqual(res.layer_e_result.total_tokens, 120)
        
        # Assertions on telemetry
        mock_telemetry.emit_sampled.assert_called_once()
        call_kwargs = mock_telemetry.emit_sampled.call_args[1]
        
        self.assertEqual(call_kwargs["workload_id"], "w1")
        payload = call_kwargs["payload"]
        self.assertEqual(payload["layer_a_verdict"], "allow")
        self.assertEqual(payload["layer_b_verdict"], "none")
        self.assertEqual(payload["layer_c_verdict"], "none")
        self.assertEqual(payload["layer_d_verdict"], "none")
        self.assertEqual(payload["layer_e_verdict"], "allow")
