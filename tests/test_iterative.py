import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pathlib
from auto_labeler.strategies.discovery import IterativeDiscoveryStrategy

class TestIterativeDiscoveryStrategy(unittest.TestCase):
    def setUp(self):
        self.llm_mock = MagicMock()
        self.strategy = IterativeDiscoveryStrategy(self.llm_mock, mode="refine")
        # Mock _load_prompt to simulate YAML loading
        self.strategy._load_prompt = MagicMock(return_value="template")
        
    @patch("auto_labeler.strategies.discovery.SimpleDiscoveryStrategy")
    @patch("jinja2.Template")
    def test_run_refine(self, mock_template, mock_simple_strategy):
        # 1. Setup mocks
        mock_simple_instance = mock_simple_strategy.return_value
        mock_simple_instance.suggest_labels.return_value = ["Label1", "Label2"]
        
        mock_template_instance = mock_template.return_value
        mock_template_instance.render.return_value = "rendered_prompt"
        
        # Mock LLM response for validation pass (finding "Other" items)
        self.llm_mock.generate_structured.side_effect = [
            {"other_items": ["Item X", "Item Y"]}, # For Phase 2 (Sweep)
            {"labels": ["Label3"]} # For Phase 3 (Refine)
        ]
        
        df = pd.DataFrame({"text": ["A", "B", "C", "D", "E"]})
        # Mocking suggest_labels of refinement_strategy properly would be complex,
        # but the current logic creates a new SimpleDiscoveryStrategy.
        # Let's mock the internal SimpleDiscoveryStrategy calls.
        
        # Execute
        labels = self.strategy.suggest_labels(df, "context", pathlib.Path("."), n_labels=5)
        
        # Verify
        self.assertIn("Label1", labels)
        self.assertIn("Label2", labels)
        # Phases: Seed -> Sweep -> Refine
        # Seed gave [Label1, Label2]
        # Sweep found [Item X, Item Y]
        # Refine on [Item X, Item Y] gave [Label3]
        # Total: [Label1, Label2, Label3]
        
    @patch("jinja2.Template")
    def test_run_evolve(self, mock_template):
        self.strategy = IterativeDiscoveryStrategy(self.llm_mock, mode="evolve", batch_size=2)
        self.strategy._load_prompt = MagicMock(return_value="template")
        
        mock_template_instance = mock_template.return_value
        mock_template_instance.render.return_value = "rendered_prompt"
        
        # 4 rows, batch_size=2 -> 2 chunks
        self.llm_mock.generate_structured.side_effect = [
            {"labels": ["L1"]},
            {"labels": ["L1", "L2"]}
        ]
        
        df = pd.DataFrame({"text": ["A", "B", "C", "D"]})
        labels = self.strategy.suggest_labels(df, "context", pathlib.Path("."), n_labels=5)
        
        self.assertEqual(len(labels), 2)
        self.assertIn("L1", labels)
        self.assertIn("L2", labels)

if __name__ == '__main__':
    unittest.main()
