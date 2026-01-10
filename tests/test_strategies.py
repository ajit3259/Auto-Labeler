import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from auto_labeler.strategies.labeling import ConsensusLabelingStrategy, SimpleLabelingStrategy
from auto_labeler.strategies.discovery import ParallelDiscoveryStrategy
from auto_labeler.llm import LLMAdapter
import pathlib

class TestStrategies(unittest.TestCase):
    def setUp(self):
        # Mock LLMAdapter to avoid API calls
        self.mock_llm_cls = patch('auto_labeler.strategies.labeling.LLMAdapter').start()
        self.mock_llm_instance = self.mock_llm_cls.return_value
        
        # Strategies often load prompts from disk. We can mock _load_prompt
        # But for integration testing, let's allow it to read real files or mock it if needed.
        # Let's mock _load_prompt to avoid file I/O dependency in unit tests
        self.patcher_load = patch.object(SimpleLabelingStrategy, '_load_prompt', return_value="Template provided context {context}")
        self.mock_load = self.patcher_load.start()
        
    def tearDown(self):
        patch.stopall()

    def test_simple_strategy_with_examples(self):
        # Verify examples are rendered
        llm_mock = MagicMock()
        llm_mock.generate_structured.return_value = {"label": "TestLabel"}
        
        strategy = SimpleLabelingStrategy(llm_mock)
        
        # Overwrite _load_prompt for this instance/class to verify rendering
        # We need to rely on the actual implementation's jinja rendering
        # But since we mocked _load_prompt in setUp, we are bypassing the yaml read.
        # Let's verify that 'examples' are passed to render.
        
        # Real jinja template required to test render?
        # Let's supply a template that uses examples logic
        strategy._load_prompt = MagicMock(return_value="Context: {context} Examples: {% if examples %}{{ examples[0].label }}{% endif %}")
        
        df = pd.DataFrame({"text": ["foo"]})
        examples = [{"text": "ex", "label": "ExLabel"}]
        
        strategy.label(df, ["A"], "ctx", pathlib.Path("."), examples=examples)
        
        # Check if the prompt generated contains the example label
        # args[0] of generate_structured is the prompt
        args, _ = llm_mock.generate_structured.call_args
        prompt = args[0]
        self.assertIn("ExLabel", prompt)

    def test_consensus_strategy_voting(self):
        # We need to mock the adapters in ConsensusStrategy
        models = ["model1", "model2", "model3"]
        
        # When specific models are initialized, they return our mock
        with patch('auto_labeler.strategies.labeling.LLMAdapter') as mock_adapter_cls:
            # We want different return values for different instances to simulate voting
            # But the strategy inits them in __init__.
            
            # Let's instantiate strategy
            strategy = ConsensusLabelingStrategy(models, "adjudicator")
            
            # Now strategy.adapters has 3 mocks. Let's configure them.
            # Mock unanimous vote
            strategy.adapters[0].generate_structured.return_value = {"label": "A"}
            strategy.adapters[1].generate_structured.return_value = {"label": "A"}
            strategy.adapters[2].generate_structured.return_value = {"label": "A"}
            
            # Mock load prompt
            strategy._load_prompt = MagicMock(return_value="Prompt")
            
            df = pd.DataFrame({"text": ["foo"]})
            result = strategy.label(df, ["A", "B"], "ctx", pathlib.Path("."))
            
            self.assertEqual(result.iloc[0]["predicted_label"], "A")
            self.assertEqual(result.iloc[0]["confidence_level"], "High (Unanimous)")

    def test_consensus_strategy_conflict(self):
        models = ["model1", "model2"]
        # We don't need to patch the class constructor if we just overwrite the adapters list
        # But we need to patch it to avoid real network calls during init
        with patch('auto_labeler.strategies.labeling.LLMAdapter'):
            strategy = ConsensusLabelingStrategy(models, "adjudicator")
            
            # Manually replace adapters with distinct MagicMocks to prevent "Singleton Mock" issue
            mock1 = MagicMock()
            mock1.generate_structured.return_value = {"label": "A"}
            
            mock2 = MagicMock()
            mock2.generate_structured.return_value = {"label": "B"}
            
            strategy.adapters = [mock1, mock2]
            
            # Adjudicator decides A
            strategy.adjudicator = MagicMock()
            strategy.adjudicator.generate_structured.return_value = {"label": "A"}
            
            strategy._load_prompt = MagicMock(return_value="Prompt")
            
            df = pd.DataFrame({"text": ["foo"]})
            result = strategy.label(df, ["A", "B"], "ctx", pathlib.Path("."))
            
            self.assertEqual(result.iloc[0]["predicted_label"], "A")
            self.assertEqual(result.iloc[0]["confidence_level"], "Medium (Adjudicated)")

if __name__ == '__main__':
    unittest.main()
