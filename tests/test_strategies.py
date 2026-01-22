import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from auto_labeler.strategies.labeling import ConsensusLabelingStrategy, SimpleLabelingStrategy
from auto_labeler.strategies.discovery import ParallelDiscoveryStrategy, SimpleDiscoveryStrategy, IterativeDiscoveryStrategy
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

    def test_simple_strategy_shuffle(self):
        # Test that shuffle works (mock df sample)
        llm_mock = MagicMock()
        strategy = SimpleDiscoveryStrategy(llm_mock, shuffle=True, sample_size=1)
        
        # Mock _load_prompt to avoid file I/O
        strategy._load_prompt = MagicMock(return_value="Prompt")
        
        df = pd.DataFrame({"text": ["A", "B", "C"]})
        
        # Mock df.sample to verify it's called
        with patch.object(pd.DataFrame, 'sample', return_value=pd.DataFrame({"text": ["A"]})) as mock_sample:
            strategy.suggest_labels(df, "ctx", pathlib.Path("."))
            mock_sample.assert_called()

    def test_iterative_strategy_refinement(self):
        # Test 'refine' mode
        llm_mock = MagicMock()
        strategy = IterativeDiscoveryStrategy(
            llm_mock, 
            mode="refine",
            seed_sample_size=2, 
            batch_size=5, 
            other_threshold=1
        )
        
        # Mock _load_prompt to avoid file I/O for classification prompt
        strategy._load_prompt = MagicMock(return_value="Prompt")

        with patch('auto_labeler.strategies.discovery.SimpleDiscoveryStrategy') as MockSimple:
            mock_simple_instance = MockSimple.return_value
            mock_simple_instance.suggest_labels.side_effect = [
                ["Label A"],  # Seed
                ["Label B"]   # Refine
            ]
            
            llm_mock.generate_structured.return_value = {"other_items": ["Unseen Item 1"]}
            
            df = pd.DataFrame({"text": ["A"] * 10})
            labels = strategy.suggest_labels(df, "ctx", pathlib.Path("."), n_labels=10)
            
            self.assertIn("Label A", labels)
            self.assertIn("Label B", labels)

    def test_iterative_strategy_aggregate(self):
        # Test 'aggregate' mode (Independent batches -> Merge)
        llm_mock = MagicMock()
        strategy = IterativeDiscoveryStrategy(
            llm_mock, 
            mode="aggregate",
            batch_size=2
        )
        
        strategy._load_prompt = MagicMock(return_value="Prompt")

        with patch('auto_labeler.strategies.discovery.SimpleDiscoveryStrategy') as MockSimple:
            mock_simple_instance = MockSimple.return_value
            # Batch 1 -> [L1], Batch 2 -> [L2]
            mock_simple_instance.suggest_labels.side_effect = [["L1"], ["L2"]]
            
            # Merge step returns combined
            llm_mock.generate_structured.return_value = {"labels": ["L1", "L2", "L3_Merged"]}
            
            df = pd.DataFrame({"text": ["A", "B", "C", "D"]})
            labels = strategy.suggest_labels(df, "ctx", pathlib.Path("."), n_labels=10)
            
            self.assertIn("L3_Merged", labels)
            # Ensure SimpleStrategy was called twice (4 items / 2 batch_size)
            self.assertEqual(mock_simple_instance.suggest_labels.call_count, 2)

if __name__ == '__main__':
    unittest.main()
