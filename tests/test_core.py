import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from auto_labeler.core import AutoLabeler

class TestAutoLabeler(unittest.TestCase):
    def setUp(self):
        self.mock_llm_adapter = patch('auto_labeler.core.LLMAdapter').start()
        self.labeler = AutoLabeler()
        self.labeler.llm.generate_structured = MagicMock()

    def tearDown(self):
        patch.stopall()

    def test_suggest_labels(self):
        df = pd.DataFrame({"text": ["sample 1", "sample 2"]})
        self.labeler.llm.generate_structured.return_value = {"labels": ["A", "B"]}
        
        labels = self.labeler.suggest_labels(df, context="test")
        
        self.assertEqual(labels, ["A", "B"])
        self.labeler.llm.generate_structured.assert_called()

    def test_label_dataset_single_label(self):
        df = pd.DataFrame({"text": ["content 1"]})
        self.labeler.llm.generate_structured.return_value = {"label": "Positive"}
        
        result_df = self.labeler.label_dataset(df, labels=["Positive", "Negative"], context="test", multi_label=False)
        
        self.assertIn("label", result_df.columns)
        self.assertEqual(result_df.iloc[0]["label"], "Positive")

    def test_label_dataset_multi_label(self):
        df = pd.DataFrame({"text": ["content 1"]})
        self.labeler.llm.generate_structured.return_value = {"label": ["Tech", "Sales"]}

        result_df = self.labeler.label_dataset(df, labels=["Tech", "Sales"], context="test", multi_label=True)

        self.assertEqual(result_df.iloc[0]["label"], ["Tech", "Sales"])

if __name__ == '__main__':
    unittest.main()
