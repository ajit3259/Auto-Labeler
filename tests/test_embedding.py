import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pathlib
from auto_labeler.strategies.embedding import EmbeddingDiscoveryStrategy
from sklearn.cluster import KMeans

class TestEmbeddingStrategy(unittest.TestCase):
    def setUp(self):
        self.llm_mock = MagicMock()
        self.strategy = EmbeddingDiscoveryStrategy(self.llm_mock, clustering_method="kmeans", n_clusters=2, text_column="text")
        # Mock prompt loading
        self.strategy._load_prompt = MagicMock(return_value="Prompt with {centroid}")
        
    def test_kmeans_clustering(self):
        # Mock embeddings: 2 clear clusters
        # Cluster 0: [0,0], [0.1, 0.1]
        # Cluster 1: [10,10], [10.1, 10.1]
        embeddings = [
            [0, 0], [0.1, 0.1], 
            [10, 10], [10.1, 10.1]
        ]
        self.llm_mock.get_embedding.return_value = embeddings
        
        # Mock LLM generation for labeling
        self.llm_mock.generate_structured.side_effect = [
            {"label": "Cluster A"},
            {"label": "Cluster B"}
        ]
        
        df = pd.DataFrame({"text": ["A1", "A2", "B1", "B2"]})
        labels = self.strategy.suggest_labels(df, "ctx", pathlib.Path("."), n_labels=2)
        
        # Should call LLM get_embedding once
        self.llm_mock.get_embedding.assert_called_once()
        
        # Should return 2 labels
        self.assertEqual(len(labels), 2)
        self.assertIn("Cluster A", labels)
        self.assertIn("Cluster B", labels)

    def test_dbscan_clustering(self):
        self.strategy = EmbeddingDiscoveryStrategy(
            self.llm_mock, 
            clustering_method="dbscan", 
            eps=0.5, 
            min_samples=2,
            text_column="text"
        )
        self.strategy._load_prompt = MagicMock(return_value="Prompt")
        
        # Embeddings in two distinct directions so they remain separate after L2 normalisation.
        # Group A points along +x, Group B points along +y.
        embeddings = [
            [1.0, 0.0], [0.95, 0.05],
            [0.0, 1.0], [0.05, 0.95],
        ]
        self.llm_mock.get_embedding.return_value = embeddings
        self.llm_mock.generate_structured.side_effect = [{"label": "L1"}, {"label": "L2"}]
        
        df = pd.DataFrame({"text": ["A1", "A2", "B1", "B2"]})
        labels = self.strategy.suggest_labels(df, "ctx", pathlib.Path("."))
        
        self.assertEqual(len(labels), 2)

if __name__ == '__main__':
    unittest.main()
