from typing import List, Protocol, Literal
import pandas as pd
import numpy as np
from ..llm import LLMAdapter
import pathlib
import yaml
from jinja2 import Template
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from ..logger import logger

class EmbeddingDiscoveryStrategy:
    """
    Discovery strategy that uses Embeddings + Clustering to find labels.
    Step 1: Embed a large sample of data.
    Step 2: Cluster vectors (KMeans or DBSCAN).
    Step 3: Summarize each cluster using its Centroid + Random Samples.
    """
    def __init__(
        self, 
        llm: LLMAdapter, 
        clustering_method: Literal["kmeans", "dbscan"] = "kmeans",
        n_clusters: int = 5,
        eps: float = 0.5,
        min_samples: int = 5,
        sample_size: int = 100,
        embedding_model: str = "text-embedding-3-small",
        text_column: str = "text"
    ):
        self.llm = llm
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.sample_size = sample_size
        self.embedding_model = embedding_model
        self.text_column = text_column

    def _load_prompt(self, prompts_dir: pathlib.Path, prompt_name: str) -> str:
        with open(prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        prompts_dir: pathlib.Path, 
        n_labels: int = 5
    ) -> List[str]:
        # 1. Sample Data
        n = min(len(df), self.sample_size)
        sample_df = df.sample(n)
        texts = sample_df[self.text_column].tolist()

        
        # 2. Embed
        embeddings = self.llm.get_embedding(texts, model=self.embedding_model)
        X = np.array(embeddings)
        
        # 2b. Normalize embeddings (Important for high-dimensional cosine similarity via KMeans)
        # This also helps avoid sklearn RuntimeWarnings with extreme values
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        X = X / norms
        
        # 3. Cluster
        if self.clustering_method == "kmeans":
            # If requested clusters > sample size, cap it
            k = min(self.n_clusters, len(X))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(X)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
        else: # DBSCAN
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(X)
            # Compute centroids manually for DBSCAN
            unique_labels = set(labels)
            centroids = []
            valid_cluster_indices = []
            
            for l in unique_labels:
                if l == -1: continue # Noise
                cluster_points = X[labels == l]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)
                valid_cluster_indices.append(l)

            if not centroids:
                return [] # No clusters found
                
            centroids = np.array(centroids)

        # 4. Summarize Clusters
        discovered_labels = set()
        cluster_prompt_template = self._load_prompt(prompts_dir, "discovery_cluster")
        
        # Find closest points to each centroid
        closest, _ = pairwise_distances_argmin_min(centroids, X)
        
        unique_cluster_ids = set(labels) if self.clustering_method == "kmeans" else set(valid_cluster_indices)
        
        for i, cluster_id in enumerate(unique_cluster_ids):
            if cluster_id == -1: continue
            
            # Get centroid text
            centroid_idx = closest[i]
            centroid_text = texts[centroid_idx]
            
            # Get random samples from cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            # Exclude centroid if possible
            other_indices = [idx for idx in cluster_indices if idx != centroid_idx]
            
            if len(other_indices) >= 5:
                # Pick 5 random
                sample_indices = np.random.choice(other_indices, 5, replace=False)
            else:
                sample_indices = other_indices
            
            cluster_samples = [texts[idx] for idx in sample_indices]
            
            # Formulate prompt
            template = Template(cluster_prompt_template)
            prompt = template.render(
                context=context,
                centroid=centroid_text,
                samples=cluster_samples
            )
            
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "A short, descriptive label for this cluster."}
                    },
                    "required": ["label"]
                }
                response = self.llm.generate_structured(prompt, response_schema=schema)
                label = response.get("label")
                if label:
                    discovered_labels.add(label)
            except Exception as e:
                logger.error(f"Error discovering label for cluster {cluster_id}: {e}")
                continue
                
        return list(discovered_labels)[:n_labels]
