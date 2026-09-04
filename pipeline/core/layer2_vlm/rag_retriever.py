import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

class RAGRetriever:
    def __init__(self, index_path, docs_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG Retriever with a pre-built FAISS index and document metadata.
        """
        print(f"Loading RAG components (Model: {model_name})...")
        self.encoder = SentenceTransformer(model_name)
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run kb_builder.py first.")
        
        self.index = faiss.read_index(index_path)
        
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        print(f"RAG Retriever ready with {len(self.documents)} profiles.")

    def retrieve(self, shap_features: dict, top_k: int = 3, threshold: float = 0.5, raw_values: dict = None):
        """
        Build query from SHAP features, retrieve relevant context, and handle Zero-day detection.

        Args:
            shap_features: Dict of {feature_name: shap_value}
            top_k: Number of documents to retrieve
            threshold: Similarity threshold (L2 distance based).
                       Note: IndexFlatL2 on L2-normalized vectors == cosine distance,
                       range [0, 4], 0.0 is exact match. Smaller is more similar.
            raw_values: Optional dict of {feature_name: original unscaled value}, so the
                        query can cite concrete numbers (e.g. "Dst Port 22") instead of
                        only "high"/"low" — SHAP values are computed on scaled features
                        and don't carry the real-world magnitude.
        """
        start_time = time.time()

        # 1. Build Query String
        query = self._build_query(shap_features, raw_values)
        
        # 2. Encode Query (L2-normalized to match the normalized index -> L2 dist == cosine dist)
        query_vec = self.encoder.encode([query], normalize_embeddings=True).astype('float32')
        
        # 3. Search Index
        # distances is a list of squared L2 distances
        distances, indices = self.index.search(query_vec, top_k)
        
        latency = (time.time() - start_time) * 1000
        
        results = []
        is_potential_zero_day = True
        
        for dist, idx in zip(distances[0], indices[0]):
            # Cosine distance (via normalized L2), lower is better, range [0, 4].
            # 0.0 is exact match; distances < 1.0 are usually good matches.
            confidence = 1.0 / (1.0 + dist) # Simple heuristic to map [0, inf] to [1, 0]
            
            doc = self.documents[idx]
            
            if confidence > threshold:
                is_potential_zero_day = False
            
            results.append({
                "doc": doc,
                "confidence": confidence,
                "distance": float(dist)
            })
            
        return {
            "query": query,
            "matches": results,
            "is_potential_zero_day": is_potential_zero_day,
            "latency_ms": latency
        }

    def _build_query(self, shap_features: dict, raw_values: dict = None) -> str:
        """
        Convert SHAP features into a natural language query for retrieval.

        When `raw_values` is provided, cites the actual unscaled feature value
        (e.g. "Dst Port = 22") instead of only "high"/"low" — SHAP direction
        alone can't distinguish e.g. SSH (port 22) from FTP (port 21) traffic.
        """
        if not shap_features:
            return "General network anomaly detection"

        # Sort features by absolute SHAP value (importance)
        sorted_features = sorted(
            shap_features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5] # Use top 5 features

        parts = []
        for feature, value in sorted_features:
            direction = "high" if value > 0 else "low"
            if raw_values and feature in raw_values:
                parts.append(f"{direction} {feature} (= {raw_values[feature]})")
            else:
                parts.append(f"{direction} {feature}")

        query = f"Network attack or anomaly characterized by: {', '.join(parts)}"
        return query

    def format_context_for_vlm(self, retrieval_result: dict) -> str:
        """
        Format the retrieved documents into a string suitable for VLM prompt injection.
        """
        context = "### RETRIEVED CYBER THREAT INTELLIGENCE (MITRE ATT&CK)\n"
        
        if retrieval_result["is_potential_zero_day"]:
            context += "> [!WARNING]\n"
            context += "> The observed behavior has LOW similarity to known attack patterns. "
            context += "This may be a NOVEL ATTACK or ZERO-DAY exploit.\n\n"
        
        for i, match in enumerate(retrieval_result["matches"]):
            doc = match["doc"]
            conf = match["confidence"]
            
            context += f"#### Match {i+1} (Confidence: {conf:.2f})\n"
            context += f"- **Technique:** {doc['title']} ({doc['mitre_id']})\n"
            context += f"- **Tactic:** {doc['mitre_tactic']}\n"
            context += f"- **Description:** {doc['description']}\n"
            context += f"- **Typical Pattern:** {doc['network_indicators']['traffic_pattern']}\n"
            context += f"- **Mitigation:** {', '.join([m['name'] for m in doc['mitigation']])}\n"
            if doc.get('mitre_url') and doc['mitre_url'] != "N/A":
                context += f"- **Reference:** {doc['mitre_url']}\n"
            context += "\n"
            
        return context

if __name__ == "__main__":
    # Unit Test / Demo
    try:
        # Paths relative to project root
        idx_path = "core/knowledge_base/embeddings/faiss.index"
        docs_path = "core/knowledge_base/embeddings/documents.json"
        
        # If running from project root, we need to handle paths
        if not os.path.exists(idx_path):
            idx_path = "project_root/" + idx_path
            docs_path = "project_root/" + docs_path

        retriever = RAGRetriever(idx_path, docs_path)
        
        # Mock SHAP features for a SYN Flood
        mock_shap = {
            "SYN Flag Cnt": 0.85,
            "Fwd Pkts/s": 0.72,
            "Flow Duration": -0.45,
            "Dst Port": 0.1
        }
        
        print("\n--- Testing Retrieval (SYN Flood pattern) ---")
        result = retriever.retrieve(mock_shap)
        print(f"Query: {result['query']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        print(f"Is Zero-day: {result['is_potential_zero_day']}")
        
        context = retriever.format_context_for_vlm(result)
        print("\nFormatted Context for VLM:")
        print(context)
        
        # Mock SHAP for something completely unknown
        mock_unknown = {
            "Atypical_Feature_X": 0.99,
            "Mystery_Metric": 0.88
        }
        print("\n--- Testing Retrieval (Unknown pattern) ---")
        result_unknown = retriever.retrieve(mock_unknown, threshold=0.6)
        print(f"Is Zero-day: {result_unknown['is_potential_zero_day']}")
        print(f"Top match confidence: {result_unknown['matches'][0]['confidence']:.2f}")

    except Exception as e:
        print(f"Error during test: {e}")
