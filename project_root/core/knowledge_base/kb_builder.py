import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_kb(docs_dir, output_dir, model_name='all-MiniLM-L6-v2'):
    """
    Load JSON attack profiles, patterns, and semantic descriptions,
    generate embeddings, and build FAISS index.
    
    Indexes three document types for richer retrieval:
    1. attack_profiles/ — Compact profiles with MITRE ATT&CK mapping
    2. patterns/        — Detailed feature patterns with RAG keywords
    3. sematic/         — Natural language semantic descriptions
    """
    print(f"Initializing embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    documents = []
    texts_to_embed = []

    # Load attack profiles (primary documents)
    profiles_dir = os.path.join(docs_dir, "attack_profiles")
    if os.path.exists(profiles_dir):
        print(f"Loading attack profiles from {profiles_dir}...")
        for filename in sorted(os.listdir(profiles_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(profiles_dir, filename), 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
                    
                    # Create a rich text representation for embedding
                    content = f"Title: {doc['title']}\n"
                    content += f"Category: {doc['category']}\n"
                    content += f"Description: {doc['description']}\n"
                    content += f"Features: {', '.join(doc['network_indicators']['key_features'])}\n"
                    content += f"Traffic Pattern: {doc['network_indicators']['traffic_pattern']}"
                    
                    # Enrich with SHAP signature if available
                    if 'shap_signature' in doc:
                        pos = ', '.join(doc['shap_signature'].get('positive_contributors', []))
                        neg = ', '.join(doc['shap_signature'].get('negative_contributors', []))
                        content += f"\nSHAP Positive: {pos}\nSHAP Negative: {neg}"
                    
                    texts_to_embed.append(content)
        print(f"  -> Loaded {len(documents)} attack profiles")

    # Enrich embeddings with pattern documents
    patterns_dir = os.path.join(docs_dir, "patterns")
    pattern_texts = {}
    if os.path.exists(patterns_dir):
        print(f"Loading pattern documents from {patterns_dir}...")
        count = 0
        for filename in sorted(os.listdir(patterns_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(patterns_dir, filename)
                if os.path.getsize(filepath) == 0:
                    print(f"  [Skip] Empty file: {filename}")
                    continue
                with open(filepath, 'r', encoding='utf-8') as f:
                    pattern = json.load(f)
                    doc_id = pattern.get('id', '')
                    
                    # Extract all RAG keywords and feature descriptions
                    keywords = []
                    for cat in pattern.get('patterns', []):
                        for feat in cat.get('features', []):
                            keywords.extend(feat.get('rag_keywords', []))
                            keywords.append(feat.get('description', ''))
                    
                    pattern_texts[doc_id] = ' '.join(keywords)
                    count += 1
        print(f"  -> Loaded {count} pattern documents")

    # Enrich embeddings with semantic descriptions
    # (supports both "sematic" typo and "semantic" folder names)
    semantic_dir = os.path.join(docs_dir, "sematic")
    if not os.path.exists(semantic_dir):
        semantic_dir = os.path.join(docs_dir, "semantic")
    semantic_texts = {}
    if os.path.exists(semantic_dir):
        print(f"Loading semantic descriptions from {semantic_dir}...")
        count = 0
        for filename in sorted(os.listdir(semantic_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(semantic_dir, filename), 'r', encoding='utf-8') as f:
                    sem = json.load(f)
                    doc_id = sem.get('id', '')
                    descriptions = sem.get('rag_semantic_descriptions', [])
                    semantic_texts[doc_id] = ' '.join(descriptions)
                    count += 1
        print(f"  -> Loaded {count} semantic documents")

    # Merge enrichments into embedding texts
    if pattern_texts or semantic_texts:
        print("Enriching embeddings with pattern & semantic data...")
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', '')
            extra = ""
            if doc_id in pattern_texts:
                extra += f"\nPatterns: {pattern_texts[doc_id]}"
            if doc_id in semantic_texts:
                extra += f"\nSemantic: {semantic_texts[doc_id]}"
            if extra:
                texts_to_embed[i] += extra
    
    if not documents:
        print("No JSON documents found. Exiting.")
        return

    # Generate embeddings (L2-normalized so IndexFlatL2 distance == cosine similarity)
    print(f"Generating embeddings for {len(texts_to_embed)} documents...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    index_path = os.path.join(output_dir, "faiss.index")
    docs_path = os.path.join(output_dir, "documents.json")
    
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving document metadata to {docs_path}...")
    with open(docs_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
        
    print(f"Knowledge Base build complete!")
    print(f"  Index: {index.ntotal} vectors, {dimension} dimensions")
    print(f"  Enriched with: {len(pattern_texts)} patterns, {len(semantic_texts)} semantic docs")

if __name__ == "__main__":
    # Auto-detect paths relative to this script or workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # documents/ is a sibling directory to this script
    docs_dir = os.path.join(script_dir, "documents")
    output_dir = os.path.join(script_dir, "embeddings")
    
    if not os.path.exists(docs_dir):
        # Fallback: try from workspace root
        for base in ["project_root/core/knowledge_base", "core/knowledge_base"]:
            candidate = os.path.join(base, "documents")
            if os.path.exists(candidate):
                docs_dir = candidate
                output_dir = os.path.join(base, "embeddings")
                break
    
    if not os.path.exists(docs_dir):
        print(f"[Error] Documents directory not found. Tried: {docs_dir}")
        sys.exit(1)
    
    print(f"Documents dir: {docs_dir}")
    print(f"Output dir: {output_dir}")
    build_kb(docs_dir, output_dir)
