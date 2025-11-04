import json
import asyncio
from typing import List, Dict, Tuple
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference
from ragas.metrics import LLMContextPrecisionWithReference
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

# Initialize LLM
llm = ChatOllama(model="llama3.1", format="json")
evaluator_llm = LangchainLLMWrapper(llm)

def load_ground_truth(path: str) -> List[Tuple[str, List[str]]]:
    """Load ground truth data and return query-texts pairs."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    query_texts_pairs = [
        (
            item["question"].strip(),
            [seg["text"].strip() for seg in item.get("ground_truth_segments", [])]
        )
        for item in data
    ]
    
    print(f"Loaded {len(query_texts_pairs)} ground truth queries")
    return query_texts_pairs

def load_dense_retrieval(path: str) -> List[Tuple[str, List[str]]]:
    """Load dense retrieval results."""
    with open(path, "r", encoding="utf-8") as f:
        rag_data = json.load(f)
    
    query_retrieved_pairs = [
        (
            query,
            [item["text"].strip() for item in texts]
        )
        for query, texts in rag_data.items()
    ]
    
    print(f"Loaded {len(query_retrieved_pairs)} dense retrieval results")
    return query_retrieved_pairs

def load_sparse_retrieval(path: str) -> List[Tuple[str, List[str]]]:
    """Load sparse (BM25) retrieval results."""
    with open(path, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)
    
    query_retrieved_pairs = [
        (
            item["query"],
            [res["window"].strip() for res in item["results"]]
        )
        for item in bm25_data
    ]
    
    print(f"Loaded {len(query_retrieved_pairs)} sparse retrieval results")
    return query_retrieved_pairs

async def evaluate_context_precision(
    query_texts_pairs: List[Tuple[str, List[str]]],
    query_retrieved_pairs: List[Tuple[str, List[str]]],
    method_name: str
) -> List[Tuple[str, float]]:
    """Evaluate context precision for a retrieval method."""
    context_precision = NonLLMContextPrecisionWithReference()
    scores = []
    
    print(f"\nEvaluating {method_name}...")
    for i, ((query_gt, gt_texts), (query_ret, ret_texts)) in enumerate(
        zip(query_texts_pairs, query_retrieved_pairs), 1
    ):
        # Sanity check
        assert query_gt == query_ret, f"Query mismatch: {query_gt} vs {query_ret}"
        
        # Build sample
        sample = SingleTurnSample(
            retrieved_contexts=ret_texts,
            reference_contexts=gt_texts
        )
        
        # Compute score
        score = await context_precision.single_turn_ascore(sample)
        scores.append((query_gt, score))
        
        print(f"  [{i}/{len(query_texts_pairs)}] Score: {score:.4f}")
    
    return scores

async def evaluate_llm_context_precision(
    query_texts_pairs: List[Tuple[str, List[str]]],
    query_retrieved_pairs: List[Tuple[str, List[str]]],
    method_name: str
) -> List[Tuple[str, float]]:
    """Evaluate context precision using LLM for a retrieval method."""
    context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
    scores = []
    
    print(f"\nEvaluating {method_name} with LLM...")
    for i, ((query_gt, gt_texts), (query_ret, ret_texts)) in enumerate(
        zip(query_texts_pairs, query_retrieved_pairs), 1
    ):
        # Sanity check
        assert query_gt == query_ret, f"Query mismatch: {query_gt} vs {query_ret}"
        
        # Build sample
        sample = SingleTurnSample(
            user_input=query_gt,
            reference=" ".join(gt_texts),
            retrieved_contexts=ret_texts
        )
        
        # Compute score
        score = await context_precision.single_turn_ascore(sample)
        scores.append((query_gt, score))
        
        print(f"  [{i}/{len(query_texts_pairs)}] Score: {score:.4f}")
    
    return scores

def print_results(scores: List[Tuple[str, float]], title: str):
    """Print evaluation results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    total_score = 0
    for query, score in scores:
        print(f"{query[:60]}... -> {score:.4f}")
        total_score += score
    
    avg_score = total_score / len(scores) if scores else 0
    print(f"\nAverage Score: {avg_score:.4f}")
    print(f"{'='*80}\n")

def save_results(dense_scores, sparse_scores, output_path: str):
    """Save results to JSON file."""
    results = {
        "dense_rag": [
            {"query": q, "score": float(s)} for q, s in dense_scores
        ],
        "sparse_rag": [
            {"query": q, "score": float(s)} for q, s in sparse_scores
        ],
        "summary": {
            "dense_avg": sum(s for _, s in dense_scores) / len(dense_scores),
            "sparse_avg": sum(s for _, s in sparse_scores) / len(sparse_scores)
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")

async def evaluate_llm_context_precision_simple(
    query_texts_pairs: List[Tuple[str, List[str]]],
    query_retrieved_pairs: List[Tuple[str, List[str]]],
    method_name: str
) -> List[Tuple[str, float]]:
    """Evaluate context precision using LLM (simple version from notebook)."""
    context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
    scores = []
    
    print(f"\nEvaluating {method_name} with LLM (simple version)...")
    for i, ((query_gt, gt_texts), (query_ret, ret_texts)) in enumerate(
        zip(query_texts_pairs, query_retrieved_pairs), 1
    ):
        # Sanity check
        assert query_gt == query_ret, f"Query mismatch: {query_gt} vs {query_ret}"
        
        # Build sample
        sample = SingleTurnSample(
            user_input=query_gt,
            reference=" ".join(gt_texts),
            retrieved_contexts=ret_texts
        )
        
        # Compute score
        score = await context_precision.single_turn_ascore(sample)
        print(f"  [{i}/{len(query_texts_pairs)}] Score for query: {query_gt[:50]}... is {score:.4f}")
        scores.append((query_gt, score))
    
    return scores

async def main():
    """Main evaluation pipeline."""
    print("Starting evaluation pipeline...")
    
    # Load data
    print("\n--- Loading Data ---")
    query_texts_pairs = load_ground_truth("C:\\Users\\nerea\\Documents\\MasterDTU\\masterThesis\\masterThesis\\testing\\gt.json")
    query_retrieved_pairs_dense = load_dense_retrieval("C:\\Users\\nerea\\Documents\\MasterDTU\\masterThesis\\masterThesis\\testing\\retrieval_results.json")
    query_retrieved_pairs_sparse = load_sparse_retrieval("C:\\Users\\nerea\\Documents\\MasterDTU\\masterThesis\\masterThesis\\testing\\retrieval_results_sparse copy.json")

    # Evaluate NonLLM Context Precision
    print("\n--- Evaluating NonLLM Context Precision ---")
    dense_scores = await evaluate_context_precision(
        query_texts_pairs,
        query_retrieved_pairs_dense,
        "Dense RAG"
    )
    
    sparse_scores = await evaluate_context_precision(
        query_texts_pairs,
        query_retrieved_pairs_sparse,
        "Sparse RAG"
    )
    
    # Print results
    print_results(dense_scores, "Dense RAG - NonLLM Context Precision")
    print_results(sparse_scores, "Sparse RAG - NonLLM Context Precision")
    
    # Evaluate with LLM Context Precision (the slow part from your notebook)
    print("\n--- Evaluating LLM Context Precision (This will take time) ---")
    dense_llm_scores = await evaluate_llm_context_precision_simple(
        query_texts_pairs,
        query_retrieved_pairs_dense,
        "Dense RAG"
    )
    print_results(dense_llm_scores, "Dense RAG - LLM Context Precision")
    
    # Save results
    save_results(dense_scores, sparse_scores, "evaluation_results.json")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())