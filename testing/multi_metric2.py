import json
import asyncio
from typing import List, Dict, Tuple
from ragas import SingleTurnSample
from ragas.metrics import (
    NonLLMContextPrecisionWithReference,
    LLMContextPrecisionWithReference,
    LLMContextPrecisionWithoutReference,
    NoiseSensitivity,
    Faithfulness,
    ContextRecall,
    ContextPrecision
)
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

import os
os.environ["OPENAI_API_KEY"] = "sk-ea25363437e1476fadd3e65759d42903"
os.environ["OPENAI_API_BASE"] = "https://chat.campusai.compute.dtu.dk/api/v1"  # or whatever DTU's endpoint is

# Output files
output_file = "comprehensive_evaluation_results.txt"
arrays_file = "metric_arrays.py"

def write_output(message):
    """Write message to both console and file"""
    print(message)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Clear the output file at the start
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== Comprehensive RAG Evaluation Results ===\n\n")

# Initialize LLM
# llm = ChatOllama(model="llama3.1")
# evaluator_llm = LangchainLLMWrapper(llm)

llm = ChatOpenAI(
    model="DeepSeek-R1",  # or "DeepSeek-R1" - try both if one doesn't work
    temperature=0,
    max_retries=3,
    request_timeout=400  # Increase timeout for complex evaluations
)
evaluator_llm = LangchainLLMWrapper(llm)


# Load ground truth data
path = r"gt.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

query_texts_pairs = [
    (
        item["question"].strip(),
        item["reference"].strip(),
        [seg.strip() for seg in item.get("context", [])]
    )
    for item in data
]

first_query, first_reference, first_contexts = query_texts_pairs[0]
write_output("Ground Truth Data Loaded:")
write_output(f"Query: {first_query}")
write_output(f"Reference answer: {first_reference}")
write_output(f"Number of context texts: {len(first_contexts)}")
write_output(f"First context snippet: {first_contexts[0][:200].replace(chr(10), ' ')}...\n")

# Load dense retrieval results
path = r"retrieval_results_dense.json"
with open(path, "r", encoding="utf-8") as f:
    rag_data_dense = json.load(f)

query_retrieved_pairs_dense = [
    (
        query,
        [item["text"].strip() for item in texts]
    )
    for query, texts in rag_data_dense.items()
]

first_query, first_texts = query_retrieved_pairs_dense[0]
write_output("\nDense Retrieval Results Loaded:")
write_output(f"Query: {first_query}")
write_output(f"Number of retrieved texts: {len(first_texts)}")
write_output(f"First retrieved text snippet: {first_texts[0][:200].replace(chr(10), ' ')}...\n")

# Load sparse retrieval results
path = r"retrieval_results_sparse.json"
with open(path, "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

query_retrieved_pairs_sparse = [
    (
        item["query"],
        [res["window"].strip() for res in item["results"]]
    )
    for item in bm25_data
]

first_query, first_texts = query_retrieved_pairs_sparse[0]
write_output("\nSparse Retrieval Results Loaded:")
write_output(f"Query: {first_query}")
write_output(f"Number of retrieved texts: {len(first_texts)}")
write_output(f"First retrieved text snippet: {first_texts[0][:200].replace(chr(10), ' ')}...\n")

# Load chat logs
path = r"chat_logs.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dense_pairs = []
sparse_pairs = []

for item in data:
    query = item.get("query", "").strip()
    mode = item.get("mode", "").strip().lower()
    response = item.get("response", "").strip()
    
    if mode == "dense rag":
        dense_pairs.append((query, response))
    elif mode == "sparse rag":
        sparse_pairs.append((query, response))

write_output("\nChat Logs Loaded:")
write_output(f"Dense pairs found: {len(dense_pairs)}")
write_output(f"Sparse pairs found: {len(sparse_pairs)}\n")

async def evaluate_metrics(mode, query_texts_pairs, query_retrieved_pairs, response_pairs):
    """
    Evaluate all metrics for a given RAG mode
    
    Args:
        mode: "Sparse" or "Dense"
        query_texts_pairs: Ground truth data
        query_retrieved_pairs: Retrieved contexts
        response_pairs: Generated responses
    
    Returns:
        Dictionary with metric scores and arrays
    """
    write_output(f"\n{'='*60}")
    write_output(f"=== Evaluating {mode} RAG - All Metrics ===")
    write_output(f"{'='*60}\n")
    
    # Initialize metrics
    metrics = {
        'faithfulness': Faithfulness(llm=evaluator_llm),
        'context_precision': ContextPrecision(llm=evaluator_llm),
        'context_recall': ContextRecall(llm=evaluator_llm),
        'noise_sensitivity': NoiseSensitivity(llm=evaluator_llm)
    }
    
    # Store all scores
    all_scores = {metric_name: [] for metric_name in metrics.keys()}
    metric_arrays = {metric_name: [] for metric_name in metrics.keys()}
    
    # Evaluate each query
    for idx, ((query_gt, gt_answer, gt_texts), (query_ret, ret_texts), (query_resp, response)) in enumerate(zip(
        query_texts_pairs, 
        query_retrieved_pairs, 
        response_pairs
    ), 1):
        # Sanity check
        assert query_gt == query_ret == query_resp, f"Query mismatch: {query_gt} vs {query_ret} vs {query_resp}"
        
        write_output(f"\n[{mode} RAG - Query {idx}/{len(query_texts_pairs)}]")
        write_output(f"Query: {query_gt[:80]}...")
        
        # Create sample
        sample = SingleTurnSample(
            user_input=query_gt,
            response=response,
            reference=gt_answer,
            retrieved_contexts=ret_texts
        )
        
        # Evaluate each metric
        query_scores = {}
        for metric_name, scorer in metrics.items():
            try:
                score = await scorer.single_turn_ascore(sample)
                query_scores[metric_name] = score
                all_scores[metric_name].append((query_gt, score))
                metric_arrays[metric_name].append(round(score, 2))
                write_output(f"  {metric_name.replace('_', ' ').title()}: {score:.4f}")
            except Exception as e:
                write_output(f"  {metric_name.replace('_', ' ').title()}: ERROR - {str(e)}")
                query_scores[metric_name] = None
                metric_arrays[metric_name].append(0)
    
    return all_scores, metric_arrays

def write_arrays_to_file(dense_arrays, sparse_arrays):
    """Write metric arrays to a Python file"""
    with open(arrays_file, "w", encoding="utf-8") as f:
        f.write("# Metric Arrays for RAG Evaluation\n")
        f.write("# Generated automatically from evaluation results\n\n")
        
        # Dense arrays
        f.write("# Dense RAG Metrics\n")
        f.write(f"faithfulness_dense = {dense_arrays['faithfulness']}\n")
        f.write(f"context_precision_dense = {dense_arrays['context_precision']}\n")
        f.write(f"context_recall_dense = {dense_arrays['context_recall']}\n")
        f.write(f"noise_sensitivity_dense = {dense_arrays['noise_sensitivity']}\n\n")
        
        # Sparse arrays
        f.write("# Sparse RAG Metrics\n")
        f.write(f"faithfulness_sparse = {sparse_arrays['faithfulness']}\n")
        f.write(f"context_precision_sparse = {sparse_arrays['context_precision']}\n")
        f.write(f"context_recall_sparse = {sparse_arrays['context_recall']}\n")
        f.write(f"noise_sensitivity_sparse = {sparse_arrays['noise_sensitivity']}\n")

async def main():
    """Main execution function"""
    
    # Evaluate Dense RAG
    dense_scores, dense_arrays = await evaluate_metrics(
        "Dense",
        query_texts_pairs,
        query_retrieved_pairs_dense,
        dense_pairs
    )
    
    # Evaluate Sparse RAG
    sparse_scores, sparse_arrays = await evaluate_metrics(
        "Sparse",
        query_texts_pairs,
        query_retrieved_pairs_sparse,
        sparse_pairs
    )
    
    # Write arrays to file
    write_arrays_to_file(dense_arrays, sparse_arrays)
    
    # Calculate and write summary statistics
    write_output("\n\n" + "="*60)
    write_output("=== SUMMARY STATISTICS ===")
    write_output("="*60 + "\n")
    
    metric_names = ['faithfulness', 'context_precision', 'context_recall', 'noise_sensitivity']
    
    write_output("Dense RAG Averages:")
    write_output("-" * 40)
    for metric_name in metric_names:
        scores = [score for _, score in dense_scores[metric_name] if score is not None]
        if scores:
            avg = sum(scores) / len(scores)
            write_output(f"  {metric_name.replace('_', ' ').title()}: {avg:.4f}")
        else:
            write_output(f"  {metric_name.replace('_', ' ').title()}: N/A")
    
    write_output("\nSparse RAG Averages:")
    write_output("-" * 40)
    for metric_name in metric_names:
        scores = [score for _, score in sparse_scores[metric_name] if score is not None]
        if scores:
            avg = sum(scores) / len(scores)
            write_output(f"  {metric_name.replace('_', ' ').title()}: {avg:.4f}")
        else:
            write_output(f"  {metric_name.replace('_', ' ').title()}: N/A")
    
    # Comparison
    write_output("\n" + "="*60)
    write_output("=== COMPARISON (Dense vs Sparse) ===")
    write_output("="*60 + "\n")
    
    for metric_name in metric_names:
        sparse_vals = [score for _, score in sparse_scores[metric_name] if score is not None]
        dense_vals = [score for _, score in dense_scores[metric_name] if score is not None]
        
        if sparse_vals and dense_vals:
            avg_sparse = sum(sparse_vals) / len(sparse_vals)
            avg_dense = sum(dense_vals) / len(dense_vals)
            diff = avg_dense - avg_sparse
            diff_pct = (diff / avg_sparse * 100) if avg_sparse != 0 else 0
            
            winner = "Dense" if diff > 0 else "Sparse" if diff < 0 else "Tie"
            
            write_output(f"{metric_name.replace('_', ' ').title()}:")
            write_output(f"  Sparse: {avg_sparse:.4f}")
            write_output(f"  Dense:  {avg_dense:.4f}")
            write_output(f"  Diff:   {diff:+.4f} ({diff_pct:+.2f}%)")
            write_output(f"  Winner: {winner}\n")
    
    write_output("\n" + "="*60)
    write_output(f"Results saved to: {output_file}")
    write_output(f"Metric arrays saved to: {arrays_file}")
    write_output("="*60)

if __name__ == "__main__":
    asyncio.run(main())