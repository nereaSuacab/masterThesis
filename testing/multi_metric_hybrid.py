"""
Comprehensive RAG Evaluation for HYBRID RAG only
Evaluates: Faithfulness, Context Precision, Context Recall, Noise Sensitivity

Uses RAGAS metrics with DeepSeek-R1 as evaluator LLM.
"""

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
os.environ["OPENAI_API_BASE"] = "https://chat.campusai.compute.dtu.dk/api/v1"

# Output files
output_file = "hybrid_evaluation_results.txt"
arrays_file = "hybrid_metric_arrays.py"

def write_output(message):
    """Write message to both console and file"""
    print(message)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Clear the output file at the start
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== Hybrid RAG Evaluation Results ===\n\n")

# Initialize LLM
llm = ChatOpenAI(
    model="DeepSeek-R1",
    temperature=0,
    max_retries=3,
    request_timeout=400
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

# Load hybrid retrieval results
path = r"retrieval_results_hybrid.json"
with open(path, "r", encoding="utf-8") as f:
    hybrid_data = json.load(f)

# Extract query-retrieved pairs from hybrid results
query_retrieved_pairs_hybrid = []
for item in hybrid_data:
    query = item["query"]
    retrieved_texts = [res["text"].strip() for res in item["results"]]
    query_retrieved_pairs_hybrid.append((query, retrieved_texts))

first_query, first_texts = query_retrieved_pairs_hybrid[0]
write_output("\nHybrid Retrieval Results Loaded:")
write_output(f"Query: {first_query}")
write_output(f"Number of retrieved texts: {len(first_texts)}")
write_output(f"First retrieved text snippet: {first_texts[0][:200].replace(chr(10), ' ')}...\n")

# Show retrieval statistics
if hybrid_data:
    first_item = hybrid_data[0]
    if "statistics" in first_item:
        stats = first_item["statistics"]
        write_output("Hybrid Retrieval Statistics:")
        write_output(f"  From Dense only: {stats.get('from_dense_only', 0)}")
        write_output(f"  From BM25 only: {stats.get('from_bm25_only', 0)}")
        write_output(f"  From Both (high confidence): {stats.get('from_both', 0)}\n")

# Load chat logs
path = r"chat_logs_hybrid.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

hybrid_pairs = []
for item in data:
    query = item.get("query", "").strip()
    mode = item.get("mode", "").strip().lower()
    response = item.get("response", "").strip()
    
    # Extract just the response text (before sources)
    if "<hr>Sources:" in response:
        response = response.split("<hr>Sources:")[0].strip()
    
    if "hybrid" in mode:
        hybrid_pairs.append((query, response))

write_output("\nChat Logs Loaded:")
write_output(f"Hybrid RAG pairs found: {len(hybrid_pairs)}")

# Show fusion configuration
if data:
    first_item = data[0]
    write_output(f"\nFusion Configuration:")
    write_output(f"  Method: {first_item.get('fusion_method', 'N/A')}")
    weights = first_item.get('weights', {})
    write_output(f"  Dense weight: {weights.get('dense', 'N/A')}")
    write_output(f"  BM25 weight: {weights.get('bm25', 'N/A')}\n")

async def evaluate_hybrid_metrics(query_texts_pairs, query_retrieved_pairs, response_pairs):
    """
    Evaluate all metrics for Hybrid RAG
    
    Args:
        query_texts_pairs: Ground truth data
        query_retrieved_pairs: Retrieved contexts from Hybrid
        response_pairs: Generated responses from Hybrid
    
    Returns:
        Dictionary with metric scores and arrays
    """
    write_output(f"\n{'='*60}")
    write_output(f"=== Evaluating Hybrid RAG - All Metrics ===")
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
        assert query_gt == query_ret == query_resp, f"Query mismatch at index {idx}: {query_gt} vs {query_ret} vs {query_resp}"
        
        write_output(f"\n[Hybrid RAG - Query {idx}/{len(query_texts_pairs)}]")
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

def write_arrays_to_file(hybrid_arrays):
    """Write metric arrays to a Python file"""
    with open(arrays_file, "w", encoding="utf-8") as f:
        f.write("# Metric Arrays for Hybrid RAG Evaluation\n")
        f.write("# Generated automatically from evaluation results\n\n")
        
        # Hybrid arrays
        f.write("# Hybrid RAG Metrics\n")
        f.write(f"faithfulness_hybrid = {hybrid_arrays['faithfulness']}\n")
        f.write(f"context_precision_hybrid = {hybrid_arrays['context_precision']}\n")
        f.write(f"context_recall_hybrid = {hybrid_arrays['context_recall']}\n")
        f.write(f"noise_sensitivity_hybrid = {hybrid_arrays['noise_sensitivity']}\n")

async def main():
    """Main execution function"""
    
    # Verify data alignment
    write_output(f"\n{'='*60}")
    write_output("=== DATA VERIFICATION ===")
    write_output(f"{'='*60}\n")
    write_output(f"Ground truth queries: {len(query_texts_pairs)}")
    write_output(f"Hybrid retrieved contexts: {len(query_retrieved_pairs_hybrid)}")
    write_output(f"Hybrid responses: {len(hybrid_pairs)}")
    
    if not (len(query_texts_pairs) == len(query_retrieved_pairs_hybrid) == len(hybrid_pairs)):
        write_output("\n⚠️  WARNING: Data counts don't match!")
        write_output("Make sure you ran the hybrid pipeline on all queries in gt.json")
        min_len = min(len(query_texts_pairs), len(query_retrieved_pairs_hybrid), len(hybrid_pairs))
        write_output(f"Proceeding with first {min_len} queries only.\n")
        
        # Truncate to minimum length
        query_texts_pairs_eval = query_texts_pairs[:min_len]
        query_retrieved_pairs_eval = query_retrieved_pairs_hybrid[:min_len]
        hybrid_pairs_eval = hybrid_pairs[:min_len]
    else:
        write_output("✓ All data counts match!\n")
        query_texts_pairs_eval = query_texts_pairs
        query_retrieved_pairs_eval = query_retrieved_pairs_hybrid
        hybrid_pairs_eval = hybrid_pairs
    
    # Evaluate Hybrid RAG
    hybrid_scores, hybrid_arrays = await evaluate_hybrid_metrics(
        query_texts_pairs_eval,
        query_retrieved_pairs_eval,
        hybrid_pairs_eval
    )
    
    # Write arrays to file
    write_arrays_to_file(hybrid_arrays)
    
    # Calculate and write summary statistics
    write_output("\n\n" + "="*60)
    write_output("=== SUMMARY STATISTICS ===")
    write_output("="*60 + "\n")
    
    metric_names = ['faithfulness', 'context_precision', 'context_recall', 'noise_sensitivity']
    
    write_output("Hybrid RAG Results:")
    write_output("-" * 40)
    for metric_name in metric_names:
        scores = [score for _, score in hybrid_scores[metric_name] if score is not None]
        if scores:
            avg = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            write_output(f"{metric_name.replace('_', ' ').title()}:")
            write_output(f"  Average: {avg:.4f}")
            write_output(f"  Min:     {min_score:.4f}")
            write_output(f"  Max:     {max_score:.4f}")
            write_output(f"  Range:   {max_score - min_score:.4f}\n")
        else:
            write_output(f"{metric_name.replace('_', ' ').title()}: N/A\n")
    
    # Detailed per-query breakdown
    write_output("\n" + "="*60)
    write_output("=== PER-QUERY BREAKDOWN ===")
    write_output("="*60 + "\n")
    
    for idx, (query, _) in enumerate(hybrid_pairs_eval, 1):
        write_output(f"Query {idx}: {query[:60]}...")
        for metric_name in metric_names:
            if idx <= len(hybrid_scores[metric_name]):
                _, score = hybrid_scores[metric_name][idx-1]
                if score is not None:
                    write_output(f"  {metric_name.replace('_', ' ').title()}: {score:.4f}")
        write_output("")
    
    # Best and worst performing queries
    write_output("\n" + "="*60)
    write_output("=== BEST & WORST QUERIES ===")
    write_output("="*60 + "\n")
    
    for metric_name in metric_names:
        scores_with_queries = [(query, score) for query, score in hybrid_scores[metric_name] if score is not None]
        if scores_with_queries:
            scores_with_queries.sort(key=lambda x: x[1])
            
            write_output(f"{metric_name.replace('_', ' ').title()}:")
            write_output(f"  Best:  {scores_with_queries[-1][1]:.4f} - {scores_with_queries[-1][0][:60]}...")
            write_output(f"  Worst: {scores_with_queries[0][1]:.4f} - {scores_with_queries[0][0][:60]}...\n")
    
    write_output("\n" + "="*60)
    write_output(f"✅ Evaluation complete!")
    write_output(f"Results saved to: {output_file}")
    write_output(f"Metric arrays saved to: {arrays_file}")
    write_output("="*60)

if __name__ == "__main__":
    asyncio.run(main())