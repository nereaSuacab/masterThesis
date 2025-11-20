import json
import asyncio
from typing import List, Dict, Tuple
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import NoiseSensitivity
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

# Output file
output_file = "noise_sensitivity_results.txt"

def write_output(message):
    """Write message to both console and file"""
    print(message)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Clear the output file at the start
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== Noise Sensitivity Evaluation Results ===\n\n")

# Initialize LLM
llm = ChatOllama(model="llama3.1")
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

async def evaluate_sparse():
    """Evaluate Sparse RAG"""
    write_output("\n=== Evaluating Sparse RAG (NoiseSensitivity) ===")
    sparse_scores = []

    for (query_gt, gt_answer, gt_texts), (query_sparse, sparse_texts), (query_response, response) in zip(
        query_texts_pairs, 
        query_retrieved_pairs_sparse, 
        sparse_pairs
    ):
        # Sanity check
        assert query_gt == query_sparse == query_response, f"Query mismatch: {query_gt} vs {query_sparse} vs {query_response}"

        sample = SingleTurnSample(
            user_input=query_gt,
            response=response,
            reference=gt_answer,
            retrieved_contexts=sparse_texts
        )

        scorer = NoiseSensitivity(llm=evaluator_llm)
        score = await scorer.single_turn_ascore(sample)
        
        result = f"Sparse - Query: {query_gt[:60]}... Score: {score}"
        write_output(result)
        sparse_scores.append((query_gt, score))
    
    return sparse_scores

async def evaluate_dense():
    """Evaluate Dense RAG"""
    write_output("\n=== Evaluating Dense RAG (NoiseSensitivity) ===")
    dense_scores = []

    for (query_gt, gt_answer, gt_texts), (query_dense, dense_texts), (query_response, response) in zip(
        query_texts_pairs, 
        query_retrieved_pairs_dense, 
        dense_pairs
    ):
        # Sanity check
        assert query_gt == query_dense == query_response, f"Query mismatch: {query_gt} vs {query_dense} vs {query_response}"
        
        sample = SingleTurnSample(
            user_input=query_gt,
            response=response,
            reference=gt_answer,
            retrieved_contexts=dense_texts
        )

        scorer = NoiseSensitivity(llm=evaluator_llm)
        score = await scorer.single_turn_ascore(sample)
        
        result = f"Dense - Query: {query_gt[:60]}... Score: {score}"
        write_output(result)
        dense_scores.append((query_gt, score))
    
    return dense_scores

async def main():
    """Main execution function"""
    sparse_scores = await evaluate_sparse()
    dense_scores = await evaluate_dense()
    
    # Calculate and write summary statistics
    write_output("\n\n=== Summary Statistics ===")
    
    if sparse_scores:
        avg_sparse = sum(score for _, score in sparse_scores) / len(sparse_scores)
        write_output(f"Average Sparse RAG Score: {avg_sparse:.4f}")
    
    if dense_scores:
        avg_dense = sum(score for _, score in dense_scores) / len(dense_scores)
        write_output(f"Average Dense RAG Score: {avg_dense:.4f}")
    
    write_output("\nResults saved to: " + output_file)

if __name__ == "__main__":
    asyncio.run(main())