from sentence_transformers import SentenceTransformer, util
import json
from statistics import mean

# Load the same embedding model your RAG uses (important for consistent results)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your JSONs
with open("testing/gt.json") as f:
    ground_truth = json.load(f)

with open("testing/retrieval_results.json") as f:
    retrieved = json.load(f)

# Helper to get retrieved windows by question
def get_retrieved_windows(question, retrieved_data):
    for q, items in retrieved_data.items():
        if q.strip().lower() == question.strip().lower():
            return [i["metadata"]["window"] for i in items if "metadata" in i and "window" in i["metadata"]]
    return []

results = []

# Iterate over questions
for gt_entry in ground_truth:
    question = gt_entry["question"]
    gt_segments = [seg["text"] for seg in gt_entry["ground_truth_segments"]]
    retrieved_windows = get_retrieved_windows(question, retrieved)

    if not retrieved_windows:
        continue

    # Encode embeddings
    gt_emb = model.encode(gt_segments, convert_to_tensor=True)
    retrieved_emb = model.encode(retrieved_windows, convert_to_tensor=True)

    # Compute cosine similarity matrix
    sim_matrix = util.cos_sim(gt_emb, retrieved_emb)

    # Take the highest similarity per GT segment
    max_sim_per_gt = [float(max(row)) for row in sim_matrix]

    avg_similarity = mean(max_sim_per_gt)
    recall_at_k = sum(1 for sim in max_sim_per_gt if sim > 0.7) / len(max_sim_per_gt)  # 0.7 threshold for "hit"

    results.append({
        "question": question,
        "avg_similarity": avg_similarity,
        "recall@k": recall_at_k
    })

# Print results
for r in results:
    print(f"Q: {r['question']}")
    print(f"  Avg similarity: {r['avg_similarity']:.3f}")
    print(f"  Recall@k: {r['recall@k']:.2f}\n")

overall_avg = mean([r["avg_similarity"] for r in results])
print(f"Overall average similarity: {overall_avg:.3f}")
