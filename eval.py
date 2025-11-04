from pathlib import Path
from typing import Dict, List, Any
import json
import argparse
import math
from pydantic import BaseModel

class MTEBDocumentModel(BaseModel):
    id: str
    content: str
    metadata: Dict[str, float]


class MTEBQueryModel(BaseModel):
    id: str
    query: str
    metadata: Dict[str, Any]
    documents: List[MTEBDocumentModel]

def compute_single_ndcg_at_k(query: MTEBQueryModel, k: int) -> float:
    ideal_gt_scores = query.metadata['gt_scores']
    sorted_scores = sorted(ideal_gt_scores, reverse=True)
    ideal_dcg = 0.0
    for i, rel in enumerate(sorted_scores[:k], 1):
        ideal_dcg += (rel) / (math.log2(i + 1))

    sorted_docs = sorted(query.documents, key=lambda d: d.metadata.get('score', 0), reverse=True)       
    dcg = 0.0
    for i, rel in enumerate(sorted_docs[:k], 1):
        dcg += (rel.metadata.get('gt_score', 0)) / (math.log2(i + 1))
    
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return ndcg

def compute_ndcg_at_k(queries: List[MTEBQueryModel], k: int) -> float:
    ncdg_scores = []
    for query in queries:
        ncdg_scores.append(compute_single_ndcg_at_k(query, k))
    average_ndcg = sum(ncdg_scores) / len(ncdg_scores) if ncdg_scores else 0.0
    return average_ndcg

def compute_single_recall_at_k(query: MTEBQueryModel, k: int) -> float:
    num_relevant = sum(query.metadata['gt_scores'])
    if num_relevant == 0:
        return 0

    sorted_docs = sorted(query.documents, key=lambda d: d.metadata.get('score', 0), reverse=True)
    retrieved_relevant = sum(doc.metadata.get('gt_score', 0) for doc in sorted_docs[:k])

    return retrieved_relevant / num_relevant

def compute_recall_at_k(queries: List[MTEBQueryModel], k: int) -> float:
    recall_scores = []
    for query in queries:        
        recall_scores.append(compute_single_recall_at_k(query, k))    
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    return average_recall
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MTEB JSONL and report appropriate metric for each dataset.")
    parser.add_argument("filename", type=str, help="Path to the input JSONL file.")
    args = parser.parse_args()

    dataset_metrics: Dict[str, str] = {}
    scores_by_dataset: Dict[str, List[float]] = {}
    # Read and parse the JSONL file
    with open(args.filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            query_model = MTEBQueryModel.model_validate_json(line)
            
            dataset_id = query_model.metadata['dataset_id']
            metric = query_model.metadata['metric']

            if dataset_id not in scores_by_dataset:
                dataset_metrics[dataset_id] = metric
                scores_by_dataset[dataset_id] = []

            if metric.startswith('ndcg_at_'):
                k = metric.removeprefix('ndcg_at_')
                assert k.isnumeric(), f"Invalid NDCG metric format: {metric}"

                ndcg_score = compute_single_ndcg_at_k(query_model, int(k))
                scores_by_dataset[dataset_id].append(ndcg_score)

            elif metric.startswith('recall_at_'):
                k = metric.removeprefix('recall_at_')
                assert k.isnumeric(), f"Invalid Recall metric format: {metric}"

                recall_score = compute_recall_at_k([query_model], int(k))
                scores_by_dataset[dataset_id].append(recall_score)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    # Report results for each dataset
    for dataset_id, metric in dataset_metrics.items():
        scores = scores_by_dataset[dataset_id]

        if metric.startswith('ndcg_at_'):
            k = metric.removeprefix('ndcg_at_')
            assert k.isnumeric(), f"Invalid NDCG metric format: {metric}"

            avg_ndcg = round(sum(scores) / len(scores) * 100, 2)
            print(f"Average NDCG@{k} for {dataset_id}: {avg_ndcg:.2f}%")

        elif metric.startswith('recall_at_'):
            k = metric.removeprefix('recall_at_')
            assert k.isnumeric(), f"Invalid Recall metric format: {metric}"

            avg_recall = round(sum(scores) / len(scores) * 100, 2)
            print(f"Average Recall@{k} for {dataset_id}: {avg_recall:.2f}%")
            
