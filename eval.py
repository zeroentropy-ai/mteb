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


def compute_ndcg_at_k(queries: List[MTEBQueryModel], k: int) -> float:
    ncdg_scores = []
    for query in queries:
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
        ncdg_scores.append(ndcg)
    average_ndcg = sum(ncdg_scores) / len(ncdg_scores) if ncdg_scores else 0.0
    return average_ndcg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MTEB JSONL and report appropriate metric for each dataset.")
    parser.add_argument("filename", type=str, help="Path to the input JSONL file.")
    args = parser.parse_args()

    queries_by_dataset: Dict[str, List[MTEBQueryModel]] = {}
    # Read and parse the JSONL file
    with open(args.filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            query_model = MTEBQueryModel.model_validate_json(line)
            
            dataset_id = query_model.metadata['dataset_id']
            if dataset_id not in queries_by_dataset:
                queries_by_dataset[dataset_id] = []
            queries_by_dataset[dataset_id].append(query_model)
    
    # Process each dataset
    for dataset_id, queries in queries_by_dataset.items():

        metric = queries[0].metadata['metric']
        if any(q.metadata['metric'] != metric for q in queries):
            print(f"SKIPPING {dataset_id}: Inconsistent scoring metrics.")
            continue 
        
        if metric.startswith('ndcg_at_'):
            k = metric.removeprefix('ndcg_at_')
            assert k.isnumeric(), f"Invalid NDCG metric format: {metric}"

            avg_ndcg = round(compute_ndcg_at_k(queries, int(k)) * 100, 2)
            print(f"Average NDCG@{k} for {dataset_id}: {avg_ndcg:.2f}%")

        elif metric.startswith('recall_at_'):
            print(f"Recall metric processing not implemented yet for {dataset_id}.")


