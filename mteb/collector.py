from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from mteb.types import (
    CorpusDatasetType,
    QueryDatasetType,
    RelevantDocumentsType,
    DocumentExportModel,
    QueryExportModel,
)
from random import Random
from .utils import ROOT

class MTEBDataCollector:
    def __init__(
        self,
        corpus: CorpusDatasetType,
        queries: QueryDatasetType,
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        dataset_name: str,
        scoring_metric: str,
        hf_split: str,
        hf_subset: str,
        top_k: int = 100,
        max_queries: int = 1000,
        **kwargs,
    ) -> None:
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.results = results

        self.dataset_name = dataset_name
        self.scoring_metric = scoring_metric
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.top_k = top_k
        self.max_queries = max_queries

        self.qmodels: List[QueryExportModel] = []


    def prepare(self):
        # Sample queries from result set
        seed = '||'.join([self.dataset_name, self.hf_split, self.hf_subset])
        self.rng = Random(seed)

        sampled_qids = self.rng.sample(
            list(self.results.keys()),
            min(len(self.results), self.max_queries)
        )
        self.sampled_results = {qid: self.results[qid] for qid in sampled_qids}

        query_id_to_idx = {q['id']: i for i, q in enumerate(self.queries)}
        doc_id_to_idx = {d['id']: i for i, d in enumerate(self.corpus)}

        # Collect queries and results
        for qid, retrieved_docs in self.sampled_results.items():
            query_gt_scores = self.qrels.get(qid, {}).values()
            query_data = QueryExportModel(
                id=qid,
                query=self.queries[query_id_to_idx[qid]]['text'],
                metadata={
                    "dataset_id": self.dataset_name,
                    "hf_split": self.hf_split,
                    "hf_subset": self.hf_subset,
                    "metric": self.scoring_metric,
                    "gt_scores": list(query_gt_scores)
                },
                documents=[]
            )
            # relevant docs sorted by descending score
            top_k_docs = sorted(retrieved_docs.items(), key=lambda item: item[1], reverse=True)[:self.top_k]

            for doc_id, cosine_score in top_k_docs:
                query_data.documents.append(
                    DocumentExportModel(
                        id=doc_id,
                        content=self.corpus[doc_id_to_idx[doc_id]]['text'],
                        metadata={
                            "score": cosine_score,
                            "gt_score": self.qrels.get(qid, {}).get(doc_id, 0),
                        }
                    )
                )
            self.qmodels.append(query_data)
        
        print(f"Prepared {len(self.qmodels)} queries for dataset {self.dataset_name}/{self.hf_split}/{self.hf_subset}.")
        print(f"Preparing to write to {self.ze_results_path()}")

    def ze_results_path(self) -> Path:
        return Path(ROOT) / "data" / self.dataset_name / self.hf_split / self.hf_subset / "ze_results.jsonl" 

    def write_jsonl(self, overwrite: bool = True):
        file_path = self.ze_results_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'w' if overwrite else 'a'
        with open(file_path, mode) as f:
            for query_data in self.qmodels:
                f.write(query_data.model_dump_json().strip() + '\n')
        print(f"Wrote {len(self.qmodels)} queries to {file_path}")