from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel

# These will be set by the main script
DATASET_NAME = None
SCORING_METRIC = None
TOP_K = 100


class MTEBDocumentModel(BaseModel):
    id: str
    content: str
    metadata: Dict[str, float]


class MTEBQueryModel(BaseModel):
    id: str
    query: str
    metadata: Dict[str, Any]
    documents: List[MTEBDocumentModel]


class MTEBDataCollector:
    def __init__(self):
        self.qmodels: Dict[str, MTEBQueryModel] = {}
        self.doc_content: Dict[str, str] = {}
        self.doc_rel_qs: Dict[str, set] = {}

    def add_query(self, query_id: str, query_text: str):
        if query_id not in self.qmodels:
            self.qmodels[query_id] = MTEBQueryModel(
                id=query_id,
                query=query_text,
                metadata={
                    "dataset_id": DATASET_NAME,
                    "metric": SCORING_METRIC,
                    "gt_scores": []
                },
                documents=[]
            )

    def add_document(self, doc_id: str, content: str):
        if doc_id in self.doc_content:
            print(f"Document {doc_id} already found.")
            return
        self.doc_content[doc_id] = content
        self.doc_rel_qs[doc_id] = set()

    def add_query_rel(self, query_id: str, doc_id: str, gt_score: float):
        if doc_id in self.doc_content and query_id in self.qmodels:
            if query_id not in self.doc_rel_qs[doc_id]:
                self.doc_rel_qs[doc_id].add(query_id)
                self.qmodels[query_id].metadata["gt_scores"].append(gt_score)

    def add_query_doc_score(self, query_id: str, doc_id: str, score: float):
        gt_score = 1 if query_id in self.doc_rel_qs.get(doc_id, set()) else 0
        if query_id in self.qmodels and doc_id in self.doc_content:
            doc_model = MTEBDocumentModel(
                id=doc_id,
                content=self.doc_content[doc_id],
                metadata={
                    "score": score,
                    "gt_score": gt_score
                }
            )
            self.qmodels[query_id].documents.append(doc_model)

    def prepare(self):
        for qid, qmodel in self.qmodels.items():
            qmodel.documents.sort(key=lambda doc: doc.metadata["score"], reverse=True)
            qmodel.documents = qmodel.documents[:TOP_K]

    def dump_to_json(self, file_path: Path, overwrite: bool = False):
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'w' if overwrite else 'a'
        with open(file_path, mode, encoding='utf-8') as f:
            for query_data in self.qmodels.values():
                f.write(query_data.model_dump_json() + '\n')
        print(f"Dumped {DATASET_NAME} retrieval data to {file_path}")

    def __repr__(self):
        return f"MTEBDataCollector(dataset={DATASET_NAME}, queries={len(self.qmodels)}, docs={len(self.doc_content)})"


# Global registry of collectors
_collectors = {}


def get_collector(name: str = 'default') -> MTEBDataCollector:
    """
    Get or create a data collector by name.

    This ensures the same collector instance is used throughout the codebase.

    Args:
        name: Name of the collector (use different names for different purposes)

    Returns:
        MTEBDataCollector instance
    """
    if name not in _collectors:
        _collectors[name] = MTEBDataCollector()
    return _collectors[name]
