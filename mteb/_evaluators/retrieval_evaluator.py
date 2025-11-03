import logging
from collections.abc import Sequence
from typing import Any

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import SearchProtocol
from mteb.types import (
    CorpusDatasetType,
    DocumentExportModel,
    QueryExportModel,
    QueryDatasetType,
    RelevantDocumentsType,
    RetrievalEvaluationResult,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from .evaluator import Evaluator
from .retrieval_metrics import (
    calculate_retrieval_scores,
)

from ..collector import MTEBDataCollector

logger = logging.getLogger(__name__)


class RetrievalEvaluator(Evaluator):
    def __init__(
        self,
        corpus: CorpusDatasetType,
        queries: QueryDatasetType,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        top_ranked: TopRankedDocumentsType | None = None,
        qid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.corpus = corpus
        self.queries = queries
        self.top_ranked = top_ranked

        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.qid = qid
        self.top_k = top_k

    def __call__(  # type: ignore[override]
        self,
        search_model: SearchProtocol,
        encode_kwargs: dict[str, Any],
    ) -> RetrievalOutputType:
        logger.info("Running retrieval task - Indexing corpus...")
        search_model.index(
            corpus=self.corpus,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            encode_kwargs=encode_kwargs,
        )
        logger.info("Running retrieval task - Searching queries...")
        return search_model.search(
            queries=self.queries,
            top_k=self.top_k,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            encode_kwargs=encode_kwargs,
            top_ranked=self.top_ranked,
        )

    def evaluate(
        self,
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        k_values: Sequence[int],
        ignore_identical_ids: bool = False,
        skip_first_result: bool = False,
    ) -> RetrievalEvaluationResult:
        if ignore_identical_ids:
            logger.debug(
                "For evaluation, ``ignore_identical_ids=True`` is set to True, the evaluator will ignore identical query and document ids."
            )
            # Remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            logger.debug(
                "For evaluation, we DO NOT ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=True`` to ignore this."
            )

        ###################################################
        ### WE SHOULD HAVE EVERYTHING THAT WE NEED HERE ###
        ###################################################
        '''
        print(f"Calculating retrieval scores for {self.hf_split}/{self.hf_subset} with {len(results)} queries and qrels for {len(qrels)} queries.")
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.corpus)}
        query_id_to_idx = {query["id"]: idx for idx, query in enumerate(self.queries)}

        # print some portion of first query/high scoring documents to verify
        printed = 0
        for qid in results:
            print(f"Query #{qid} is part of dataset {self.task_metadata.name}, evaluated with metric {self.task_metadata.main_score}.")
            print(f" - Query text snippet: {repr(self.queries[query_id_to_idx[qid]]['text'])[:150]}...")
            print(f" - Ground truth scores: {qrels.get(qid, {}).items()}")
            print(f" - Top retrieved documents:")
            sorted_rels = sorted(results[qid].items(), key=lambda item: item[1], reverse=True)
            for pid, score in sorted_rels[:10]:
                print(f"    ---> Document #{pid} with (score, gt) ({score}, {qrels.get(qid, {}).get(pid, 0)}) -- text snippet: {repr(self.corpus[doc_id_to_idx[pid]]['text'])[:150]}...")
            printed += 1
            if printed >= 5:
                break
        '''

        collector = MTEBDataCollector(
            corpus=self.corpus,
            queries=self.queries,
            qrels=qrels,
            results=results,
            dataset_name=self.task_metadata.name,
            scoring_metric=self.task_metadata.main_score,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            top_k=self.top_k
        )

        collector.prepare()
        collector.write_jsonl()

        return calculate_retrieval_scores(
            results, qrels, list(k_values), skip_first_result
        )
