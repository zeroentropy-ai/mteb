from typing import Any, Dict, List
from pydantic import BaseModel

class DocumentExportModel(BaseModel):
    id: str
    content: str
    metadata: Dict[str, float]


class QueryExportModel(BaseModel):
    id: str
    query: str
    metadata: Dict[str, Any]
    documents: List[DocumentExportModel]