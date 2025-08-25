from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class GeneralRequest(BaseModel):
    target_column: str
    task_type: Literal['regression', 'classification']
    feature_columns: List[str]
    columns: Dict[str, Literal['int', 'float', 'str']]
