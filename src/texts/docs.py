from abc import abstractmethod, ABC
from typing import Dict, List
class DocumentAbstract(ABC):
    def __init__(self, content: str):
        self.content = content

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: Dict):
        raise NotImplementedError

class Article(DocumentAbstract):
    def __init__(self, content: str, title: str, labels: List[str]|None = None):
        super().__init__(content)
        self.title = title
        self.labels = labels or []

    @classmethod
    def from_json(cls, json_data: Dict):
        return cls(json_data['content'], json_data['title'], labels=json_data.get('labels'))