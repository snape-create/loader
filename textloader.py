"""纯文本（.txt）加载器。"""

from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class TextLoader(BaseLoader):
    """纯文本加载器：读取文本文件并产出单个 Document。"""

    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        """初始化加载器。

        Args:
            file_path: 待加载的文本文件路径。
            encoding: 文件编码，默认 utf-8。
        """
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载，产出单个包含全文的 Document。"""
        text = Path(self.file_path).read_text(encoding=self.encoding)
        yield Document(page_content=text, metadata={"source": self.file_path})
