"""图片（.png/.jpg/.jpeg）OCR 加载器。"""

from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .ocr import extract_ocr_text, get_ocr


class OCRIMGLoader(BaseLoader):
    """图片加载器：对整张图片做 OCR 并产出文本。"""

    def __init__(self, file_path: str) -> None:
        """初始化加载器。

        Args:
            file_path: 待加载的图片路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载，产出单个包含全文的 Document。"""
        text = self.img2text()
        yield Document(page_content=text, metadata={"source": self.file_path})

    def img2text(self) -> str:
        """提取图片中的文字。"""
        ocr = get_ocr()
        result, _ = ocr(self.file_path)
        return extract_ocr_text(result)
