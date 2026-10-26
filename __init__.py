"""LangChain 文档加载器集合：按扩展名分发到对应 OCR 加载器。"""

from .docloader import OCRDOCLoader
from .imgloader import OCRIMGLoader
from .pdfloader import OCRPDFLoader
from .pptloader import OCRPPTLoader
from .textloader import TextLoader

__all__ = [
    "OCRDOCLoader",
    "OCRIMGLoader",
    "OCRPDFLoader",
    "OCRPPTLoader",
    "document_loader",
]

document_loader = {
    "doc": OCRDOCLoader,
    "docx": OCRDOCLoader,
    "ppt": OCRPPTLoader,
    "pptx": OCRPPTLoader,
    "pdf": OCRPDFLoader,
    "txt": TextLoader,
    "png": OCRIMGLoader,
    "jpg": OCRIMGLoader,
    "jpeg": OCRIMGLoader,
}
