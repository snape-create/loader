"""Word 文档（.doc/.docx）OCR 加载器：提取正文、表格及图片内文字。"""

from io import BytesIO
from typing import Iterator

import numpy as np
from docx import Document  # 工厂函数，打开 .docx 文件
from docx.document import Document as DocxDocument  # 文档部件类型，用于 isinstance 判断
from docx import ImagePart
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from PIL import Image
from tqdm import tqdm

from .ocr import extract_ocr_text, get_ocr


def _iter_block_items(parent):
    """按文档流顺序迭代段落与表格块。"""
    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError(f"不支持的文档容器类型: {type(parent)!r}")
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


class OCRDOCLoader(BaseLoader):
    """Word 文档加载器：解析段落、表格，并对文档内嵌图片做 OCR。"""

    def __init__(self, file_path: str) -> None:
        """初始化加载器。

        Args:
            file_path: 待加载的 Word 文档路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[LCDocument]:
        """惰性加载，产出单个包含全文的 Document。"""
        text = self.doc2text(self.file_path)
        yield LCDocument(page_content=text, metadata={"source": self.file_path})

    def doc2text(self, file_path: str) -> str:
        """提取 Word 文档文本（含表格与图片 OCR 结果）。"""
        ocr = get_ocr()
        doc = Document(file_path)
        resp = ""
        blocks = list(_iter_block_items(doc))
        b_unit = tqdm(total=len(blocks), desc="OCRDOCLoader block index: 0")
        for i, block in enumerate(blocks):
            b_unit.set_description(f"OCRDOCLoader block index: {i}")
            b_unit.refresh()

            if isinstance(block, Paragraph):
                resp += block.text.strip() + "\n"
                # 提取段落中的内嵌图片并 OCR
                for image in block._element.xpath(".//pic:pic"):
                    for img_id in image.xpath(".//a:blip/@r:embed"):
                        part = doc.part.related_parts[img_id]
                        if isinstance(part, ImagePart):
                            img = Image.open(BytesIO(part._blob))
                            result, _ = ocr(np.array(img))
                            resp += extract_ocr_text(result)
            elif isinstance(block, Table):
                for row in block.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            resp += paragraph.text.strip() + "\n"

            b_unit.update(1)
        b_unit.close()
        return resp
