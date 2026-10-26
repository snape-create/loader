"""PPT 演示文稿（.ppt/.pptx）OCR 加载器：提取文本、表格及图片内文字。"""

from io import BytesIO
from typing import Iterator

import numpy as np
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from tqdm import tqdm

from .ocr import extract_ocr_text, get_ocr


class OCRPPTLoader(BaseLoader):
    """PPT 加载器：按阅读顺序提取文本框、表格，并对图片做 OCR。"""

    def __init__(self, file_path: str) -> None:
        """初始化加载器。

        Args:
            file_path: 待加载的 PPT 文件路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载，产出单个包含全文的 Document。"""
        text = self.ppt2text(self.file_path)
        yield Document(page_content=text, metadata={"source": self.file_path})

    def ppt2text(self, file_path: str) -> str:
        """提取 PPT 文本（含图片 OCR 结果）。"""
        prs = Presentation(file_path)
        ocr = get_ocr()
        resp = ""

        def extract_text(shape):
            # nonlocal 指明 resp 是外部嵌套函数中的变量
            nonlocal resp

            if shape.has_text_frame:
                resp += shape.text.strip() + "\n"

            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            resp += paragraph.text.strip() + "\n"

            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image = Image.open(BytesIO(shape.image.blob))
                result, _ = ocr(np.array(image))
                resp += extract_ocr_text(result)
            elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for child_shape in shape.shapes:
                    extract_text(child_shape)

        b_unit = tqdm(total=len(prs.slides), desc="OCRPPTLoader slide index: 1")
        for slide_number, slide in enumerate(prs.slides, start=1):
            b_unit.set_description(f"OCRPPTLoader slide index: {slide_number}")
            b_unit.refresh()

            # 按照从上到下、从左到右的顺序遍历形状
            sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))
            for shape in sorted_shapes:
                extract_text(shape)

            b_unit.update(1)
        b_unit.close()
        return resp


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("用法: python pptloader.py <ppt文件路径>")
    ppt_loader = OCRPPTLoader(file_path=sys.argv[1])
    doc = ppt_loader.load()
    print(doc)
