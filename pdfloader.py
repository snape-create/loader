"""PDF OCR 加载器：提取文本层，并对大图（含扫描页）做 OCR。"""

import cv2
import numpy as np
import pymupdf  # 原 fitz 包，PyMuPDF 1.24+ 推荐使用 pymupdf 导入名
from PIL import Image
from tqdm import tqdm
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .ocr import extract_ocr_text, get_ocr

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)


class OCRPDFLoader(BaseLoader):
    """PDF 加载器：提取文本层，并对超过尺寸阈值的图片做 OCR。"""

    def __init__(self, file_path: str) -> None:
        """初始化加载器。

        Args:
            file_path: 待加载的 PDF 文件路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载，产出单个包含全文的 Document。"""
        text = self.pdf2text()
        yield Document(page_content=text, metadata={"source": self.file_path})

    def pdf2text(self) -> str:
        """提取 PDF 文本（含图片 OCR 结果）。"""
        ocr = get_ocr()
        doc = pymupdf.open(self.file_path)
        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            b_unit.set_description(f"OCRPDFLoader context page index: {i}")
            b_unit.refresh()

            # 提取文本层（默认 "text" 模式）
            resp += page.get_text("text") + "\n"

            # 获取页面上显示的所有图像元信息
            img_list = page.get_image_info(xrefs=True)
            for img in img_list:
                # xref 指向图像对象在 PDF 文件中的位置，用于快速定位和提取图像数据
                xref = img.get("xref")
                if not xref:
                    continue
                # bbox 为图像在页面上的位置和尺寸，跳过尺寸低于阈值的小图
                bbox = img["bbox"]
                if ((bbox[2] - bbox[0]) / page.rect.width < PDF_OCR_THRESHOLD[0]
                        or (bbox[3] - bbox[1]) / page.rect.height < PDF_OCR_THRESHOLD[1]):
                    continue

                pix = pymupdf.Pixmap(doc, xref)
                # 统一转换为 RGB 数组（去掉 alpha 通道），保证两条路径通道顺序一致
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                img_rgb = np.array(Image.fromarray(img_array).convert("RGB"))
                if page.rotation:  # 页面有旋转角度时，先校正图片
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    rot_img = self.rotate_img(img=img_bgr, angle=360 - page.rotation)
                    img_rgb = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)

                result, _ = ocr(img_rgb)
                resp += extract_ocr_text(result)

            b_unit.update(1)
        b_unit.close()
        return resp

    def rotate_img(self, img, angle):
        """旋转图片并调整边界，避免裁剪。

        Args:
            img: 待旋转的图片数组。
            angle: 旋转角度，正值逆时针、负值顺时针。
        """
        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        # 参数依次为：旋转中心、旋转角度、各向同性缩放比例
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算旋转后的新边界
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        return cv2.warpAffine(img, M, (new_w, new_h))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("用法: python pdfloader.py <pdf文件路径>")
    pdf_loader = OCRPDFLoader(file_path=sys.argv[1])
    doc = pdf_loader.load()
    print(doc)
