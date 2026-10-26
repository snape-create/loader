# loader

基于 **LangChain** 与 **RapidOCR** 的多格式文档 OCR 加载器，面向 RAG 场景：不仅提取文档文本层，还能识别 Word / PPT / PDF / 图片中嵌入图片里的文字，统一输出 LangChain `Document`。

## 特性

- 兼容 LangChain `BaseLoader`，可直接接入文档流水线（分块、向量化、检索）
- 按文件扩展名自动分发到对应加载器（`document_loader` 注册表）
- OCR 引擎可插拔：优先 Paddle 版 RapidOCR（GPU 推荐），缺失时自动回退 ONNXRuntime 版（CPU）
- OCR 引擎进程内单例缓存，避免每个文件重复加载模型
- PDF 支持文本层提取 + 大图 OCR，并自动校正旋转页面上的图片
- 处理过程带 tqdm 进度条

## 支持格式
v
| 扩展名 | 加载器 | 文本来源 |
| --- | --- | --- |
| `.doc` / `.docx` | `OCRDOCLoader` | 段落 + 表格 + 内嵌图片 OCR |
| `.ppt` / `.pptx` | `OCRPPTLoader` | 文本框 + 表格 + 图片 OCR（含组合形状） |
| `.pdf` | `OCRPDFLoader` | 文本层 + 超过尺寸阈值的图片 OCR |
| `.png` / `.jpg` / `.jpeg` | `OCRIMGLoader` | 整图 OCR |
| `.txt` | `TextLoader` | 纯文本直接读取（内部轻量实现） |

## 安装

使用 [uv](https://docs.astral.sh/uv/)（推荐，Python ≥ 3.13）：

```bash
uv sync
# OCR 引擎二选一：
uv sync --extra ocr-paddle   # GPU / 推荐
uv sync --extra ocr-onnx     # 纯 CPU
```

或使用 pip：

```bash
pip install -e .
pip install rapidocr-paddle        # 或 rapidocr-onnxruntime
```

> 未安装任何 OCR 引擎时，加载含图片识别需求的文档会抛出 `RuntimeError` 并提示安装方式。

## 快速上手

```python
from loader import document_loader

ext = "pdf"
loader_cls = document_loader[ext]
loader = loader_cls(file_path="example.pdf")
docs = loader.load()
print(docs[0].page_content)
```

也可以直接使用具体加载器：

```python
from loader import OCRDOCLoader, OCRIMGLoader, OCRPDFLoader, OCRPPTLoader

docs = OCRIMGLoader(file_path="scan.png").load()
```

## 命令行手工测试

各 PDF / PPT 加载器内置了 `__main__` 测试入口：

```bash
uv run python pdfloader.py path/to/file.pdf
uv run python pptloader.py path/to/file.pptx
```

## 配置

| 环境变量 | 说明 |
| --- | --- |
| `LOADER_DEVICE` | 设定计算设备：`cuda` 或 `cpu`。不设置时自动探测（检测到 CUDA 则用 GPU，否则 CPU） |

PDF 图片 OCR 阈值可通过修改 `pdfloader.PDF_OCR_THRESHOLD`（默认 `(0.6, 0.6)`，即图片宽/高均超过页面 60% 才 OCR）调整，用于跳过小图、提升非扫描版 PDF 处理速度。

## 项目结构

```
loader/
├── __init__.py     # 包入口，document_loader 扩展名注册表
├── ocr.py          # OCR 引擎管理（单例缓存、Paddle/ONNX 双引擎回退）
├── docloader.py    # Word (.doc/.docx) 加载器
├── pptloader.py    # PPT (.ppt/.pptx) 加载器
├── pdfloader.py    # PDF 加载器
├── imgloader.py    # 图片 (.png/.jpg/.jpeg) 加载器
├── textloader.py   # 纯文本 (.txt) 加载器
└── pyproject.toml  # 项目配置与依赖
```

## 开发

```bash
# 代码格式化 / 静态检查（如需）
uv add --dev ruff
uv run ruff check .
```

欢迎补充单元测试（`tests/` 目录尚未建立）。

## License

按需补充。
