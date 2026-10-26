"""OCR 引擎管理：进程内单例缓存，支持 Paddle / ONNXRuntime 双引擎。"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """探测计算设备：优先环境变量 LOADER_DEVICE，否则自动探测 CUDA。"""
    device = os.environ.get("LOADER_DEVICE", "").lower()
    if device in ("cuda", "cpu"):
        return device
    try:
        import onnxruntime

        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _instantiate(engine_cls: Any, use_cuda: bool) -> Any:
    """实例化指定引擎。

    Paddle 和 ONNXRuntime 两个版本的 RapidOCR 都支持 det/cls/rec_use_cuda 三个开关，
    但版本间签名有差异；不支持显式传参时退回默认构造，避免直接崩掉。
    """
    cuda_kwargs = {
        "det_use_cuda": use_cuda,
        "cls_use_cuda": use_cuda,
        "rec_use_cuda": use_cuda,
    }
    try:
        return engine_cls(**cuda_kwargs)
    except TypeError:
        logger.warning(
            "ocr.engine_ignores_cuda engine=%s",
            getattr(engine_cls, "__module__", "unknown"),
        )
        return engine_cls()


def _build_ocr(use_cuda: bool) -> Any:
    """构建 OCR 实例：优先 Paddle 引擎，缺失时回退 ONNXRuntime。"""
    try:
        from rapidocr_paddle import RapidOCR as PaddleRapidOCR  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.warning("ocr.paddle_unavailable fallback=rapidocr_onnxruntime")
    else:
        logger.info("ocr.engine engine=rapidocr_paddle use_cuda=%s", use_cuda)
        return _instantiate(PaddleRapidOCR, use_cuda)

    try:
        from rapidocr_onnxruntime import RapidOCR as OnnxRapidOCR  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "未安装任何 OCR 引擎，请任选其一："
            "uv add rapidocr-paddle（GPU 推荐）或 uv add rapidocr-onnxruntime（CPU）"
        ) from exc

    logger.info("ocr.engine engine=rapidocr_onnxruntime use_cuda=%s", use_cuda)
    return _instantiate(OnnxRapidOCR, use_cuda)


@lru_cache(maxsize=None)
def get_ocr(use_cuda: bool | None = None) -> Any:
    """获取全局唯一的 OCR 实例。

    模型加载开销大，进程内只初始化一次并缓存，避免每个文件重复加载。

    Args:
        use_cuda: 是否启用 GPU 加速；None 表示自动探测（无 CUDA 时回退 CPU）。
    """
    if use_cuda is None:
        use_cuda = _get_device() == "cuda"
    return _build_ocr(use_cuda)


def extract_ocr_text(result: Any) -> str:
    """将 RapidOCR 识别结果整理为多行文本。

    Args:
        result: RapidOCR 返回的识别结果（每项含文本框坐标、文本、置信度）。
    """
    if not result:
        return ""
    return "\n".join(line[1] for line in result)
