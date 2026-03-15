import json
import asyncio
import logging
import httpx

from .exceptions import (
    NetworkError,
    ServiceUnavailableError,
    ValidationError,
    TimeoutError,
    with_fallback,
    MessageFallback,
    retry_on_error,
    RetryConfig,
    convert_httpx_error,
)

logger = logging.getLogger("lelamp.integrations.qwen")

class Qwen3VLClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    @with_fallback(
        MessageFallback("视觉服务暂时不可用，请稍后再试"),
        log_error=True,
    )
    @retry_on_error(config=RetryConfig(max_attempts=2, base_delay=1.0))
    async def describe(self, *, image_jpeg_b64: str, question: str) -> str:
        """
        描述图片内容（带重试和降级）

        Args:
            image_jpeg_b64: Base64 编码的 JPEG 图片
            question: 用户问题

        Returns:
            图片描述文本

        Raises:
            NetworkError: 网络连接失败
            ServiceUnavailableError: 服务不可用
            ValidationError: 响应格式错误
        """
        if not self._api_key:
            raise ValidationError(
                "未配置 MODELSCOPE_API_KEY",
                provider="modelscope",
                field="api_key",
            )

        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "你现在可以看到用户。请基于图片内容，用中文简洁回答用户问题。",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_jpeg_b64}"},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            "temperature": 0.2,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

        except Exception as e:
            # 转换为统一的集成错误
            raise convert_httpx_error(e, provider="modelscope")

        # 解析响应
        try:
            choice0 = (data.get("choices") or [])[0]
            msg = (choice0 or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
                if texts:
                    return "\n".join(texts)

        except Exception as e:
            raise ValidationError(
                f"视觉模型响应格式错误: {str(e)}",
                provider="modelscope",
                original=e,
            )

        # 如果无法解析，返回原始 JSON
        logger.warning("无法解析视觉模型响应，返回原始数据")
        return json.dumps(data, ensure_ascii=False)
