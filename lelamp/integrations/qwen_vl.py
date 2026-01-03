import json
import asyncio
import logging
import httpx

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

    async def describe(self, *, image_jpeg_b64: str, question: str) -> str:
        if not self._api_key:
            return "未配置 MODELSCOPE_API_KEY，无法调用视觉模型。"

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
        except httpx.HTTPStatusError as e:
            try:
                err_body = e.response.text
            except Exception:
                err_body = ""
            return f"视觉模型请求失败（HTTP {e.response.status_code}）：{err_body}"
        except Exception as e:
            return f"视觉模型请求失败：{str(e)}"

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
        except Exception:
            pass

        return json.dumps(data, ensure_ascii=False)
