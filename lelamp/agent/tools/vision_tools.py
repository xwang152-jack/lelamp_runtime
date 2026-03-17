"""
视觉工具模块
"""
import asyncio
import base64
import json
import logging
import os
import time
import uuid
import urllib.request
from typing import TYPE_CHECKING

from livekit.agents import function_tool

if TYPE_CHECKING:
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.service.vision.vision_service import VisionService
    from lelamp.integrations.qwen_vl import Qwen3VLClient
    from lelamp.agent.states import StateManager


logger = logging.getLogger("lelamp.agent.tools.vision")


class VisionTools:
    """视觉工具类 - 管理视觉问答、作业检查和飞书推送"""

    def __init__(
        self,
        vision_service: "VisionService",
        qwen_client: "Qwen3VLClient",
        rgb_service: "RGBService",
        motors_service: "MotorsService",
        state_manager: "StateManager",
        rate_limiter,
    ):
        """
        初始化视觉工具

        Args:
            vision_service: 视觉服务实例
            qwen_client: Qwen 视觉客户端实例
            rgb_service: RGB 服务实例
            motors_service: 电机服务实例
            state_manager: 状态管理器实例
            rate_limiter: 速率限制器实例
        """
        self._vision_service = vision_service
        self._qwen_client = qwen_client
        self.rgb_service = rgb_service
        self.motors_service = motors_service
        self.state_manager = state_manager
        self._rate_limiter = rate_limiter
        self._motion_locked = False
        self.logger = logger

    @function_tool
    async def vision_answer(self, question: str) -> str:
        """
        Ask a question about what the lamp can see through its camera.
        询问关于灯通过摄像头能看到什么的问题。
        """
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

        # 应用速率限制
        if not await self._rate_limiter.acquire(tokens=1, timeout=10.0):
            return "视觉 API 调用太频繁了，让我休息一下眼睛。"

        # 保存并覆盖灯光状态
        prev_override_until_ts = self._save_and_set_light_override(duration_s=3600.0)

        try:
            from lelamp.service import Priority

            self.rgb_service.dispatch("solid", (255, 255, 255), priority=Priority.HIGH)

            latest = await self._vision_service.get_latest_jpeg_b64()
            if not latest:
                return "当前没有可用画面。请确保摄像头已启用并在刷新。"

            jpeg_b64, _ = latest
            return await self._qwen_client.describe(image_jpeg_b64=jpeg_b64, question=question)
        finally:
            # 恢复灯光状态
            self._restore_light_override(prev_override_until_ts)

    @function_tool
    async def check_homework(self) -> str:
        """
        帮用户检查画面中的作业（数学、口算、填空等）。
        Analyze and check homework in the camera view (math, corrections, etc.).
        """
        if not self._vision_service or not self._qwen_client:
            return "视觉能力未初始化。"

        # 应用速率限制
        if not await self._rate_limiter.acquire(tokens=1, timeout=10.0):
            return "作业检查太频繁了，让我也喘口气。"

        # 1. 补光 - 保存并覆盖灯光状态
        prev_override_until_ts = self._save_and_set_light_override(duration_s=3600.0)

        try:
            from lelamp.service import Priority

            self.rgb_service.dispatch("solid", (255, 255, 255), priority=Priority.HIGH)

            # 2. 获取最清晰、最实时的画面
            latest = await self._vision_service.get_fresh_jpeg_b64(timeout_s=5.0)
            if not latest:
                return "拍照失败，无法看清作业。"

            jpeg_b64, _ = latest

            # 3. 调用视觉模型，使用老师人设
            prompt = (
                "你现在是一位认真负责且幽默的老师。请检查图片中的作业（通常是数学题、口算或填空）。"
                "指出错误的地方并给出正确答案或解析，鼓励用户改进。如果看不清，请提示用户调整作业位置或补光。"
                "请用中文回答，保持简洁。最后别忘了以LeLamp的性格损一下用户。"
            )
            return await self._qwen_client.describe(image_jpeg_b64=jpeg_b64, question=prompt)
        finally:
            # 恢复灯光状态
            self._restore_light_override(prev_override_until_ts)

    @function_tool
    async def capture_to_feishu(self) -> str:
        """
        拍照并通过飞书机器人推送（直接上传图片），拍照前会锁定动作并停止以确保清晰度。
        Capture a photo and push it to Feishu bot (direct upload), stops motion for clarity.
        """
        if not self._vision_service:
            return "视觉能力未初始化。"

        # 1. 锁定动作并停止当前所有动作（保持扭矩）
        self._motion_locked = True
        from lelamp.service import Priority

        self.motors_service.dispatch("stop", None, priority=Priority.CRITICAL)
        await asyncio.sleep(1.5)  # 等待 1.5 秒让机械臂完全静止

        try:
            # 2. 获取飞书配置
            app_id = os.getenv("FEISHU_APP_ID")
            app_secret = os.getenv("FEISHU_APP_SECRET")
            receive_id = os.getenv("FEISHU_RECEIVE_ID")
            receive_id_type = os.getenv("FEISHU_RECEIVE_ID_TYPE") or "open_id"

            if not all([app_id, app_secret, receive_id]):
                return "飞书配置不完整，请检查环境变量 (FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_RECEIVE_ID)"

            # 3. 获取 Token
            async def _do_req(req):
                return await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=15).read())

            token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            token_payload = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
            token_req = urllib.request.Request(
                token_url, data=token_payload, headers={"Content-Type": "application/json"}, method="POST"
            )

            token_resp = json.loads(await _do_req(token_req))
            token = token_resp.get("tenant_access_token")
            if not token:
                return f"获取飞书 Token 失败: {token_resp.get('msg')}"

            # 4. 获取当前画面 (确保是机械臂停止后的最新画面)
            latest = await self._vision_service.get_fresh_jpeg_b64(timeout_s=5.0)
            if not latest:
                return "拍照失败，无可用画面"
            jpeg_b64, _ = latest
            jpeg_data = base64.b64decode(jpeg_b64)

            # 5. 上传图片到飞书
            upload_url = "https://open.feishu.cn/open-apis/im/v1/images"
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

            body = []
            body.append(f"--{boundary}".encode("utf-8"))
            body.append(b'Content-Disposition: form-data; name="image_type"')
            body.append(b"")
            body.append(b"message")
            body.append(f"--{boundary}".encode("utf-8"))
            body.append(b'Content-Disposition: form-data; name="image"; filename="photo.jpg"')
            body.append(b"Content-Type: image/jpeg")
            body.append(b"")
            body.append(jpeg_data)
            body.append(f"--{boundary}--".encode("utf-8"))

            payload = b"\r\n".join(body)
            upload_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            }
            upload_req = urllib.request.Request(upload_url, data=payload, headers=upload_headers, method="POST")
            upload_resp = json.loads(await _do_req(upload_req))

            image_key = upload_resp.get("data", {}).get("image_key")
            if not image_key:
                return f"上传图片到飞书失败: {upload_resp.get('msg')}"

            # 6. 发送消息
            msg_url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}"
            msg_payload = json.dumps({
                "receive_id": receive_id,
                "msg_type": "image",
                "content": json.dumps({"image_key": image_key})
            }).encode("utf-8")

            msg_req = urllib.request.Request(
                msg_url, data=msg_payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST"
            )
            msg_resp = json.loads(await _do_req(msg_req))

            if msg_resp.get("code") != 0:
                return f"发送飞书消息失败: {msg_resp.get('msg')}"

            return "照片已成功推送到飞书。"

        except Exception as e:
            return f"飞书推送过程发生异常: {str(e)}"
        finally:
            # 7. 解锁动作
            self._motion_locked = False

    def _save_and_set_light_override(self, duration_s: float) -> float:
        """
        保存当前灯光覆盖时间戳并设置新的覆盖

        Args:
            duration_s: 新的覆盖时长（秒）

        Returns:
            之前的覆盖时间戳
        """
        # 使用 state_manager 的内部方法来保存和设置
        # 由于需要访问私有成员，我们通过 state_manager 的公共接口
        with self.state_manager._timestamps_lock:
            prev = self.state_manager._light_override_until_ts
            # 如果是 None，使用 0.0 作为默认值
            prev_override_until_ts = float(prev) if prev is not None else 0.0
            self.state_manager._light_override_until_ts = time.time() + duration_s
            return prev_override_until_ts

    def _restore_light_override(self, prev_override_until_ts: float) -> None:
        """
        恢复之前的灯光覆盖时间戳

        Args:
            prev_override_until_ts: 之前保存的覆盖时间戳
        """
        with self.state_manager._timestamps_lock:
            self.state_manager._light_override_until_ts = prev_override_until_ts
