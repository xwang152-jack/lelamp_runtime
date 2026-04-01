"""
Captive Portal API 服务
为首次设置提供 Web API 接口
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from lelamp.api.services.setup_state import SetupStateManager
from lelamp.api.services.wifi_scanner import WiFiScanner
from lelamp.api.services.network_manager import NetworkConnectionManager

logger = logging.getLogger(__name__)


# Pydantic 模型
class WiFiConnectRequest(BaseModel):
    ssid: str
    password: str


class CompleteSetupRequest(BaseModel):
    success: bool = True
    ip_address: str = None


# 创建 FastAPI 应用
def create_captive_portal_app() -> FastAPI:
    """创建 Captive Portal 应用"""
    app = FastAPI(
        title="LeLamp Setup Portal",
        description="LeLamp 首次设置向导",
        version="1.0.0"
    )

    # 初始化服务
    state_manager = SetupStateManager()
    wifi_scanner = WiFiScanner()
    network_manager = NetworkConnectionManager()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """主页面 - 返回设置向导 HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LeLamp 设置向导</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }
                .container {
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    padding: 40px;
                    width: 100%;
                    max-width: 480px;
                }
                h1 { text-align: center; color: #333; margin-bottom: 10px; }
                .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
                .btn {
                    display: block;
                    width: 100%;
                    padding: 14px;
                    margin: 10px 0;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 500;
                    transition: all 0.3s ease;
                }
                .btn-primary {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
                .btn-secondary { background: #f0f0f0; color: #333; }
                .btn-secondary:hover { background: #e0e0e0; }
                #networks, #connecting, #result, #error { display: none; }
                .network {
                    padding: 15px;
                    margin: 10px 0;
                    border: 2px solid #eee;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                .network:hover { border-color: #667eea; background: #f8f9ff; }
                .network-info { display: flex; align-items: center; gap: 10px; }
                .signal { font-size: 20px; }
                .security { color: #28a745; }
                .loading { text-align: center; padding: 40px; }
                .spinner {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 20px;
                }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                .success-icon { font-size: 60px; text-align: center; }
                .error-icon { font-size: 60px; text-align: center; }
                .ip-address { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; margin: 20px 0; font-family: monospace; font-size: 18px; }
                input[type="password"], input[type="text"] {
                    width: 100%;
                    padding: 14px;
                    margin: 10px 0;
                    border: 2px solid #eee;
                    border-radius: 8px;
                    font-size: 16px;
                }
                input:focus { outline: none; border-color: #667eea; }
                .step { display: none; }
                .step.active { display: block; }
            </style>
        </head>
        <body>
            <div class="container">
                <div id="step-welcome" class="step active">
                    <h1>🪔 LeLamp</h1>
                    <p class="subtitle">欢迎使用 LeLamp 智能台灯</p>
                    <div id="ap-password-hint" style="display:none; background:#e8f4fd; border-radius:8px; padding:12px; margin:15px 0; text-align:center;">
                        <p style="margin:0; font-size:13px; color:#666;">当前热点密码</p>
                        <p id="ap-password-display" style="margin:4px 0 0; font-size:20px; font-weight:bold; font-family:monospace; letter-spacing:2px; color:#333;"></p>
                    </div>
                    <p style="text-align: center; margin-bottom: 20px;">让我们帮您连接到 WiFi 网络</p>
                    <button class="btn btn-primary" onclick="scanNetworks()">开始设置</button>
                </div>

                <div id="step-networks" class="step">
                    <h2>选择 WiFi 网络</h2>
                    <button class="btn btn-secondary" onclick="scanNetworks()">🔍 刷新</button>
                    <div id="networks"></div>
                    <button class="btn btn-secondary" onclick="showStep('step-welcome')">返回</button>
                </div>

                <div id="step-password" class="step">
                    <h2>输入密码</h2>
                    <p id="selected-network" style="text-align: center; margin: 15px 0; color: #666;"></p>
                    <input type="text" id="ssid-input" hidden>
                    <input type="password" id="password-input" placeholder="请输入 WiFi 密码">
                    <button class="btn btn-primary" onclick="connectWifi()">连接</button>
                    <button class="btn btn-secondary" onclick="showStep('step-networks')">返回</button>
                </div>

                <div id="step-connecting" class="step">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>正在连接...</p>
                        <p id="connecting-status" style="color: #666; margin-top: 10px;"></p>
                    </div>
                </div>

                <div id="step-result" class="step">
                    <div class="success-icon">✅</div>
                    <h2 style="text-align: center;">设置完成！</h2>
                    <p style="text-align: center; color: #666;">您的台灯已连接到 WiFi 网络</p>
                    <div class="ip-address" id="ip-display"></div>
                    <p style="text-align: center; color: #666; font-size: 14px;">
                        下次访问: <span id="access-url"></span>
                    </p>
                </div>

                <div id="step-error" class="step">
                    <div class="error-icon">❌</div>
                    <h2 style="text-align: center;">连接失败</h2>
                    <p id="error-message" style="text-align: center; color: #666; margin: 20px 0;"></p>
                    <button class="btn btn-primary" onclick="showStep('step-networks')">重试</button>
                    <button class="btn btn-secondary" onclick="resetSetup()">恢复出厂设置</button>
                </div>
            </div>

            <script>
                function showStep(stepId) {
                    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
                    document.getElementById(stepId).classList.add('active');
                }

                async function scanNetworks() {
                    showStep('step-connecting');
                    document.getElementById('connecting-status').textContent = '正在扫描 WiFi 网络...';

                    try {
                        const response = await fetch('/api/setup/networks');
                        const data = await response.json();

                        const networksDiv = document.getElementById('networks');
                        networksDiv.innerHTML = '';

                        if (data.networks.length === 0) {
                            networksDiv.innerHTML = '<p style="text-align: center; color: #666;">未找到 WiFi 网络</p>';
                        } else {
                            data.networks.forEach(network => {
                                const div = document.createElement('div');
                                div.className = 'network';
                                div.onclick = () => selectNetwork(network.ssid);

                                const signalIcons = ['📡', '📶', '📶', '📶', '📶'];
                                const signalIdx = Math.min(Math.floor(network.signal_strength / 20), 4);
                                const signal = signalIcons[signalIdx];

                                div.innerHTML = `
                                    <div class="network-info">
                                        <span class="signal">${signal}</span>
                                        <div>
                                            <div style="font-weight: 500;">${network.ssid}</div>
                                            <div style="font-size: 12px; color: #666;">信号: ${network.signal_strength}%</div>
                                        </div>
                                    </div>
                                    <span class="security">${network.encryption !== 'Open' ? '🔒' : '🔓'}</span>
                                `;
                                networksDiv.appendChild(div);
                            });
                        }

                        showStep('step-networks');
                    } catch (error) {
                        alert('扫描失败: ' + error.message);
                        showStep('step-welcome');
                    }
                }

                function selectNetwork(ssid) {
                    document.getElementById('selected-network').textContent = '连接到: ' + ssid;
                    document.getElementById('ssid-input').value = ssid;
                    document.getElementById('password-input').value = '';
                    showStep('step-password');
                }

                async function connectWifi() {
                    const ssid = document.getElementById('ssid-input').value;
                    const password = document.getElementById('password-input').value;

                    if (!password) {
                        alert('请输入密码');
                        return;
                    }

                    showStep('step-connecting');
                    document.getElementById('connecting-status').textContent = '正在连接到 ' + ssid + '...';

                    try {
                        const response = await fetch('/api/setup/connect', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ssid, password})
                        });
                        const data = await response.json();

                        if (data.success) {
                            document.getElementById('ip-display').textContent = data.ip_address || '连接成功';
                            document.getElementById('access-url').textContent = 'http://' + (data.ip_address || 'lelamp.local') + ':5173';
                            showStep('step-result');
                        } else {
                            document.getElementById('error-message').textContent = data.message || '连接失败，请重试';
                            showStep('step-error');
                        }
                    } catch (error) {
                        document.getElementById('error-message').textContent = '连接失败: ' + error.message;
                        showStep('step-error');
                    }
                }

                async function resetSetup() {
                    if (confirm('确定要恢复出厂设置吗？这将清除所有配置。')) {
                        try {
                            await fetch('/api/setup/reset', {method: 'POST'});
                            alert('已恢复出厂设置，请重启设备');
                        } catch (error) {
                            alert('重置失败: ' + error.message);
                        }
                    }
                }

                // 检查是否已完成设置
                fetch('/api/setup/status')
                    .then(r => r.json())
                    .then(data => {
                        if (data.ap_password) {
                            document.getElementById('ap-password-display').textContent = data.ap_password;
                            document.getElementById('ap-password-hint').style.display = 'block';
                        }
                        if (data.setup_completed) {
                            document.getElementById('ip-display').textContent = data.last_ip_address || '未知';
                            document.getElementById('access-url').textContent = 'http://' + (data.last_ip_address || 'lelamp.local') + ':5173';
                        }
                    });
            </script>
        </body>
        </html>
        """

    @app.get("/api/setup/status")
    async def get_setup_status():
        """获取设置状态"""
        try:
            state = state_manager.load_state()
            # 读取 AP 密码
            ap_password = None
            try:
                import json
                from pathlib import Path
                status_file = Path("/var/lib/lelamp/setup_status.json")
                if status_file.exists():
                    setup_data = json.loads(status_file.read_text())
                    ap_password = setup_data.get("ap_password")
            except Exception:
                pass
            # 回退到 APManager 当前密码
            if not ap_password:
                from lelamp.api.services.ap_manager import ap_manager
                ap_password = ap_manager.current_password

            return {
                "setup_completed": state.get("setup_completed", False),
                "current_step": state.get("current_step", "welcome"),
                "error_message": state.get("error_message"),
                "connection_attempts": state.get("connection_attempts", 0),
                "last_ip_address": state.get("last_ip_address"),
                "wifi_ssid": state.get("wifi_ssid"),
                "ap_password": ap_password,
            }
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/setup/networks")
    async def scan_networks():
        """扫描可用 WiFi 网络"""
        try:
            networks = await wifi_scanner.async_scan_networks()
            return {
                "networks": networks,
                "count": len(networks)
            }
        except Exception as e:
            logger.error(f"扫描网络失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/setup/connect")
    async def connect_wifi(request: WiFiConnectRequest):
        """连接到指定 WiFi 网络"""
        try:
            # 增加尝试次数
            attempts = state_manager.increment_attempts()

            if attempts > 3:
                return {
                    "success": False,
                    "error": "max_attempts",
                    "message": "连接尝试次数过多，请重试网络设置"
                }

            # 更新状态
            state_manager.update_step("connecting")

            # 尝试连接
            result = await network_manager.async_connect_wifi(
                request.ssid,
                request.password
            )

            if result["success"]:
                # 更新状态
                state_manager.set_wifi_ssid(request.ssid)
                state_manager.complete_setup(result.get("ip_address", ""))
                state_manager.reset_attempts()

                return {
                    "success": True,
                    "ip_address": result.get("ip_address"),
                    "message": "连接成功"
                }
            else:
                # 设置错误信息
                state_manager.set_error(result.get("error", "连接失败"))
                return result

        except Exception as e:
            logger.error(f"连接 WiFi 失败: {e}")
            state_manager.set_error(str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/setup/test-connection")
    async def test_connection():
        """测试网络连接"""
        try:
            result = network_manager.test_connection()
            return result
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/setup/complete")
    async def complete_setup(request: CompleteSetupRequest):
        """完成设置"""
        try:
            if request.success:
                state = state_manager.load_state()
                return {
                    "success": True,
                    "ip_address": state.get("last_ip_address"),
                    "message": "设置完成"
                }
            else:
                return {
                    "success": False,
                    "message": "设置未完成"
                }
        except Exception as e:
            logger.error(f"完成设置失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/setup/reset")
    async def reset_setup():
        """重置设置（恢复出厂设置）"""
        try:
            state_manager.reset()
            return {
                "success": True,
                "message": "设置已重置"
            }
        except Exception as e:
            logger.error(f"重置设置失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/setup/config")
    async def get_portal_config():
        """获取 Portal 配置"""
        return {
            "ssid": "LeLamp-Setup",
            "password": "lelamp123",
            "ip_address": "192.168.4.1",
            "port": 8080
        }

    return app


app = create_captive_portal_app()

def main():
    """启动 Captive Portal 服务"""
    import uvicorn

    app = create_captive_portal_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )


if __name__ == "__main__":
    main()
