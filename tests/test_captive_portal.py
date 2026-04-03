"""Captive Portal 中间件测试"""
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from lelamp.api.middleware.captive_portal import CaptivePortalMiddleware


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(CaptivePortalMiddleware, setup_url="/setup")

    @app.get("/setup")
    async def setup_page():
        return {"page": "setup"}

    @app.get("/api/test")
    async def api_test():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app, follow_redirects=False)


class TestCaptivePortal:
    def test_apple_captive_redirect(self, client):
        """Apple 设备的 captive.apple.com 请求应被重定向"""
        resp = client.get("/", headers={"Host": "captive.apple.com"})
        assert resp.status_code == 302
        assert "/setup" in resp.headers["location"]

    def test_android_generate_204_redirect(self, client):
        """Android 的 generate_204 应被重定向"""
        resp = client.get("/generate_204", headers={"Host": "connectivitycheck.gstatic.com"})
        assert resp.status_code == 302

    def test_windows_redirect(self, client):
        """Windows 的连接测试应被重定向"""
        resp = client.get("/connecttest.txt", headers={"Host": "www.msftconnecttest.com"})
        assert resp.status_code == 302

    def test_local_requests_pass_through(self, client):
        """本机请求不应被拦截"""
        resp = client.get("/api/test", headers={"Host": "192.168.4.1"})
        assert resp.status_code == 200

    def test_api_requests_not_redirected(self, client):
        """API 路径不应被重定向"""
        resp = client.get("/api/test", headers={"Host": "some-random-host.com"})
        assert resp.status_code == 200

    def test_unknown_host_non_api_redirected(self, client):
        """非 API 请求的未知 host 应被重定向"""
        resp = client.get("/some/page", headers={"Host": "example.com"})
        assert resp.status_code == 302
