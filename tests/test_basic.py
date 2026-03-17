"""
Simple test to verify pytest configuration works
"""
import pytest


@pytest.mark.unit
def test_basic():
    """Basic sanity test"""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_fixture(mock_config):
    """Test that fixtures work"""
    assert mock_config.livekit_url == "wss://test.livekit.io"
    assert mock_config.deepseek_model == "deepseek-chat"
