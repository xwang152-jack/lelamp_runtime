import asyncio
import os
os.environ["LIVEKIT_URL"] = "ws://dummy"
os.environ["LIVEKIT_API_KEY"] = "dummy"
os.environ["LIVEKIT_API_SECRET"] = "dummy"
os.environ["DEEPSEEK_API_KEY"] = "dummy"
os.environ["BAIDU_SPEECH_API_KEY"] = "dummy"
os.environ["BAIDU_SPEECH_SECRET_KEY"] = "dummy"
os.environ["LELAMP_PORT"] = "dummy_port"

from lelamp.agent.lelamp_agent import LeLamp
from lelamp.config import load_motor_config

async def main():
    motor_config = load_motor_config()
    agent = LeLamp(motor_config=motor_config)
    print("Agent created!")
    
    # Try a simple command
    res = await agent._execute_command("set_rgb_solid", {"r": 255, "g": 0, "b": 0})
    print(res)

asyncio.run(main())
