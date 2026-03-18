from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)

from lelamp.config import load_motor_config
from lelamp.service.motors.motors_service import MotorsService
from lelamp.service.rgb.rgb_service import RGBService

@asynccontextmanager
async def lifespan(app: FastAPI):
    motor_config = load_motor_config()
    motors_service = MotorsService(port="/dev/ttyACM0", config=motor_config)
    rgb_service = RGBService()
    
    motors_service.start()
    rgb_service.start()
    
    app.state.motors_service = motors_service
    app.state.rgb_service = rgb_service
    
    yield
    
    motors_service.stop()
    rgb_service.stop()

app = FastAPI(lifespan=lifespan)
