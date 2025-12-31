import os
import csv
import time
import threading
from typing import Any, List
from ..base import ServiceBase, Priority
from lelamp.follower import LeLampFollowerConfig, LeLampFollower


class MotorsService(ServiceBase):
    def __init__(self, port: str, lamp_id: str, fps: int = 30):
        super().__init__("motors")
        self.port = port
        self.lamp_id = lamp_id
        self.fps = fps
        self.robot_config = LeLampFollowerConfig(port=port, id=lamp_id)
        self.robot: LeLampFollower = None
        self.recordings_dir = os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
        self._cancel_playback = threading.Event()

    def dispatch(self, event_type: str, payload: Any, priority: Priority = Priority.NORMAL):
        if event_type == "stop":
            self._cancel_playback.set()
        return super().dispatch(event_type, payload, priority=priority)
    
    def start(self):
        super().start()
        self.robot = LeLampFollower(self.robot_config)
        self.robot.connect(calibrate=False)
        self.logger.info(f"Motors service connected to {self.port}")

    def stop(self, timeout: float = 5.0):
        if self.robot:
            self.robot.disconnect()
            self.robot = None
        super().stop(timeout)
    
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        elif event_type == "stop":
            self.logger.info("Stopping motors playback")
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_play(self, recording_name: str):
        """Play a recording by name"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return

        self._cancel_playback.clear()
        
        csv_filename = f"{recording_name}.csv"
        csv_path = os.path.join(self.recordings_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            self.logger.error(f"Recording not found: {csv_path}")
            return
        
        try:
            with open(csv_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                actions = list(csv_reader)
            
            self.logger.info(f"Playing {len(actions)} actions from {recording_name}")
            
            for row in actions:
                if self._cancel_playback.is_set():
                    self.logger.info(f"Playback cancelled: {recording_name}")
                    break
                t0 = time.perf_counter()
                
                # Extract action data (exclude timestamp column)
                action = {key: float(value) for key, value in row.items() if key != 'timestamp'}
                self.robot.send_action(action)
                
                # Use time.sleep instead of busy_wait to avoid blocking other threads
                sleep_time = 1.0 / self.fps - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.logger.info(f"Finished playing recording: {recording_name}")
            
        except Exception as e:
            self.logger.error(f"Error playing recording {recording_name}: {e}")
    
    def get_available_recordings(self) -> List[str]:
        """Get list of recording names available for this lamp ID"""
        if not os.path.exists(self.recordings_dir):
            return []
        
        recordings = []
        suffix = f".csv"
        
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(suffix):
                # Remove the lamp_id suffix to get the recording name
                recording_name = filename[:-len(suffix)]
                recordings.append(recording_name)
        
        return sorted(recordings)
