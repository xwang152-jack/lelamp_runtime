import os
import csv
import time
import threading
from typing import Any, List
from ..base import ServiceBase, Priority
from lelamp.follower import LeLampFollowerConfig, LeLampFollower


class MotorsService(ServiceBase):
    # Valid joint names for the LeLamp robot
    VALID_JOINTS = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
    
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
        elif event_type == "move_joint":
            self._handle_move_joint(payload)
        elif event_type == "stop":
            self.logger.info("Stopping motors playback")
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_move_joint(self, payload: dict):
        """Move a single joint to a specified angle"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return
        
        joint_name = payload.get("joint_name")
        angle = payload.get("angle")
        
        if joint_name not in self.VALID_JOINTS:
            self.logger.error(f"Invalid joint name: {joint_name}")
            return
        
        try:
            # Get current positions of all joints first
            obs = self.robot.get_observation()
            action = {}
            for key, value in obs.items():
                if key.endswith(".pos"):
                    action[key] = value
            
            # Update only the target joint
            action[f"{joint_name}.pos"] = float(angle)
            
            # Send full action with all joints
            self.robot.send_action(action)
            self.logger.info(f"Moved joint {joint_name} to {angle} degrees")
        except Exception as e:
            self.logger.error(f"Error moving joint {joint_name}: {e}")
    
    def get_joint_positions(self) -> dict[str, float]:
        """Get current positions of all joints"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return {}
        
        try:
            obs = self.robot.get_observation()
            # Extract joint positions (remove ".pos" suffix from keys)
            positions = {}
            for key, value in obs.items():
                if key.endswith(".pos"):
                    joint_name = key.removesuffix(".pos")
                    if joint_name in self.VALID_JOINTS:
                        positions[joint_name] = value
            return positions
        except Exception as e:
            self.logger.error(f"Error getting joint positions: {e}")
            return {}
    
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
