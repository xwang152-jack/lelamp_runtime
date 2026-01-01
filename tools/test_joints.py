#!/usr/bin/env python
import sys
import os
import time

# Add the project root to the python path so we can import lelamp
sys.path.append(os.getcwd())

from lelamp.follower.lelamp_follower import LeLampFollower
from lelamp.follower.config_lelamp_follower import LeLampFollowerConfig

def test_joint(robot, joint_name, description):
    print(f"\n--- Testing Joint: {joint_name} ---")
    print(f"Description from config: {description}")
    
    current_pos_dict = robot.get_observation()
    # Key in observation usually has .pos suffix
    current_angle = current_pos_dict.get(f"{joint_name}.pos")
    
    if current_angle is None:
        print(f"Error: Could not read position for {joint_name}")
        return

    print(f"Current Angle: {current_angle:.2f}")

    # Test moves
    move_delta = 10.0
    
    # Positive Move
    target_pos = current_angle + move_delta
    print(f"\nPreparing to move +{move_delta} degrees -> Target: {target_pos:.2f}")
    input(f"Press ENTER to move {joint_name} POSITIVE (+). Observe direction...")
    
    action = {f"{joint_name}.pos": target_pos}
    robot.send_action(action)
    time.sleep(1.0) # Wait for move
    
    print("Action sent. Please note the direction.")
    
    # Return to original
    print(f"Returning to original: {current_angle:.2f}")
    action = {f"{joint_name}.pos": current_angle}
    robot.send_action(action)
    time.sleep(1.0)
    
    # Negative Move
    target_pos = current_angle - move_delta
    print(f"\nPreparing to move -{move_delta} degrees -> Target: {target_pos:.2f}")
    input(f"Press ENTER to move {joint_name} NEGATIVE (-). Observe direction...")
    
    action = {f"{joint_name}.pos": target_pos}
    robot.send_action(action)
    time.sleep(1.0)
    
    print("Action sent. Please note the direction.")

    # Return to original
    print(f"Returning to original: {current_angle:.2f}")
    action = {f"{joint_name}.pos": current_angle}
    robot.send_action(action)
    time.sleep(1.0)

def main():
    print("LeLamp Joint Direction Tester")
    print("-----------------------------")
    port = os.getenv("LELAMP_PORT", "/dev/ttyACM0")
    print(f"Connecting to LeLamp on {port}...")
    
    config = LeLampFollowerConfig(port=port, id="lelamp_tester")
    robot = LeLampFollower(config)
    
    try:
        robot.connect(calibrate=False)
        print("Connected.")
        
        # Disable torque on exit? Robot class handles disconnect usually.
        
        # Joint definitions from main_deepseek.py
        joint_descriptions = {
            "base_yaw": "Positive = LEFT, Negative = RIGHT",
            "base_pitch": "Positive = FORWARD/DOWN, Negative = BACKWARD/UP",
            "elbow_pitch": "Unknown (Check: +Down?)",
            "wrist_roll": "Unknown (Check: +CW?)",
            "wrist_pitch": "Positive = DOWN, Negative = UP"
        }
        
        joints_to_test = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
        
        for joint in joints_to_test:
            desc = joint_descriptions.get(joint, "Unknown")
            test_joint(robot, joint, desc)
            
            cont = input("\nContinue to next joint? (y/n): ")
            if cont.lower() != 'y':
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Disconnecting...")
        if robot.is_connected:
            robot.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()
