from lelamp.service.base import ServiceBase


class NoOpMotorsService(ServiceBase):
    def __init__(self):
        super().__init__("motors")
        self.robot = None

    def handle_event(self, event_type, payload):
        return

    def get_available_recordings(self):
        return []

    def get_motor_health_summary(self):
        return {"error": "Health monitoring not enabled"}

    def reset_health_statistics(self, motor_name=None):
        return

    def clear_health_history(self, motor_name=None):
        return

    def check_motor_stall(self, motor_name):
        return False
