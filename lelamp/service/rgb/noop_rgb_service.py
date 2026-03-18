from lelamp.service.base import ServiceBase


class NoOpRGBService(ServiceBase):
    def __init__(self):
        super().__init__("rgb")

    def handle_event(self, event_type, payload):
        return
