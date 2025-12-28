#  밤/낮 판단기
class DayNightDetector:
    def __init__(self, brightness_threshold=100):
        self.brightness_threshold = brightness_threshold

    def predict(self, brightness_value):
        return "NIGHT" if brightness_value < self.brightness_threshold else "DAY"

