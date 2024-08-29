import time

class FPSCounter:
    def __init__(self):
        self.pTime = 0

    def update(self):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        return int(fps)
