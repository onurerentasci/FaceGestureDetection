import time
import cv2

class FPSCounter:
    def __init__(self):
        self.pTime = 0

    def display_fps(self, img):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img
