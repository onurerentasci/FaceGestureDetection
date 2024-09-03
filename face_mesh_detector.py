import cv2
import mediapipe as mp
import math

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for lm in faceLms.landmark:
                    ih, iw, _ = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def calculate_ratio(self, point1, point2, point3, point4):
        dist1 = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        dist2 = math.sqrt((point4[0] - point3[0]) ** 2 + (point4[1] - point3[1]) ** 2)
        return dist1 / dist2

    def calculate_smile_ratio(self, face):
        left_lip_corner = face[61]  # Sol dudak köşesi
        right_lip_corner = face[291]  # Sağ dudak köşesi

        upper_lip = face[13]  # Dudakların üst noktası
        lower_lip = face[14]  # Dudakların alt noktası

        horizontal_dist = self.calculate_distance(left_lip_corner, right_lip_corner)

        vertical_dist = self.calculate_distance(upper_lip, lower_lip)

        smile_ratio = horizontal_dist / vertical_dist if vertical_dist != 0 else 0

        return smile_ratio

    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)