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

    def calculate_eyebrow_movement(self, face):
        # Sağ kaş noktaları
        right_eyebrow = [face[70], face[63], face[105]]
        right_eye_top = face[159]  # Sağ gözün üst noktası

        # Sol kaş noktaları
        left_eyebrow = [face[336], face[296], face[334]]
        left_eye_top = face[386]  # Sol gözün üst noktası

        # Sağ kaş hareketi (sağ kaşın ortalama pozisyonu ile sağ gözün üst noktası arasındaki mesafe)
        right_eyebrow_movement = self.calculate_average_distance(right_eyebrow, right_eye_top)

        # Sol kaş hareketi (sol kaşın ortalama pozisyonu ile sol gözün üst noktası arasındaki mesafe)
        left_eyebrow_movement = self.calculate_average_distance(left_eyebrow, left_eye_top)

        return left_eyebrow_movement, right_eyebrow_movement

    def calculate_average_distance(self, points, reference_point):
        distances = [self.calculate_distance(point, reference_point) for point in points]
        return sum(distances) / len(distances)

    def calculate_cheek_movement(self, face):
        # Sağ yanak noktaları
        right_cheek_points = [face[50], face[101], face[205]]
        right_nose_point = face[1]  # Sağ yanaktaki referans burun noktası

        # Sol yanak noktaları
        left_cheek_points = [face[280], face[330], face[425]]
        left_nose_point = face[1]  # Sol yanaktaki referans burun noktası

        # Sağ yanak hareketi (sağ yanaktaki noktaların buruna olan ortalama mesafesi)
        right_cheek_movement = self.calculate_average_distance(right_cheek_points, right_nose_point)

        # Sol yanak hareketi (sol yanaktaki noktaların buruna olan ortalama mesafesi)
        left_cheek_movement = self.calculate_average_distance(left_cheek_points, left_nose_point)

        return left_cheek_movement, right_cheek_movement

    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)