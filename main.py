import cv2
from face_mesh_detector import FaceMeshDetector
from fps_counter import FPSCounter

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)
    fps_counter = FPSCounter()
    blink_count = 0
    blink_threshold = 0.2
    closed_eyes = False

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img, draw=True)
        if faces:
            face = faces[0]

            left_eye_ratio = detector.calculate_ratio(face[159], face[145], face[33], face[133])
            right_eye_ratio = detector.calculate_ratio(face[386], face[374], face[362], face[263])
            eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if eye_ratio < blink_threshold:
                if not closed_eyes:
                    blink_count += 1
                    closed_eyes = True
            else:
                closed_eyes = False

            cv2.putText(img, f"Goz Kirpma: {blink_count}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            smile_ratio = detector.calculate_smile_ratio(face)

            cv2.putText(img, f"Gulumseme: {smile_ratio:.2f}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # img = fps_counter.display_fps(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

