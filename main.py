import cv2
from face_mesh_detector import FaceMeshDetector
from fps_counter import FPSCounter

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps_counter = FPSCounter()
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from video.")
            break

        img, faces = detector.findFaceMesh(img, False)
        if len(faces) != 0:
            print(faces[0])

        fps = fps_counter.update()
        cv2.putText(img, f"FPS: {fps}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
