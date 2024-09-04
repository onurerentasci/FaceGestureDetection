import cv2
from face_mesh_detector import FaceMeshDetector
from fps_counter import FPSCounter

def main():
    cap = cv2.VideoCapture("Videos/3.mp4")
    detector = FaceMeshDetector(maxFaces=1)
    fps_counter = FPSCounter()
    blink_count = 0
    blink_threshold = 0.2 # Göz kırpma oranı için eşik değer
    closed_eyes = False

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img, draw=True)
        if faces:
            face = faces[0]

            # Göz kırpma oranını hesapla
            left_eye_ratio = detector.calculate_ratio(face[159], face[145], face[33], face[133])
            right_eye_ratio = detector.calculate_ratio(face[386], face[374], face[362], face[263])
            eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # Göz kırpma sayısını tespit et
            if eye_ratio < blink_threshold:
                if not closed_eyes:
                    blink_count += 1
                    closed_eyes = True
            else:
                closed_eyes = False

            # Göz kırpma sayısını ekrana yazdır
            cv2.putText(img, f"Goz Kirpma: {blink_count}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Gülümseme oranını hesapla
            smile_ratio = detector.calculate_smile_ratio(face)

            # Gülümseme oranını ekrana yazdır
            cv2.putText(img, f"Gulumseme: {smile_ratio:.2f}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Kaş hareketlerini hesapla
            left_eyebrow_movement, right_eyebrow_movement = detector.calculate_eyebrow_movement(face)

            # Kaş hareketlerini ekrana yazdır
            cv2.putText(img, f"Sol Kas: {left_eyebrow_movement:.2f}", (20, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(img, f"Sag Kas: {right_eyebrow_movement:.2f}", (20, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Yanak hareketlerini hesapla
            left_cheek_movement, right_cheek_movement = detector.calculate_cheek_movement(face)

            # Yanak hareketlerini ekrana yazdır
            cv2.putText(img, f"Sol Yanak: {left_cheek_movement:.2f}", (20, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(img, f"Sag Yanak: {right_cheek_movement:.2f}", (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # FPS ve diğer bilgileri ekrana yazdır
        # img = fps_counter.display_fps(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


