import cv2
import mediapipe as mp
import time

# Initialize mediapipe modules
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # ---------------- FACE ----------------
    # if results.face_landmarks:
    #     # DRAW ONLY FACE CONTOURS (no green mesh)
    #     mp_drawing.draw_landmarks(
    #         image,
    #         results.face_landmarks,
    #         mp.solutions.face_mesh.FACEMESH_CONTOURS,   # CLEAN OUTLINE ONLY
    #         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
    #         mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
    #     )

    # ---------------- POSE ----------------
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    # ---------------- RIGHT HAND ----------------
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    # ---------------- LEFT HAND ----------------
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # Use DirectShow backend for Windows
    print("Camera opened:", cap.isOpened())

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read() 

            if not ret:
                print("Frame read failed")
                break


            frame = cv2.flip(frame, 1)

            # Mediapipe processing
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Display output
            cv2.imshow('OpenCV Feed - press q to quit', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
