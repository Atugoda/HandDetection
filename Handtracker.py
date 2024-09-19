import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize system volume control via pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Capture video from the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Initialize MediaPipe Hands with specific configurations
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Only detect one hand
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert the image from BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        results = hands.process(img_rgb)

        # If a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get the positions of the thumb tip (landmark 4) and index finger tip (landmark 8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Get coordinates for the thumb tip and index finger tip
                h, w, _ = img.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circles on thumb and index finger tips
                cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)

                # Draw a line between the thumb tip and index finger tip
                cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)

                # Calculate the distance between the thumb and index finger
                length = math.hypot(index_x - thumb_x, index_y - thumb_y)

                # Map the length (distance) to the volume range
                vol = np.interp(length, [20, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                # Add text to show the distance (optional)
                cv2.putText(img, f'Volume: {int(np.interp(vol, [min_vol, max_vol], [0, 100]))} %', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image with hand landmarks and volume control
        cv2.imshow("Hand Tracker - Volume Control", img)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
























