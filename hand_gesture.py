import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

# Define a function to detect hand gestures based on landmarks
def detect_gesture(landmarks):
    # Get the positions of specific landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Example of detecting a "thumb up" gesture
    if index_tip.y < thumb_tip.y:
        return "Thumb Up"
    
    # Example of detecting an "open hand" gesture (all fingers are spread)
    if index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return "Open Hand"
    
    return "Unknown Gesture"

# Loop to continuously capture frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a more intuitive experience
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect hand gestures
            gesture = detect_gesture(landmarks.landmark)
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame with the hand landmarks and detected gesture
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyqAllWindows()
