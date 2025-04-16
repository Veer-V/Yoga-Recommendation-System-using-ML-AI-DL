import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the model
model = load_model('keras_model.h5')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def preprocess_frame(frame):
    frame = cv2.resize(frame, (150, 150))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_pose(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    predicted_classes = np.argsort(prediction[0])[-3:][::-1]  # Get top 3 predictions

    # Map predicted class indices to pose names
    pose_names = ['Cobra Pose', 'Dog Pose', 'Padmasana', 'Tree Pose', 'Triangle Pose', 'Tension Headache', 'Migraine', 'Sitting Normal']
    detected_poses = [pose_names[i] for i in predicted_classes if prediction[0][i] > 0.5]  # Only include poses with confidence > 0.5

    return detected_poses

def evaluate_pose(landmarks):
    if landmarks:
        # Example of detecting "neutral" or "resting" pose
        left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        
        # Check for specific pose conditions
        if left_shoulder_y > 0.6 and right_shoulder_y > 0.6:
            return "Standing Normal"  # Indicates sitting normally
        elif left_shoulder_y < 0.4 and right_shoulder_y < 0.4:
            return "Cobra Pose Detected"  # Example for Cobra Pose
        # Add more conditions for other poses as needed

        return "Pose Analysis In Progress"  # Placeholder for actual pose evaluation

    return "Pose Not Detected"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for pose estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Evaluate the pose
        feedback = evaluate_pose(results.pose_landmarks)
        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    detected_poses = predict_pose(frame)
    cv2.putText(frame, f'Poses: {", ".join(detected_poses)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Yoga Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
