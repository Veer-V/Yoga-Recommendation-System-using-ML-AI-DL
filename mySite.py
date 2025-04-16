from flask import Flask, request, render_template, jsonify
import json
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)

# Load the yoga dataset
with open('Dataset/yoga_dataset.json') as f:
    yoga_data = json.load(f)

# Load the pose detection model
model = load_model('keras_model.h5')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_problem = request.form['problem'].lower()
    recommendations = []

    # Find matching solutions
    for item in yoga_data['poses']:
        if user_problem in item['problem'].lower():
            recommendation = {
                'problem': item['problem'],
                'solutions': [],
                'image_links': []
            }
            # Split each solution and add a link for Google search and YouTube
            for solution in item['solution']:
                solution_data = {
                    'solution': solution,
                    'google_link': f"https://www.google.com/search?hl=en&tbm=isch&q={solution.replace(' ', '+')}+Yoga",
                    'youtube_link': f"https://www.youtube.com/results?search_query={solution.replace(' ', '+')}+Yoga"
                }
                recommendation['solutions'].append(solution_data)
            recommendations.append(recommendation)

    return render_template('index.html', recommendations=recommendations)

@app.route('/pose_detection')
def pose_detection():
    cap = cv2.VideoCapture(0)  # Start the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            feedback = evaluate_pose(results.pose_landmarks)
            cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Yoga Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Pose detection ended."

def evaluate_pose(landmarks):
    if landmarks:
        left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        
        if left_shoulder_y > 0.6 and right_shoulder_y > 0.6:
            return "Sitting Normal"
        elif left_shoulder_y < 0.4 and right_shoulder_y < 0.4:
            return "Cobra Pose Detected"

        return "Pose Analysis In Progress"

    return "Pose Not Detected"

if __name__ == '__main__':
    app.run(debug=True)
