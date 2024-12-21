# HAR
Human Activity Recognition
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Directory for storing data
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define activities and parameters
ACTIVITIES = ["laughing","waving","clapping","calling"] #"standing", "walking", "waving"
NUM_CLASSES = len(ACTIVITIES)
SEQUENCE_LENGTH = 30  # Number of frames per sequence
FRAME_RATE = 3  # Collect data every nth frame

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to collect data
def collect_data(activity_name):
    print(f"Collecting data for: {activity_name}")
    cap = cv2.VideoCapture(0)
    data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the camera!")
            break

        frame = cv2.flip(frame, 1)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks and frame_count % FRAME_RATE == 0:
            # Extract landmarks
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            data.append(np.array(landmarks).flatten())

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_count += 1
        cv2.putText(frame, f"Collecting: {activity_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save collected data
    data = np.array(data)
    np.save(os.path.join(DATA_DIR, f"{activity_name}.npy"), data)
    print(f"Data for {activity_name} saved with shape {data.shape}")

# Collect data for each activity
for activity in ACTIVITIES:
    collect_data(activity)

# Load and preprocess data
def load_data():
    X, y = [], []
    for label, activity in enumerate(ACTIVITIES):
        data = np.load(os.path.join(DATA_DIR, f"{activity}.npy"))
        sequences = [data[i:i + SEQUENCE_LENGTH] for i in range(len(data) - SEQUENCE_LENGTH)]
        X.extend(sequences)
        y.extend([label] * len(sequences))
    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = load_data()
print(f"Loaded data: X.shape={X.shape}, y.shape={y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    LSTM(32),
    Dense(32, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

# Train the model
callbacks = [EarlyStopping(monitor="val_loss", patience=5)]
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Save the model
MODEL_PATH = "human_activity_recognition_model.h5"
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Real-time prediction
def real_time_prediction():
    print("Starting real-time prediction...")
    model = load_model(MODEL_PATH)
    sequence = []  # Buffer to hold a sequence of frames
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # Extract landmarks
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            keypoints = np.array(landmarks).flatten()
            sequence.append(keypoints)

            # Maintain sequence length
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # Predict activity if sequence is ready
            if len(sequence) == SEQUENCE_LENGTH:
                prediction = model.predict(np.expand_dims(sequence, axis=0))
                activity = ACTIVITIES[np.argmax(prediction)]

                # Display prediction on frame
                cv2.putText(frame, f"Activity: {activity}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video feed
        cv2.imshow("Real-Time Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time prediction
real_time_prediction()
