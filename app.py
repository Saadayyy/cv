import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import List, Deque, Tuple
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize deques to store drawing points
if "points" not in st.session_state:
    st.session_state.points: List[Deque[Tuple[int, int]]] = [
        deque(maxlen=1024) for _ in range(4)
    ]
if "paintWindow" not in st.session_state:
    st.session_state.paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

colorIndex = 0
is_drawing = False

# Set up the drawing colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]


def main():
    global colorIndex, is_drawing

    st.title("Air Drawing with Hand Gestures")
    run = st.checkbox("Run", value=False, key="run_checkbox")
    record = st.checkbox("Record", value=False, key="record_checkbox")
    clear_canvas = st.button("Clear Canvas", key="clear_button")
    color_index = st.radio(
        "Choose color", ("BLUE", "GREEN", "RED", "YELLOW"), key="color_radio"
    )

    # Set the color index based on radio button
    if color_index == "BLUE":
        colorIndex = 0
    elif color_index == "GREEN":
        colorIndex = 1
    elif color_index == "RED":
        colorIndex = 2
    elif color_index == "YELLOW":
        colorIndex = 3

    if clear_canvas:
        st.session_state.points = [deque(maxlen=1024) for _ in range(4)]
        st.session_state.paintWindow[67:, :, :] = 255

    cap = None
    out = None

    if run:
        # Start capturing video
        cap = cv2.VideoCapture(0)
        paint_placeholder = st.empty()
        frame_placeholder = st.empty()

        # Set up video writer if recording is enabled
        if record:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

        while run:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            hand_landmarks = result.multi_hand_landmarks

            if hand_landmarks:
                for landmarks in hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Get the landmarks for the index finger tip and thumb tip
                    index_finger_tip = landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    index_finger_mcp = landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_MCP
                    ]

                    # Calculate the positions
                    index_finger_tip_pos = (
                        int(index_finger_tip.x * frame.shape[1]),
                        int(index_finger_tip.y * frame.shape[0]),
                    )

                    # Check if the index finger is up or making a fist
                    if index_finger_tip.y < index_finger_mcp.y:
                        is_drawing = True
                    else:
                        is_drawing = False

                    if is_drawing:
                        st.session_state.points[colorIndex].appendleft(
                            index_finger_tip_pos
                        )
                    else:
                        st.session_state.points[colorIndex].appendleft(None)

            for i, color_points in enumerate(st.session_state.points):
                for j in range(1, len(color_points)):
                    if color_points[j - 1] is None or color_points[j] is None:
                        continue
                    cv2.line(frame, color_points[j - 1], color_points[j], colors[i], 2)
                    cv2.line(
                        st.session_state.paintWindow,
                        color_points[j - 1],
                        color_points[j],
                        colors[i],
                        2,
                    )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            paintWindow_rgb = cv2.cvtColor(
                st.session_state.paintWindow, cv2.COLOR_BGR2RGB
            )

            # Display the frame and paint window using Streamlit
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)
            paint_placeholder.image(
                paintWindow_rgb, channels="RGB", use_column_width=True
            )

            # Write the frame to the video file if recording is enabled
            if record:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Check if the 'Run' checkbox is still checked
            run = st.session_state.get("run_checkbox", False)

            # Add a short delay to reduce CPU usage
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.05 - elapsed_time))

        cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
