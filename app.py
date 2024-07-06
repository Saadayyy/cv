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


def air_drawing():
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
                    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                    # Calculate the positions
                    index_finger_tip_pos = (
                        int(index_finger_tip.x * frame.shape[1]),
                        int(index_finger_tip.y * frame.shape[0]),
                    )
                    thumb_tip_pos = (
                        int(thumb_tip.x * frame.shape[1]),
                        int(thumb_tip.y * frame.shape[0]),
                    )

                    # Calculate the Euclidean distance between index finger tip and thumb tip
                    distance = np.sqrt(
                        (index_finger_tip_pos[0] - thumb_tip_pos[0]) ** 2
                        + (index_finger_tip_pos[1] - thumb_tip_pos[1]) ** 2
                    )

                    # Check if the distance is below a threshold (adjust the threshold as needed)
                    if distance < 30:
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


def object_tracking():
    st.title("Object Tracking with HSV Masking")
    run = st.checkbox("Run", value=False, key="run_checkbox_hsv")

    # HSV range sliders
    st.sidebar.title("HSV Range for Object Tracking")
    l_hue = st.sidebar.slider("Lower Hue", 0, 180, 64)
    l_saturation = st.sidebar.slider("Lower Saturation", 0, 255, 72)
    l_value = st.sidebar.slider("Lower Value", 0, 255, 49)
    u_hue = st.sidebar.slider("Upper Hue", 0, 180, 153)
    u_saturation = st.sidebar.slider("Upper Saturation", 0, 255, 255)
    u_value = st.sidebar.slider("Upper Value", 0, 255, 255)

    # Arrays to handle colour points of different colours
    points = [deque(maxlen=1024) for _ in range(4)]
    colorIndex = 0

    # Set up the paint window
    paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)
    cv2.putText(
        paintWindow,
        "CLEAR",
        (49, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        paintWindow,
        "BLUE",
        (185, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        paintWindow,
        "GREEN",
        (298, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        paintWindow,
        "RED",
        (420, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        paintWindow,
        "YELLOW",
        (520, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        2,
        cv2.LINE_AA,
    )
    cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

    cap = None
    kernel = np.ones((5, 5), np.uint8)

    if run:
        cap = cv2.VideoCapture(0)
        paint_placeholder = st.empty()
        frame_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break

            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            Upper_hsv = np.array([u_hue, u_saturation, u_value])
            Lower_hsv = np.array([l_hue, l_saturation, l_value])
            Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
            Mask = cv2.erode(Mask, None, iterations=2)
            Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
            Mask = cv2.dilate(Mask, None, iterations=1)

            # Find contours for the pointer after identifying it
            cnts, _ = cv2.findContours(
                Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            center = None

            if len(cnts) > 0:
                # Get the largest contour and its center
                cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                M = cv2.moments(cnt)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if center[1] <= 65:
                    if 40 <= center[0] <= 140:  # Clear All
                        points = [deque(maxlen=1024) for _ in range(4)]
                        paintWindow[67:, :, :] = 255
                    elif 160 <= center[0] <= 255:
                        colorIndex = 0  # Blue
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1  # Green
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2  # Red
                    elif 505 <= center[0] <= 600:
                        colorIndex = 3  # Yellow
                else:
                    points[colorIndex].appendleft(center)
            else:
                points[colorIndex].appendleft(None)

            # Draw lines of all the colors on the canvas and frame
            for i in range(len(points)):
                for j in range(len(points[i])):
                    if points[i][j] is None or points[i][j - 1] is None:
                        continue
                    cv2.line(frame, points[i][j - 1], points[i][j], colors[i], 2)
                    cv2.line(paintWindow, points[i][j - 1], points[i][j], colors[i], 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            paintWindow_rgb = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame, channels="RGB", use_column_width=True)
            paint_placeholder.image(
                paintWindow_rgb, channels="RGB", use_column_width=True
            )

            run = st.session_state.get("run_checkbox_hsv", False)

        cap.release()
        cv2.destroyAllWindows()


mode = st.sidebar.selectbox("Select Mode", ["Air Drawing", "Object Tracking"])

if mode == "Air Drawing":
    air_drawing()
else:
    object_tracking()
