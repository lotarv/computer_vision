import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 200, 100])
    red_upper1 = np.array([5, 255, 255])
    red_lower2 = np.array([175, 200, 100])
    red_upper2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(frame_hsv, red_lower1, red_upper1) + cv2.inRange(
        frame_hsv, red_lower2, red_upper2
    )

    kernel = np.ones((15, 15), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    area_text = "Area: 0"

    if valid_contours:
        all_points = np.vstack(valid_contours)
        total_area = sum(cv2.contourArea(cnt) for cnt in valid_contours)
        area_text = f"Area: {int(total_area)}"

        M = cv2.moments(all_points)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)  # только центр

    cv2.putText(
        frame,
        area_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Red_object_center", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
