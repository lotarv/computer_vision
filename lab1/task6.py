import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    arm_width = 60    
    arm_length = 400   

    color = (0, 0, 255)       

    # Вертикальный прямоугольник
    top_left_v = (center_x - arm_width // 2, center_y - arm_length // 2)
    bottom_right_v = (center_x + arm_width // 2, center_y + arm_length // 2)
    cv2.rectangle(frame, top_left_v, bottom_right_v, color)

    # Горизонтальный прямоугольник
    top_left_h = (center_x - arm_length // 2, center_y - arm_width // 2)
    bottom_right_h = (center_x + arm_length // 2, center_y + arm_width // 2)
    cv2.rectangle(frame, top_left_h, bottom_right_h, color)

    cv2.imshow("Red Cross", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
