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
    color = (0, 0, 255)  # красный (BGR)

    # Вертикальный прямоугольник
    top_left_v = (center_x - arm_width // 2, center_y - arm_length // 2)
    bottom_right_v = (center_x + arm_width // 2, center_y + arm_length // 2)
    cv2.rectangle(frame, top_left_v, bottom_right_v, color, 2)

    #Горизонтальный прямоугольник
    top_left_h = (center_x - arm_length // 2, center_y - arm_width // 2)
    bottom_right_h = (center_x + arm_length // 2, center_y + arm_width // 2)

    # Вырезаем горизонтальную область
    roi = frame[top_left_h[1]:bottom_right_h[1], top_left_h[0]:bottom_right_h[0]]

    # Применяем размытие
    blurred_roi = cv2.GaussianBlur(roi, (35, 35), 0)

    # Вставляем размытую часть обратно
    frame[top_left_h[1]:bottom_right_h[1], top_left_h[0]:bottom_right_h[0]] = blurred_roi

    # Рисуем контур горизонтального прямоугольника поверх
    cv2.rectangle(frame, top_left_h, bottom_right_h, color, 2)

    cv2.imshow("Red Cross with Blur", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
