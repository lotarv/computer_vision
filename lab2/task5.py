import cv2
import numpy as np

camera = cv2.VideoCapture(0)
trajectory = []
while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создаем маску для красного цвета (два диапазона из-за цикличности Hue)
    red_lower1 = np.array([0, 200, 100])
    red_upper1 = np.array([5, 255, 255])
    red_lower2 = np.array([175, 200, 100])
    red_upper2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(frame_hsv, red_lower1, red_upper1) + cv2.inRange(
        frame_hsv, red_lower2, red_upper2
    )

    # Морфологические операции для объединения близких областей и удаления шума
    kernel = np.ones((15, 15), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # ВЫЧИСЛЕНИЕ МОМЕНТОВ ИЗОБРАЖЕНИЯ
    # Моменты вычисляются напрямую из маски без использования контуров
    moments = cv2.moments(red_mask)

    # m00 - нулевой момент (площадь)
    area = moments["m00"]

    area_text = f"Area: {int(area)}"

    # Если найдены красные пиксели и площадь достаточная
    if area > 500:
        # ЦЕНТР МАСС через моменты первого порядка
        # m10 - момент первого порядка по X
        # m01 - момент первого порядка по Y
        # Центр масс: (m10/m00, m01/m00)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])

        # Для bounding box находим координаты пикселей
        y_coords, x_coords = np.where(red_mask > 0)

        x_min = x_coords.min()
        x_max = x_coords.max()
        y_min = y_coords.min()
        y_max = y_coords.max()

        trajectory.append([cX, cY])
        # Рисуем bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)

        # Рисуем центр масс (вычисленный через моменты!)
        cv2.circle(frame, (cX, cY), 3, (255, 0, 0), -1)

        trajectory_formatted = np.array(trajectory, np.int32)
        cv2.polylines(frame, [trajectory_formatted], False, (0, 0, 255), 4)

        # Дополнительно: выводим информацию о моментах для понимания
        cv2.putText(
            frame,
            f"m00: {int(moments['m00'])}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Center: ({cX}, {cY})",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Выводим площадь в левом верхнем углу
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
