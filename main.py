import os
import cv2
import wget
import argparse
import numpy as np
import pandas as pd
import tkinter as tk
from ultralytics import YOLO
from collections import deque

# --- 1. НАСТРОЙКИ И КОНСТАНТЫ ---
STATE_GREEN = 'green'   # Стол пустой (людей в зоне нет)
STATE_YELLOW = 'yellow' # Подход к столу (переходное состояние)
STATE_RED = 'red'       # Стол занят (есть человек в зоне)

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
MODEL_PATH = "yolo26n.pt"

def ensure_model_exists():
    """
    Проверяет наличие файла модели YOLO в рабочей директории.
    Если файл отсутствует, скачивает его из официального репозитория Ultralytics.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Модель не найдена. Скачивание {os.path.basename(MODEL_PATH)}...")
        wget.download(MODEL_URL, MODEL_PATH)
        print(f"\nМодель успешно скачана.")
    else:
        print(f"Модель уже существует. Пропускаем скачивание.")

def fit_frame_to_screen(frame):
    """
    Масштабирует входной кадр видео для отображения на экране с разрешением 1920x1080.
    Масштабирование происходит с сохранением исходных пропорций кадра,
    чтобы избежать искажения изображения.
    """
    screen_res = (1920, 1080)
    frame_h, frame_w = frame.shape[:2]
    scale = min(screen_res[0] / frame_w, screen_res[1] / frame_h)
    new_size = (int(frame_w * scale), int(frame_h * scale))
    return cv2.resize(frame, new_size)

def calculate_intersection_area(box1, box2):
    """
    Вычисляет площадь пересечения (overlap area) между двумя прямоугольниками.
    Прямоугольники задаются координатами верхнего левого угла и размерами [x, y, width, height].
    Используется для определения факта пересечения ограничивающей рамки человека
    с зоной столика.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)

def get_scaled_frame(frame, target_width, target_height):
    """
    Масштабирует кадр до заданных размеров (target_width, target_height) и центрирует его
    на черном холсте. Это предотвращает искажения (растяжение) видео при изменении
    размера окна приложения.
    """
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_size = (int(w * scale), int(h * scale))
    resized_frame = cv2.resize(frame, new_size)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_size[0]) // 2
    y_offset = (target_height - new_size[1]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized_frame
    return canvas

def main(video_path):
    window_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- 2. ПЕРЕМЕННЫЕ ДЛЯ АНАЛИТИКИ ---
    analytics = {
        'total_empty_time': 0.0,
        'total_occupied_time': 0.0,
        'empty_intervals_count': 0,
        'occupied_intervals_count': 0,
        'empty_start_time': None,
        'occupied_start_time': None,
        'table_state': STATE_GREEN,
        'pending_occupied_time': 0.0,
        'pending_empty_time': 0.0,
    }

    # --- 3. ИНИЦИАЛИЗАЦИЯ ВИДЕО И МОДЕЛИ ---
    screen_width, screen_height = 1920, 1080 
    
    model = YOLO('yolo26n.pt')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame_original = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать кадр для выбора ROI.")
        return

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.withdraw()

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    screen_ratio = screen_width / screen_height
    video_ratio = original_width / original_height

    if video_ratio > screen_ratio:
        scale = screen_width / original_width
    else:
        scale = screen_height / original_height

    preview_width = int(original_width * scale)
    preview_height = int(original_height * scale)

    scaled_frame = cv2.resize(frame_original, (preview_width, preview_height))

    roi_scaled = cv2.selectROI("Выберите столик", scaled_frame, fromCenter=False, showCrosshair=True)

    if roi_scaled == (0, 0, 0, 0):
        print("ROI не выбран. Завершение работы.")
        return

    cv2.destroyAllWindows()

    rx_s, ry_s, rw_s, rh_s = roi_scaled
    roi_original = (
        int(rx_s / scale),
        int(ry_s / scale),
        int(rw_s / scale),
        int(rh_s / scale)
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (original_width, original_height))

    BUFFER_SIZE = 1200 
    state_buffer = deque([STATE_GREEN] * BUFFER_SIZE, maxlen=BUFFER_SIZE) 
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_width, screen_height) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0

    while True:
        ret, frame_original = cap.read()
        if not ret:
            break

        results = model(frame_original, classes=[0], conf=0.3)
        
        # --- ОПРЕДЕЛЕНИЕ ФАКТИЧЕСКОГО СОСТОЯНИЯ ПО ПЕРЕСЕЧЕНИЮ ---
        current_frame_state_by_intersection = STATE_GREEN # по умолчанию пусто

        rx_o, ry_o, rw_o, rh_o = roi_original

        closest_person = None
        min_distance_to_center = float('inf')

        for result in results[0].boxes:
            pb = list(map(int, result.xyxy[0]))
            person_box_wh = [pb[0], pb[1], pb[2] - pb[0], pb[3] - pb[1]]
            
            intersection_area = calculate_intersection_area(person_box_wh, roi_original)
            
            if intersection_area > 0:
                person_center_x = pb[0] + person_box_wh[2] / 2
                person_center_y = pb[1] + person_box_wh[3] / 2
                roi_center_x = rx_o + rw_o / 2
                roi_center_y = ry_o + rh_o / 2
                
                dist = ((person_center_x - roi_center_x) ** 2 + (person_center_y - roi_center_y) ** 2) ** 0.5
                
                if dist < min_distance_to_center:
                    min_distance_to_center = dist
                    closest_person = {
                        'box': person_box_wh,
                        'area': person_box_wh[2] * person_box_wh[3],
                        'intersection_area': intersection_area
                    }
        
        if closest_person is not None:
            person_area = closest_person['area']
            intersection_area = closest_person['intersection_area']
            
            if intersection_area >= (person_area / 2):
                current_frame_state_by_intersection = STATE_RED # занят
            else: 
                current_frame_state_by_intersection = STATE_YELLOW # переходное

        # --- ВИЗУАЛЬНОЕ СОСТОЯНИЕ (по буферу) ---
        state_buffer.append(current_frame_state_by_intersection)
        
        count_green = list(state_buffer).count(STATE_GREEN)
        count_yellow = list(state_buffer).count(STATE_YELLOW)
        count_red = list(state_buffer).count(STATE_RED)
        
        dominant_state_counts = {
            STATE_GREEN: count_green,
            STATE_YELLOW: count_yellow,
            STATE_RED: count_red
        }
        
        final_color_state_for_drawing = max(dominant_state_counts, key=dominant_state_counts.get)
        
        # --- НОВАЯ ЛОГИКА: УЧЁТ ВРЕМЕНИ ДО СМЕНЫ ЦВЕТА РАМКИ ---
        current_time_sec = frame_idx / fps

        # Если фактическое состояние — "занят", а визуальное ещё нет — накапливаем время в pending_occupied_time
        if current_frame_state_by_intersection == STATE_RED and final_color_state_for_drawing != STATE_RED:
            analytics['pending_occupied_time'] += 1/fps

        # Если визуальное состояние сменилось на "занят" (RED)
        if final_color_state_for_drawing == STATE_RED:
            # Если предыдущее состояние было "пусто" (GREEN), вычитаем pending_empty_time
            if analytics['table_state'] == STATE_GREEN:
                analytics['total_empty_time'] -= analytics['pending_occupied_time']
                analytics['pending_empty_time'] = 0.0

            # Переносим накопленное время в основной таймер занятости
            analytics['total_occupied_time'] += analytics['pending_occupied_time']
            analytics['pending_occupied_time'] = 0.0

            # Начало нового интервала занятости
            if analytics['occupied_start_time'] is None:
                analytics['occupied_start_time'] = current_time_sec
                analytics['occupied_intervals_count'] += 1

            # Если был отсчёт простоя — закрываем его и переносим накопленное время пустоты
            if analytics['empty_start_time'] is not None:
                analytics['total_empty_time'] += current_time_sec - analytics['empty_start_time']
                analytics['total_empty_time'] += analytics['pending_empty_time']
                analytics['pending_empty_time'] = 0.0
                analytics['empty_start_time'] = None

            analytics['table_state'] = STATE_RED

        # Если визуальное состояние сменилось на "пусто" (GREEN)
        elif final_color_state_for_drawing == STATE_GREEN:
            # Если предыдущее состояние было "занято" (RED), вычитаем pending_occupied_time
            if analytics['table_state'] == STATE_RED:
                analytics['total_occupied_time'] -= analytics['pending_empty_time']
                analytics['pending_occupied_time'] = 0.0

            # Переносим накопленное время в основной таймер пустоты
            analytics['total_empty_time'] += analytics['pending_empty_time']
            analytics['pending_empty_time'] = 0.0

            # Начало нового интервала пустоты
            if analytics['empty_start_time'] is None:
                analytics['empty_start_time'] = current_time_sec
                analytics['empty_intervals_count'] += 1

            # Если был отсчёт занятости — закрываем его и переносим накопленное время занятости
            if analytics['occupied_start_time'] is not None:
                analytics['total_occupied_time'] += current_time_sec - analytics['occupied_start_time']
                analytics['total_occupied_time'] += analytics['pending_occupied_time']
                analytics['pending_occupied_time'] = 0.0
                analytics['occupied_start_time'] = None

            analytics['table_state'] = STATE_GREEN 
         
        
        final_roi_color_for_drawing = {
            STATE_GREEN: COLOR_GREEN,
            STATE_YELLOW: COLOR_YELLOW,
            STATE_RED: COLOR_RED
        }[final_color_state_for_drawing]
         
        cv2.rectangle(frame_original, (rx_o, ry_o), (rx_o + rw_o, ry_o + rh_o), final_roi_color_for_drawing, 3)
         
        out.write(frame_original) 

        window_width = cv2.getWindowImageRect(window_name)[2]
        window_height = cv2.getWindowImageRect(window_name)[3]

        frame_to_show = get_scaled_frame(frame_original.copy(), window_width, window_height)
         
        cv2.imshow(window_name, frame_to_show)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_frames = frame_idx
    total_video_time_sec = total_frames / fps if fps > 0 else 0

    current_time_sec = total_video_time_sec

    # Финальное закрытие таймеров и добавление "висящего" времени из pending_
    
    # Если видео закончилось на "занято"
    if analytics['table_state'] == STATE_RED and analytics['occupied_start_time'] is not None:
         analytics['total_occupied_time'] += current_time_sec - analytics['occupied_start_time']
         analytics['total_occupied_time'] += analytics['pending_occupied_time']
         analytics['pending_occupied_time'] = 0.0

    # Если видео закончилось на "пусто"
    if analytics['table_state'] == STATE_GREEN and analytics['empty_start_time'] is not None:
        analytics['total_empty_time'] += current_time_sec - analytics['empty_start_time']
        analytics['total_empty_time'] += analytics['pending_empty_time']
        analytics['pending_empty_time'] = 0.0

    # Если видео закончилось в "жёлтом" состоянии — добавляем время к тому таймеру, который был активен до этого.
    # Это реализует правило: "жёлтый цвет и буфер считаем временем переходным и в это время продолжает идти таймер"
    elif final_color_state_for_drawing == STATE_YELLOW:
        if analytics['table_state'] == STATE_RED and analytics['occupied_start_time'] is not None:
            analytics['total_occupied_time'] += current_time_sec - analytics['occupied_start_time']
            analytics['total_occupied_time'] += analytics['pending_occupied_time']
            analytics['pending_occupied_time'] = 0.0
        elif analytics['table_state'] == STATE_GREEN and analytics['empty_start_time'] is not None:
            analytics['total_empty_time'] += current_time_sec - analytics['empty_start_time']
            analytics['total_empty_time'] += analytics['pending_empty_time']
            analytics['pending_empty_time'] = 0.0
      
    if total_video_time_sec > 0:
        occupancy_percent = (analytics['total_occupied_time'] / total_video_time_sec) * 100
        
        avg_empty_duration = analytics['total_empty_time'] / analytics['empty_intervals_count'] if analytics['empty_intervals_count'] > 0 else None
        avg_occupied_duration = analytics['total_occupied_time'] / analytics['occupied_intervals_count'] if analytics['occupied_intervals_count'] > 0 else None

    print("\n--- ИТОГОВАЯ АНАЛИТИКА ---")
    print(f"Общая длительность видео: {total_video_time_sec:.1f} сек")
    print(f"Количество периодов простоя: {analytics['empty_intervals_count']}")
    print(f"Средняя длительность простоя: {avg_empty_duration:.1f} сек" if avg_empty_duration is not None else "Нет данных для расчета")
    print(f"Количество периодов занятости: {analytics['occupied_intervals_count']}")
    print(f"Средняя длительность наличия клиента: {avg_occupied_duration:.1f} сек" if avg_occupied_duration is not None else "Нет данных для расчета")
     
if __name__ == "__main__":
    ensure_model_exists()
    
    parser = argparse.ArgumentParser(description="Прототип детекции уборки столиков по видео.")
    parser.add_argument("--video", type=str, required=True,
                        help="Путь к видеофайлу")
     
    args = parser.parse_args()
    
    main(args.video)

