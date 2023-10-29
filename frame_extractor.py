from deepface import DeepFace
import cv2
import os
import argparse
from time import perf_counter
from datetime import datetime

parser = argparse.ArgumentParser(description="frame extractor")
parser.add_argument("-video_path", "--video_path", help="specify path to video to extract frames from",
                    default="C:\\Users\\milen\\test\\test2.mp4", type=str)
args = parser.parse_args()

video_path = args.video_path
video_name = video_path.split('\\')[-1].split(".")[0]

# сохраняем данные о работе программы в файл
f = open(f"{video_name}-extraction-log.txt", "w")
f.write(f"{video_name}.mp4 frame extraction log {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

isExist = os.path.exists(f"{video_name}_frames")
if not isExist:
    os.mkdir(f"{video_name}_frames")

# открываем видеофайл
cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

n = 10
n_counter = 0
saved_frame_counter = 1
frame_counter = 0
start_time = perf_counter()

while (cap.isOpened()):
    print(f"n_counter: {n_counter}")
    if n_counter == 0:
        frame_counter += 1
        # захват кадра
        ret, frame = cap.read()
        if ret == True:
            try:
                # обнаружение лиц на текущем кадре
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="retinaface",
                    target_size=(224, 224)
                )
                if (len(faces) > 1):
                    print("too many faces found on that frame, skipping")
                    continue

                # сохраненяем текущий кадр
                cv2.imwrite(f"{video_name}_frames/{saved_frame_counter}.jpg", frame)
                saved_frame_counter += 1

            except ValueError as e:
                # если лицо в кадре не обнаружено
                print(f"No face found for current frame, error message: {str(e)}")
                n_counter = n / 2
                continue
        # прерываем цикл
        else:
            time_elapsed = perf_counter() - start_time

            print(f"Time elapsed: {time_elapsed}, FPS: {frame_counter / time_elapsed}\n")
            f.write(f"Time elapsed: {time_elapsed}, FPS: {frame_counter / time_elapsed}\n")
            print(f"frames processed: {frame_counter}")
            f.write(f"frames processed: {frame_counter}")
            print("video ended, exiting")
            f.write("video ended, exiting")
            break
        n_counter = n
    else:
        n_counter -= 1
        _ = cap.grab()

# когда все сделано, освобождаем видеозахват и закрываем файл для записи данных
cap.release()
f.close()
