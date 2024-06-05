from deepface import DeepFace
import cv2
import os
import numpy as np
import pandas as pd

model_names = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "ArcFace",
    "SFace",
]
detector_backends = ["retinaface"]

i = 0
k = 0
# путь к директории с изображениями
frames_directory = r"C:\Users\milen\proverka1"
db_image = r"C:\Users\milen\ameliseenko\ameliseenko_glasses.JPG"
ref_image = os.path.basename(db_image)
email = "ameliseenko@edu.hse.ru"
frames_list = os.listdir(frames_directory)
false_count = 0
true_count = 0

# функция для визуализации результатов проверки лица
def visualise_deepface_verify_obj(image, verification_results = {}):
    color = (128, 0, 128)
    face_pixel_height = verification_results['facial_areas']['img1']['h']
    img = cv2.putText(image, f"verified: {verification_results['verified']}", (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    img = cv2.putText(img, f"distance: {verification_results['distance']:1.3f}", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    img = cv2.putText(img, f"threshold: {verification_results['threshold']:1.3f}", (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    img = cv2.putText(img, f"model: {verification_results['model']}", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    img = cv2.putText(img, f"detector_backend: {verification_results['detector_backend']}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    img = cv2.putText(img, f"similarity_metric: {verification_results['similarity_metric']}", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    img = cv2.putText(img, f"face_pixel_height: {face_pixel_height}", (50,210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    return img

# проверка существования директория, если ее нет, создаем
isExist = os.path.exists(frames_directory+"/verify_results")
if not isExist:
        os.makedirs(frames_directory + "/verify_results", exist_ok=True)

# извлекаем лица из эталонных изображений
face_objs_db = DeepFace.extract_faces(
    img_path=db_image, detector_backend="retinaface",
    target_size= (224,224)
)
face_db = face_objs_db[0]["face"]

# создаем df для хранения результатов
results_df = pd.DataFrame(columns=["email", "file_name", "verified", "distance", "model", "detector", "metric", "height", "reference_image"])

def save_results_to_csv(results_df, output_csv_file):
    results_df.to_csv(output_csv_file, index=False)

# сравниваем лица в кадре с эталонными и сохранаям результат проверки
for frame in frames_list:
    frame_path = frames_directory + r'\\' + frame
    try:
        result = DeepFace.verify(img1_path=frame_path,
                                 img2_path=db_image,
                                 enforce_detection=True,
                                 detector_backend="retinaface",
                                 model_name="VGG-Face",
                                 distance_metric="euclidean_l2"
                                 )
        # извлекаем лица из кадра
        face_objs = DeepFace.extract_faces(img_path=frame_path,
                                           detector_backend="retinaface",
                                           target_size=(224,224)
                                           )

        print(result)
        for face_obj in face_objs:
            k+=1
            face = face_obj["face"]
            concatendated_faces = np.concatenate((face,face_db), axis = 1)*255
            resized_concatenated_faces = cv2.resize(concatendated_faces, (224*6,112*6),interpolation = cv2.INTER_AREA)
            output_image = visualise_deepface_verify_obj(resized_concatenated_faces, result)

            cv2.imwrite(f'{frames_directory}/verify_results/{frame}_{result["distance"]}_{result["verified"]}.jpg',
                        cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

            print("-----------")

            if result['verified']:
                true_count+=1
            else:
                false_count+=1

            results_df = pd.concat([results_df, pd.DataFrame(
                {"email": [email], "file_name": [frame], "verified": [result['verified']], "distance": [round(result['distance'], 5)],
                 "model": [result['model']], "detector": [result['detector_backend']],
                 "metric": [result['similarity_metric']], "height": [result['facial_areas']], "reference_image": [ref_image]})], ignore_index=True)

    except ValueError as e:
        # если в кадре не найдены лица
        print(f"No face found for {frame}, error message: {str(e)}")
        continue

results_csv_filename = os.path.join(frames_directory, "verification_results.csv")
save_results_to_csv(results_df, results_csv_filename)

print(f"True positive:{true_count}, false negative:{false_count}")
