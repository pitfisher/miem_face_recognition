import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

model_names = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    # "DeepID",
    # "Dlib",
    "ArcFace",
    "SFace",
]
detector_backends = ["retinaface"]

# font = ImageFont.truetype(r"C:\Users\Student\Documents\face_recognition\VCR OSD Mono Cyr.ttf", 18)
# img_pil = Image.fromarray(img)
# draw = ImageDraw.Draw(img_pil)
# draw.text((50, 100),  "国庆节/中秋节 快乐!", font = font, fill = (b, g, r, a))
# img = np.array(img_pil)

i = 0
k = 0
# frames_directory = r"C:\Users\Student\Pictures\vlc_snapshots\slastnikov_sa_test"
frames_directory = r"C:\Users\Student\Documents\face_recognition\wide_cam_308_face_test_8mp_ameliseenko_glasses_frames"
db_image = r"C:\Users\Student\Documents\face_recognition\reference_faces\ameliseenko\ameliseenko_glasses.JPG"
frames_list = os.listdir(frames_directory)
false_count = 0
true_count = 0

def visualise_deepface_verify_obj(image, verification_results = {}):
    #{'verified': True, 
    # 'distance': 0.6994153135006677, 
    # 'threshold': 0.86, 
    # 'model': 'VGG-Face', 
    # 'detector_backend': 'retinaface', 
    # 'similarity_metric': 'euclidean_l2', 
    # 'facial_areas': {'img1': {'x': 0, 'y': 624, 'w': 65, 'h': 142}, 'img2': {'x': 142, 'y': 82, 'w': 204, 'h': 265}}, 'time': 3.12}
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

isExist = os.path.exists(frames_directory+"/verify_results")
if not isExist:
    os.mkdir(frames_directory+"/verify_results")
    
face_objs_db = DeepFace.extract_faces(
    img_path=db_image, detector_backend="retinaface",
    target_size= (224,224)
)
face_db = face_objs_db[0]["face"]
        
for frame in frames_list:
    frame_path = frames_directory + r'\\' + frame
    try:
        # print(DeepFace.verify(img1_path = frame_path, img2_path = db_image, enforce_detection=False, detector_backend="retinaface", model_name="SFace"))
        result = DeepFace.verify(img1_path = frame_path, img2_path = db_image, enforce_detection=True, detector_backend="retinaface", model_name="VGG-Face", distance_metric = "euclidean_l2")
        face_objs = DeepFace.extract_faces(
            img_path=frame_path, detector_backend="retinaface",
            target_size = (224,224)
        )
        
        print(result)
        for face_obj in face_objs:
            k+=1
            face = face_obj["face"]
            concatendated_faces = np.concatenate((face,face_db), axis = 1)*255
            resized_concatenated_faces = cv2.resize(concatendated_faces, (224*6,112*6), interpolation = cv2.INTER_AREA)
            output_image = visualise_deepface_verify_obj(resized_concatenated_faces, result)
            # plt.imshow(face)
            # plt.imshow(np.concatenate((face,face_db), axis = 1))
            # plt.axis("off")
            # plt.show()
            cv2.imwrite(f'{frames_directory}/verify_results/{frame}_{result["distance"]}_{result["verified"]}.jpg', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            
            print("-----------")
        if result['verified']:
            true_count+=1
        else:
            false_count+=1
    except ValueError as e:
        print(f"No face found for {frame}, error message: {str(e)}")
        continue
print(f"True positive:{true_count}, false negative:{false_count}")
# # extract faces
# for detector_backend in detector_backends:
#     i+=1
#     try:
#         face_objs = DeepFace.extract_faces(
#             # img_path="C:/Users/Student/Pictures/vlc_snapshots/slastnikov_sa/vlcsnap-2023-07-12-20h05m22s174.png", detector_backend=detector_backend
#             # img_path="C:/Users/Student/Pictures/vlc_snapshots/slastnikov_sa/vlcsnap-2023-07-12-20h06m15s846.png", detector_backend=detector_backend,
#             img_path=r"C:\Users\Student\Documents\face_recognition\hse_faces_miem\301_miem.jpeg", detector_backend=detector_backend,
#             target_size= (112,112)
            
#         )
#     except ValueError as e:
#         print(f"No face found for {detector_backend}, error message: {str(e)}")
#         continue
#     for face_obj in face_objs:
#         k+=1
#         face = face_obj["face"]
#         print(detector_backend)
#         plt.imshow(face)
#         plt.axis("off")
#         plt.show()
#         cv2.imwrite(f'dataset/test_results/{i}_{k}.jpg', cv2.cvtColor(face*255, cv2.COLOR_BGR2RGB))
#         print("-----------")
