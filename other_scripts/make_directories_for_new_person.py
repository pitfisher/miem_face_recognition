import os

dataset_path = r"C:\Users\HYPERPC\Documents\miem-project-1712\dataset_clean"
new_person_folder_path = r"C:\Users\HYPERPC\Downloads\Датасет лиц\Датасет лиц\gyuvasilev"
os.startfile(new_person_folder_path)
new_person_mail = new_person_folder_path.split("\\")[-1]
people_hse_mails = [new_person_mail]
vertical_folders = ["up", "center", "down"]
horizontal_folders = ["45_left", "center", "45_right"]
for vertical_folder_name in vertical_folders:
    vertical_folder_path = os.path.join(dataset_path, vertical_folder_name)
    os.makedirs(vertical_folder_path, exist_ok=True)

    for horizontal_folder_name in horizontal_folders:
        horizontal_folder_path = os.path.join(vertical_folder_path, horizontal_folder_name)
        os.makedirs(horizontal_folder_path, exist_ok=True)

        for person_hse_mail in people_hse_mails:
            new_person_folder_path = os.path.join(horizontal_folder_path, person_hse_mail)
            os.makedirs(new_person_folder_path, exist_ok=True)
            os.startfile(new_person_folder_path)