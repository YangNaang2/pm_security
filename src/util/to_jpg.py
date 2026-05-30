import os
from PIL import Image

def convert_dataset_to_jpg(target_dir):
    target_extensions = ('.png', '.webp', '.avif', '.jpeg', '.bmp', '.tiff')
    
    for file_name in os.listdir(target_dir):
        base_name, ext = os.path.splitext(file_name)
        ext = ext.lower()
        
        if ext in target_extensions:
            old_path = os.path.join(target_dir, file_name)
            new_path = os.path.join(target_dir, base_name + '.jpg')
            
            if ext == '.jpeg':
                os.rename(old_path, new_path)
                continue
            
            try:
                with Image.open(old_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(new_path, 'JPEG', quality=95)
                os.remove(old_path)
            except Exception as e:
                print(f"Error {file_name}: {e}")

if __name__ == "__main__":
    IMAGE_DIRECTORIES = [
        '../../data/raw/driving/abnormal/helmet_no/',
        '../../data/raw/driving/abnormal/many_drivers/',
        '../../data/driving_algorithm_test/abnormal/',
        '../../data/driving_algorithm_test/normal/'
       
    ]
    for directory in IMAGE_DIRECTORIES:
        if os.path.exists(directory):
            print(f"==== 변환 시작: {directory} ====")
            convert_dataset_to_jpg(directory)
            print(f"==== 변환 완료: {directory} ====\n")
        else:
            print(f"경로를 찾을 수 없습니다: {directory}\n")
