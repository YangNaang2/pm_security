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
    IMAGE_DIRECTORY = '../../data/raw/parking/normal/' 
    convert_dataset_to_jpg(IMAGE_DIRECTORY)