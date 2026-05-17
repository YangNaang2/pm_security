import os
import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

target_dirs = [
    '../../data/raw/driving/abnormal/helmet_no/',
    '../../data/raw/driving/abnormal/many_drivers/'
]
for target_dir in target_dirs:
    print(f"==== 🔍 검사 시작: {target_dir} ====")
    
    if not os.path.exists(target_dir):
        print(f"경로를 찾을 수 없습니다: {target_dir}\n")
        continue

    hash_dict = {}
    duplicates = []

    for file_name in os.listdir(target_dir):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
            file_path = os.path.join(target_dir, file_name)
            file_hash = get_file_hash(file_path)
            
            if file_hash in hash_dict:
                print(f"🚨 중복 발견! \n  - 원본: {hash_dict[file_hash]}\n  - 중복: {file_name}")
                duplicates.append(file_path)
            else:
                hash_dict[file_hash] = file_name

    print(f"✅ 검사 완료! 총 {len(duplicates)}개의 중복 파일을 찾았습니다.\n")

# 💡 중복 파일 자동 삭제
# for dup_path in duplicates:
#     os.remove(dup_path)
# print("🗑️ 중복 파일 삭제 완료!")