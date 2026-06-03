import json
import os

splits = ['train', 'valid', 'test']
base_dir = "./"

for split in splits:
    coco_path = os.path.join(base_dir, split, "_annotations.coco.json")
    
    if not os.path.exists(coco_path):
        print(f"⚠️ {split} 폴더에 '_annotations.coco.json' 파일이 없습니다. 패스합니다.")
        continue
        
    with open(coco_path, 'r') as f:
        data = json.load(f)

    old_categories = data['categories']
    fixed_categories = []
    id_mapping = {}
    new_id = 0
    for cat in old_categories:
        if cat['name'] == 'PM-Parking' or cat['id'] == 0:
            print(f"[{split}] 유령 클래스 제거 완료: {cat['name']}")
            continue
            
        old_id = cat['id']
        id_mapping[old_id] = new_id
        
        new_cat = cat.copy()
        new_cat['id'] = new_id
        fixed_categories.append(new_cat)
        new_id += 1

    data['categories'] = fixed_categories

    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in id_mapping:
            ann['category_id'] = id_mapping[old_cat_id]

    with open(coco_path, 'w') as f:
        json.dump(data, f, indent=2)

print("\n✨ 데이터 수정 완료!")