import os
import glob
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

if __name__ == "__main__":
    CHECKPOINT_PATH = "logs/checkpoint_best_regular.pth" 
    INPUT_DIR = "../../data/labeled/parking/coco/test"
    OUTPUT_DIR = "logs/test_results"
    
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    CAPTION = "crosswalk . ebike_no . ebike_ok . human . kickboard_no . kickboard_ok . parking_zone . tactile_paving"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔄 Grounding DINO 로드 중... ({device})")
    
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    image_paths = []
    for ext in ["*.jpg"]:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not image_paths:
        print(f"⚠️ 에러: '{INPUT_DIR}' 경로에 이미지가 없습니다.")
        exit(1)

    print(f"🚀 총 {len(image_paths)}장 허깅페이스 파이프라인 추론 시작...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)

        image_cv = cv2.imread(img_path)
        if image_cv is None:
            continue
        
        h, w, _ = image_cv.shape
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image_pil, text=CAPTION, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)]
        )[0]

        if len(results["boxes"]) == 0:
            cv2.imwrite(save_path, image_cv)
            continue

        annotated_image = image_cv.copy()
        boxes = results["boxes"].cpu().numpy().astype(int)
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            caption_text = f"{label} {score:.2f}"
            cv2.putText(annotated_image, caption_text, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(save_path, annotated_image)

    print(f"🎉결과: '{OUTPUT_DIR}/'")