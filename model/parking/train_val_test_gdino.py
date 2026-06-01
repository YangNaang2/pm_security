import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image

BASE_DIR = "../../data/labeled/parking/coco"
MODEL_ID = "IDEA-Research/grounding-dino-base"
BATCH_SIZE = 1
ACCUMULATION_STEPS = 4
EPOCHS = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

with open(os.path.join(BASE_DIR, "train/_annotations.coco.json"), "r") as f:
    train_coco = json.load(f)

id_to_name_map = {cat["id"]: cat["name"].lower().strip() for cat in train_coco["categories"]}

category_names = [cat["name"].lower().strip() for cat in train_coco["categories"]]
TEXT_PROMPT = " . ".join(category_names) + " ."
print(f"Generated TEXT_PROMPT: '{TEXT_PROMPT}'")

processor = AutoProcessor.from_pretrained(MODEL_ID)

class CustomCocoDataset(Dataset):
    def __init__(self, json_path, img_dir):
        with open(json_path, "r") as f:
            self.coco = json.load(f)
        self.img_dir = img_dir
        
        self.images = {img["id"]: img for img in self.coco["images"]}
        
        self.img_to_anns = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        anns = self.img_to_anns.get(img_id, [])
        
        return {"img_info": img_info, "annotations": anns, "img_dir": self.img_dir}

train_dataset = CustomCocoDataset(os.path.join(BASE_DIR, "train/_annotations.coco.json"), os.path.join(BASE_DIR, "train"))
valid_dataset = CustomCocoDataset(os.path.join(BASE_DIR, "valid/_annotations.coco.json"), os.path.join(BASE_DIR, "valid"))
test_dataset = CustomCocoDataset(os.path.join(BASE_DIR, "test/_annotations.coco.json"), os.path.join(BASE_DIR, "test"))

def collate_fn(batch):
    images = []
    targets = []
    
    for item in batch:
        img_info = item["img_info"]
        anns = item["annotations"]
        img_dir = item["img_dir"]
        
        img_path = os.path.join(img_dir, img_info["file_name"])
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
        except FileNotFoundError:
            continue
        if len(anns) == 0:
            continue

        normalized_boxes = []
        class_labels = []
        
        for ann in anns:
            box = ann["bbox"]
            cat_id = ann["category_id"]
            
            cx = min(max((box[0] + box[2] / 2) / w, 0.0), 1.0)
            cy = min(max((box[1] + box[3] / 2) / h, 0.0), 1.0)
            nw = min(max(box[2] / w, 0.0), 1.0)
            nh = min(max(box[3] / h, 0.0), 1.0)
            
            normalized_boxes.append([cx, cy, nw, nh])
            class_labels.append(category_names.index(id_to_name_map[cat_id]))  # phrase index
            
        images.append(img)
        targets.append({
            "boxes": torch.tensor(normalized_boxes, dtype=torch.float32),
            "class_labels": torch.tensor(class_labels, dtype=torch.long)
        })

    if len(images) == 0:
        return None

    inputs = processor(
        images=images,
        text=[TEXT_PROMPT] * len(images),
        return_tensors="pt",
        padding=True
    )
    inputs["labels"] = targets
    return inputs

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="⚡ Training")):
        if batch is None: continue
        
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "labels" in batch:
            inputs["labels"] = [{tk: tv.to(DEVICE) for tk, tv in t.items()} for t in batch["labels"]]
        
        outputs = model(**inputs)
        loss = outputs.loss / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * ACCUMULATION_STEPS
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, phase="Validation"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"🔍 {phase}"):
            if batch is None: continue
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE)
                elif k == "labels":
                    inputs[k] = [{tk: tv.to(DEVICE) for tk, tv in t.items()} for t in v]
            
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            if loss is not None:
                total_loss += loss.item()
                
    print(f"[{phase}] Completed. Average Loss: {total_loss / len(dataloader):.4f}")
    
if __name__ == "__main__":
    print("--- Start Grounding DINO Pipeline ---")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Train Loss: {train_loss:.4f}")
        
        evaluate(model, valid_loader, phase="Validation")
        
    print("\n--- Final Test Evaluation ---")
    evaluate(model, test_loader, phase="Test")
    
    save_path = "./gdino"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Model saved to {save_path}")