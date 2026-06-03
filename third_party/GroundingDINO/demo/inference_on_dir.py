import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from tqdm import tqdm
from torchvision.ops import nms


CLASS_COLORS = {
    "kickboard_no":   (220, 50,  50),   # red
    "ebike_no":       (220, 50,  50),   # red
    "kickboard_ok":   (50,  180, 80),   # green
    "ebike_ok":       (50,  180, 80),   # green
    "crosswalk":      (220, 180, 0),    # yellow
    "tactile_paving": (220, 180, 0),    # yellow
    "parking_zone":   (30,  150, 200),  # blue
    "human":          (120, 120, 120),  # gray
}
DEFAULT_COLOR = (200, 200, 200)
FONT_SIZE = 30

def get_class_name(label: str) -> str:
    return label.split("(")[0].strip()


def get_color(label: str):
    cls = get_class_name(label)
    cls_normalized = cls.replace(" ", "_")
    for key, color in CLASS_COLORS.items():
        if key in cls_normalized or cls_normalized in key:
            return color
    return DEFAULT_COLOR


def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    font = get_font(FONT_SIZE)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        color = get_color(label)
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

        if hasattr(font, "getbbox"):
            text_bbox = font.getbbox(str(label))
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
        else:
            tw, th = draw.textsize(str(label), font=font)

        pad = 3
        label_y0 = max(y0 - th - pad * 2, 0)
        label_y1 = label_y0 + th + pad * 2
        draw.rectangle([x0, label_y0, x0 + tw + pad * 2, label_y1], fill=color)
        draw.text((x0 + pad, label_y0 + pad), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=8)

    return image_pil, mask


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=False, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans should not be None at the same time!"
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device)

        logits_for_phrases = positive_maps @ logits.T
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO Directory Batch Inference", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--text_prompt", "-t", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--token_spans", type=str, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ Error: {args.input_dir}")
        sys.exit(1)

    image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in valid_extensions]
    if not image_paths:
        print(f"❌ Error: no images in {args.input_dir}")
        sys.exit(1)

    print(f"📦 Images: {len(image_paths)}")
    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)

    token_spans = args.token_spans
    text_threshold = None if token_spans is not None else args.text_threshold

    for img_path_obj in tqdm(image_paths, desc="Batch Processing"):
        img_path = str(img_path_obj)
        filename = img_path_obj.name
        stem_name = img_path_obj.stem

        try:
            image_pil, image = load_image(img_path)
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, args.text_prompt, args.box_threshold, text_threshold,
                cpu_only=args.cpu_only,
                token_spans=eval(f"{token_spans}") if token_spans else None
            )

            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],
                "labels": pred_phrases,
            }
            
            if len(boxes_filt) > 0:
                # cxcywh → xyxy 변환
                W, H = size
                boxes_xyxy = boxes_filt.clone()
                boxes_xyxy[:, 0] = (boxes_filt[:, 0] - boxes_filt[:, 2] / 2) * W
                boxes_xyxy[:, 1] = (boxes_filt[:, 1] - boxes_filt[:, 3] / 2) * H
                boxes_xyxy[:, 2] = (boxes_filt[:, 0] + boxes_filt[:, 2] / 2) * W
                boxes_xyxy[:, 3] = (boxes_filt[:, 1] + boxes_filt[:, 3] / 2) * H

                # confidence 추출
                scores = torch.tensor([
                    float(p.split("(")[-1].replace(")", "")) if "(" in p else 1.0
                    for p in pred_phrases
                ])

                keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
                boxes_filt = boxes_filt[keep]
                pred_phrases = [pred_phrases[i] for i in keep]

            image_with_box, _ = plot_boxes_to_image(image_pil, pred_dict)
            output_file_name = f"res_{stem_name}.jpg"
            image_with_box.save(os.path.join(args.output_dir, output_file_name))

        except Exception as e:
            print(f"\n⚠️ Skip -> {filename}: {e}")

    print(f"\n🎉 Done -> '{args.output_dir}'")