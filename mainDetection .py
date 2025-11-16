# Deformable DETR Object Detection (Image + Video)
# Clean Python Script for VS Code / Local Machine

# ---------------------
# IMPORTS
# ---------------------
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import warnings
warnings.filterwarnings("ignore")

# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔧 Device:", device)

# ---------------------
# LOAD MODEL
# ---------------------
print("📦 Loading Deformable DETR model...")

processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = DeformableDetrForObjectDetection.from_pretrained(
    "SenseTime/deformable-detr"
).to(device)
model.eval()

print("✅ Model loaded successfully!\n")


# DRAW BOUNDING BOXES

def draw_boxes(image_pil, results):
    draw = ImageDraw.Draw(image_pil)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

        score = float(score)
        if score < 0.5:
            continue

        box = box.tolist()
        cls_name = model.config.id2label[int(label)]

        draw.rectangle(box, outline="red", width=3)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{cls_name} {score:.2f}",
            fill="yellow"
        )

    return image_pil


# IMAGE DETECTION

def detect_image(image_path, save_output=True):

    print(f"🖼 Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]

    # Move to CPU
    results = {
        "scores": results["scores"].cpu(),
        "labels": results["labels"].cpu(),
        "boxes": results["boxes"].cpu(),
    }

    img_out = draw_boxes(image.copy(), results)

    if save_output:
        out_name = "detected_output.jpg"
        img_out.save(out_name)
        print(f" Saved detection -> {out_name}")

    return img_out


# VIDEO DETECTION
def process_video(input_path, output_path="detected_output.mp4"):

    print(" Starting video detection...")

    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb)

        inputs = processor(images=image_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]], device=device)

        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]

        # Move to CPU
        results = {
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
            "boxes": results["boxes"].cpu(),
        }

        # Draw boxes
        out_img = draw_boxes(image_pil.copy(), results)

        # Convert back to BGR for video writing
        bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
        out.write(bgr)

    cap.release()
    out.release()
    print(f"🎉 Video saved -> {output_path}")
    return output_path


# MAIN MENU
if __name__ == "__main__":

    print("==========================================")
    print("  Deformable DETR Object Detection Script ")
    print("==========================================")
    print("1 → Detect Image")
    print("2 → Detect Video")
    print("==========================================")

    choice = input("Select option (1/2): ").strip()

    if choice == "1":
        img = input("Enter image path: ").strip()
        detect_image(img)

    elif choice == "2":
        vid = input("Enter video path: ").strip()
        process_video(vid)

    else:
        print("❌ Invalid choice. Exiting.")
