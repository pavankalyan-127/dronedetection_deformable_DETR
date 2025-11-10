import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import tempfile
import os
from datetime import datetime

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Drone Object Detection", layout="wide")
st.title("🚁 Drone Object Detection (Deformable DETR)")
st.write("Detect objects in live drone videos using the pre-trained Transformer-based Deformable DETR model.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)
    return processor, model, device

processor, model, device = load_model()

# -------------------- SIDEBAR OPTIONS --------------------
st.sidebar.header("⚙️ Options")
input_type = st.sidebar.radio("Select Input Type", ["🎞️ Video Upload", "📷 Live Camera"])
save_output = st.sidebar.radio("💾 Do you want to save detected video?", ["Yes", "No"])
capture_images = st.sidebar.checkbox("📸 Capture Photos During Detection", value=True)

# Create output directory
output_dir = "drone_outputs"
os.makedirs(output_dir, exist_ok=True)

# -------------------- VIDEO UPLOAD MODE --------------------
if input_type == "🎞️ Video Upload":
    video_file = st.file_uploader("Upload Drone Video", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # Setup output saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(output_dir, f"detected_{timestamp}.avi")
        output_image_dir = os.path.join(output_dir, f"captures_{timestamp}")
        os.makedirs(output_image_dir, exist_ok=True)

        if save_output == "Yes":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0,
                                  (int(cap.get(3)), int(cap.get(4))))
            st.info("💾 Saving detected video...")

        frame_count = 0
        st.info("🚀 Running object detection... Please wait...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.6)[0]

            draw = ImageDraw.Draw(image_pil)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} {round(score.item(), 2)}", fill="white")

            annotated = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            stframe.image(annotated, channels="BGR", use_container_width=True)

            # Save video frame
            if save_output == "Yes":
                out.write(annotated)

            # Capture frame image every 50 frames
            if capture_images and frame_count % 50 == 0:
                capture_path = os.path.join(output_image_dir, f"capture_{frame_count}.jpg")
                cv2.imwrite(capture_path, annotated)

        cap.release()
        if save_output == "Yes":
            out.release()
            st.success(f"✅ Detected video saved at: `{output_video_path}`")

        if capture_images:
            st.info(f"📸 Captured frames saved in: `{output_image_dir}`")

        st.success("🎉 Detection complete!")

# -------------------- LIVE CAMERA MODE --------------------
elif input_type == "📷 Live Camera":
    st.warning("⚠️ Live camera access is not supported in Streamlit Cloud. Please run locally to use webcam.")
    st.write("To run locally:")
    st.code("streamlit run app.py", language="bash")
