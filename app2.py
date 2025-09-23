import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import cv2
import streamlit as st

# YOLOv11 via ultralytics
from ultralytics import YOLO

# OCR engine
import easyocr

# Optional: PDF
try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except Exception:
    HAS_PDFIUM = False

st.set_page_config(page_title="Simple Resume Parser",
                   page_icon="📄",
                   layout="wide")

# Kelas yang tidak perlu di-OCR
SKIP_CLASSES = {
    "background", "image", "photo", "profile_photo", "avatar", "profil-foto"
}

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    model = YOLO(weights_path)
    return model

@st.cache_resource(show_spinner=False)
def load_ocr_reader(langs: Tuple[str, ...] = ("en","id")):
    return easyocr.Reader(list(langs), gpu=False, verbose=False)

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def render_pdf_page(file_bytes: bytes, page_index: int = 0, dpi: int = 300) -> Image.Image:
    if not HAS_PDFIUM:
        raise RuntimeError("PDF support is disabled.")
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
    page = pdf.get_page(page_index)
    # 72 dpi adalah baseline PDF; scale = dpi/72
    scale = dpi / 72
    return page.render(scale=scale).to_pil()

def enhance_image_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Normalisasi kontras lembut
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced  # tanpa adaptive threshold


def extract_best_text(reader: easyocr.Reader, crop_bgr: np.ndarray) -> str:
    """Extract the best OCR result from image crop"""
    try:
        # Upscale image for better OCR
        upscaled = cv2.resize(crop_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        # Enhance image
        enhanced = enhance_image_for_ocr(upscaled)
        
        # Get OCR results
        results = reader.readtext(enhanced, detail=0, paragraph=True)
        
        # Join all text results
        text = "\n".join(results)
        
        # Basic text cleaning
        text = text.replace("—", "-")
        text = text.replace(" rn", " m")
        
        # Clean up spaces
        text = "\n".join(" ".join(line.split()) for line in text.splitlines())
        
        return text.strip()
        
    except Exception as e:
        return f"[OCR Error: {str(e)}]"

def run_detection(model: "YOLO", image_pil: Image.Image, conf: float = 0.45):
    """Run YOLO detection on image"""
    image_bgr = pil_to_cv2(image_pil)
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    
    r0 = results[0]
    if r0.boxes is None:
        return []
    
    # Get class names
    names = getattr(model, "names", {})
    if isinstance(names, list):
        class_map = {i: names[i] for i in range(len(names))}
    else:
        class_map = dict(names) if names else {}
    
    # Extract detection data
    boxes_xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
    scores = r0.boxes.conf.cpu().numpy()
    cls_idx = r0.boxes.cls.cpu().numpy().astype(int)
    
    detections = []
    for i, (box, score, cls) in enumerate(zip(boxes_xyxy, scores, cls_idx)):
        class_name = class_map.get(cls, f"class_{cls}")
        x1, y1, x2, y2 = box
        
        detections.append({
            "class": class_name,
            "confidence": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "crop": image_bgr[y1:y2, x1:x2].copy()
        })
    
    return detections

def parse_resume(image_pil: Image.Image, model_path: str = "best.pt") -> Dict[str, str]:
    """
    Main function: Parse resume and return best OCR result for each class
    
    Args:
        image_pil: PIL Image of the resume
        model_path: Path to YOLO model weights
    
    Returns:
        Dict with class names as keys and best OCR text as values
    """
    
    # Load models
    model = load_model(model_path)
    reader = load_ocr_reader(("en", "id"))
    
    # Run detection
    detections = run_detection(model, image_pil)
    
    # Extract text for each detection
    results = {}
    
    for det in detections:
        class_name = det["class"]
        
        # Skip certain classes
        if class_name.lower() in SKIP_CLASSES:
            continue
        
        # Extract best OCR text
        best_text = extract_best_text(reader, det["crop"])
        
        # If class already exists, keep the one with higher confidence
        if class_name in results:
            if det["confidence"] > results[class_name]["confidence"]:
                results[class_name] = {
                    "text": best_text,
                    "confidence": det["confidence"]
                }
        else:
            results[class_name] = {
                "text": best_text,
                "confidence": det["confidence"]
            }
    
    # Return only the text for each class (simplified output)
    return {class_name: data["text"] for class_name, data in results.items()}

def main():
    st.title("📄 Simple Resume Parser")
    st.caption("Upload gambar resume dan dapatkan hasil OCR terbaik untuk setiap kelas")
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model Path", "best.pt")
        confidence = st.slider("Confidence Threshold", 0.1, 0.95, 0.45, 0.05)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Resume (JPG/PNG/PDF)", 
        type=["jpg", "jpeg", "png", "pdf"]
    )
    
    if uploaded_file is None:
        st.info("Please upload a resume file to continue.")
        return
    
    # Process uploaded file
    if uploaded_file.name.lower().endswith(".pdf"):
        if not HAS_PDFIUM:
            st.error("PDF support not available. Please upload as image.")
            return
        image_pil = render_pdf_page(uploaded_file.read(), page_index=0)
    else:
        image_pil = Image.open(uploaded_file).convert("RGB")
    
    # Display original image
    st.subheader("Original Resume")
    st.image(image_pil, use_container_width=True)
    
    # Process button
    if st.button("🚀 Parse Resume"):
        with st.spinner("Processing resume..."):
            try:
                # Parse resume
                results = parse_resume(image_pil, model_path)
                
                # Display results
                st.subheader("Parsed Results")
                
                if not results:
                    st.warning("No text detected in the resume.")
                else:
                    for class_name, text in results.items():
                        with st.expander(f"📋 {class_name.title()}"):
                            if text:
                                st.text_area(
                                    f"{class_name} content:",
                                    value=text,
                                    height=100,
                                    key=f"text_{class_name}"
                                )
                            else:
                                st.info("No text extracted for this section.")
                
                # Export results as JSON
                import json
                json_result = json.dumps(results, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Download Results (JSON)",
                    data=json_result,
                    file_name=f"resume_parsed_{uploaded_file.name}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")

if __name__ == "__main__":
    main()