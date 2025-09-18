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

st.set_page_config(page_title="TemanHire CV Reader (YOLOv11 + OCR)",
                   page_icon="🧠",
                   layout="wide")

DEFAULT_CLASS_NAMES = {
    0: "part1",
    1: "part2",
    2: "part3",      # photo (skip OCR)
    3: "part4",
    4: "part5",
    5: "part6",
    6: "part7",
    7: "background"
}
PART_ORDER = {f"part{i}": i for i in range(1, 8)}

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    model = YOLO(weights_path)
    return model

@st.cache_resource(show_spinner=False)
def load_ocr_reader(langs: Tuple[str, ...] = ("en","id")):
    return easyocr.Reader(list(langs), gpu=False, verbose=False)

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def render_pdf_page(file_bytes: bytes, page_index: int = 0) -> Image.Image:
    if not HAS_PDFIUM:
        raise RuntimeError("PDF support is disabled (pypdfium2 not installed).")
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
    page = pdf.get_page(page_index)
    return page.render(scale=2).to_pil()

def annotate(image_bgr: np.ndarray,
             boxes: List[Tuple[int, int, int, int]],
             labels: List[str],
             scores: List[float]) -> np.ndarray:
    canvas = image_bgr.copy()
    for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
        color = (36, 142, 255) if label.startswith("part") else (180, 180, 180)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return canvas

# --- OCR Enhancers ---
def enhance_for_ocr(img_bgr: np.ndarray, upscale: float = 1.7, denoise: bool = True, clahe: bool = True):
    img = img_bgr.copy()
    # Upscale
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe_op.apply(gray)
    # Adaptive threshold + Otsu candidate (will be mixed later)
    ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return gray, ada, otsu

COMMON_FIXES = [
    (" fndustr", " industri"),
    (" Taknik", " Teknik"),
    (" morniliki", " memiliki"),
    (" skilt", " skill"),
    (" dlslplin", " disiplin"),
    (" plonning", " planning"),
    (" terr", " ter"),
    (" socara", " secara"),
    (" dengen", " dengan"),
    (" Univorsitas", " Universitas"),
    (" Trunojoyo", " Trunojoyo"),
    (" Sakretaris", " Sekretaris"),
    (" industrl", " industri"),
    (" Tabun", " Tahun"),
    (" neraca sakdo", " neraca saldo"),
    (" tridl bolance", " trial balance"),
    (" rugl", " rugi"),
    (" Intetn", " Intern"),
    (" Noverber", " November"),
    (" Januzri", " Januari"),
]

def post_correction(text: str) -> str:
    # Basic character confusions
    repl = (("0", "O"), (" rn", " m"), (" l", " I"), (" y", " y"), ("—","-"))
    out = text
    for a,b in repl:
        out = out.replace(a,b)
    for a,b in COMMON_FIXES:
        out = out.replace(a,b)
    # tidy spaces
    out = "\n".join(" ".join(line.split()) for line in out.splitlines())
    return out

def ocr_with_variants(reader: easyocr.Reader, crop_bgr: np.ndarray, high_accuracy: bool = False) -> str:
    gray, ada, otsu = enhance_for_ocr(crop_bgr, upscale=1.7 if high_accuracy else 1.3)
    # Try 2–3 variants and merge
    variants = [ada, otsu, gray]
    texts = []
    for v in variants:
        try:
            t = reader.readtext(v, detail=0, paragraph=True)
            texts.append("\n".join(t))
        except Exception:
            pass
    merged = "\n".join([t for t in texts if t]).strip()
    return post_correction(merged)

def run_inference(model: "YOLO", image_pil: Image.Image, conf: float, iou: float):
    image_bgr = pil_to_cv2(image_pil)
    results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    r0 = results[0]
    names = getattr(model, "names", None) or {}
    if isinstance(names, list):
        class_map = {i: names[i] for i in range(len(names))}
    else:
        class_map = dict(names) if names else DEFAULT_CLASS_NAMES
    boxes_xyxy = r0.boxes.xyxy.cpu().numpy().astype(int) if r0.boxes is not None else np.zeros((0,4), dtype=int)
    scores = r0.boxes.conf.cpu().numpy().tolist() if r0.boxes is not None else []
    cls_idx = r0.boxes.cls.cpu().numpy().astype(int).tolist() if r0.boxes is not None else []
    labels = [class_map.get(i, f"class_{i}") for i in cls_idx]
    return image_bgr, boxes_xyxy.tolist(), labels, scores

def main():
    st.title("🧠 TemanHire — CV/Resume Reader (YOLOv11 + OCR)")
    st.caption("Deteksi layout CV dengan YOLOv11 dan ekstrak teks per-bagian menggunakan EasyOCR.")

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        weights_src = st.radio("Sumber model (.pt)", ["Default (best.pt)", "Upload"], index=0)
        if weights_src == "Default (best.pt)":
            weights_path = st.text_input("Path weights", "best.pt")
            weights_bytes = None
        else:
            weights_file = st.file_uploader("Upload YOLO weights (.pt)", type=["pt"])
            weights_path = None
            weights_bytes = weights_file.read() if weights_file else None
        conf = st.slider("Confidence threshold", 0.1, 0.95, 0.45, 0.05)
        iou = st.slider("IoU NMS threshold", 0.1, 0.95, 0.65, 0.05)
        langs = st.multiselect("Bahasa OCR", options=["en","id"], default=["en","id"])
        high_acc = st.checkbox("🔍 High-accuracy OCR (lebih lambat)", value=True)
        run_btn = st.button("🚀 Jalankan")

    col1, col2 = st.columns([1,1])
    uploaded = st.file_uploader("Unggah gambar CV (JPG/PNG) atau PDF", type=["jpg","jpeg","png","pdf"])
    if uploaded is None:
        st.info("Unggah file terlebih dahulu untuk memulai.")
        return

    # Prepare image
    if uploaded.name.lower().endswith(".pdf"):
        if not HAS_PDFIUM:
            st.error("Dukungan PDF tidak aktif. Tambahkan 'pypdfium2' di requirements.txt atau unggah sebagai gambar.")
            return
        image_pil = render_pdf_page(uploaded.read(), page_index=0)
    else:
        image_pil = Image.open(uploaded).convert("RGB")

    with col1:
        st.subheader("➊ Pratinjau")
        st.image(image_pil, use_container_width=True)

    # Load model & OCR
    try:
        if weights_src == "Default (best.pt)":
            model = load_model(weights_path)
        else:
            if not weights_bytes:
                st.warning("Silakan upload file weights .pt terlebih dahulu.")
                return
            tmp_path = Path(st.secrets.get("YOLO_TMP", "tmp_weights.pt"))
            with open(tmp_path, "wb") as f:
                f.write(weights_bytes)
            model = load_model(str(tmp_path))
    except Exception as e:
        st.exception(e)
        return

    reader = load_ocr_reader(tuple(langs) if langs else ("en","id"))

    if not run_btn:
        st.stop()

    with st.spinner("Menjalankan deteksi..."):
        image_bgr, boxes, labels, scores = run_inference(model, image_pil, conf, iou)

    annotated = annotate(image_bgr, boxes, labels, scores)

    results: List[Dict] = []
    for (x1,y1,x2,y2), label, score in zip(boxes, labels, scores):
        if label.lower() in {"background"}:
            continue
        if label.lower() == "part3":
            results.append({"part": label, "confidence": float(score),
                            "text": "", "note": "Photo region (OCR skipped)",
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]})
            continue
        # Crop
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(image_bgr.shape[1]-1, x2), min(image_bgr.shape[0]-1, y2)
        crop = image_bgr[y1c:y2c, x1c:x2c].copy()
        try:
            text = ocr_with_variants(reader, crop, high_accuracy=high_acc)
        except Exception as e:
            text = f"[OCR error] {e}"
        results.append({
            "part": label, "confidence": float(score), "text": text,
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

    # Sort by PART_ORDER then y
    def sort_key(item):
        p = item["part"].lower()
        order = PART_ORDER.get(p, 999)
        y_top = item["bbox"][1]
        return (order, y_top)
    results.sort(key=sort_key)

    with col2:
        st.subheader("➋ Hasil Deteksi")
        st.image(cv2_to_pil(annotated), use_container_width=True)

    st.subheader("➌ Teks per-bagian")
    for r in results:
        with st.expander(f"{r['part']} • conf={r['confidence']:.2f}"):
            if r.get("note"):
                st.info(r["note"])
            st.code(r["text"] or "(kosong)")

    # Export
    import json, csv, io as pyio
    export = {"file_name": uploaded.name, "parts": results}
    st.download_button("⬇️ Unduh JSON", json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="cv_ocr.json", mime="application/json")
    buf = pyio.StringIO()
    writer = csv.writer(buf, delimiter="\t")
    writer.writerow(["part","confidence","x1","y1","x2","y2","text"])
    for r in results:
        x1,y1,x2,y2 = r["bbox"]
        writer.writerow([r["part"], f"{r['confidence']:.3f}", x1,y1,x2,y2, r["text"].replace("\n","\\n")])
    st.download_button("⬇️ Unduh TSV", buf.getvalue().encode("utf-8"),
                       file_name="cv_ocr.tsv", mime="text/tab-separated-values")

    st.caption("Tip akurasi: aktifkan 'High-accuracy OCR', atur conf 0.45–0.6, IoU 0.6–0.7, dan pastikan gambar CV berkualitas ≥150 DPI.")
if __name__ == "__main__":
    main()