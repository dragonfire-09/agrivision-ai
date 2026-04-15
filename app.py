%%writefile app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import tempfile
import time
import io
import json
import cv2
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriVision AI - Video Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main { padding-top: 0; background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0f766e 100%);
        padding: 28px 32px; border-radius: 20px; color: white;
        margin-bottom: 20px; box-shadow: 0 16px 48px rgba(15,23,42,0.12);
        position: relative; overflow: hidden;
    }
    .hero-container::before {
        content: ''; position: absolute; top: -50%; right: -15%;
        width: 350px; height: 350px;
        background: radial-gradient(circle, rgba(34,197,94,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block; background: rgba(34,197,94,0.2);
        border: 1px solid rgba(34,197,94,0.3); color: #4ade80;
        padding: 4px 14px; border-radius: 20px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 6px; position: relative; z-index: 1; }
    .hero-subtitle { font-size: 0.95rem; color: #94a3b8; line-height: 1.6; position: relative; z-index: 1; }

    .metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 20px; }
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 18px 20px; box-shadow: 0 4px 16px rgba(15,23,42,0.04);
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 32px rgba(15,23,42,0.08); }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 3px; border-radius: 18px 18px 0 0;
    }
    .metric-card.weed::before { background: linear-gradient(90deg, #ef4444, #f97316); }
    .metric-card.crop::before { background: linear-gradient(90deg, #22c55e, #10b981); }
    .metric-card.total::before { background: linear-gradient(90deg, #3b82f6, #6366f1); }
    .metric-card.fps::before { background: linear-gradient(90deg, #8b5cf6, #a855f7); }
    .metric-card.density::before { background: linear-gradient(90deg, #f59e0b, #eab308); }

    .metric-icon { font-size: 1.3rem; margin-bottom: 10px; }
    .metric-label {
        font-size: 0.75rem; color: #64748b; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
    }
    .metric-value { font-size: 1.7rem; font-weight: 800; line-height: 1; }
    .metric-value.weed { color: #ef4444; }
    .metric-value.crop { color: #16a34a; }
    .metric-value.total { color: #3b82f6; }
    .metric-value.fps { color: #8b5cf6; }
    .metric-value.density { color: #d97706; }

    .metric-sub {
        font-size: 0.72rem; color: #94a3b8; margin-top: 6px; font-weight: 500;
    }

    .panel-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 22px; box-shadow: 0 4px 16px rgba(15,23,42,0.04); margin-bottom: 18px;
    }
    .panel-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;
    }
    .panel-title {
        font-size: 1.05rem; font-weight: 700; color: #0f172a;
        display: flex; align-items: center; gap: 8px;
    }
    .panel-badge {
        display: inline-block; padding: 3px 10px; border-radius: 10px;
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .panel-badge.live { background: rgba(34,197,94,0.1); color: #16a34a; }
    .panel-badge.video { background: rgba(139,92,246,0.1); color: #8b5cf6; }

    .status-alert {
        padding: 14px 18px; border-radius: 14px; display: flex;
        align-items: center; gap: 12px; margin-bottom: 14px; font-weight: 500;
    }
    .status-alert.danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(239,68,68,0.03) 100%);
        border: 1px solid rgba(239,68,68,0.2); color: #dc2626;
    }
    .status-alert.success {
        background: linear-gradient(135deg, rgba(34,197,94,0.08) 0%, rgba(34,197,94,0.03) 100%);
        border: 1px solid rgba(34,197,94,0.2); color: #16a34a;
    }
    .status-alert.warning {
        background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%);
        border: 1px solid rgba(245,158,11,0.2); color: #d97706;
    }
    .status-alert.info {
        background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(59,130,246,0.03) 100%);
        border: 1px solid rgba(59,130,246,0.2); color: #2563eb;
    }
    .status-icon { font-size: 1.3rem; }
    .status-title { font-weight: 700; font-size: 0.9rem; }
    .status-desc { font-size: 0.78rem; opacity: 0.8; margin-top: 2px; }

    .timeline-bar {
        display: flex; width: 100%; height: 28px; border-radius: 8px;
        overflow: hidden; margin: 8px 0; background: #f1f5f9;
    }
    .timeline-segment {
        height: 100%; transition: all 0.3s ease; position: relative;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.6rem; font-weight: 700; color: white;
    }
    .timeline-segment.clean { background: #22c55e; }
    .timeline-segment.low { background: #f59e0b; }
    .timeline-segment.high { background: #ef4444; }

    .legend-container {
        display: flex; align-items: center; gap: 16px;
        padding: 10px 14px; background: #f8fafc; border-radius: 10px; margin-top: 10px;
    }
    .legend-item {
        display: flex; align-items: center; gap: 5px;
        font-size: 0.78rem; color: #475569; font-weight: 500;
    }
    .legend-dot { width: 10px; height: 10px; border-radius: 3px; }
    .legend-dot.weed { background: #ef4444; }
    .legend-dot.crop { background: #22c55e; }
    .legend-dot.clean { background: #22c55e; }
    .legend-dot.low-w { background: #f59e0b; }
    .legend-dot.high-w { background: #ef4444; }

    .rec-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 16px 18px; margin-bottom: 10px; display: flex;
        align-items: flex-start; gap: 12px; transition: all 0.2s ease;
    }
    .rec-card:hover { box-shadow: 0 6px 20px rgba(15,23,42,0.06); transform: translateX(3px); }
    .rec-icon {
        width: 38px; height: 38px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; flex-shrink: 0;
    }
    .rec-icon.spray { background: rgba(239,68,68,0.1); }
    .rec-icon.monitor { background: rgba(59,130,246,0.1); }
    .rec-icon.healthy { background: rgba(34,197,94,0.1); }
    .rec-title { font-weight: 700; font-size: 0.85rem; color: #0f172a; margin-bottom: 3px; }
    .rec-desc { font-size: 0.76rem; color: #64748b; line-height: 1.5; }

    .footer {
        text-align: center; padding: 18px; color: #94a3b8;
        font-size: 0.75rem; margin-top: 30px; border-top: 1px solid #e2e8f0;
    }

    .frame-counter {
        background: rgba(15,23,42,0.75); color: white; padding: 4px 12px;
        border-radius: 8px; font-size: 0.75rem; font-weight: 600;
        display: inline-block; margin-bottom: 6px; backdrop-filter: blur(8px);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# CORE ENGINE (model + detection + drawing)
# ═══════════════════════════════════════════════════════
MODEL_PATH =  "/content/best_float32.tflite"


@st.cache_resource
def load_model():
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp


def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, iou_thr):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    selected = []
    while boxes:
        best = boxes.pop(0)
        selected.append(best)
        boxes = [b for b in boxes
                 if b[5] != best[5] or iou(best[:4], b[:4]) < iou_thr]
    return selected


def draw_label(draw, x1, y1, text, color, font_size=18):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x1, y1), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = 8, 4
    top = max(0, y1 - th - 2*py - 4)
    draw.rounded_rectangle([x1+2, top+2, x1+tw+2*px+2, top+th+2*py+2], radius=7, fill=(0,0,0,70))
    draw.rounded_rectangle([x1, top, x1+tw+2*px, top+th+2*py], radius=7, fill=color)
    draw.text((x1+px, top+py), text, fill="white", font=font)


def detect_frame(interpreter, input_details, output_details,
                 pil_img, threshold, iou_thr,
                 show_weeds=True, show_crops=True,
                 show_labels=True, show_confidence=True,
                 box_thickness=4, label_size=18):
    """Run detection on a single PIL image. Return annotated image + stats dict."""
    img_arr = np.array(pil_img)
    h, w = img_arr.shape[:2]
    inp_h = int(input_details[0]["shape"][1])
    inp_w = int(input_details[0]["shape"][2])

    resized = np.array(pil_img.resize((inp_w, inp_h)))
    tensor = np.expand_dims(resized, 0).astype(np.float32)
    if input_details[0]["dtype"] == np.float32:
        tensor /= 255.0

    interpreter.set_tensor(input_details[0]["index"], tensor)
    t0 = time.time()
    interpreter.invoke()
    inf_ms = (time.time() - t0) * 1000

    output = interpreter.get_tensor(output_details[0]["index"])
    preds = np.transpose(output[0], (1, 0))  # (8400, 6)

    raw = []
    for row in preds:
        x, y, bw, bh, ws, cs = row
        x1 = int((x - bw/2) * w); y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w); y2 = int((y + bh/2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2-x1)*(y2-y1)
        total_area = w*h
        if area < total_area*0.001 or area > total_area*0.50:
            continue
        if ws >= threshold:
            raw.append([x1, y1, x2, y2, float(ws), 0])
        if cs >= threshold:
            raw.append([x1, y1, x2, y2, float(cs), 1])

    boxes = nms(raw, iou_thr)

    # Draw
    annotated = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    weed_cnt, crop_cnt = 0, 0
    weed_positions = []

    for bx in boxes:
        x1, y1, x2, y2, score, cls = bx
        if cls == 0 and not show_weeds: continue
        if cls == 1 and not show_crops: continue

        if cls == 0:
            label, color, fill = "Weed", "#ef4444", (239,68,68,30)
            weed_cnt += 1
            weed_positions.append(((x1+x2)//2, (y1+y2)//2, score))
        else:
            label, color, fill = "Crop", "#22c55e", (34,197,94,30)
            crop_cnt += 1

        draw.rounded_rectangle([x1,y1,x2,y2], radius=5, fill=fill)
        draw.rounded_rectangle([x1,y1,x2,y2], radius=5, outline=color, width=box_thickness)
        if show_labels:
            txt = f"{label} {score:.0%}" if show_confidence else label
            draw_label(draw, x1, y1, txt, color, label_size)

    result = Image.alpha_composite(annotated, overlay).convert("RGB")

    total = weed_cnt + crop_cnt
    density = (weed_cnt / total * 100) if total > 0 else 0.0

    stats = {
        "weed_count": weed_cnt,
        "crop_count": crop_cnt,
        "total": total,
        "density": density,
        "inference_ms": inf_ms,
        "weed_positions": weed_positions,
        "img_w": w, "img_h": h,
    }
    return result, stats


def build_cumulative_heatmap(all_positions, img_w, img_h, decay=0.85):
    """
    Build a density heatmap from accumulated weed positions across frames.
    Older frames contribute less (temporal decay).
    """
    heatmap = np.zeros((img_h, img_w), dtype=np.float64)

    n_frames = len(all_positions)
    for fidx, frame_pos in enumerate(all_positions):
        weight = decay ** (n_frames - 1 - fidx)  # newer frames → higher weight
        for cx, cy, score in frame_pos:
            radius = int(min(img_w, img_h) * 0.06)
            y_lo = max(0, cy - radius)
            y_hi = min(img_h, cy + radius)
            x_lo = max(0, cx - radius)
            x_hi = min(img_w, cx + radius)
            for yy in range(y_lo, y_hi, 2):
                for xx in range(x_lo, x_hi, 2):
                    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                    if dist < radius:
                        val = (1 - dist/radius) * score * weight
                        heatmap[yy, xx] += val

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def heatmap_to_image(base_img, heatmap, alpha=0.55):
    """Overlay heatmap on base image."""
    h, w = heatmap.shape
    color_map = np.zeros((h, w, 4), dtype=np.uint8)

    for y in range(0, h, 1):
        for x in range(0, w, 1):
            v = heatmap[y, x]
            if v > 0.01:
                # red channel scales with intensity
                r = int(min(255, 200 + v * 55))
                g = int(max(0, 80 - v * 80))
                b = 0
                a = int(v * 255 * alpha)
                color_map[y, x] = [r, g, b, a]

    overlay = Image.fromarray(color_map, "RGBA")
    base = base_img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def heatmap_to_image_fast(base_img, heatmap, alpha=0.55):
    """Vectorized heatmap overlay — much faster than pixel loop."""
    h, w = heatmap.shape
    mask = heatmap > 0.01

    r = np.clip(200 + heatmap * 55, 0, 255).astype(np.uint8)
    g = np.clip(80 - heatmap * 80, 0, 255).astype(np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)
    a = np.clip(heatmap * 255 * alpha, 0, 255).astype(np.uint8)

    # zero out where no heat
    r[~mask] = 0; g[~mask] = 0; a[~mask] = 0

    color_map = np.stack([r, g, b, a], axis=-1)
    overlay = Image.fromarray(color_map, "RGBA")
    base = base_img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 20px 0;">
        <div style="font-size:2.2rem;">🌱</div>
        <div style="font-size:1.15rem; font-weight:800; color:#f0fdf4; margin-top:2px;">AgriVision AI</div>
        <div style="font-size:0.7rem; color:#64748b; margin-top:3px;">Video Analysis Platform v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">📹 Input Mode</p>', unsafe_allow_html=True)
    input_mode = st.radio("Select input", ["🖼️ Image", "🎬 Video"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">⚙️ Detection</p>', unsafe_allow_html=True)
    threshold = st.slider("Confidence", 0.10, 0.90, 0.55, 0.05)
    iou_threshold = st.slider("NMS IoU", 0.10, 0.80, 0.35, 0.05)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">👁️ Visualization</p>', unsafe_allow_html=True)
    show_weeds = st.checkbox("🔴 Show Weeds", True)
    show_crops = st.checkbox("🟢 Show Crops", True)
    show_labels = st.checkbox("🏷️ Labels", True)
    show_confidence = st.checkbox("📊 Confidence", True)
    show_debug = st.checkbox("🔍 Debug", False)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">🎨 Style</p>', unsafe_allow_html=True)
    box_thickness = st.select_slider("Box Width", [2,3,4,5,6], value=4)
    label_size = st.select_slider("Label Size", [12,14,16,18,20,22], value=18)

    if input_mode == "🎬 Video":
        st.markdown("---")
        st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">🎬 Video Settings</p>', unsafe_allow_html=True)
        frame_skip = st.slider("Process every Nth frame", 1, 30, 5, 1,
                               help="Higher = faster but less granular")
        max_frames = st.slider("Max frames to process", 10, 500, 100, 10,
                               help="Limit total frames for speed")
        heatmap_decay = st.slider("Heatmap temporal decay", 0.50, 1.00, 0.85, 0.05,
                                  help="Lower = older detections fade faster")

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:12px;
                border:1px solid rgba(255,255,255,0.06); font-size:0.72rem;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:#94a3b8;">Architecture</span>
            <span style="color:#e2e8f0;font-weight:600;">YOLOv8</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:#94a3b8;">Format</span>
            <span style="color:#e2e8f0;font-weight:600;">TFLite FP32</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:#94a3b8;">Classes</span>
            <span style="color:#e2e8f0;font-weight:600;">Weed, Crop</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;">Input</span>
            <span style="color:#e2e8f0;font-weight:600;">640×640</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════
mode_text = "Video Analysis" if input_mode == "🎬 Video" else "Image Detection"
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">● {mode_text} Mode</div>
    <div class="hero-title">🌱 Precision Agriculture AI</div>
    <div class="hero-subtitle">
        {'Frame-by-frame video analysis with temporal weed density heatmaps and field health timeline.' if input_mode == '🎬 Video'
         else 'Advanced weed and crop detection powered by edge AI. Upload field images for instant analysis.'}
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"⚠️ Model load error: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════
# ─── IMAGE MODE ───
# ═══════════════════════════════════════════════════════
if input_mode == "🖼️ Image":
    uploaded = st.file_uploader("Upload field image", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if not uploaded:
        st.markdown("""
        <div style="background:#f8fafc; border:2px dashed #cbd5e1; border-radius:20px;
                    padding:50px 30px; text-align:center;">
            <div style="font-size:3rem; margin-bottom:12px;">📸</div>
            <div style="font-size:1.2rem; font-weight:700; color:#0f172a;">Drop your field image here</div>
            <div style="font-size:0.85rem; color:#64748b;">JPG, JPEG, PNG supported</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    original = image.copy()

    with st.spinner("🔬 Analyzing..."):
        annotated, stats = detect_frame(
            interpreter, input_details, output_details,
            image, threshold, iou_threshold,
            show_weeds, show_crops, show_labels, show_confidence,
            box_thickness, label_size
        )

    wc = stats["weed_count"]; cc = stats["crop_count"]
    total = stats["total"]; density = stats["density"]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card weed">
            <div class="metric-icon">🔴</div>
            <div class="metric-label">Weeds</div>
            <div class="metric-value weed">{wc}</div>
        </div>
        <div class="metric-card crop">
            <div class="metric-icon">🟢</div>
            <div class="metric-label">Crops</div>
            <div class="metric-value crop">{cc}</div>
        </div>
        <div class="metric-card total">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Total</div>
            <div class="metric-value total">{total}</div>
        </div>
        <div class="metric-card density">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Weed Density</div>
            <div class="metric-value density">{density:.1f}%</div>
        </div>
        <div class="metric-card fps">
            <div class="metric-icon">⏱️</div>
            <div class="metric-label">Inference</div>
            <div class="metric-value fps">{stats['inference_ms']:.0f}ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Heatmap for single image
    if stats["weed_positions"]:
        heat = build_cumulative_heatmap(
            [stats["weed_positions"]], stats["img_w"], stats["img_h"]
        )
        heat_img = heatmap_to_image_fast(original, heat)
    else:
        heat_img = original.copy()

    tab1, tab2, tab3 = st.tabs(["📸 Detection", "🔥 Heatmap", "🔄 Compare"])
    with tab1:
        st.image(annotated, use_container_width=True)
    with tab2:
        st.image(heat_img, use_container_width=True)
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="frame-counter">Original</div>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with c2:
            st.markdown('<div class="frame-counter">Detection</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

    # Export
    buf = io.BytesIO(); annotated.save(buf, format="PNG")
    st.download_button("📥 Download Result", buf.getvalue(),
                       f"detection_{datetime.now():%Y%m%d_%H%M%S}.png", "image/png",
                       use_container_width=True)


# ═══════════════════════════════════════════════════════
# ─── VIDEO MODE ───
# ═══════════════════════════════════════════════════════
else:
    uploaded_video = st.file_uploader("Upload field video", type=["mp4","avi","mov","mkv"],
                                      label_visibility="collapsed")
    if not uploaded_video:
        st.markdown("""
        <div style="background:#f8fafc; border:2px dashed #cbd5e1; border-radius:20px;
                    padding:50px 30px; text-align:center;">
            <div style="font-size:3rem; margin-bottom:12px;">🎬</div>
            <div style="font-size:1.2rem; font-weight:700; color:#0f172a;">Drop your field video here</div>
            <div style="font-size:0.85rem; color:#64748b; margin-top:6px;">
                MP4, AVI, MOV, MKV supported<br>
                <span style="color:#94a3b8;">Drone footage or ground-level video — we'll analyze frame by frame</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Save uploaded video to temp file for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_video_frames / video_fps if video_fps > 0 else 0

    # Video info
    st.markdown(f"""
    <div class="status-alert info">
        <div class="status-icon">🎬</div>
        <div>
            <div class="status-title">Video Loaded: {uploaded_video.name}</div>
            <div class="status-desc">
                {vid_w}×{vid_h} • {video_fps:.0f} FPS • {total_video_frames} frames
                • {duration_sec:.1f}s duration
                • Processing every {frame_skip}{'st' if frame_skip==1 else 'th'} frame
                (≈{min(max_frames, total_video_frames // max(1, frame_skip))} frames to analyze)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️  Start Video Analysis", type="primary", use_container_width=True):

        # Placeholders
        progress_bar = st.progress(0, text="Initializing...")
        frame_display = st.empty()
        metrics_display = st.empty()
        timeline_display = st.empty()

        # Accumulators
        all_weed_positions = []
        frame_stats_list = []
        processed = 0
        frame_idx = 0
        first_frame_pil = None
        last_annotated = None

        total_to_process = min(max_frames, total_video_frames // max(1, frame_skip))

        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            if first_frame_pil is None:
                first_frame_pil = pil_frame.copy()

            annotated_frame, fstats = detect_frame(
                interpreter, input_details, output_details,
                pil_frame, threshold, iou_threshold,
                show_weeds, show_crops, show_labels, show_confidence,
                box_thickness, label_size
            )

            all_weed_positions.append(fstats["weed_positions"])
            frame_stats_list.append(fstats)
            last_annotated = annotated_frame
            processed += 1
            frame_idx += 1

            # Update progress
            pct = processed / total_to_process
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Frame {processed}/{total_to_process} — "
                     f"{fstats['weed_count']} weeds, {fstats['crop_count']} crops "
                     f"({fstats['inference_ms']:.0f}ms)"
            )

            # Live frame preview (update every 3rd processed frame for speed)
            if processed % 3 == 1 or processed == total_to_process:
                with frame_display.container():
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f'<div class="frame-counter">📷 Frame {processed}/{total_to_process}</div>',
                                    unsafe_allow_html=True)
                        st.image(annotated_frame, use_container_width=True)
                    with c2:
                        # Mini running stats
                        total_weeds = sum(f["weed_count"] for f in frame_stats_list)
                        total_crops = sum(f["crop_count"] for f in frame_stats_list)
                        avg_density = np.mean([f["density"] for f in frame_stats_list])
                        avg_inf = np.mean([f["inference_ms"] for f in frame_stats_list])

                        st.markdown(f"""
                        <div class="panel-card">
                            <div class="panel-header">
                                <div class="panel-title">📊 Running Stats</div>
                                <div class="panel-badge live">● Live</div>
                            </div>
                            <table style="width:100%;font-size:0.85rem;">
                                <tr><td style="color:#64748b;padding:4px 0;">Frames Processed</td>
                                    <td style="font-weight:700;text-align:right;">{processed}</td></tr>
                                <tr><td style="color:#64748b;padding:4px 0;">Total Weeds</td>
                                    <td style="font-weight:700;color:#ef4444;text-align:right;">{total_weeds}</td></tr>
                                <tr><td style="color:#64748b;padding:4px 0;">Total Crops</td>
                                    <td style="font-weight:700;color:#16a34a;text-align:right;">{total_crops}</td></tr>
                                <tr><td style="color:#64748b;padding:4px 0;">Avg Density</td>
                                    <td style="font-weight:700;color:#d97706;text-align:right;">{avg_density:.1f}%</td></tr>
                                <tr><td style="color:#64748b;padding:4px 0;">Avg Inference</td>
                                    <td style="font-weight:700;color:#8b5cf6;text-align:right;">{avg_inf:.0f}ms</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

        cap.release()
        progress_bar.progress(1.0, text="✅ Analysis Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        frame_display.empty()

        # ═══════════════════════════════════════════════════
        # FINAL RESULTS
        # ═══════════════════════════════════════════════════
        if not frame_stats_list:
            st.warning("No frames were processed.")
            st.stop()

        total_weeds = sum(f["weed_count"] for f in frame_stats_list)
        total_crops = sum(f["crop_count"] for f in frame_stats_list)
        total_det = total_weeds + total_crops
        avg_density = np.mean([f["density"] for f in frame_stats_list])
        avg_inf = np.mean([f["inference_ms"] for f in frame_stats_list])
        max_density_frame = max(frame_stats_list, key=lambda f: f["density"])
        peak_density = max_density_frame["density"]

        # ── Summary Metrics ──
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card weed">
                <div class="metric-icon">🔴</div>
                <div class="metric-label">Total Weeds</div>
                <div class="metric-value weed">{total_weeds}</div>
                <div class="metric-sub">across {processed} frames</div>
            </div>
            <div class="metric-card crop">
                <div class="metric-icon">🟢</div>
                <div class="metric-label">Total Crops</div>
                <div class="metric-value crop">{total_crops}</div>
                <div class="metric-sub">across {processed} frames</div>
            </div>
            <div class="metric-card total">
                <div class="metric-icon">📊</div>
                <div class="metric-label">Detections</div>
                <div class="metric-value total">{total_det}</div>
                <div class="metric-sub">{total_det/max(1,processed):.1f} per frame</div>
            </div>
            <div class="metric-card density">
                <div class="metric-icon">🎯</div>
                <div class="metric-label">Avg Density</div>
                <div class="metric-value density">{avg_density:.1f}%</div>
                <div class="metric-sub">peak: {peak_density:.1f}%</div>
            </div>
            <div class="metric-card fps">
                <div class="metric-icon">⚡</div>
                <div class="metric-label">Avg Speed</div>
                <div class="metric-value fps">{avg_inf:.0f}ms</div>
                <div class="metric-sub">{1000/max(1,avg_inf):.1f} FPS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Status Alert ──
        if avg_density > 50:
            alert_cls, icon, title = "danger", "🚨", "Critical Weed Infestation"
            desc = f"Average density {avg_density:.1f}% across video. Immediate action required."
        elif avg_density > 20:
            alert_cls, icon, title = "warning", "⚡", "Moderate Weed Presence"
            desc = f"Average density {avg_density:.1f}%. Targeted treatment recommended."
        elif total_weeds > 0:
            alert_cls, icon, title = "info", "ℹ️", "Low Weed Presence"
            desc = f"{total_weeds} total weed detections. Standard monitoring sufficient."
        else:
            alert_cls, icon, title = "success", "✅", "Field Healthy"
            desc = "No weeds detected throughout the video."

        st.markdown(f"""
        <div class="status-alert {alert_cls}">
            <div class="status-icon">{icon}</div>
            <div><div class="status-title">{title}</div>
                 <div class="status-desc">{desc}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Field Health Timeline ──
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-title">📈 Field Health Timeline</div>
                <div class="panel-badge video">● Video Analysis</div>
            </div>
        """, unsafe_allow_html=True)

        # Timeline bar
        segments_html = ""
        for i, fs in enumerate(frame_stats_list):
            d = fs["density"]
            seg_cls = "clean" if d < 10 else "low" if d < 40 else "high"
            w_pct = 100 / len(frame_stats_list)
            segments_html += f'<div class="timeline-segment {seg_cls}" style="width:{w_pct:.2f}%" title="Frame {i+1}: {d:.0f}%"></div>'

        st.markdown(f"""
            <div style="font-size:0.78rem;color:#64748b;margin-bottom:4px;">
                Weed density per frame (left → right = start → end of video)
            </div>
            <div class="timeline-bar">{segments_html}</div>
            <div class="legend-container">
                <div class="legend-item"><div class="legend-dot clean"></div> Clean (&lt;10%)</div>
                <div class="legend-item"><div class="legend-dot low-w"></div> Low (10-40%)</div>
                <div class="legend-item"><div class="legend-dot high-w"></div> High (&gt;40%)</div>
            </div>
        """, unsafe_allow_html=True)

        # Time series chart
        import pandas as pd
        chart_df = pd.DataFrame({
            "Frame": list(range(1, len(frame_stats_list)+1)),
            "Weeds": [f["weed_count"] for f in frame_stats_list],
            "Crops": [f["crop_count"] for f in frame_stats_list],
            "Density (%)": [f["density"] for f in frame_stats_list],
        })
        st.line_chart(chart_df.set_index("Frame")[["Weeds","Crops"]], height=220,
                      color=["#ef4444","#22c55e"])

        st.line_chart(chart_df.set_index("Frame")[["Density (%)"]], height=180,
                      color=["#f59e0b"])

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Cumulative Heatmap ──
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-title">🔥 Cumulative Weed Density Heatmap</div>
                <div class="panel-badge video">● Temporal Analysis</div>
            </div>
        """, unsafe_allow_html=True)

        base_for_heatmap = first_frame_pil if first_frame_pil else Image.new("RGB", (vid_w, vid_h), (0,0,0))
        hmap = build_cumulative_heatmap(all_weed_positions, vid_w, vid_h, decay=heatmap_decay)
        heat_img = heatmap_to_image_fast(base_for_heatmap, hmap)

        tab_h1, tab_h2 = st.tabs(["🔥 Heatmap Overlay", "📸 Last Detection"])
        with tab_h1:
            st.image(heat_img, use_container_width=True)
            st.markdown("""
            <div class="legend-container">
                <span style="font-size:0.78rem;color:#64748b;">
                    🔥 Brighter red = higher weed recurrence across frames.
                    Temporal decay factor weights recent frames more heavily.
                </span>
            </div>
            """, unsafe_allow_html=True)
        with tab_h2:
            if last_annotated:
                st.image(last_annotated, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Recommendations ──
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-title">💡 AI Recommendations</div>
            </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            if avg_density > 50:
                rec_icon, rec_bg, rec_title, rec_desc = (
                    "🧪", "spray", "Immediate Herbicide Application",
                    f"Critical density {avg_density:.0f}%. Deploy GPS-guided precision spraying on hotspot zones shown in heatmap."
                )
            elif avg_density > 20:
                rec_icon, rec_bg, rec_title, rec_desc = (
                    "👁️", "monitor", "Targeted Spot Treatment",
                    f"Moderate density {avg_density:.0f}%. Focus mechanical/chemical treatment on red zones in the heatmap."
                )
            else:
                rec_icon, rec_bg, rec_title, rec_desc = (
                    "✅", "healthy", "Continue Monitoring",
                    "Low weed presence. Maintain current management. Re-scan in 5-7 days."
                )
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-icon {rec_bg}">{rec_icon}</div>
                <div><div class="rec-title">{rec_title}</div><div class="rec-desc">{rec_desc}</div></div>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            hotspot_pct = sum(1 for f in frame_stats_list if f["density"] > 30) / len(frame_stats_list) * 100
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-icon monitor">📍</div>
                <div>
                    <div class="rec-title">Hotspot Analysis</div>
                    <div class="rec-desc">
                        {hotspot_pct:.0f}% of video frames show elevated weed density (&gt;30%).
                        {'Concentrate resources on persistent hotspot areas.' if hotspot_pct > 30
                         else 'Weed distribution is sparse — localized treatment preferred.'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rc3:
            next_scan = "24-48 hours" if avg_density > 50 else "3-5 days" if avg_density > 20 else "5-7 days"
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-icon spray">📅</div>
                <div>
                    <div class="rec-title">Next Scan: {next_scan}</div>
                    <div class="rec-desc">
                        Based on {avg_density:.0f}% average weed density.
                        {processed} frames analyzed over {duration_sec:.1f}s of footage.
                        Track density trend to assess treatment effectiveness.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Exports ──
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-title">💾 Export Results</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            buf = io.BytesIO(); heat_img.save(buf, format="PNG")
            st.download_button("📥 Heatmap Image", buf.getvalue(),
                               f"heatmap_{datetime.now():%Y%m%d_%H%M%S}.png", "image/png",
                               use_container_width=True)

        with ex2:
            if last_annotated:
                buf2 = io.BytesIO(); last_annotated.save(buf2, format="PNG")
                st.download_button("📥 Last Detection", buf2.getvalue(),
                                   f"last_frame_{datetime.now():%Y%m%d_%H%M%S}.png", "image/png",
                                   use_container_width=True)

        with ex3:
            report = {
                "timestamp": datetime.now().isoformat(),
                "video": uploaded_video.name,
                "resolution": f"{vid_w}x{vid_h}",
                "fps": video_fps,
                "duration_sec": round(duration_sec, 2),
                "frames_processed": processed,
                "frame_skip": frame_skip,
                "settings": {
                    "threshold": threshold,
                    "iou_threshold": iou_threshold,
                    "heatmap_decay": heatmap_decay,
                },
                "summary": {
                    "total_weeds": total_weeds,
                    "total_crops": total_crops,
                    "avg_density_pct": round(avg_density, 2),
                    "peak_density_pct": round(peak_density, 2),
                    "avg_inference_ms": round(avg_inf, 2),
                },
                "per_frame": [
                    {
                        "frame": i+1,
                        "weeds": f["weed_count"],
                        "crops": f["crop_count"],
                        "density": round(f["density"], 2),
                        "inference_ms": round(f["inference_ms"], 2),
                    }
                    for i, f in enumerate(frame_stats_list)
                ],
            }
            st.download_button("📥 JSON Report", json.dumps(report, indent=2),
                               f"video_report_{datetime.now():%Y%m%d_%H%M%S}.json",
                               "application/json", use_container_width=True)

        # ── Debug ──
        if show_debug:
            with st.expander("🔍 Debug", expanded=False):
                st.json({
                    "model_input": str(input_details[0]["shape"]),
                    "model_output": str(output_details[0]["shape"]),
                    "video_frames": total_video_frames,
                    "processed_frames": processed,
                    "frame_skip": frame_skip,
                    "all_weed_positions_count": sum(len(p) for p in all_weed_positions),
                })


# ═══════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
    🌱 <strong>AgriVision AI</strong> — Precision Agriculture Platform<br>
    Streamlit • TensorFlow Lite • YOLOv8 • OpenCV •
    {datetime.now():%Y-%m-%d %H:%M:%S}
</div>
""", unsafe_allow_html=True)
