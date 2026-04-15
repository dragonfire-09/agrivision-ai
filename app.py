from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

# Glassmorphism CSS (aynı kalıyor)
GLASS_CSS = """
<style>
.glass-card { background: rgba(255,255,255,0.25); backdrop-filter: blur(10px); 
              border-radius: 20px; border: 1px solid rgba(255,255,255,0.18); 
              box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37); padding: 1.5rem; 
              text-align: center; transition: transform 0.3s ease; }
.glass-card:hover { transform: translateY(-5px); }
.metric-icon { font-size: 3em; margin-bottom: 0.5rem; }
.metric-value { font-size: 2.5em; font-weight: bold; margin: 0; }
.metric-label { color: rgba(255,255,255,0.8); margin: 0; }
.download-btn { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); 
                border: none; border-radius: 15px; padding: 1rem 2rem; 
                color: white; font-size: 1.1em; font-weight: bold; 
                box-shadow: 0 8px 32px rgba(102,126,234,0.4); transition: all 0.3s ease; }
.download-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(102,126,234,0.6); }
</style>
"""

st.set_page_config(page_title="🌱 AgriVision AI Pro", layout="wide", page_icon="🌱")
st.markdown(GLASS_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1, area2 = (x2-x1)*(y2-y1), (x2b-x1b)*(y2b-y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if not boxes: return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if indices.size == 1: break
        remaining = indices[1:]
        iou_scores = np.array([iou(boxes[current], boxes[i]) for i in remaining])
        indices = remaining[iou_scores <= iou_threshold]
    return keep

def draw_detections(img, boxes_weed, scores_weed, keep_weed, 
                   boxes_crop, scores_crop, keep_crop):
    """Dual-class drawing: WEED(red) + CROP(green)"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # DRAW WEEDS (RED)
    for i in keep_weed:
        x1, y1, x2, y2 = boxes_weed[i]
        conf = scores_weed[i]
        color, class_name, bg_color = "#FF4757", "🌿 WEED", "#FF6B7A"
        
        draw.rectangle([x1,y1,x2,y2], outline=color, width=5)
        label = f"{class_name} {conf:.0%}"
        bbox = draw.textbbox((0,0), label, font=font)
        label_w = bbox[2] - bbox[0] + 16
        label_h = bbox[3] - bbox[1] + 10
        label_x, label_y = x1, y1 - label_h
        draw.rectangle([label_x, label_y, label_x+label_w, label_y+label_h], fill=bg_color)
        draw.text((label_x+8, label_y+5), label, fill="white", font=font)
    
    # DRAW CROPS (GREEN)
    for i in keep_crop:
        x1, y1, x2, y2 = boxes_crop[i]
        conf = scores_crop[i]
        color, class_name, bg_color = "#2ED573", "🌾 CROP", "#51CF66"
        
        draw.rectangle([x1,y1,x2,y2], outline=color, width=5)
        label = f"{class_name} {conf:.0%}"
        bbox = draw.textbbox((0,0), label, font=font)
        label_w = bbox[2] - bbox[0] + 16
        label_h = bbox[3] - bbox[1] + 10
        label_x, label_y = x1, y1 - label_h
        draw.rectangle([label_x, label_y, label_x+label_w, label_y+label_h], fill=bg_color)
        draw.text((label_x+8, label_y+5), label, fill="white", font=font)
    
    return img

# LOAD MODEL
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except:
    st.error("❌ Model yüklenemedi!")
    st.stop()

# UI
st.markdown("""
<div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); 
            border-radius: 20px; padding: 2rem; margin-bottom: 2rem;'>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,1,1])
with col1:
    uploaded = st.file_uploader("📁 Tarla Fotoğrafı Yükle", type=["jpg","png","jpeg"])
with col2:
    threshold = st.slider("🎯 Güven Eşiği", 0.3, 0.9, 0.5, 0.05)
with col3:
    iou_thresh = st.slider("🔗 NMS IoU", 0.3, 0.8, 0.5, 0.05)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    
    with st.spinner("🔍 Tarla analizi yapılıyor..."):
        # RESIZE & NORMALIZE
        resized = original_img.resize((640, 640))
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        # INFERENCE
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        preds = np.transpose(output, (1, 0))
    
    # 🔥 YENİ: ÇİFT SINIF DETECTION
    boxes_weed, scores_weed, areas_weed = [], [], []
    boxes_crop, scores_crop, areas_crop = [], [], []
    
    for row in preds:
        x, y, bw, bh, ws, cs = row  # ws=weed_score, cs=crop_score
        
        # WEED DETECTION: weed_score > crop_score AND > threshold
        if ws > cs and ws >= threshold:
            x1 = max(0, int((x-bw/2)*w))
            y1 = max(0, int((y-bh/2)*h))
            x2 = min(w, int((x+bw/2)*w))
            y2 = min(h, int((y+bh/2)*h))
            boxes_weed.append([x1,y1,x2,y2])
            scores_weed.append(ws)
            areas_weed.append((x2-x1)*(y2-y1))
        
        # CROP DETECTION: crop_score > weed_score AND > threshold  
        elif cs > ws and cs >= threshold:
            x1 = max(0, int((x-bw/2)*w))
            y1 = max(0, int((y-bh/2)*h))
            x2 = min(w, int((x+bw/2)*w))
            y2 = min(h, int((y+bh/2)*h))
            boxes_crop.append([x1,y1,x2,y2])
            scores_crop.append(cs)
            areas_crop.append((x2-x1)*(y2-y1))
    
    # NMS UYGULA
    keep_weed = non_max_suppression(boxes_weed, scores_weed, iou_thresh) if boxes_weed else []
    keep_crop = non_max_suppression(boxes_crop, scores_crop, iou_thresh) if boxes_crop else []
    
    # DRAW
    result_img = original_img.copy()
    result_img = draw_detections(result_img, boxes_weed, scores_weed, keep_weed,
                                boxes_crop, scores_crop, keep_crop)
    
    # METRICS
    total_area = w * h
    weed_density = sum(areas_weed[i] for i in keep_weed) / total_area * 100 if keep_weed else 0
    crop_density = sum(areas_crop[i] for i in keep_crop) / total_area * 100 if keep_crop else 0
    
    # GLASS CARDS
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌿</div>
            <h1 class="metric-value" style="color: #FF4757;">{len(keep_weed)}</h1>
            <p class="metric-label">WEED COUNT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌾</div>
            <h1 class="metric-value" style="color: #2ED573;">{len(keep_crop)}</h1>
            <p class="metric-label">CROP COUNT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">📊</div>
            <h1 class="metric-value" style="color: #FFA502;">{weed_density:.1f}%</h1>
            <p class="metric-label">WEED DENSITY</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🎯</div>
            <h1 class="metric-value" style="color: #3742FA;">
                {np.mean(scores_weed + scores_crop)*100:.0f}%</h1>
            <p class="metric-label">AVG CONFIDENCE</p>
        </div>
        """, unsafe_allow_html=True)
    
    # IMAGES + DOWNLOAD
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original_img, caption="📸 Original", use_container_width=True)
    with col_img2:
        st.image(result_img, caption="🎯 Detected", use_container_width=True)
    
    # DOWNLOAD
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button(
        label=f"💾 Download {timestamp}",
        data=buf.getvalue(),
        file_name=f"agri_vision_{timestamp}.png",
        mime="image/png",
        use_container_width=True
    )
    
    # CONFIDENCE DEBUG (isteğe bağlı)
    st.subheader("🔍 Confidence Debug")
    st.write(f"Weeds detected: {len(keep_weed)} | Crops detected: {len(keep_crop)}")
    st.write(f"Weed scores: {np.mean(scores_weed):.2f} ± {np.std(scores_weed):.2f}")
    st.write(f"Crop scores: {np.mean(scores_crop):.2f} ± {np.std(scores_crop):.2f}")
