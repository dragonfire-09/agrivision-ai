from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

GLASS_CSS = """
<style>
.glass-card { 
    background: rgba(255,255,255,0.15); 
    backdrop-filter: blur(12px); 
    border-radius: 20px; 
    border: 1px solid rgba(255,255,255,0.2); 
    box-shadow: 0 8px 32px rgba(31,38,135,0.37); 
    padding: 1.5rem; 
    text-align: center; 
    transition: all 0.3s ease; 
}
.glass-card:hover { 
    transform: translateY(-5px); 
    box-shadow: 0 12px 40px rgba(31,38,135,0.5); 
}
.metric-icon { font-size: 3em; margin-bottom: 0.3rem; }
.metric-value { font-size: 2.8em; font-weight: 800; margin: 0; }
.metric-label { 
    font-size: 0.9em; 
    text-transform: uppercase; 
    letter-spacing: 2px; 
    margin-top: 0.3rem; 
    opacity: 0.8; 
}
.detection-tag {
    display: inline-block; 
    padding: 0.5rem 1.2rem; 
    margin: 0.3rem;
    border-radius: 12px; 
    font-weight: bold; 
    font-size: 1em;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
"""

st.set_page_config(page_title="🌱 AgriVision AI", layout="wide", page_icon="🌱")
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# HEADER
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
    <h1 style='color: #e94560; margin: 0; font-size: 2.8em;'>🌱 AgriVision AI</h1>
    <p style='color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; font-size: 1.1em;'>
        Precision Weed Detection System</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def iou(box1, box2):
    x1,y1,x2,y2 = box1
    x1b,y1b,x2b,y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def containment(box1, box2):
    x1,y1,x2,y2 = box1
    x1b,y1b,x2b,y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    smaller = min(max(1,(x2-x1)*(y2-y1)), max(1,(x2b-x1b)*(y2b-y1b)))
    return inter / smaller if smaller > 0 else 0

def smart_nms(boxes, scores, iou_threshold=0.3, containment_threshold=0.6):
    """Aggressive NMS: IoU + Containment"""
    if not boxes: return []
    indices = np.argsort(scores)[::-1].tolist()
    keep = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        remove = []
        for idx in indices:
            if iou(boxes[current], boxes[idx]) > iou_threshold:
                remove.append(idx)
            elif containment(boxes[current], boxes[idx]) > containment_threshold:
                remove.append(idx)
        for r in remove:
            indices.remove(r)
    return keep

def draw_weed_boxes(img, boxes, scores, keep_indices):
    """Draw WEED detections with red boxes"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except:
        font = ImageFont.load_default()
    
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        
        # WEED = Always RED
        draw.rectangle([x1,y1,x2,y2], outline="#FF4757", width=5)
        
        label = f"🌿 WEED {conf:.0%}"
        bbox = draw.textbbox((0,0), label, font=font)
        lw = bbox[2] - bbox[0] + 16
        lh = bbox[3] - bbox[1] + 12
        lx, ly = x1, max(0, y1 - lh)
        
        draw.rectangle([lx, ly, lx+lw, ly+lh], fill="#FF6B7A")
        draw.text((lx+8, ly+5), label, fill="white", font=font)
    
    return img

# Load model
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"❌ Model hatası: {e}")
    st.stop()

# CONTROLS
col1, col2, col3 = st.columns([2,1,1])
with col1:
    uploaded = st.file_uploader("📁 Tarla Fotoğrafı Yükle", type=["jpg","png","jpeg"])
with col2:
    threshold = st.slider("🎯 Confidence", 0.1, 0.9, 0.4, 0.05)
with col3:
    nms_iou = st.slider("🔗 NMS Hassasiyet", 0.1, 0.7, 0.3, 0.05)

if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    
    with st.spinner("🔍 Weed Analizi Yapılıyor..."):
        resized = original_img.resize((640, 640))
        arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, 0)
        
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        preds = np.transpose(output_data, (1, 0))
    
    # ═══════════════════════════════════════════════════════════════
    # 🔥 WEED-ONLY DETECTION
    # Model tek sınıf: WEED
    # ═══════════════════════════════════════════════════════════════
    boxes_all, scores_all, areas_all = [], [], []
    
    num_classes = preds.shape[1] - 4  # İlk 4 = x,y,w,h
    
    for row in preds:
        x, y, bw, bh = row[0], row[1], row[2], row[3]
        
        # Tüm class score'larının max'ını al
        class_scores = row[4:]
        best_score = float(np.max(class_scores))
        
        if best_score < threshold:
            continue
        
        x1 = max(0, int((x - bw/2) * w))
        y1 = max(0, int((y - bh/2) * h))
        x2 = min(w, int((x + bw/2) * w))
        y2 = min(h, int((y + bh/2) * h))
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        if box_w > 10 and box_h > 10:
            boxes_all.append([x1, y1, x2, y2])
            scores_all.append(best_score)
            areas_all.append(box_w * box_h)
    
    # SMART NMS
    keep_indices = smart_nms(boxes_all, scores_all, 
                            iou_threshold=nms_iou, 
                            containment_threshold=0.6)
    
    # DRAW
    result_img = original_img.copy()
    if keep_indices:
        result_img = draw_weed_boxes(result_img, boxes_all, scores_all, keep_indices)
    
    weed_count = len(keep_indices)
    total_area = w * h
    weed_area = sum(areas_all[i] for i in keep_indices)
    weed_density = (weed_area / total_area) * 100
    avg_conf = np.mean([scores_all[i] for i in keep_indices]) if keep_indices else 0
    
    # ═══════════════════════════════════════════════════════════════
    # GLASSMORPHISM CARDS
    # ═══════════════════════════════════════════════════════════════
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(255,71,87,0.3), rgba(255,107,122,0.2));">
            <div class="metric-icon">🌿</div>
            <h1 class="metric-value" style="color: #FF4757;">{weed_count}</h1>
            <p class="metric-label" style="color: #FF6B7A;">Weed Count</p>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        # Severity level
        if weed_density < 5:
            severity, sev_color, sev_emoji = "LOW", "#2ED573", "✅"
        elif weed_density < 15:
            severity, sev_color, sev_emoji = "MEDIUM", "#FFA502", "⚠️"
        else:
            severity, sev_color, sev_emoji = "HIGH", "#FF4757", "🚨"
        
        st.markdown(f"""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(46,213,115,0.3), rgba(46,213,115,0.1));">
            <div class="metric-icon">{sev_emoji}</div>
            <h1 class="metric-value" style="color: {sev_color};">{severity}</h1>
            <p class="metric-label" style="color: #2ED573;">Severity</p>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(255,165,2,0.3), rgba(255,165,2,0.1));">
            <div class="metric-icon">📊</div>
            <h1 class="metric-value" style="color: #FFA502;">{weed_density:.1f}%</h1>
            <p class="metric-label" style="color: #FFB84D;">Weed Density</p>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(55,66,250,0.3), rgba(55,66,250,0.1));">
            <div class="metric-icon">🎯</div>
            <h1 class="metric-value" style="color: #3742FA;">{avg_conf:.0%}</h1>
            <p class="metric-label" style="color: #5B6BFF;">Confidence</p>
        </div>""", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # IMAGE COMPARISON
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original_img, caption="📸 Orijinal Görüntü", use_container_width=True)
    with col_img2:
        st.image(result_img, caption="🎯 Weed Tespiti", use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════
    # DETECTION TAGS
    # ═══════════════════════════════════════════════════════════════
    if keep_indices:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Tespit Edilen Yabani Otlar")
        
        tags_html = ""
        for idx, i in enumerate(keep_indices):
            conf = scores_all[i]
            area_pct = (areas_all[i] / total_area) * 100
            tags_html += f"""
            <div class="detection-tag" style="background: linear-gradient(135deg, #FF4757, #FF6B7A); color: white;">
                🌿 Weed #{idx+1} — {conf:.0%} conf — {area_pct:.1f}% alan
            </div>
            """
        st.markdown(tags_html, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # DOWNLOAD
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    st.download_button(
        label=f"💾 Sonucu İndir — agri_vision_{timestamp}.png",
        data=buf.getvalue(),
        file_name=f"agri_vision_{timestamp}.png",
        mime="image/png",
        use_container_width=True
    )

# FOOTER
st.markdown("""
<div style='text-align: center; padding: 2rem; margin-top: 3rem;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
    <p style='color: rgba(255,255,255,0.6); margin: 0;'>
        🌾 AgriVision AI — Precision Weed Detection
    </p>
</div>
""", unsafe_allow_html=True)
