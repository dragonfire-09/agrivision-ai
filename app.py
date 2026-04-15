from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

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
</style>
"""

st.set_page_config(page_title="🌱 AgriVision AI Pro", layout="wide", page_icon="🌱")
st.markdown(GLASS_CSS, unsafe_allow_html=True)

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102,126,234,0.4);'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🌱 AgriVision AI Pro</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0;'>Weed & Crop Detection</p>
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
    area1 = max(1, (x2 - x1) * (y2 - y1))
    area2 = max(1, (x2b - x1b) * (y2b - y1b))
    smaller_area = min(area1, area2)
    return inter / smaller_area if smaller_area > 0 else 0

def class_aware_nms(boxes, scores, classes, iou_threshold=0.3, containment_threshold=0.6):
    """Sınıf bazlı NMS: WEED ve CROP ayrı ayrı"""
    if not boxes: return []
    
    unique_classes = list(set(classes))
    all_keep = []
    
    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        cls_boxes = [boxes[i] for i in cls_indices]
        cls_scores = [scores[i] for i in cls_indices]
        
        if not cls_boxes: continue
        
        indices = np.argsort(cls_scores)[::-1].tolist()
        keep = []
        
        while indices:
            current = indices.pop(0)
            keep.append(current)
            remove = []
            for idx in indices:
                if iou(cls_boxes[current], cls_boxes[idx]) > iou_threshold:
                    remove.append(idx)
                elif containment(cls_boxes[current], cls_boxes[idx]) > containment_threshold:
                    remove.append(idx)
            for r in remove:
                indices.remove(r)
        
        all_keep.extend([cls_indices[k] for k in keep])
    
    return all_keep

def draw_detections(img, boxes, scores, classes, keep_indices):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cls = classes[i]
        
        if cls == "WEED":
            color, bg_color, emoji = "#FF4757", "#FF6B7A", "🌿"
        else:
            color, bg_color, emoji = "#2ED573", "#51CF66", "🌾"
        
        draw.rectangle([x1,y1,x2,y2], outline=color, width=5)
        
        label = f"{emoji} {cls} {conf:.0%}"
        bbox = draw.textbbox((0,0), label, font=font)
        lw = bbox[2] - bbox[0] + 16
        lh = bbox[3] - bbox[1] + 10
        lx, ly = x1, y1 - lh
        
        draw.rectangle([lx, ly, lx+lw, ly+lh], fill=bg_color)
        draw.text((lx+8, ly+5), label, fill="white", font=font)
    
    return img

# Load
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# CONTROLS
col1, col2, col3 = st.columns(3)
with col1:
    uploaded = st.file_uploader("📁 Fotoğraf", type=["jpg","png","jpeg"])
with col2:
    threshold = st.slider("🎯 Confidence", 0.1, 0.9, 0.3, 0.05)
with col3:
    # 🔥 BOYUT EŞİĞİ: Bu değerden küçük = WEED, büyük = CROP
    size_threshold = st.slider("📏 Boyut Eşiği (%)", 5, 50, 15, 1,
                               help="Bu yüzdeden küçük kutular = WEED, büyük = CROP")

if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    total_area = w * h
    
    with st.spinner("🔍 Analiz..."):
        resized = original_img.resize((640, 640))
        arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, 0)
        
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        preds = np.transpose(output_data, (1, 0))
    
    # ═══════════════════════════════════════════════════════════════
    # 🔥 TÜM DETECTION'LARI TOPLA (class fark etmez)
    # ═══════════════════════════════════════════════════════════════
    raw_boxes, raw_scores = [], []
    
    for row in preds:
        x, y, bw, bh = row[0], row[1], row[2], row[3]
        class_scores = row[4:]
        
        # En yüksek confidence
        best_score = np.max(class_scores)
        
        if best_score < threshold:
            continue
        
        x1 = max(0, int((x - bw/2) * w))
        y1 = max(0, int((y - bh/2) * h))
        x2 = min(w, int((x + bw/2) * w))
        y2 = min(h, int((y + bh/2) * h))
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            raw_boxes.append([x1, y1, x2, y2])
            raw_scores.append(float(best_score))
    
    # ═══════════════════════════════════════════════════════════════
    # 🔥 BOYUTA GÖRE SINIFLANDIR
    # ═══════════════════════════════════════════════════════════════
    boxes_all, scores_all, classes_all, areas_all = [], [], [], []
    size_limit = total_area * (size_threshold / 100)
    
    for i in range(len(raw_boxes)):
        x1, y1, x2, y2 = raw_boxes[i]
        box_area = (x2 - x1) * (y2 - y1)
        
        # 🔥 KÜÇÜK = WEED, BÜYÜK = CROP
        if box_area < size_limit:
            detected_class = "WEED"
        else:
            detected_class = "CROP"
        
        boxes_all.append(raw_boxes[i])
        scores_all.append(raw_scores[i])
        classes_all.append(detected_class)
        areas_all.append(box_area)
    
    # CLASS-AWARE NMS
    keep_indices = class_aware_nms(
        boxes_all, scores_all, classes_all,
        iou_threshold=0.3, containment_threshold=0.6
    )
    
    # DRAW
    result_img = original_img.copy()
    if keep_indices:
        result_img = draw_detections(
            result_img, boxes_all, scores_all, classes_all, keep_indices
        )
    
    # METRICS
    weed_count = sum(1 for i in keep_indices if classes_all[i] == "WEED")
    crop_count = sum(1 for i in keep_indices if classes_all[i] == "CROP")
    weed_area = sum(areas_all[i] for i in keep_indices if classes_all[i] == "WEED")
    weed_density = (weed_area / total_area) * 100 if total_area > 0 else 0
    avg_conf = np.mean([scores_all[i] for i in keep_indices]) if keep_indices else 0
    
    # GLASS CARDS
    st.markdown("### 📊 Sonuçlar")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌿</div>
            <h1 class="metric-value" style="color: #FF4757;">{weed_count}</h1>
            <p class="metric-label">WEEDS</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌾</div>
            <h1 class="metric-value" style="color: #2ED573;">{crop_count}</h1>
            <p class="metric-label">CROPS</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">📊</div>
            <h1 class="metric-value" style="color: #FFA502;">{weed_density:.1f}%</h1>
            <p class="metric-label">WEED DENSITY</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🎯</div>
            <h1 class="metric-value" style="color: #3742FA;">{avg_conf:.0%}</h1>
            <p class="metric-label">CONFIDENCE</p>
        </div>""", unsafe_allow_html=True)
    
    # IMAGES
    st.markdown("### 🖼️ Görüntü Karşılaştırma")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original_img, caption="📸 Orijinal", use_container_width=True)
    with col_img2:
        st.image(result_img, caption="🎯 Tespit", use_container_width=True)
    
    # DETECTION TABLE
    if keep_indices:
        st.markdown("### 📋 Detaylar")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            for i in keep_indices:
                if classes_all[i] == "WEED":
                    area_pct = (areas_all[i] / total_area) * 100
                    st.markdown(f"""
                    <div style='padding:0.5rem 1rem; margin:0.3rem; background:#FF4757; 
                                color:white; border-radius:10px; font-weight:bold;'>
                        🌿 WEED {scores_all[i]:.0%} | Alan: {area_pct:.1f}%
                    </div>""", unsafe_allow_html=True)
        with col_d2:
            for i in keep_indices:
                if classes_all[i] == "CROP":
                    area_pct = (areas_all[i] / total_area) * 100
                    st.markdown(f"""
                    <div style='padding:0.5rem 1rem; margin:0.3rem; background:#2ED573; 
                                color:white; border-radius:10px; font-weight:bold;'>
                        🌾 CROP {scores_all[i]:.0%} | Alan: {area_pct:.1f}%
                    </div>""", unsafe_allow_html=True)
    
    # DOWNLOAD
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button(
        label=f"💾 Sonuçları İndir ({timestamp})",
        data=buf.getvalue(),
        file_name=f"agri_vision_{timestamp}.png",
        mime="image/png",
        use_container_width=True
    )

st.markdown("""
<div style='text-align:center; padding:2rem; margin-top:3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius:20px; color:white;'>
    <h3>🌾 Precision Agriculture AI</h3>
    <p>🌿 Küçük = WEED (kırmızı) | 🌾 Büyük = CROP (yeşil)</p>
</div>
""", unsafe_allow_html=True)
