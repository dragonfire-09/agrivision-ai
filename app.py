from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

# Glassmorphism CSS
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

st.set_page_config(page_title="🌱 AgriVision AI Pro - FIXED", layout="wide", page_icon="🌱")
st.markdown(GLASS_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def iou(box1, box2):
    """IoU calculation for NMS"""
    x1,y1,x2,y2 = box1
    x1b,y1b,x2b,y2b = box2
    xi1,yi1 = max(x1,x1b), max(y1,y1b)
    xi2,yi2 = min(x2,x2b), min(y2,y2b)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1, area2 = (x2-x1)*(y2-y1), (x2b-x1b)*(y2b-y1b)
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """NMS implementation"""
    if not boxes: return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1: break
        remaining = indices[1:]
        iou_scores = np.array([iou(boxes[current], boxes[i]) for i in remaining])
        indices = remaining[iou_scores <= iou_threshold]
    return keep

def draw_detections(img, boxes, scores, classes, keep_indices):
    """Draw both WEED(red) and CROP(green) detections"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        class_name = classes[i]
        
        # Color & emoji by class
        if class_name == "WEED":
            color, bg_color, emoji = "#FF4757", "#FF6B7A", "🌿"
        else:  # CROP
            color, bg_color, emoji = "#2ED573", "#51CF66", "🌾"
        
        # Bounding box
        draw.rectangle([x1,y1,x2,y2], outline=color, width=5)
        
        # Confidence label
        label = f"{emoji} {class_name} {conf:.0%}"
        bbox = draw.textbbox((0,0), label, font=font)
        label_w = bbox[2] - bbox[0] + 16
        label_h = bbox[3] - bbox[1] + 10
        label_x = x1
        label_y = y1 - label_h
        
        # Label background
        draw.rectangle([label_x, label_y, label_x+label_w, label_y+label_h], fill=bg_color)
        draw.text((label_x+8, label_y+5), label, fill="white", font=font)
    
    return img

# Load model
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("✅ Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"❌ Model hatası: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLS
st.markdown("""
<div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); 
            border-radius: 20px; padding: 2rem; margin-bottom: 2rem;'>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    uploaded = st.file_uploader("📁 Tarla Fotoğrafı", type=["jpg","png","jpeg"])
with col2:
    threshold = st.slider("🎯 Threshold", 0.1, 0.9, 0.4, 0.05)
with col3:
    iou_thresh = st.slider("🔗 NMS IoU", 0.3, 0.8, 0.5, 0.05)
with col4:
    debug_mode = st.checkbox("🔍 Debug Modu", value=False)
st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING
if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    
    with st.spinner("🔍 AI Analizi Yapılıyor..."):
        # Preprocess
        resized = original_img.resize((640, 640))
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        preds = np.transpose(output_data, (1, 0))
    
    # 🔥 UNIVERSAL DETECTION (ÇALIŞIR!)
    boxes_all, scores_all, classes_all, areas_all = [], [], [], []
    
    st.info(f"📊 Toplam {len(preds)} prediction işlendi")
    
    for row_idx, row in enumerate(preds):
        x, y, bw, bh, ws, cs = row  # ws=weed_score, cs=crop_score
        
        # Max confidence al + class belirle
        if ws > cs:
            max_conf = ws
            detected_class = "WEED"
        else:
            max_conf = cs
            detected_class = "CROP"
        
        if max_conf >= threshold:
            # Box koordinatları
            x1 = max(0, int((x - bw/2) * w))
            y1 = max(0, int((y - bh/2) * h))
            x2 = min(w, int((x + bw/2) * w))
            y2 = min(h, int((y + bh/2) * h))
            
            boxes_all.append([x1, y1, x2, y2])
            scores_all.append(max_conf)
            classes_all.append(detected_class)
            areas_all.append((x2-x1) * (y2-y1))
    
    # NMS uygula
    keep_indices = non_max_suppression(boxes_all, scores_all, iou_thresh) if boxes_all else []
    
    # Metrics hesapla
    weed_count = sum(1 for i in keep_indices if classes_all[i] == "WEED")
    crop_count = sum(1 for i in keep_indices if classes_all[i] == "CROP")
    total_detections = len(keep_indices)
    
    total_area = w * h
    weed_area = sum(areas_all[i] for i in keep_indices if classes_all[i] == "WEED")
    weed_density = (weed_area / total_area) * 100 if total_area > 0 else 0
    
    # Draw detections
    result_img = original_img.copy()
    if keep_indices:
        result_img = draw_detections(result_img, boxes_all, scores_all, classes_all, keep_indices)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # GLASS METRIC CARDS
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌿</div>
            <h1 class="metric-value" style="color: #FF4757;">{weed_count}</h1>
            <p class="metric-label">WEEDS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌾</div>
            <h1 class="metric-value" style="color: #2ED573;">{crop_count}</h1>
            <p class="metric-label">CROPS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">📦</div>
            <h1 class="metric-value">{total_detections}</h1>
            <p class="metric-label">TOTAL</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">📊</div>
            <h1 class="metric-value" style="color: #FFA502;">{weed_density:.1f}%</h1>
            <p class="metric-label">WEED DENSITY</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # IMAGE GALLERY
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original_img, caption="📸 Orijinal", use_container_width=True)
    with col_img2:
        st.image(result_img, caption="🎯 Sonuç", use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DOWNLOAD
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button(
        label=f"💾 İndir ({timestamp})",
        data=buf.getvalue(),
        file_name=f"agri_vision_{timestamp}.png",
        mime="image/png",
        use_container_width=True
    )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DEBUG MODU
    if debug_mode or weed_count + crop_count == 0:
        st.subheader("🔍 DEBUG BİLGİLERİ")
        
        # Model shape
        st.write("**Model Çıktısı Shape:**", preds.shape)
        st.write("**Örnek predictions (ilk 5): **")
        st.dataframe(preds[:5])
        
        # Confidence histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        weed_scores = preds[:, 4]  # ws
        crop_scores = preds[:, 5]  # cs
        
        ax1.hist(weed_scores, bins=30, alpha=0.7, color='red', label='Weed Scores')
        ax1.axvline(threshold, color='orange', linestyle='--', label=f'Threshold {threshold}')
        ax1.set_title('🔴 Weed Confidence')
        ax1.legend()
        
        ax2.hist(crop_scores, bins=30, alpha=0.7, color='green', label='Crop Scores')
        ax2.axvline(threshold, color='orange', linestyle='--', label=f'Threshold {threshold}')
        ax2.set_title('🟢 Crop Confidence')
        ax2.legend()
        
        st.pyplot(fig)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Max Weed", f"{np.max(weed_scores):.3f}")
        with col2: st.metric("Max Crop", f"{np.max(crop_scores):.3f}")
        with col3: st.metric("Avg Weed", f"{np.mean(weed_scores):.3f}")
        with col4: st.metric("Avg Crop", f"{np.mean(crop_scores):.3f}")
        
        st.info(f"""
        **Debug Sonuçları:**
        - Toplam detection: {len(boxes_all)} 
        - NMS sonrası: {total_detections}
        - Weed: {weed_count}, Crop: {crop_count}
        - Öneri: Threshold {'düşür' if total_detections==0 else 'artır'}
        """)

st.markdown("""
<div style='text-align: center; padding: 3rem; color: rgba(255,255,255,0.8); 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px; margin: 3rem 0;'>
    <h2>🌾 Precision Farming AI</h2>
    <p>Ot & Mahsul Ayrımı • Gerçek Zamanlı • Üretim Hazır</p>
</div>
""", unsafe_allow_html=True)
