from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import streamlit.components.v1 as components

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

# Glassmorphism CSS
GLASS_CSS = """
<style>
.glass-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-5px);
}
.metric-icon {
    font-size: 3em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    margin: 0;
}
.metric-label {
    color: rgba(255,255,255,0.8);
    margin: 0;
}
.download-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 15px;
    padding: 1rem 2rem;
    color: white;
    font-size: 1.1em;
    font-weight: bold;
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    transition: all 0.3s ease;
}
.download-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(102,126,234,0.6);
}
</style>
"""

st.set_page_config(page_title="🌱 AgriVision AI Pro", layout="wide", page_icon="🌱")
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# Title with glass effect
st.markdown("""
<div style='
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
'>
    <h1 style='color: white; margin: 0; font-size: 3em;'>🌱 AgriVision AI Pro</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 1.2em;'>Precision Weed Detection & Crop Intelligence</p>
</div>
""", unsafe_allow_html=True)

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

def draw_detections(img, boxes, scores, keep_indices):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        
        # 🌿 WEED vs 🌾 CROP
        if conf > 0.5:
            color, class_name, bg_color = "#FF4757", "🌿 WEED", "#FF6B7A"
        else:
            color, class_name, bg_color = "#2ED573", "🌾 CROP", "#51CF66"
        
        # Bounding box
        draw.rectangle([x1,y1,x2,y2], outline=color, width=5)
        
        # Glass label
        label = f"{class_name} {conf:.0%}"
        bbox = draw.textbbox((0,0), label, font=font)
        label_w = bbox[2] - bbox[0] + 16
        label_h = bbox[3] - bbox[1] + 10
        label_x, label_y = x1, y1 - label_h
        
        # Glass background
        draw.rectangle([label_x, label_y, label_x+label_w, label_y+label_h], 
                      fill=bg_color)
        draw.text((label_x+8, label_y+5), label, fill="white", font=font)
    
    return img

# Load model
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"❌ Model Error: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# GLASS CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); 
            border-radius: 20px; padding: 2rem; margin-bottom: 2rem;'>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,1,1])

with col1:
    uploaded = st.file_uploader("📁 Upload Field Image", type=["jpg","png","jpeg"], 
                               help="Upload high-resolution field photo")

with col2:
    threshold = st.slider("🎯 Confidence Threshold", 0.3, 0.9, 0.5, 0.05)

with col3:
    iou_thresh = st.slider("🔗 NMS IoU Threshold", 0.3, 0.8, 0.5, 0.05)

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    
    with st.spinner("🔍 AI Analysis in Progress..."):
        resized = original_img.resize((640, 640))
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        preds = np.transpose(output, (1, 0))
    
    # Collect detections
    boxes, scores, areas = [], [], []
    for row in preds:
        x,y,bw,bh,ws,_ = row
        if ws >= threshold:
            x1 = max(0, int((x-bw/2)*w))
            y1 = max(0, int((y-bh/2)*h))
            x2 = min(w, int((x+bw/2)*w))
            y2 = min(h, int((y+bh/2)*h))
            boxes.append([x1,y1,x2,y2])
            scores.append(ws)
            areas.append((x2-x1)*(y2-y1))
    
    keep_indices = non_max_suppression(boxes, scores, iou_thresh) if boxes else []
    result_img = original_img.copy()
    result_img = draw_detections(result_img, boxes, scores, keep_indices)
    
    # Density calculation
    total_area = w * h
    weed_pixels = sum(areas[i] for i in keep_indices if scores[i] > 0.5)
    crop_pixels = sum(areas[i] for i in keep_indices if scores[i] <= 0.5)
    density_pct = (weed_pixels / total_area) * 100
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # GLASSMORPHISM METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════════════
    col1, col2, col3, col4 = st.columns(4)
    
    weeds_count = len([i for i in keep_indices if scores[i]>0.5])
    crops_count = len([i for i in keep_indices if scores[i]<=0.5])
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌿</div>
            <h1 class="metric-value" style="color: #FF4757;">{weeds_count}</h1>
            <p class="metric-label">WEEDS DETECTED</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🌾</div>
            <h1 class="metric-value" style="color: #2ED573;">{crops_count}</h1>
            <p class="metric-label">HEALTHY CROPS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">📊</div>
            <h1 class="metric-value" style="color: #FFA502;">{density_pct:.1f}%</h1>
            <p class="metric-label">WEED DENSITY</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-icon">🎯</div>
            <h1 class="metric-value" style="color: #3742FA;">{np.mean(scores)*100:.0f}%</h1>
            <p class="metric-label">AVG CONFIDENCE</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # IMAGE GALLERY
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
        """, unsafe_allow_html=True)
        st.image(original_img, caption="📸 Original Field", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_img2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
        """, unsafe_allow_html=True)
        st.image(result_img, caption="🎯 AI Analysis Result", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SHINY DOWNLOAD + HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════════
    col_dl1, col_dl2 = st.columns([1,2])
    
    with col_dl1:
        # Timestamped Download
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.markdown(f"""
        <div style='text-align: center; margin-top: 1rem;'>
            <button class="download-btn">
                💾 Download Analysis<br>
                <small>agri_vision_{timestamp}.png</small>
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label=f"💾 Download {timestamp}",
            data=buf.getvalue(),
            file_name=f"agri_vision_{timestamp}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col_dl2:
        # 🔥 HEATMAP DENSITY
        if keep_indices and weeds_count > 0:
            density_map = np.zeros((h//32, w//32))
            for i in keep_indices:
                if scores[i] > 0.5:
                    x1,y1,x2,y2 = [int(coord//32) for coord in boxes[i]]
                    density_map[max(0,y1):min(density_map.shape[0],y2), 
                               max(0,x1):min(density_map.shape[1],x2)] += 1
            
            fig = px.imshow(density_map, 
                          color_continuous_scale='Reds',
                          title="🔴 Weed Density Heatmap (Red = High Concentration)",
                          labels={'color': 'Weed Density'})
            fig.update_layout(height=400, margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style='text-align: center; padding: 3rem; color: rgba(255,255,255,0.7); 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px; margin-top: 3rem;'>
    <h3>🌾 Precision Agriculture Intelligence</h3>
    <p>Real-time detection • Enterprise ready • Built for farmers</p>
</div>
""", unsafe_allow_html=True)
