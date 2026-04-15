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

st.set_page_config(page_title="AgriVision AI Pro", layout="wide", page_icon="🌱")
st.markdown("""
# 🌱 **AgriVision AI Pro** 
### Precision Weed Detection & Crop Analysis
""")

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
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        
        # WEED (Red) vs CROP (Green)
        if conf > 0.5:  # Weed threshold
            color, class_name = "red", "🌿 WEED"
            color_bg = "#FF4444"
        else:
            color, class_name = "lime", "🌾 CROP" 
            color_bg = "#44FF44"
        
        # Bounding box
        draw.rectangle([x1,y1,x2,y2], outline=color, width=4)
        
        # Label background
        label = f"{class_name} {conf:.1%}"
        bbox = draw.textbbox((0,0), label, font=font)
        label_w = bbox[2] - bbox[0] + 12
        label_h = bbox[3] - bbox[1] + 8
        label_x, label_y = x1, y1 - label_h
        
        draw.rectangle([label_x, label_y, label_x+label_w, label_y+label_h], fill=color_bg)
        draw.text((label_x+6, label_y+4), label, fill="white", font=font)
    
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
# CONTROLS & UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
col1, col2, col3 = st.columns([2,1,1])

with col1:
    uploaded = st.file_uploader("📁 Upload Field Image", type=["jpg","png","jpeg"], 
                               help="Upload high-res field photo")

with col2:
    threshold = st.slider("🎯 Confidence", 0.3, 0.9, 0.5, 0.05)

with col3:
    iou_thresh = st.slider("🔗 NMS IoU", 0.3, 0.8, 0.5, 0.05)

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded:
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    
    # PROCESS IMAGE
    with st.spinner("🔍 Analyzing field..."):
        resized = original_img.resize((640, 640))
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        preds = np.transpose(output, (1, 0))
    
    # DETECTION COLLECTION
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
    
    # NMS
    keep_indices = non_max_suppression(boxes, scores, iou_thresh) if boxes else []
    
    # FINAL IMAGE
    result_img = original_img.copy()
    result_img = draw_detections(result_img, boxes, scores, keep_indices)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SHINY METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════════════
    col_a, col_b, col_c, col_d = st.columns(4)
    
    # WEED DENSITY (NEW!)
    total_area = w * h
    weed_pixels = sum(areas[i] for i in keep_indices if scores[i] > 0.5)
    density = (weed_pixels / total_area) * 100
    
    with col_a:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #FF6B6B, #FF8E8E); 
                    border-radius: 15px; box-shadow: 0 8px 32px rgba(255,107,107,0.3);'>
            <h2 style='color: white; margin: 0;'>🌿 WEEDS</h2>
            <h1 style='color: white; margin: 0; font-size: 2.5em;'>
                """ + str(len([i for i in keep_indices if scores[i]>0.5])) + """
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #4ECDC4, #6EE7B7); 
                    border-radius: 15px; box-shadow: 0 8px 32px rgba(78,205,196,0.3);'>
            <h2 style='color: white; margin: 0;'>🌾 CROPS</h2>
            <h1 style='color: white; margin: 0; font-size: 2.5em;'>
                {len([i for i in keep_indices if scores[i]<=0.5])}
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #FFD23F, #FFED4E); 
                    border-radius: 15px; box-shadow: 0 8px 32px rgba(255,210,63,0.3);'>
            <h2 style='color: #333; margin: 0;'>📊 DENSITY</h2>
            <h1 style='color: #333; margin: 0; font-size: 2.5em;'>
                {density:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col_d:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #667eea, #764ba2); 
                    border-radius: 15px; box-shadow: 0 8px 32px rgba(102,126,234,0.3);'>
            <h2 style='color: white; margin: 0;'>🎯 CONFIDENCE</h2>
            <h1 style='color: white; margin: 0; font-size: 2.5em;'>
                {np.mean(scores):.0%}
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # IMAGE GALLERY + DOWNLOAD
    # ═══════════════════════════════════════════════════════════════════════════════
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.image(original_img, caption="📸 Original Field", use_container_width=True)
    
    with col_img2:
        st.image(result_img, caption="🎯 Detected", use_container_width=True)
    
    # DOWNLOAD BUTTON (Shiny!)
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    st.download_button(
        label="💾 Download Annotated Image",
        data=buf.getvalue(),
        file_name=f"agri_vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        use_container_width=True,
        help="Download high-res annotated image"
    )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DENSITY VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("📈 Weed Density Heatmap")
    
    if keep_indices:
        # Create density map
        density_map = np.zeros((h//20, w//20))
        for i in keep_indices:
            if scores[i] > 0.5:  # Only weeds
                x1,y1,x2,y2 = [coord//20 for coord in boxes[i]]
                density_map[y1:y2, x1:x2] += 1
        
        fig = px.imshow(density_map, color_continuous_scale='Reds', 
                       title="🔴 Red = High Weed Density")
        fig.update_layout(width=800, height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <h3>🌾 Built for Precision Agriculture</h3>
    <p>Real-time weed detection | Multi-model support | Enterprise ready</p>
</div>
""", unsafe_allow_html=True)
