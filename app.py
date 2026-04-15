from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import io

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

st.set_page_config(page_title="AgriVision AI", layout="wide")
st.title("🌱 AgriVision AI — Professional Weed Detection")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def iou(box1, box2):
    """Intersection over Union for NMS"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def non_max_suppression(boxes, scores, threshold=0.5):
    """Apply NMS to filter overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        remaining = indices[1:]
        iou_scores = [iou(boxes[current], boxes[i]) for i in remaining]
        
        indices = remaining[np.array(iou_scores) <= threshold]
    
    return keep

def draw_detections(img, boxes, scores, keep_indices, class_names):
    """Draw bounding boxes with labels and confidence"""
    draw = ImageDraw.Draw(img)
    
    # Try to load font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        confidence = scores[idx]
        class_name = class_names[0] if confidence > threshold else class_names[1]
        color = "lime" if class_name == "Crop" else "red"
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw filled rectangle for label background
        label = f"{class_name}: {confidence:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_x = x1
        label_y = y1 - label_h - 5
        
        draw.rectangle([label_x, label_y, label_x + label_w + 10, label_y + label_h + 5], 
                      fill=color)
        draw.text((label_x + 5, label_y + 2), label, fill="white", font=font)
    
    return img

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# UI Controls
col1, col2, col3 = st.columns(3)
with col1:
    uploaded = st.file_uploader("📁 Upload Image", type=["jpg","png","jpeg"])
with col2:
    threshold = st.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
with col3:
    iou_threshold = st.slider("🔗 NMS IoU Threshold", 0.1, 0.9, 0.5, 0.05)

class_names = ["Weed", "Crop"]  # Adjust based on your model classes

if uploaded:
    # Original image
    original_img = Image.open(uploaded).convert("RGB")
    w, h = original_img.size
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_img, caption="📸 Original", use_container_width=True)
    
    with col2:
        # Process image
        resized = original_img.resize((640, 640))
        arr = np.array(resized).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        preds = np.transpose(output[0], (1, 0))
        
        # Collect all detections
        boxes = []
        scores = []
        
        for row in preds:
            x, y, bw, bh, ws, cs = row
            
            # Weed confidence
            if ws >= threshold:
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                
                boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
                scores.append(ws)
        
        # Apply NMS
        if boxes:
            keep_indices = non_max_suppression(boxes, scores, iou_threshold)
        else:
            keep_indices = []
        
        # Draw detections
        result_img = original_img.copy()
        result_img = draw_detections(result_img, boxes, scores, keep_indices, class_names)
        
        st.image(result_img, caption="🎯 Detection Result", use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    weed_count = sum(1 for i in keep_indices if scores[i] > threshold)
    crop_count = len(keep_indices) - weed_count
    
    with col1:
        st.metric("🌿 Weeds", weed_count)
    with col2:
        st.metric("🌾 Crops", crop_count)
    with col3:
        st.metric("📦 Total Detections", len(keep_indices))
    with col4:
        st.metric("🎯 Avg Confidence", f"{np.mean(scores):.2f}")
    
    # Detection Details
    if keep_indices:
        st.subheader("📋 Detection Details")
        details_data = []
        for i in keep_indices:
            class_name = "Weed" if scores[i] > threshold else "Crop"
            details_data.append({
                "Class": class_name,
                "Confidence": f"{scores[i]:.2f}",
                "Position": f"({boxes[i][0]:.0f},{boxes[i][1]:.0f})"
            })
        
        st.dataframe(details_data)
    
    # Download result
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    st.download_button(
        label="💾 Download Result",
        data=buf.getvalue(),
        file_name="weed_detection_result.png",
        mime="image/png"
    )

# Future Features Teaser
st.markdown("---")
st.markdown("""
## 🚀 **Coming Soon**
- **🎥 Video Detection** 
- **📈 Weed Density Heatmap**
- **🌍 Multi-field Analysis**
- **📱 Mobile App**
- **⚡ Real-time Processing**

**Built with ❤️ for Precision Agriculture**
""")
