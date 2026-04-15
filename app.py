from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import cv2
import pandas as pd

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
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
.metric-icon { font-size: 2.5em; margin-bottom: 0.3rem; }
.metric-value { font-size: 2.2em; font-weight: 700; margin: 0; }
.metric-label { color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9em; }
.tab-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 0.8rem 1.5rem;
    border-radius: 15px;
    color: white;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: center;
}
.detection-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    border-radius: 10px;
    font-weight: bold;
    color: white;
}
</style>
"""

st.set_page_config(
    page_title="🌱 AgriVision AI Pro", 
    layout="wide", 
    page_icon="🌱",
    initial_sidebar_state="expanded"
)
st.markdown(GLASS_CSS, unsafe_allow_html=True)

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102,126,234,0.4);'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🌱 AgriVision AI Pro</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
        Weed & Crop Detection | Heatmap | Video | PDF Report | GPS
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


# ═══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def containment(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(1, (x2 - x1) * (y2 - y1))
    area2 = max(1, (x2b - x1b) * (y2b - y1b))
    smaller_area = min(area1, area2)
    return inter / smaller_area if smaller_area > 0 else 0

def class_aware_nms(boxes, scores, classes, iou_threshold=0.2, containment_threshold=0.6):
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        label = f"{emoji} {cls} {conf:.0%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0] + 16
        lh = bbox[3] - bbox[1] + 10
        lx, ly = x1, y1 - lh
        draw.rectangle([lx, ly, lx + lw, ly + lh], fill=bg_color)
        draw.text((lx + 8, ly + 5), label, fill="white", font=font)
    return img

def process_image(img, interpreter, input_details, output_details,
                  threshold, size_threshold, nms_iou):
    w, h = img.size
    total_area = w * h
    resized = img.resize((640, 640))
    arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, 0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    preds = np.transpose(output_data, (1, 0))
    
    raw_boxes, raw_scores = [], []
    for row in preds:
        x, y, bw, bh = row[0], row[1], row[2], row[3]
        class_scores = row[4:]
        best_score = np.max(class_scores)
        if best_score < threshold: continue
        x1 = max(0, int((x - bw / 2) * w))
        y1 = max(0, int((y - bh / 2) * h))
        x2 = min(w, int((x + bw / 2) * w))
        y2 = min(h, int((y + bh / 2) * h))
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            raw_boxes.append([x1, y1, x2, y2])
            raw_scores.append(float(best_score))
    
    boxes_all, scores_all, classes_all, areas_all = [], [], [], []
    size_limit = total_area * (size_threshold / 100)
    for i in range(len(raw_boxes)):
        x1, y1, x2, y2 = raw_boxes[i]
        box_area = (x2 - x1) * (y2 - y1)
        detected_class = "WEED" if box_area < size_limit else "CROP"
        boxes_all.append(raw_boxes[i])
        scores_all.append(raw_scores[i])
        classes_all.append(detected_class)
        areas_all.append(box_area)
    
    keep_indices = class_aware_nms(boxes_all, scores_all, classes_all,
                                    iou_threshold=nms_iou, containment_threshold=0.6)
    return boxes_all, scores_all, classes_all, areas_all, keep_indices

def generate_heatmap(boxes, scores, classes, keep_indices, w, h):
    grid_size = 20
    rows = h // grid_size + 1
    cols = w // grid_size + 1
    density_map = np.zeros((rows, cols))
    for i in keep_indices:
        if classes[i] == "WEED":
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            r1 = max(0, y1 // grid_size)
            r2 = min(rows, y2 // grid_size + 1)
            c1 = max(0, x1 // grid_size)
            c2 = min(cols, x2 // grid_size + 1)
            density_map[r1:r2, c1:c2] += conf
    return density_map

def generate_pdf_report(result_img, weed_count, crop_count, weed_density,
                        avg_conf, boxes, scores, classes, keep_indices,
                        gps_lat, gps_lon):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "AgriVision AI Pro - Field Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    if gps_lat and gps_lon:
        pdf.cell(0, 8, f"GPS: {gps_lat:.6f}, {gps_lon:.6f}", ln=True)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        result_img.save(tmp.name)
        pdf.image(tmp.name, x=10, w=190)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Detection Summary", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(95, 8, f"Weeds Detected: {weed_count}", border=1)
    pdf.cell(95, 8, f"Crops Detected: {crop_count}", border=1, ln=True)
    pdf.cell(95, 8, f"Weed Density: {weed_density:.1f}%", border=1)
    pdf.cell(95, 8, f"Avg Confidence: {avg_conf:.0%}", border=1, ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detection Details", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(20, 8, "#", border=1, align="C")
    pdf.cell(40, 8, "Class", border=1, align="C")
    pdf.cell(40, 8, "Confidence", border=1, align="C")
    pdf.cell(50, 8, "Position", border=1, align="C")
    pdf.cell(40, 8, "Size", border=1, align="C", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    for idx, i in enumerate(keep_indices):
        x1, y1, x2, y2 = boxes[i]
        pdf.cell(20, 8, str(idx + 1), border=1, align="C")
        pdf.cell(40, 8, classes[i], border=1, align="C")
        pdf.cell(40, 8, f"{scores[i]:.0%}", border=1, align="C")
        pdf.cell(50, 8, f"({x1},{y1})-({x2},{y2})", border=1, align="C")
        pdf.cell(40, 8, f"{(x2-x1)}x{(y2-y1)}", border=1, align="C", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    if weed_density > 20:
        pdf.multi_cell(0, 7, "HIGH ALERT: Weed density above 20%. Immediate herbicide application recommended.")
    elif weed_density > 10:
        pdf.multi_cell(0, 7, "WARNING: Moderate weed presence. Targeted spot treatment recommended.")
    elif weed_count > 0:
        pdf.multi_cell(0, 7, "LOW RISK: Minor weed presence. Monitor and manual removal suggested.")
    else:
        pdf.multi_cell(0, 7, "ALL CLEAR: No weeds detected. Field is healthy.")
    
    return pdf.output(dest='S').encode('latin-1')


# ═══════════════════════════════════════════════════════════════
# MODEL YÜKLE
# ═══════════════════════════════════════════════════════════════
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"❌ Model hatası: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem;'>
        <h2>🎛️ Ayarlar</h2>
    </div>
    """, unsafe_allow_html=True)
    
    threshold = st.slider("🎯 Confidence", 0.1, 0.9, 0.60, 0.05)
    size_threshold = st.slider("📏 Boyut Eşiği (%)", 5, 50, 42, 1)
    nms_iou = st.slider("🔗 NMS IoU", 0.1, 0.7, 0.20, 0.05)
    
    st.markdown("---")
    st.markdown("### 🗺️ GPS Koordinatları")
    gps_lat = st.number_input("📍 Enlem", value=39.9334, format="%.6f")
    gps_lon = st.number_input("📍 Boylam", value=32.8597, format="%.6f")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:1rem; background:rgba(255,255,255,0.1); 
                border-radius:10px;'>
        <p style='margin:0; font-size:0.8em;'>
            🌿 Küçük = WEED<br>
            🌾 Büyük = CROP
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SEKMELER
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📸 Fotoğraf Analizi", "🎥 Video Analizi", "🗺️ GPS Harita"])


# ═══════════════════════════════════════════════════════════════
# TAB 1: FOTOĞRAF
# ═══════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("📁 Tarla Fotoğrafı Yükle", type=["jpg", "png", "jpeg"])
    
    if uploaded:
        original_img = Image.open(uploaded).convert("RGB")
        w, h = original_img.size
        total_area = w * h
        
        with st.spinner("🔍 AI Analizi..."):
            boxes_all, scores_all, classes_all, areas_all, keep_indices = process_image(
                original_img, interpreter, input_details, output_details,
                threshold, size_threshold, nms_iou
            )
        
        result_img = original_img.copy()
        if keep_indices:
            result_img = draw_detections(result_img, boxes_all, scores_all,
                                         classes_all, keep_indices)
        
        weed_count = sum(1 for i in keep_indices if classes_all[i] == "WEED")
        crop_count = sum(1 for i in keep_indices if classes_all[i] == "CROP")
        weed_area = sum(areas_all[i] for i in keep_indices if classes_all[i] == "WEED")
        weed_density = (weed_area / total_area) * 100 if total_area > 0 else 0
        avg_conf = np.mean([scores_all[i] for i in keep_indices]) if keep_indices else 0
        
        # Glass Kartlar
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
        
        # Görüntüler
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(original_img, caption="📸 Orijinal", use_container_width=True)
        with col_img2:
            st.image(result_img, caption="🎯 Tespit", use_container_width=True)
        
        # Heatmap
        st.markdown("### 📈 Yabancı Ot Yoğunluk Haritası")
        if weed_count > 0:
            density_map = generate_heatmap(boxes_all, scores_all, classes_all,
                                           keep_indices, w, h)
            fig = px.imshow(
                density_map,
                color_continuous_scale='RdYlGn_r',
                title="🔴 Kırmızı = Yüksek | 🟢 Yeşil = Düşük",
                labels={'color': 'Ot Yoğunluğu'}
            )
            fig.update_layout(height=400, margin=dict(r=0, t=40, l=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Yabancı ot bulunamadı!")
        
        # Detaylar
        if keep_indices:
            st.markdown("### 📋 Tespit Detayları")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**🌿 Yabancı Otlar:**")
                weed_found = False
                for i in keep_indices:
                    if classes_all[i] == "WEED":
                        weed_found = True
                        area_pct = (areas_all[i] / total_area) * 100
                        st.markdown(f"""
                        <div class="detection-badge" style="background:#FF4757;">
                            🌿 WEED {scores_all[i]:.0%} | Alan: {area_pct:.1f}%
                        </div>""", unsafe_allow_html=True)
                if not weed_found:
                    st.success("✅ Ot yok!")
            with col_d2:
                st.markdown("**🌾 Mahsuller:**")
                crop_found = False
                for i in keep_indices:
                    if classes_all[i] == "CROP":
                        crop_found = True
                        area_pct = (areas_all[i] / total_area) * 100
                        st.markdown(f"""
                        <div class="detection-badge" style="background:#2ED573;">
                            🌾 CROP {scores_all[i]:.0%} | Alan: {area_pct:.1f}%
                        </div>""", unsafe_allow_html=True)
                if not crop_found:
                    st.warning("⚠️ Mahsul bulunamadı!")
        
        # İndirme
        st.markdown("### 💾 İndirme Seçenekleri")
        col_dl1, col_dl2 = st.columns(2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with col_dl1:
            buf_img = io.BytesIO()
            result_img.save(buf_img, format='PNG')
            st.download_button(
                label="🖼️ Görüntü İndir (PNG)",
                data=buf_img.getvalue(),
                file_name=f"agrivision_{timestamp}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_dl2:
            try:
                pdf_bytes = generate_pdf_report(
                    result_img, weed_count, crop_count, weed_density, avg_conf,
                    boxes_all, scores_all, classes_all, keep_indices,
                    gps_lat, gps_lon
                )
                if pdf_bytes:
                    st.download_button(
                        label="📊 PDF Rapor İndir",
                        data=pdf_bytes,
                        file_name=f"agrivision_report_{timestamp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.info("📦 PDF için: `pip install fpdf2`")
            except Exception as e:
                st.warning(f"PDF hatası: {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 2: VİDEO
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="tab-header">🎥 Video Analizi</div>
    """, unsafe_allow_html=True)
    
    video_file = st.file_uploader("🎥 Video Yükle", type=["mp4", "avi", "mov"])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        
        st.info(f"📹 {total_frames} kare | {fps} FPS | ~{total_frames // fps}s")
        
        frame_skip = st.slider("⏭️ Her kaç karede analiz?", 5, 60, 30, 5)
        
        if st.button("🚀 Video Analizi Başlat", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            result_placeholder = st.empty()
            
            all_results = []
            frame_idx = 0
            processed = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    
                    boxes, sc, cls, areas, keep = process_image(
                        pil_frame, interpreter, input_details, output_details,
                        threshold, size_threshold, nms_iou
                    )
                    
                    wc = sum(1 for i in keep if cls[i] == "WEED")
                    cc = sum(1 for i in keep if cls[i] == "CROP")
                    
                    all_results.append({
                        'frame': frame_idx,
                        'time': frame_idx / fps,
                        'weeds': wc,
                        'crops': cc
                    })
                    
                    if keep:
                        rf = pil_frame.copy()
                        rf = draw_detections(rf, boxes, sc, cls, keep)
                        result_placeholder.image(rf, caption=f"Kare {frame_idx}",
                                                use_container_width=True)
                    
                    processed += 1
                    progress.progress(min(frame_idx / total_frames, 1.0))
                    status.text(f"İşlenen: {processed} | Weed: {wc} | Crop: {cc}")
                
                frame_idx += 1
            
            cap.release()
            progress.progress(1.0)
            status.success(f"✅ {processed} kare analiz edildi!")
            
            if all_results:
                st.markdown("### 📈 Video Sonuçları")
                times = [r['time'] for r in all_results]
                weeds = [r['weeds'] for r in all_results]
                crops = [r['crops'] for r in all_results]
                
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=times, y=weeds, mode='lines+markers',
                                           name='🌿 Weeds', line=dict(color='#FF4757', width=3)))
                fig_v.add_trace(go.Scatter(x=times, y=crops, mode='lines+markers',
                                           name='🌾 Crops', line=dict(color='#2ED573', width=3)))
                fig_v.update_layout(title="Zaman İçinde Tespit",
                                    xaxis_title="Saniye", yaxis_title="Sayı", height=400)
                st.plotly_chart(fig_v, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("🌿 Toplam Weed", sum(weeds))
                with col2: st.metric("🌾 Toplam Crop", sum(crops))
                with col3: st.metric("📊 Max Weed/Kare", max(weeds) if weeds else 0)
    else:
        st.info("🎥 MP4, AVI veya MOV dosyası yükleyin")


# ═══════════════════════════════════════════════════════════════
# TAB 3: GPS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="tab-header">🗺️ GPS Konum & Tarla Haritası</div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 Tarla Konumu")
    st.map(pd.DataFrame({'lat': [gps_lat], 'lon': [gps_lon]}), zoom=14)
    
    st.markdown(f"""
    <div class="glass-card" style="margin-top:1rem;">
        <h3>📍 Koordinatlar</h3>
        <p><b>Enlem:</b> {gps_lat:.6f} | <b>Boylam:</b> {gps_lon:.6f}</p>
        <p><b>Tarih:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Çoklu Alan Takibi")
    num_fields = st.number_input("Kaç tarla?", 1, 10, 3)
    
    field_data = []
    for i in range(num_fields):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(f"Tarla {i+1} Adı", value=f"Tarla-{i+1}", key=f"n_{i}")
        with col2:
            lat = st.number_input(f"Enlem {i+1}", value=gps_lat + (i * 0.002),
                                  format="%.6f", key=f"la_{i}")
        with col3:
            lon = st.number_input(f"Boylam {i+1}", value=gps_lon + (i * 0.002),
                                  format="%.6f", key=f"lo_{i}")
        field_data.append({'lat': lat, 'lon': lon})
    
    if field_data:
        st.map(pd.DataFrame(field_data), zoom=13)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:2rem; margin-top:3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius:20px; color:white;'>
    <h3>🌾 AgriVision AI Pro v2.0</h3>
    <p>📸 Fotoğraf | 🎥 Video | 📈 Heatmap | 📊 PDF | 🗺️ GPS</p>
</div>
""", unsafe_allow_html=True)
