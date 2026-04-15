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
.history-card {
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
}
</style>
"""

st.set_page_config(
    page_title="AgriVision AI Pro",
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
        Weed & Crop Detection | Heatmap | Video | Analytics | GPS
    </p>
</div>
""", unsafe_allow_html=True)

# Session state for history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []


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
    if not boxes:
        return []
    unique_classes = list(set(classes))
    all_keep = []
    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        cls_boxes = [boxes[i] for i in cls_indices]
        cls_scores = [scores[i] for i in cls_indices]
        if not cls_boxes:
            continue
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
        if best_score < threshold:
            continue
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


def generate_csv_report(weed_count, crop_count, weed_density, avg_conf,
                        boxes, scores, classes, keep_indices, gps_lat, gps_lon):
    """CSV rapor olustur (PDF yerine!)"""
    rows = []
    for idx, i in enumerate(keep_indices):
        x1, y1, x2, y2 = boxes[i]
        rows.append({
            'No': idx + 1,
            'Class': classes[i],
            'Confidence': f"{scores[i]:.2f}",
            'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2,
            'Width': x2 - x1,
            'Height': y2 - y1,
            'Area': (x2 - x1) * (y2 - y1)
        })
    
    df = pd.DataFrame(rows)
    
    # Ozet satiri ekle
    summary = pd.DataFrame([{
        'No': '---',
        'Class': 'SUMMARY',
        'Confidence': f"Avg: {avg_conf:.2f}",
        'X1': f"Weeds: {weed_count}",
        'Y1': f"Crops: {crop_count}",
        'X2': f"Density: {weed_density:.1f}%",
        'Y2': f"GPS Lat: {gps_lat:.6f}",
        'Width': f"GPS Lon: {gps_lon:.6f}",
        'Height': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Area': '---'
    }])
    
    df = pd.concat([df, summary], ignore_index=True)
    return df.to_csv(index=False).encode('utf-8')


# ═══════════════════════════════════════════════════════════════
# MODEL YUKLE
# ═══════════════════════════════════════════════════════════════
try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model hatasi: {e}")
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
    size_threshold = st.slider("📏 Boyut Esigi (%)", 5, 50, 42, 1)
    nms_iou = st.slider("🔗 NMS IoU", 0.1, 0.7, 0.20, 0.05)

    st.markdown("---")
    st.markdown("### 🗺️ GPS Koordinatlari")
    gps_lat = st.number_input("📍 Enlem", value=39.9334, format="%.6f")
    gps_lon = st.number_input("📍 Boylam", value=32.8597, format="%.6f")

    st.markdown("---")
    
    # Tarama gecmisi
    st.markdown("### 📜 Tarama Gecmisi")
    if st.session_state.scan_history:
        for scan in st.session_state.scan_history[-5:][::-1]:
            weed_color = "#FF4757" if scan['weeds'] > 0 else "#2ED573"
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05); padding:0.5rem; 
                        border-radius:8px; margin:0.3rem 0; border-left:3px solid {weed_color};'>
                <small>{scan['time']}</small><br>
                <b>🌿 {scan['weeds']}</b> | <b>🌾 {scan['crops']}</b> | <b>{scan['density']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Henuz tarama yok")
    
    if st.button("🗑️ Gecmisi Temizle", use_container_width=True):
        st.session_state.scan_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:1rem; background:rgba(255,255,255,0.1); 
                border-radius:10px;'>
        <p style='margin:0; font-size:0.8em;'>
            🌿 Kucuk = WEED<br>
            🌾 Buyuk = CROP
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SEKMELER
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Fotograf Analizi", 
    "🎥 Video Analizi", 
    "📊 Analitik Dashboard",
    "🗺️ GPS Harita"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: FOTOGRAF
# ═══════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("📁 Tarla Fotografi Yukle", type=["jpg", "png", "jpeg"])

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

        # Gecmise kaydet
        st.session_state.scan_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'weeds': weed_count,
            'crops': crop_count,
            'density': weed_density,
            'confidence': avg_conf
        })

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

        # Goruntuler
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(original_img, caption="📸 Orijinal", use_container_width=True)
        with col_img2:
            st.image(result_img, caption="🎯 Tespit", use_container_width=True)

        # Heatmap
        st.markdown("### 📈 Yabanci Ot Yogunluk Haritasi")
        if weed_count > 0:
            density_map = generate_heatmap(boxes_all, scores_all, classes_all,
                                           keep_indices, w, h)
            fig = px.imshow(
                density_map,
                color_continuous_scale='RdYlGn_r',
                title="🔴 Kirmizi = Yuksek | 🟢 Yesil = Dusuk",
                labels={'color': 'Ot Yogunlugu'}
            )
            fig.update_layout(height=400, margin=dict(r=0, t=40, l=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Yabanci ot bulunamadi!")

        # Detaylar
        if keep_indices:
            st.markdown("### 📋 Tespit Detaylari")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**🌿 Yabanci Otlar:**")
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
                    st.warning("⚠️ Mahsul bulunamadi!")

        # Oneri Sistemi
        st.markdown("### 💡 AI Onerisi")
        if weed_density > 20:
            st.error("🚨 **YUKSEK RİSK:** Ot yogunlugu %20 uzerinde! Acil herbisit uygulamasi onerilir.")
        elif weed_density > 10:
            st.warning("⚠️ **ORTA RİSK:** Hedefli ilaclama onerilir.")
        elif weed_count > 0:
            st.info("💚 **DUSUK RİSK:** Elle temizlik yeterli.")
        else:
            st.success("✅ **TEMİZ:** Yabanci ot bulunamadi! Tarla saglikli.")

        # Indirme Secenekleri
        st.markdown("### 💾 Indirme Secenekleri")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with col_dl1:
            buf_img = io.BytesIO()
            result_img.save(buf_img, format='PNG')
            st.download_button(
                label="🖼️ Goruntu Indir (PNG)",
                data=buf_img.getvalue(),
                file_name=f"agrivision_{timestamp}.png",
                mime="image/png",
                use_container_width=True
            )

        with col_dl2:
            csv_bytes = generate_csv_report(
                weed_count, crop_count, weed_density, avg_conf,
                boxes_all, scores_all, classes_all, keep_indices,
                gps_lat, gps_lon
            )
            st.download_button(
                label="📊 CSV Rapor Indir",
                data=csv_bytes,
                file_name=f"agrivision_report_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_dl3:
            # JSON rapor
            import json
            json_report = {
                "timestamp": timestamp,
                "gps": {"lat": gps_lat, "lon": gps_lon},
                "summary": {
                    "weeds": weed_count,
                    "crops": crop_count,
                    "weed_density": round(weed_density, 2),
                    "avg_confidence": round(float(avg_conf), 2)
                },
                "detections": [
                    {
                        "class": classes_all[i],
                        "confidence": round(scores_all[i], 3),
                        "box": boxes_all[i],
                        "area": areas_all[i]
                    }
                    for i in keep_indices
                ]
            }
            st.download_button(
                label="📋 JSON Rapor Indir",
                data=json.dumps(json_report, indent=2),
                file_name=f"agrivision_report_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════════════
# TAB 2: VIDEO
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="tab-header">🎥 Video Analizi</div>
    """, unsafe_allow_html=True)

    video_file = st.file_uploader("🎥 Video Yukle", type=["mp4", "avi", "mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

        st.info(f"📹 {total_frames} kare | {fps} FPS | ~{total_frames // fps}s")
        frame_skip = st.slider("⏭️ Her kac karede analiz?", 5, 60, 30, 5)

        if st.button("🚀 Video Analizi Baslat", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            result_placeholder = st.empty()

            all_results = []
            frame_idx = 0
            processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

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
                    status.text(f"Islenen: {processed} | Weed: {wc} | Crop: {cc}")

                frame_idx += 1

            cap.release()
            progress.progress(1.0)
            status.success(f"✅ {processed} kare analiz edildi!")

            if all_results:
                st.markdown("### 📈 Video Sonuclari")
                times = [r['time'] for r in all_results]
                weeds = [r['weeds'] for r in all_results]
                crops = [r['crops'] for r in all_results]

                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(
                    x=times, y=weeds, mode='lines+markers',
                    name='🌿 Weeds', line=dict(color='#FF4757', width=3)
                ))
                fig_v.add_trace(go.Scatter(
                    x=times, y=crops, mode='lines+markers',
                    name='🌾 Crops', line=dict(color='#2ED573', width=3)
                ))
                fig_v.update_layout(
                    title="Zaman Icinde Tespit",
                    xaxis_title="Saniye", yaxis_title="Sayi", height=400
                )
                st.plotly_chart(fig_v, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌿 Toplam Weed", sum(weeds))
                with col2:
                    st.metric("🌾 Toplam Crop", sum(crops))
                with col3:
                    st.metric("📊 Max Weed/Kare", max(weeds) if weeds else 0)
    else:
        st.info("🎥 MP4, AVI veya MOV dosyasi yukleyin")


# ═══════════════════════════════════════════════════════════════
# TAB 3: ANALİTİK DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="tab-header">📊 Analitik Dashboard</div>
    """, unsafe_allow_html=True)

    if st.session_state.scan_history:
        history = st.session_state.scan_history

        # Zaman serisi grafigi
        st.markdown("### 📈 Tarama Gecmisi Grafigi")
        times = list(range(1, len(history) + 1))
        weeds_hist = [h['weeds'] for h in history]
        crops_hist = [h['crops'] for h in history]
        density_hist = [h['density'] for h in history]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=times, y=weeds_hist, name='🌿 Weeds',
            marker_color='#FF4757'
        ))
        fig_hist.add_trace(go.Bar(
            x=times, y=crops_hist, name='🌾 Crops',
            marker_color='#2ED573'
        ))
        fig_hist.update_layout(
            title="Tarama Bazinda Tespit Sayisi",
            xaxis_title="Tarama No",
            yaxis_title="Sayi",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Yogunluk trendi
        st.markdown("### 📉 Ot Yogunluk Trendi")
        fig_density = go.Figure()
        fig_density.add_trace(go.Scatter(
            x=times, y=density_hist, mode='lines+markers+text',
            name='Density %',
            line=dict(color='#FFA502', width=3),
            text=[f"{d:.1f}%" for d in density_hist],
            textposition="top center"
        ))
        fig_density.update_layout(
            title="Tarama Bazinda Ot Yogunlugu",
            xaxis_title="Tarama No",
            yaxis_title="Yogunluk (%)",
            height=350
        )
        st.plotly_chart(fig_density, use_container_width=True)

        # Pasta grafigi
        st.markdown("### 🥧 Genel Dagilim")
        col_pie1, col_pie2 = st.columns(2)

        total_weeds = sum(weeds_hist)
        total_crops = sum(crops_hist)

        with col_pie1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['🌿 Weeds', '🌾 Crops'],
                values=[total_weeds, total_crops],
                hole=0.4,
                marker_colors=['#FF4757', '#2ED573']
            )])
            fig_pie.update_layout(title="Toplam Tespit Dagilimi", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_pie2:
            avg_density = np.mean(density_hist)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_density,
                title={'text': "Ortalama Ot Yogunlugu (%)"},
                delta={'reference': 10},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FFA502"},
                    'steps': [
                        {'range': [0, 10], 'color': "#2ED573"},
                        {'range': [10, 20], 'color': "#FFD23F"},
                        {'range': [20, 100], 'color': "#FF4757"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Ozet istatistikler
        st.markdown("### 📊 Ozet Istatistikler")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("📷 Toplam Tarama", len(history))
        with col_s2:
            st.metric("🌿 Toplam Weed", total_weeds)
        with col_s3:
            st.metric("🌾 Toplam Crop", total_crops)
        with col_s4:
            st.metric("📊 Ort. Yogunluk", f"{avg_density:.1f}%")

        # CSV indirme
        df_history = pd.DataFrame(history)
        csv_hist = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Gecmis Verilerini Indir (CSV)",
            data=csv_hist,
            file_name=f"agrivision_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("📸 Oncelikle Fotograf Analizi sekmesinden tarama yapin!")


# ═══════════════════════════════════════════════════════════════
# TAB 4: GPS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="tab-header">🗺️ GPS Konum & Tarla Haritasi</div>
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

    st.markdown("### 🗺️ Coklu Alan Takibi")
    num_fields = st.number_input("Kac tarla?", 1, 10, 3)

    field_data = []
    for i in range(num_fields):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(f"Tarla {i + 1} Adi", value=f"Tarla-{i + 1}", key=f"n_{i}")
        with col2:
            lat = st.number_input(f"Enlem {i + 1}", value=gps_lat + (i * 0.002),
                                  format="%.6f", key=f"la_{i}")
        with col3:
            lon = st.number_input(f"Boylam {i + 1}", value=gps_lon + (i * 0.002),
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
    <p>📸 Fotograf | 🎥 Video | 📈 Heatmap | 📊 Analitik | 🗺️ GPS</p>
</div>
""", unsafe_allow_html=True)
