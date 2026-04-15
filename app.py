from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import cv2
import pandas as pd

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
}
.metric-icon { font-size: 2.2em; }
.metric-value { font-size: 2em; font-weight: 700; margin: 0; }
.metric-label { color: rgba(255,255,255,0.85); margin: 0; font-size: 0.9em; }
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
    padding: 0.45rem 0.9rem;
    margin: 0.25rem;
    border-radius: 10px;
    font-weight: bold;
    color: white;
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

if "scan_history" not in st.session_state:
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
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
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
    return inter / min(area1, area2)


def class_aware_nms(boxes, scores, classes, iou_threshold=0.2, containment_threshold=0.6):
    if not boxes:
        return []

    all_keep = []
    unique_classes = list(set(classes))

    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        cls_boxes = [boxes[i] for i in cls_indices]
        cls_scores = [scores[i] for i in cls_indices]

        indices = np.argsort(cls_scores)[::-1].tolist()
        keep_local = []

        while indices:
            current = indices.pop(0)
            keep_local.append(current)
            remove = []
            for idx in indices:
                if iou(cls_boxes[current], cls_boxes[idx]) > iou_threshold:
                    remove.append(idx)
                elif containment(cls_boxes[current], cls_boxes[idx]) > containment_threshold:
                    remove.append(idx)
            for r in remove:
                if r in indices:
                    indices.remove(r)

        all_keep.extend([cls_indices[k] for k in keep_local])

    return all_keep


def draw_detections(img, boxes, scores, classes, keep_indices):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
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

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        label = f"{emoji} {cls} {conf:.0%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0] + 12
        lh = bbox[3] - bbox[1] + 8
        lx, ly = x1, max(0, y1 - lh)

        draw.rectangle([lx, ly, lx + lw, ly + lh], fill=bg_color)
        draw.text((lx + 6, ly + 4), label, fill="white", font=font)

    return img


def process_image(img, interpreter, input_details, output_details, threshold, size_threshold, nms_iou):
    w, h = img.size
    total_area = w * h

    resized = img.resize((640, 640))
    arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, 0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    preds = np.transpose(output_data, (1, 0))

    raw_boxes, raw_scores = [], []

    for row in preds:
        x, y, bw, bh = row[0], row[1], row[2], row[3]
        class_scores = row[4:]
        best_score = float(np.max(class_scores))

        if best_score < threshold:
            continue

        x1 = max(0, int((x - bw / 2) * w))
        y1 = max(0, int((y - bh / 2) * h))
        x2 = min(w, int((x + bw / 2) * w))
        y2 = min(h, int((y + bh / 2) * h))

        if (x2 - x1) > 10 and (y2 - y1) > 10:
            raw_boxes.append([x1, y1, x2, y2])
            raw_scores.append(best_score)

    boxes_all, scores_all, classes_all, areas_all = [], [], [], []
    size_limit = total_area * (size_threshold / 100)

    for i in range(len(raw_boxes)):
        x1, y1, x2, y2 = raw_boxes[i]
        area = (x2 - x1) * (y2 - y1)
        cls = "WEED" if area < size_limit else "CROP"

        boxes_all.append(raw_boxes[i])
        scores_all.append(raw_scores[i])
        classes_all.append(cls)
        areas_all.append(area)

    keep_indices = class_aware_nms(
        boxes_all, scores_all, classes_all,
        iou_threshold=nms_iou,
        containment_threshold=0.6
    )

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
    rows = []
    for idx, i in enumerate(keep_indices):
        x1, y1, x2, y2 = boxes[i]
        rows.append({
            "No": idx + 1,
            "Class": classes[i],
            "Confidence": round(float(scores[i]), 3),
            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2,
            "Width": x2 - x1,
            "Height": y2 - y1,
            "Area": (x2 - x1) * (y2 - y1)
        })

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "No": "---",
        "Class": "SUMMARY",
        "Confidence": round(float(avg_conf), 3),
        "X1": f"Weeds: {weed_count}",
        "Y1": f"Crops: {crop_count}",
        "X2": f"Density: {weed_density:.1f}%",
        "Y2": f"Lat: {gps_lat:.6f}",
        "Width": f"Lon: {gps_lon:.6f}",
        "Height": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Area": "---"
    }])

    df = pd.concat([df, summary], ignore_index=True)
    return df.to_csv(index=False).encode("utf-8")


try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model hatası: {e}")
    st.stop()


with st.sidebar:
    st.markdown("## 🎛️ Ayarlar")
    threshold = st.slider("🎯 Confidence", 0.1, 0.9, 0.60, 0.05)
    size_threshold = st.slider("📏 Boyut Eşiği (%)", 5, 50, 42, 1)
    nms_iou = st.slider("🔗 NMS IoU", 0.1, 0.7, 0.20, 0.05)

    st.markdown("---")
    st.markdown("### 🗺️ GPS")
    gps_lat = st.number_input("📍 Enlem", value=39.9334, format="%.6f")
    gps_lon = st.number_input("📍 Boylam", value=32.8597, format="%.6f")

    st.markdown("---")
    st.markdown("### 📜 Geçmiş")
    if st.session_state.scan_history:
        for scan in st.session_state.scan_history[-5:][::-1]:
            st.markdown(
                f"• {scan['time']} | 🌿 {scan['weeds']} | 🌾 {scan['crops']} | 📊 {scan['density']:.1f}%"
            )
    else:
        st.info("Henüz tarama yok")

    if st.button("🗑️ Geçmişi Temizle", use_container_width=True):
        st.session_state.scan_history = []
        st.rerun()


tab1, tab2, tab3, tab4 = st.tabs(["📸 Fotoğraf", "🎥 Video", "📊 Analitik", "🗺️ GPS"])


with tab1:
    uploaded = st.file_uploader("📁 Tarla Fotoğrafı", type=["jpg", "png", "jpeg"], key="photo_upload")

    if uploaded:
        original_img = Image.open(uploaded).convert("RGB")
        w, h = original_img.size
        total_area = w * h

        with st.spinner("🔍 Analiz yapılıyor..."):
            boxes_all, scores_all, classes_all, areas_all, keep_indices = process_image(
                original_img, interpreter, input_details, output_details,
                threshold, size_threshold, nms_iou
            )

        result_img = original_img.copy()
        if keep_indices:
            result_img = draw_detections(result_img, boxes_all, scores_all, classes_all, keep_indices)

        weed_count = sum(1 for i in keep_indices if classes_all[i] == "WEED")
        crop_count = sum(1 for i in keep_indices if classes_all[i] == "CROP")
        weed_area = sum(areas_all[i] for i in keep_indices if classes_all[i] == "WEED")
        weed_density = (weed_area / total_area) * 100 if total_area > 0 else 0
        avg_conf = np.mean([scores_all[i] for i in keep_indices]) if keep_indices else 0

        st.session_state.scan_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "weeds": weed_count,
            "crops": crop_count,
            "density": float(weed_density),
            "confidence": float(avg_conf)
        })

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌿</div><div class='metric-value' style='color:#FF4757'>{weed_count}</div><div class='metric-label'>WEEDS</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌾</div><div class='metric-value' style='color:#2ED573'>{crop_count}</div><div class='metric-label'>CROPS</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>📊</div><div class='metric-value' style='color:#FFA502'>{weed_density:.1f}%</div><div class='metric-label'>DENSITY</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🎯</div><div class='metric-value' style='color:#3742FA'>{avg_conf:.0%}</div><div class='metric-label'>CONFIDENCE</div></div>", unsafe_allow_html=True)

        i1, i2 = st.columns(2)
        with i1:
            st.image(original_img, caption="📸 Orijinal", use_container_width=True)
        with i2:
            st.image(result_img, caption="🎯 Sonuç", use_container_width=True)

        st.markdown("### 📈 Ot Yoğunluk Haritası")
        if weed_count > 0:
            density_map = generate_heatmap(boxes_all, scores_all, classes_all, keep_indices, w, h)
            fig = px.imshow(
                density_map,
                color_continuous_scale="RdYlGn_r",
                title="🔴 Kırmızı = Yüksek | 🟢 Yeşil = Düşük"
            )
            fig.update_layout(height=400, margin=dict(r=0, t=40, l=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Ot bulunamadı")

        st.markdown("### 💡 AI Önerisi")
        if weed_density > 20:
            st.error("🚨 YÜKSEK RİSK: Acil herbisit uygulaması önerilir.")
        elif weed_density > 10:
            st.warning("⚠️ ORTA RİSK: Hedefli ilaçlama önerilir.")
        elif weed_count > 0:
            st.info("💚 DÜŞÜK RİSK: Elle temizlik yeterli olabilir.")
        else:
            st.success("✅ TEMİZ: Tarla sağlıklı görünüyor.")

        st.markdown("### 💾 İndir")
        d1, d2, d3 = st.columns(3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with d1:
            buf_img = io.BytesIO()
            result_img.save(buf_img, format="PNG")
            st.download_button(
                "🖼️ PNG İndir",
                data=buf_img.getvalue(),
                file_name=f"agrivision_{timestamp}.png",
                mime="image/png",
                use_container_width=True
            )

        with d2:
            csv_data = generate_csv_report(
                weed_count, crop_count, weed_density, avg_conf,
                boxes_all, scores_all, classes_all, keep_indices,
                gps_lat, gps_lon
            )
            st.download_button(
                "📊 CSV İndir",
                data=csv_data,
                file_name=f"agrivision_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d3:
            json_report = {
                "timestamp": timestamp,
                "gps": {"lat": gps_lat, "lon": gps_lon},
                "summary": {
                    "weeds": weed_count,
                    "crops": crop_count,
                    "density": round(float(weed_density), 2),
                    "confidence": round(float(avg_conf), 2)
                },
                "detections": [
                    {
                        "class": classes_all[i],
                        "confidence": round(float(scores_all[i]), 3),
                        "box": boxes_all[i],
                        "area": areas_all[i]
                    }
                    for i in keep_indices
                ]
            }
            st.download_button(
                "📋 JSON İndir",
                data=json.dumps(json_report, indent=2),
                file_name=f"agrivision_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )


with tab2:
    st.markdown("<div class='tab-header'>🎥 Video Analizi</div>", unsafe_allow_html=True)
    video_file = st.file_uploader("🎥 Video Yükle", type=["mp4", "avi", "mov"], key="video_upload")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

        st.info(f"📹 {total_frames} kare | {fps} FPS | ~{total_frames // fps}s")
        frame_skip = st.slider("⏭️ Kare aralığı", 5, 60, 30, 5)

        if st.button("🚀 Video Analizini Başlat", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()
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
                        "time": frame_idx / fps,
                        "weeds": wc,
                        "crops": cc
                    })

                    if keep:
                        rf = pil_frame.copy()
                        rf = draw_detections(rf, boxes, sc, cls, keep)
                        preview.image(rf, caption=f"Kare {frame_idx}", use_container_width=True)

                    processed += 1
                    progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                    status.text(f"İşlenen: {processed} | Weed: {wc} | Crop: {cc}")

                frame_idx += 1

            cap.release()
            progress.progress(1.0)
            status.success(f"✅ {processed} kare analiz edildi")

            if all_results:
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(
                    x=[r["time"] for r in all_results],
                    y=[r["weeds"] for r in all_results],
                    mode="lines+markers",
                    name="🌿 Weeds",
                    line=dict(color="#FF4757", width=3)
                ))
                fig_v.add_trace(go.Scatter(
                    x=[r["time"] for r in all_results],
                    y=[r["crops"] for r in all_results],
                    mode="lines+markers",
                    name="🌾 Crops",
                    line=dict(color="#2ED573", width=3)
                ))
                fig_v.update_layout(height=400, xaxis_title="Saniye", yaxis_title="Sayı")
                st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info("🎥 MP4, AVI veya MOV yükleyin")


with tab3:
    st.markdown("<div class='tab-header'>📊 Analitik Dashboard</div>", unsafe_allow_html=True)

    if st.session_state.scan_history:
        history = st.session_state.scan_history
        x_vals = list(range(1, len(history) + 1))
        weeds_hist = [h["weeds"] for h in history]
        crops_hist = [h["crops"] for h in history]
        density_hist = [h["density"] for h in history]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(x=x_vals, y=weeds_hist, name="🌿 Weeds", marker_color="#FF4757"))
        fig_hist.add_trace(go.Bar(x=x_vals, y=crops_hist, name="🌾 Crops", marker_color="#2ED573"))
        fig_hist.update_layout(barmode="group", height=380, xaxis_title="Tarama No", yaxis_title="Sayı")
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_density = go.Figure()
        fig_density.add_trace(go.Scatter(
            x=x_vals,
            y=density_hist,
            mode="lines+markers+text",
            text=[f"{d:.1f}%" for d in density_hist],
            textposition="top center",
            line=dict(color="#FFA502", width=3)
        ))
        fig_density.update_layout(height=350, xaxis_title="Tarama No", yaxis_title="Yoğunluk (%)")
        st.plotly_chart(fig_density, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📷 Tarama", len(history))
        with c2:
            st.metric("🌿 Toplam Weed", sum(weeds_hist))
        with c3:
            st.metric("🌾 Toplam Crop", sum(crops_hist))
        with c4:
            st.metric("📊 Ort. Yoğunluk", f"{np.mean(density_hist):.1f}%")
    else:
        st.info("📸 Önce fotoğraf sekmesinden tarama yapın")


with tab4:
    st.markdown("<div class='tab-header'>🗺️ GPS Harita</div>", unsafe_allow_html=True)

    st.map(pd.DataFrame({"lat": [gps_lat], "lon": [gps_lon]}), zoom=14)

    st.markdown(f"""
    <div class='glass-card' style='margin-top:1rem;'>
        <h3>📍 Konum</h3>
        <p><b>Enlem:</b> {gps_lat:.6f} | <b>Boylam:</b> {gps_lon:.6f}</p>
        <p><b>Tarih:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center; padding:2rem; margin-top:3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius:20px; color:white;'>
    <h3>🌾 AgriVision AI Pro</h3>
    <p>📸 Fotoğraf | 🎥 Video | 📈 Heatmap | 📊 Analitik | 🗺️ GPS</p>
</div>
""", unsafe_allow_html=True)
