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

# ═══════════════════════════════════════════════════════════════
# AI BİLGİ TABANI
# ═══════════════════════════════════════════════════════════════
WEED_DATABASE = {
    "genel": {
        "name": "Yabanci Ot (Genel)",
        "icon": "🌿",
        "description": "Tarlada tespit edilen yabanci ot.",
        "diseases": [
            {"name": "Kurutucu Mantari", "risk": "Yuksek", "icon": "🍄",
             "detail": "Nemli ortamlarda yayilir. Yapraklarda kahverengi lekeler olusturur."},
            {"name": "Yaprak Leke Hastaligi", "risk": "Orta", "icon": "🦠",
             "detail": "Yapraklarda sari-kahverengi lekeler. Hava sirkülasyonu onemli."},
            {"name": "Kok Curumesi", "risk": "Dusuk", "icon": "🪱",
             "detail": "Asiri sulamada gorulur. Bitki solgunlugu ve sararma yapar."}
        ],
        "treatment": [
            {"method": "Herbisit Uygulama", "icon": "💊",
             "detail": "Secici herbisit ile hedefli uygulama. Ruzgarsiz havada, sabah erken saatlerde uygulayin.",
             "timing": "Ot 5-10 cm boyundayken"},
            {"method": "Mekanik Capa", "icon": "⛏️",
             "detail": "Elle veya makineli capalama. Kok sistemi cikarilmali.",
             "timing": "Haftada 1-2 kez"},
            {"method": "Malclama", "icon": "🌾",
             "detail": "Toprak yuzeyini organik malc ile ortun. Ot cimlenmesini engeller.",
             "timing": "Ekim sonrasi"},
            {"method": "Sulama Kontrolu", "icon": "💧",
             "detail": "Damla sulama tercih edin. Gereksiz sulama ot buyumesini tesvik eder.",
             "timing": "Surekli"}
        ],
        "prevention": [
            "Ekim oncesi toprak hazirligi yapin",
            "Sertifikali tohum kullanin",
            "Ekim normu yakin tutun",
            "Munavebe (ekim nöbeti) uygulayin",
            "Tarla kenarlarini temiz tutun"
        ]
    }
}

CROP_DATABASE = {
    "genel": {
        "name": "Mahsul (Genel)",
        "icon": "🌾",
        "description": "Tarlada tespit edilen mahsul bitkisi.",
        "health_tips": [
            {"tip": "Toprak Analizi", "icon": "🧪",
             "detail": "Yilda 1 kez toprak analizi yaptirin. pH, azot, fosfor, potasyum degerlerini kontrol edin."},
            {"tip": "Gubreleme", "icon": "🌱",
             "detail": "Toprak analizine gore dengeli gubreleme yapin. Asiri azot yaprak buyutup meyve azaltir."},
            {"tip": "Sulama Programi", "icon": "💧",
             "detail": "Damla sulama en verimli yontemdir. Sabah erken veya aksam gec sulayin."},
            {"tip": "Hastalik Takibi", "icon": "🔍",
             "detail": "Haftada 1 tarla gezisi yapin. Yaprak rengi, leke, solgunluk kontrol edin."},
            {"tip": "Budama", "icon": "✂️",
             "detail": "Hasta ve kurumus dallari budayin. Hava sirkülasyonunu artirin."}
        ],
        "growth_stages": [
            {"stage": "Cimlendirme", "icon": "🌱", "days": "0-14 gun",
             "advice": "Toprak nemini koruyun. Sicaklik 18-25°C ideal."},
            {"stage": "Vejetatif Buyume", "icon": "🌿", "days": "15-45 gun",
             "advice": "Azotlu gubre verin. Yabanci ot kontrolu yapin."},
            {"stage": "Ciceklenme", "icon": "🌸", "days": "45-70 gun",
             "advice": "Fosforlu gubre verin. Sulama duzenli olmali."},
            {"stage": "Meyve/Tohum", "icon": "🍎", "days": "70-120 gun",
             "advice": "Potasyumlu gubre verin. Hasat zamanlama onemli."},
            {"stage": "Hasat", "icon": "🌾", "days": "120+ gun",
             "advice": "Olgunluk kontrolu yapin. Uygun hava kosullarinda hasat edin."}
        ]
    }
}

SEASONAL_ADVICE = {
    1: {"season": "Kis", "icon": "❄️", "advice": "Toprak hazirligi ve planlama zamani. Tohum secimi yapin."},
    2: {"season": "Kis", "icon": "❄️", "advice": "Sera ici uretim baslatilabilir. Toprak analizi yaptirin."},
    3: {"season": "Ilkbahar", "icon": "🌸", "advice": "Erken ekim baslayabilir. Toprak sicakligi olcun."},
    4: {"season": "Ilkbahar", "icon": "🌸", "advice": "Ana ekim donemi. Gubreleme ve sulama planlayin."},
    5: {"season": "Ilkbahar", "icon": "🌸", "advice": "Yabanci ot mucadelesi kritik. Duzenli kontrol yapin."},
    6: {"season": "Yaz", "icon": "☀️", "advice": "Sulama kritik. Hastalik ve zararli takibi yogunlastirin."},
    7: {"season": "Yaz", "icon": "☀️", "advice": "Sicaklik stresi riski. Golgeleme ve sulama artirin."},
    8: {"season": "Yaz", "icon": "☀️", "advice": "Erken hasat baslayabilir. Depolama hazirligi yapin."},
    9: {"season": "Sonbahar", "icon": "🍂", "advice": "Ana hasat donemi. Kurutma ve depolama onemli."},
    10: {"season": "Sonbahar", "icon": "🍂", "advice": "Sonbahar ekimi yapilabilir. Toprak isleme zamani."},
    11: {"season": "Sonbahar", "icon": "🍂", "advice": "Tarla temizligi yapin. Kis hazirligi baslatin."},
    12: {"season": "Kis", "icon": "❄️", "advice": "Yillik degerlendirme yapin. Gelecek sezon planlayin."}
}


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
.advice-card {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
}
.disease-card {
    background: rgba(255,0,0,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #FF4757;
}
.treatment-card {
    background: rgba(0,255,0,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #2ED573;
}
.stage-card {
    background: rgba(100,100,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
}
</style>
"""

st.set_page_config(page_title="AgriVision AI Pro", layout="wide", page_icon="🌱",
                   initial_sidebar_state="expanded")
st.markdown(GLASS_CSS, unsafe_allow_html=True)

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102,126,234,0.4);'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🌱 AgriVision AI Pro</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
        Detection | AI Advisor | Disease Control | Heatmap | Video | GPS
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
    for cls in list(set(classes)):
        ci = [i for i, c in enumerate(classes) if c == cls]
        cb = [boxes[i] for i in ci]
        cs = [scores[i] for i in ci]
        indices = np.argsort(cs)[::-1].tolist()
        keep = []
        while indices:
            cur = indices.pop(0)
            keep.append(cur)
            rm = []
            for idx in indices:
                if iou(cb[cur], cb[idx]) > iou_threshold:
                    rm.append(idx)
                elif containment(cb[cur], cb[idx]) > containment_threshold:
                    rm.append(idx)
            for r in rm:
                if r in indices:
                    indices.remove(r)
        all_keep.extend([ci[k] for k in keep])
    return all_keep


def draw_detections(img, boxes, scores, classes, keep_indices):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        conf, cls = scores[i], classes[i]
        if cls == "WEED":
            color, bg, emoji = "#FF4757", "#FF6B7A", "🌿"
        else:
            color, bg, emoji = "#2ED573", "#51CF66", "🌾"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{emoji} {cls} {conf:.0%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        lw, lh = bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 8
        lx, ly = x1, max(0, y1 - lh)
        draw.rectangle([lx, ly, lx + lw, ly + lh], fill=bg)
        draw.text((lx + 6, ly + 4), label, fill="white", font=font)
    return img


def process_image(img, interpreter, input_details, output_details, threshold, size_threshold, nms_iou):
    w, h = img.size
    total_area = w * h
    resized = img.resize((640, 640))
    arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, 0)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = np.transpose(interpreter.get_tensor(output_details[0]["index"])[0], (1, 0))

    raw_boxes, raw_scores = [], []
    for row in preds:
        x, y, bw, bh = row[0], row[1], row[2], row[3]
        best_score = float(np.max(row[4:]))
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
        boxes_all.append(raw_boxes[i])
        scores_all.append(raw_scores[i])
        classes_all.append("WEED" if area < size_limit else "CROP")
        areas_all.append(area)

    keep = class_aware_nms(boxes_all, scores_all, classes_all, nms_iou, 0.6)
    return boxes_all, scores_all, classes_all, areas_all, keep


def generate_heatmap(boxes, scores, classes, keep_indices, w, h):
    gs = 20
    rows, cols = h // gs + 1, w // gs + 1
    dm = np.zeros((rows, cols))
    for i in keep_indices:
        if classes[i] == "WEED":
            x1, y1, x2, y2 = boxes[i]
            dm[max(0, y1//gs):min(rows, y2//gs+1), max(0, x1//gs):min(cols, x2//gs+1)] += scores[i]
    return dm


def generate_csv_report(wc, cc, wd, ac, boxes, scores, classes, keep, lat, lon):
    rows = []
    for idx, i in enumerate(keep):
        x1, y1, x2, y2 = boxes[i]
        rows.append({"No": idx+1, "Class": classes[i], "Confidence": round(float(scores[i]), 3),
                     "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                     "Width": x2-x1, "Height": y2-y1, "Area": (x2-x1)*(y2-y1)})
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{"No": "---", "Class": "SUMMARY", "Confidence": round(float(ac), 3),
                             "X1": f"Weeds:{wc}", "Y1": f"Crops:{cc}",
                             "X2": f"Density:{wd:.1f}%", "Y2": f"Lat:{lat:.6f}",
                             "Width": f"Lon:{lon:.6f}",
                             "Height": datetime.now().strftime("%Y-%m-%d %H:%M"), "Area": "---"}])
    return pd.concat([df, summary], ignore_index=True).to_csv(index=False).encode("utf-8")


def get_ai_advice(weed_count, crop_count, weed_density, avg_conf):
    """AI tabanli akilli oneri sistemi"""
    advice = {
        "risk_level": "",
        "risk_color": "",
        "risk_icon": "",
        "summary": "",
        "weed_info": WEED_DATABASE["genel"],
        "crop_info": CROP_DATABASE["genel"],
        "seasonal": SEASONAL_ADVICE.get(datetime.now().month, {}),
        "actions": []
    }

    if weed_density > 30:
        advice["risk_level"] = "KRITIK"
        advice["risk_color"] = "#FF0000"
        advice["risk_icon"] = "🚨"
        advice["summary"] = "Tarla kritik seviyede ot istilasinda! Acil mudahale gerekli."
        advice["actions"] = [
            "🚨 Acil herbisit uygulamasi baslatin",
            "📞 Tarim danismaniniza haber verin",
            "🔄 2 gun icinde tekrar tarama yapin",
            "💊 Sistemik herbisit tercih edin",
            "⏰ Sabah 06:00-09:00 arasi uygulayin"
        ]
    elif weed_density > 20:
        advice["risk_level"] = "YUKSEK"
        advice["risk_color"] = "#FF4757"
        advice["risk_icon"] = "🔴"
        advice["summary"] = "Ot yogunlugu yuksek! Herbisit uygulamasi onerilir."
        advice["actions"] = [
            "💊 Secici herbisit uygulayin",
            "⛏️ Mekanik capalama yapin",
            "📅 1 hafta icinde tekrar kontrol edin",
            "💧 Sulama programini gozden gecirin",
            "🌾 Malclama uygulamayi degerlendindir"
        ]
    elif weed_density > 10:
        advice["risk_level"] = "ORTA"
        advice["risk_color"] = "#FFA502"
        advice["risk_icon"] = "🟡"
        advice["summary"] = "Orta duzeyde ot varligi. Hedefli mudahale yeterli."
        advice["actions"] = [
            "⛏️ Elle veya mekanik temizlik yapin",
            "🎯 Noktasal herbisit uygulayin",
            "📅 2 haftada bir kontrol edin",
            "🌱 Mahsul guclendiricisi kullanin"
        ]
    elif weed_count > 0:
        advice["risk_level"] = "DUSUK"
        advice["risk_color"] = "#2ED573"
        advice["risk_icon"] = "🟢"
        advice["summary"] = "Az miktarda ot tespit edildi. Koruyucu onlemler yeterli."
        advice["actions"] = [
            "👁️ Duzenli gozlem yapin",
            "✋ Elle cekme yeterli",
            "🌾 Malclama uygulayin",
            "📅 Ayda 2 kez kontrol edin"
        ]
    else:
        advice["risk_level"] = "TEMIZ"
        advice["risk_color"] = "#00D2D3"
        advice["risk_icon"] = "✅"
        advice["summary"] = "Tarlada yabanci ot bulunmadi! Tarla saglikli."
        advice["actions"] = [
            "✅ Mevcut bakim programina devam edin",
            "🌱 Gubreleme takvimini takip edin",
            "💧 Sulama programini surdurn",
            "📅 Aylik kontrol yeterli"
        ]

    return advice


try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model hatasi: {e}")
    st.stop()


with st.sidebar:
    st.markdown("## 🎛️ Ayarlar")
    threshold = st.slider("🎯 Confidence", 0.1, 0.9, 0.60, 0.05)
    size_threshold = st.slider("📏 Boyut Esigi (%)", 5, 50, 42, 1)
    nms_iou = st.slider("🔗 NMS IoU", 0.1, 0.7, 0.20, 0.05)
    st.markdown("---")
    st.markdown("### 🗺️ GPS")
    gps_lat = st.number_input("📍 Enlem", value=39.9334, format="%.6f")
    gps_lon = st.number_input("📍 Boylam", value=32.8597, format="%.6f")
    st.markdown("---")
    st.markdown("### 📜 Gecmis")
    if st.session_state.scan_history:
        for s in st.session_state.scan_history[-5:][::-1]:
            st.markdown(f"• {s['time']} | 🌿{s['weeds']} 🌾{s['crops']} 📊{s['density']:.1f}%")
    if st.button("🗑️ Temizle", use_container_width=True):
        st.session_state.scan_history = []
        st.rerun()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Fotograf", "🧠 AI Danismanlik", "🎥 Video", "📊 Analitik", "🗺️ GPS"
])


with tab1:
    uploaded = st.file_uploader("📁 Tarla Fotografi", type=["jpg", "png", "jpeg"], key="photo")

    if uploaded:
        original_img = Image.open(uploaded).convert("RGB")
        w, h = original_img.size
        total_area = w * h

        with st.spinner("🔍 Analiz..."):
            boxes_all, scores_all, classes_all, areas_all, keep_indices = process_image(
                original_img, interpreter, input_details, output_details,
                threshold, size_threshold, nms_iou)

        result_img = original_img.copy()
        if keep_indices:
            result_img = draw_detections(result_img, boxes_all, scores_all, classes_all, keep_indices)

        weed_count = sum(1 for i in keep_indices if classes_all[i] == "WEED")
        crop_count = sum(1 for i in keep_indices if classes_all[i] == "CROP")
        weed_area = sum(areas_all[i] for i in keep_indices if classes_all[i] == "WEED")
        weed_density = (weed_area / total_area) * 100 if total_area > 0 else 0
        avg_conf = float(np.mean([scores_all[i] for i in keep_indices])) if keep_indices else 0

        st.session_state.scan_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "weeds": weed_count, "crops": crop_count,
            "density": float(weed_density), "confidence": avg_conf
        })

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌿</div><div class='metric-value' style='color:#FF4757'>{weed_count}</div><div class='metric-label'>WEEDS</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌾</div><div class='metric-value' style='color:#2ED573'>{crop_count}</div><div class='metric-label'>CROPS</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>📊</div><div class='metric-value' style='color:#FFA502'>{weed_density:.1f}%</div><div class='metric-label'>DENSITY</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🎯</div><div class='metric-value' style='color:#3742FA'>{avg_conf:.0%}</div><div class='metric-label'>CONF</div></div>", unsafe_allow_html=True)

        i1, i2 = st.columns(2)
        with i1:
            st.image(original_img, caption="📸 Orijinal", use_container_width=True)
        with i2:
            st.image(result_img, caption="🎯 Sonuc", use_container_width=True)

        if weed_count > 0:
            st.markdown("### 📈 Ot Yogunluk Haritasi")
            dm = generate_heatmap(boxes_all, scores_all, classes_all, keep_indices, w, h)
            fig = px.imshow(dm, color_continuous_scale="RdYlGn_r",
                          title="🔴 Kirmizi = Yuksek | 🟢 Yesil = Dusuk")
            fig.update_layout(height=400, margin=dict(r=0, t=40, l=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💾 Indir")
        d1, d2, d3 = st.columns(3)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with d1:
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button("🖼️ PNG", buf.getvalue(), f"agrivision_{ts}.png", "image/png", use_container_width=True)
        with d2:
            csv = generate_csv_report(weed_count, crop_count, weed_density, avg_conf,
                                     boxes_all, scores_all, classes_all, keep_indices, gps_lat, gps_lon)
            st.download_button("📊 CSV", csv, f"agrivision_{ts}.csv", "text/csv", use_container_width=True)
        with d3:
            jr = {"timestamp": ts, "gps": {"lat": gps_lat, "lon": gps_lon},
                  "summary": {"weeds": weed_count, "crops": crop_count,
                             "density": round(weed_density, 2), "confidence": round(avg_conf, 2)},
                  "detections": [{"class": classes_all[i], "confidence": round(float(scores_all[i]), 3),
                                 "box": boxes_all[i]} for i in keep_indices]}
            st.download_button("📋 JSON", json.dumps(jr, indent=2), f"agrivision_{ts}.json",
                             "application/json", use_container_width=True)


with tab2:
    st.markdown("<div class='tab-header'>🧠 AI Tarim Danismanligi</div>", unsafe_allow_html=True)

    if st.session_state.scan_history:
        last = st.session_state.scan_history[-1]
        advice = get_ai_advice(last["weeds"], last["crops"], last["density"], last["confidence"])

        # Risk Seviyesi
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {advice["risk_color"]}44, {advice["risk_color"]}22);
                    padding: 1.5rem; border-radius: 20px; border-left: 6px solid {advice["risk_color"]};
                    margin-bottom: 1.5rem;'>
            <h2 style='margin:0;'>{advice["risk_icon"]} Risk Seviyesi: {advice["risk_level"]}</h2>
            <p style='margin:0.5rem 0 0 0; font-size:1.1em;'>{advice["summary"]}</p>
        </div>
        """, unsafe_allow_html=True)

        # Aksiyon Plani
        st.markdown("### 📋 Aksiyon Plani")
        for idx, action in enumerate(advice["actions"]):
            st.markdown(f"""
            <div class='advice-card' style='border-color: {advice["risk_color"]};'>
                <b>Adim {idx+1}:</b> {action}
            </div>
            """, unsafe_allow_html=True)

        # Mevsimsel Oneri
        seasonal = advice["seasonal"]
        if seasonal:
            st.markdown(f"""
            ### {seasonal.get('icon', '📅')} Mevsimsel Oneri ({seasonal.get('season', '')})
            <div class='advice-card' style='border-color: #667eea;'>
                {seasonal.get('advice', '')}
            </div>
            """, unsafe_allow_html=True)

        # Hastalik Bilgisi
        if last["weeds"] > 0:
            st.markdown("### 🦠 Olasi Hastaliklar & Riskler")
            weed_info = advice["weed_info"]
            for disease in weed_info["diseases"]:
                risk_color = "#FF4757" if disease["risk"] == "Yuksek" else "#FFA502" if disease["risk"] == "Orta" else "#2ED573"
                st.markdown(f"""
                <div class='disease-card'>
                    <h4>{disease["icon"]} {disease["name"]} 
                        <span style='background:{risk_color}; color:white; padding:2px 8px; 
                              border-radius:5px; font-size:0.8em;'>{disease["risk"]}</span>
                    </h4>
                    <p>{disease["detail"]}</p>
                </div>
                """, unsafe_allow_html=True)

            # Tedavi Yontemleri
            st.markdown("### 💊 Tedavi & Mucadele Yontemleri")
            for treatment in weed_info["treatment"]:
                st.markdown(f"""
                <div class='treatment-card'>
                    <h4>{treatment["icon"]} {treatment["method"]}</h4>
                    <p>{treatment["detail"]}</p>
                    <small>⏰ <b>Zamanlama:</b> {treatment["timing"]}</small>
                </div>
                """, unsafe_allow_html=True)

            # Onleme
            st.markdown("### 🛡️ Onleme Tedbirleri")
            for p in weed_info["prevention"]:
                st.markdown(f"✅ {p}")

        # Mahsul Bakimi
        if last["crops"] > 0:
            st.markdown("### 🌾 Mahsul Bakim Onerileri")
            crop_info = advice["crop_info"]
            for tip in crop_info["health_tips"]:
                st.markdown(f"""
                <div class='treatment-card'>
                    <h4>{tip["icon"]} {tip["tip"]}</h4>
                    <p>{tip["detail"]}</p>
                </div>
                """, unsafe_allow_html=True)

            # Buyume Asamalari
            st.markdown("### 📅 Bitki Buyume Asamalari")
            for stage in crop_info["growth_stages"]:
                st.markdown(f"""
                <div class='stage-card'>
                    <h4>{stage["icon"]} {stage["stage"]} ({stage["days"]})</h4>
                    <p>{stage["advice"]}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📸 Once Fotograf sekmesinden tarama yapin! AI danismanlik icin veri gerekli.")


with tab3:
    st.markdown("<div class='tab-header'>🎥 Video Analizi</div>", unsafe_allow_html=True)
    video_file = st.file_uploader("🎥 Video", type=["mp4", "avi", "mov"], key="video")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        st.info(f"📹 {total_frames} kare | {fps} FPS | ~{total_frames // fps}s")
        frame_skip = st.slider("⏭️ Kare araligi", 5, 60, 30, 5)

        if st.button("🚀 Baslat", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()
            results = []
            fi = 0
            pc = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if fi % frame_skip == 0:
                    pf = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    b, s, cl, a, k = process_image(pf, interpreter, input_details, output_details,
                                                    threshold, size_threshold, nms_iou)
                    wc = sum(1 for i in k if cl[i] == "WEED")
                    cc = sum(1 for i in k if cl[i] == "CROP")
                    results.append({"time": fi/fps, "weeds": wc, "crops": cc})
                    if k:
                        rf = pf.copy()
                        rf = draw_detections(rf, b, s, cl, k)
                        preview.image(rf, caption=f"Kare {fi}", use_container_width=True)
                    pc += 1
                    progress.progress(min(fi / max(total_frames, 1), 1.0))
                    status.text(f"Islenen: {pc} | 🌿{wc} | 🌾{cc}")
                fi += 1
            cap.release()
            progress.progress(1.0)
            status.success(f"✅ {pc} kare analiz edildi")
            if results:
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=[r["time"] for r in results], y=[r["weeds"] for r in results],
                                           mode="lines+markers", name="🌿 Weeds", line=dict(color="#FF4757", width=3)))
                fig_v.add_trace(go.Scatter(x=[r["time"] for r in results], y=[r["crops"] for r in results],
                                           mode="lines+markers", name="🌾 Crops", line=dict(color="#2ED573", width=3)))
                fig_v.update_layout(height=400, xaxis_title="Saniye", yaxis_title="Sayi")
                st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info("🎥 MP4, AVI veya MOV yukleyin")


with tab4:
    st.markdown("<div class='tab-header'>📊 Analitik Dashboard</div>", unsafe_allow_html=True)
    if st.session_state.scan_history:
        h = st.session_state.scan_history
        x = list(range(1, len(h)+1))
        wh = [i["weeds"] for i in h]
        ch = [i["crops"] for i in h]
        dh = [i["density"] for i in h]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=x, y=wh, name="🌿 Weeds", marker_color="#FF4757"))
        fig1.add_trace(go.Bar(x=x, y=ch, name="🌾 Crops", marker_color="#2ED573"))
        fig1.update_layout(barmode="group", height=380)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=dh, mode="lines+markers+text",
                                  text=[f"{d:.1f}%" for d in dh], textposition="top center",
                                  line=dict(color="#FFA502", width=3)))
        fig2.update_layout(height=350, yaxis_title="Yogunluk (%)")
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("📷 Tarama", len(h))
        with c2: st.metric("🌿 Toplam", sum(wh))
        with c3: st.metric("🌾 Toplam", sum(ch))
        with c4: st.metric("📊 Ort.", f"{np.mean(dh):.1f}%")
    else:
        st.info("📸 Fotograf sekmesinden tarama yapin")


with tab5:
    st.markdown("<div class='tab-header'>🗺️ GPS Harita</div>", unsafe_allow_html=True)
    st.map(pd.DataFrame({"lat": [gps_lat], "lon": [gps_lon]}), zoom=14)
    st.markdown(f"""
    <div class='glass-card' style='margin-top:1rem;'>
        <h3>📍 Konum</h3>
        <p><b>Enlem:</b> {gps_lat:.6f} | <b>Boylam:</b> {gps_lon:.6f}</p>
        <p><b>Tarih:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>""", unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center; padding:2rem; margin-top:3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius:20px; color:white;'>
    <h3>🌾 AgriVision AI Pro v3.0</h3>
    <p>📸 Detection | 🧠 AI Advisor | 🦠 Disease Control | 📈 Heatmap | 🎥 Video | 🗺️ GPS</p>
</div>
""", unsafe_allow_html=True)
