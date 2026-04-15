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
import requests

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

# ═══════════════════════════════════════════════════════════════
# HAVA DURUMU FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════
def get_weather(lat, lon, api_key):
    """OpenWeatherMap'den hava durumu al"""
    try:
        # Guncel hava
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr"
        current = requests.get(current_url, timeout=5).json()

        # 5 gunluk tahmin
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr"
        forecast = requests.get(forecast_url, timeout=5).json()

        return current, forecast
    except Exception:
        return None, None


def analyze_spray_conditions(weather_current, weather_forecast):
    """Ilacalama kosullarini analiz et"""
    if not weather_current or "main" not in weather_current:
        return None

    temp = weather_current["main"]["temp"]
    humidity = weather_current["main"]["humidity"]
    wind_speed = weather_current["wind"]["speed"] * 3.6  # m/s -> km/h
    description = weather_current["weather"][0]["description"]
    icon_code = weather_current["weather"][0]["icon"]

    # Yagmur kontrolu
    rain_now = "rain" in weather_current or "Rain" in description or "yagmur" in description.lower()

    # Gelecek 24 saat yagmur kontrolu
    rain_tomorrow = False
    rain_hours = []
    if weather_forecast and "list" in weather_forecast:
        for item in weather_forecast["list"][:8]:  # 24 saat
            if "rain" in item or any("rain" in w["main"].lower() for w in item["weather"]):
                rain_tomorrow = True
                dt = datetime.fromtimestamp(item["dt"])
                rain_hours.append(dt.strftime("%H:%M"))

    # Skor hesapla (0-100)
    score = 100

    # Ruzgar kontrolu
    if wind_speed > 25:
        score -= 50
        wind_status = "🔴 TEHLIKELI"
        wind_advice = "Ilacalama YAPMAAYIN! Ruzgar cok kuvvetli."
    elif wind_speed > 15:
        score -= 30
        wind_status = "🟡 RISKLI"
        wind_advice = "Ilacalama onerilmez. Ruzgar yuksek."
    elif wind_speed > 8:
        score -= 10
        wind_status = "🟢 DIKKATLI"
        wind_advice = "Ilacalama yapilabilir ama dikkatli olun."
    else:
        wind_status = "✅ IDEAL"
        wind_advice = "Ruzgar ilacalama icin uygun."

    # Sicaklik kontrolu
    if temp > 35:
        score -= 30
        temp_status = "🔴 COK SICAK"
        temp_advice = "Ilac buharlasmasi riski! Sabah erken veya aksam uygulayin."
    elif temp > 28:
        score -= 10
        temp_status = "🟡 SICAK"
        temp_advice = "Sabah 06:00-09:00 arasi uygulayin."
    elif temp < 5:
        score -= 30
        temp_status = "🔴 COK SOGUK"
        temp_advice = "Ilac etkisi azalir. Sicaklik yukselmesini bekleyin."
    elif temp < 12:
        score -= 10
        temp_status = "🟡 SERIN"
        temp_advice = "Ogle saatlerinde uygulayin."
    else:
        temp_status = "✅ IDEAL"
        temp_advice = "Sicaklik ilacalama icin uygun."

    # Nem kontrolu
    if humidity > 90:
        score -= 20
        humidity_status = "🟡 COK NEMLI"
        humidity_advice = "Ilac yapismasi zayif olabilir."
    elif humidity < 40:
        score -= 15
        humidity_status = "🟡 KURU"
        humidity_advice = "Ilac hizla kuruyabilir. Daha fazla su kullanin."
    else:
        humidity_status = "✅ IDEAL"
        humidity_advice = "Nem seviyesi uygun."

    # Yagmur kontrolu
    if rain_now:
        score -= 40
        rain_status = "🔴 YAGMURLU"
        rain_advice = "Simdi ilacalama YAPMAYIN! Yagmur ilaci yikayacak."
    elif rain_tomorrow:
        score -= 20
        rain_status = "🟡 YAGMUR BEKLENIYOR"
        rain_advice = f"Yakin saatlerde yagmur bekleniyor ({', '.join(rain_hours[:3])}). Ilaci en az 4 saat once uygulayin."
    else:
        rain_status = "✅ YAGMUR YOK"
        rain_advice = "24 saat icinde yagmur beklenmiyor. Ilacalama icin uygun."

    score = max(0, min(100, score))

    # Genel oneri
    if score >= 80:
        overall = "✅ MUKEMMEL"
        overall_color = "#2ED573"
        overall_advice = "Kosullar ilacalama icin ideal! Hemen uygulamayi baslatin."
    elif score >= 60:
        overall = "🟢 UYGUN"
        overall_color = "#7BED9F"
        overall_advice = "Ilacalama yapilabilir. Bazi kosullara dikkat edin."
    elif score >= 40:
        overall = "🟡 DIKKATLI"
        overall_color = "#FFA502"
        overall_advice = "Ilacalama riskli. Kosullarin iyilesmesini bekleyin."
    elif score >= 20:
        overall = "🟠 RISKLI"
        overall_color = "#FF6348"
        overall_advice = "Ilacalama onerilmez. Ertelemeniz onemle tavsiye edilir."
    else:
        overall = "🔴 UYGUN DEGIL"
        overall_color = "#FF4757"
        overall_advice = "Ilacalama YAPMAYIN! Kosullar cok uygunsuz."

    # En iyi zamanlama onerisi
    best_times = []
    if weather_forecast and "list" in weather_forecast:
        for item in weather_forecast["list"][:16]:  # 48 saat
            ft = item["main"]["temp"]
            fw = item["wind"]["speed"] * 3.6
            fh = item["main"]["humidity"]
            f_rain = any("rain" in w["main"].lower() for w in item["weather"])

            if not f_rain and fw < 15 and 12 < ft < 30 and 40 < fh < 85:
                dt = datetime.fromtimestamp(item["dt"])
                best_times.append(dt.strftime("%d/%m %H:%M"))

    return {
        "score": score,
        "overall": overall,
        "overall_color": overall_color,
        "overall_advice": overall_advice,
        "temp": temp,
        "temp_status": temp_status,
        "temp_advice": temp_advice,
        "humidity": humidity,
        "humidity_status": humidity_status,
        "humidity_advice": humidity_advice,
        "wind_speed": round(wind_speed, 1),
        "wind_status": wind_status,
        "wind_advice": wind_advice,
        "rain_now": rain_now,
        "rain_tomorrow": rain_tomorrow,
        "rain_status": rain_status,
        "rain_advice": rain_advice,
        "description": description,
        "icon_code": icon_code,
        "best_times": best_times[:5]
    }


# ═══════════════════════════════════════════════════════════════
# AI BİLGİ TABANI (ayni kalıyor)
# ═══════════════════════════════════════════════════════════════
WEED_DATABASE = {
    "genel": {
        "name": "Yabanci Ot",
        "diseases": [
            {"name": "Kurutucu Mantari", "risk": "Yuksek", "icon": "🍄",
             "detail": "Nemli ortamlarda yayilir. Yapraklarda kahverengi lekeler."},
            {"name": "Yaprak Leke Hastaligi", "risk": "Orta", "icon": "🦠",
             "detail": "Sari-kahverengi lekeler. Hava sirkülasyonu onemli."},
            {"name": "Kok Curumesi", "risk": "Dusuk", "icon": "🪱",
             "detail": "Asiri sulamada gorulur. Solgunluk ve sararma."}
        ],
        "treatment": [
            {"method": "Herbisit", "icon": "💊",
             "detail": "Secici herbisit ile hedefli uygulama. Sabah erken saatlerde.",
             "timing": "Ot 5-10 cm boyundayken"},
            {"method": "Mekanik Capa", "icon": "⛏️",
             "detail": "Elle veya makineli capalama. Kok cikarilmali.",
             "timing": "Haftada 1-2 kez"},
            {"method": "Malclama", "icon": "🌾",
             "detail": "Organik malc ile ortun. Cimlendirmeyi engeller.",
             "timing": "Ekim sonrasi"}
        ],
        "prevention": [
            "Ekim oncesi toprak hazirligi",
            "Sertifikali tohum kullanin",
            "Munavebe uygulayin",
            "Tarla kenarlarini temiz tutun"
        ]
    }
}

CROP_DATABASE = {
    "genel": {
        "health_tips": [
            {"tip": "Toprak Analizi", "icon": "🧪",
             "detail": "Yilda 1 kez toprak analizi yaptirin."},
            {"tip": "Gubreleme", "icon": "🌱",
             "detail": "Toprak analizine gore dengeli gubreleme."},
            {"tip": "Sulama", "icon": "💧",
             "detail": "Damla sulama en verimli yontemdir."},
            {"tip": "Hastalik Takibi", "icon": "🔍",
             "detail": "Haftada 1 tarla gezisi yapin."}
        ],
        "growth_stages": [
            {"stage": "Cimlendirme", "icon": "🌱", "days": "0-14",
             "advice": "Toprak nemini koruyun. 18-25C ideal."},
            {"stage": "Vejetatif", "icon": "🌿", "days": "15-45",
             "advice": "Azotlu gubre. Ot kontrolu."},
            {"stage": "Ciceklenme", "icon": "🌸", "days": "45-70",
             "advice": "Fosforlu gubre. Duzenli sulama."},
            {"stage": "Hasat", "icon": "🌾", "days": "120+",
             "advice": "Olgunluk kontrolu yapin."}
        ]
    }
}

SEASONAL_ADVICE = {
    1: {"season": "Kis", "icon": "❄️", "advice": "Toprak hazirligi ve planlama."},
    2: {"season": "Kis", "icon": "❄️", "advice": "Sera uretimi. Toprak analizi."},
    3: {"season": "Ilkbahar", "icon": "🌸", "advice": "Erken ekim. Toprak sicakligi."},
    4: {"season": "Ilkbahar", "icon": "🌸", "advice": "Ana ekim. Gubreleme planla."},
    5: {"season": "Ilkbahar", "icon": "🌸", "advice": "Ot mucadelesi kritik!"},
    6: {"season": "Yaz", "icon": "☀️", "advice": "Sulama kritik. Zararli takibi."},
    7: {"season": "Yaz", "icon": "☀️", "advice": "Sicaklik stresi. Golgeleme."},
    8: {"season": "Yaz", "icon": "☀️", "advice": "Erken hasat. Depolama."},
    9: {"season": "Sonbahar", "icon": "🍂", "advice": "Ana hasat. Kurutma."},
    10: {"season": "Sonbahar", "icon": "🍂", "advice": "Sonbahar ekimi. Toprak isleme."},
    11: {"season": "Sonbahar", "icon": "🍂", "advice": "Tarla temizligi."},
    12: {"season": "Kis", "icon": "❄️", "advice": "Yillik degerlendirme."}
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
.weather-card {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
}
.spray-score {
    font-size: 3em;
    font-weight: 800;
    text-align: center;
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
            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🌱 AgriVision AI Pro</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
        Detection | AI Advisor | Weather | Disease Control | GPS
    </p>
</div>
""", unsafe_allow_html=True)

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


@st.cache_resource
def load_model():
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp


def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0


def containment(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = max(1, (b1[2]-b1[0]) * (b1[3]-b1[1]))
    a2 = max(1, (b2[2]-b2[0]) * (b2[3]-b2[1]))
    return inter / min(a1, a2)


def class_aware_nms(boxes, scores, classes, iou_t=0.2, cont_t=0.6):
    if not boxes: return []
    keep_all = []
    for cls in set(classes):
        ci = [i for i, c in enumerate(classes) if c == cls]
        cb, cs = [boxes[i] for i in ci], [scores[i] for i in ci]
        idx = np.argsort(cs)[::-1].tolist()
        kp = []
        while idx:
            cur = idx.pop(0)
            kp.append(cur)
            rm = [j for j in idx if iou(cb[cur], cb[j]) > iou_t or containment(cb[cur], cb[j]) > cont_t]
            for r in rm:
                if r in idx: idx.remove(r)
        keep_all.extend([ci[k] for k in kp])
    return keep_all


def draw_detections(img, boxes, scores, classes, keep):
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 22)
    except: font = ImageFont.load_default()
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cls = classes[i]
        if cls == "WEED": color, bg, em = "#FF4757", "#FF6B7A", "🌿"
        else: color, bg, em = "#2ED573", "#51CF66", "🌾"
        draw.rectangle([x1,y1,x2,y2], outline=color, width=4)
        label = f"{em} {cls} {scores[i]:.0%}"
        bb = draw.textbbox((0,0), label, font=font)
        lw, lh = bb[2]-bb[0]+12, bb[3]-bb[1]+8
        draw.rectangle([x1, max(0,y1-lh), x1+lw, max(0,y1-lh)+lh], fill=bg)
        draw.text((x1+6, max(0,y1-lh)+4), label, fill="white", font=font)
    return img


def process_image(img, interp, inp, out, thresh, size_t, nms_iou):
    w, h = img.size
    ta = w * h
    arr = np.expand_dims(np.array(img.resize((640,640)), dtype=np.float32)/255.0, 0)
    interp.set_tensor(inp[0]["index"], arr)
    interp.invoke()
    preds = np.transpose(interp.get_tensor(out[0]["index"])[0], (1,0))
    rb, rs = [], []
    for row in preds:
        bs = float(np.max(row[4:]))
        if bs < thresh: continue
        x,y,bw,bh = row[0],row[1],row[2],row[3]
        x1,y1 = max(0,int((x-bw/2)*w)), max(0,int((y-bh/2)*h))
        x2,y2 = min(w,int((x+bw/2)*w)), min(h,int((y+bh/2)*h))
        if (x2-x1) > 10 and (y2-y1) > 10:
            rb.append([x1,y1,x2,y2]); rs.append(bs)
    ba, sa, ca, aa = [], [], [], []
    sl = ta * (size_t / 100)
    for i in range(len(rb)):
        a = (rb[i][2]-rb[i][0]) * (rb[i][3]-rb[i][1])
        ba.append(rb[i]); sa.append(rs[i])
        ca.append("WEED" if a < sl else "CROP"); aa.append(a)
    return ba, sa, ca, aa, class_aware_nms(ba, sa, ca, nms_iou, 0.6)


def generate_heatmap(boxes, scores, classes, keep, w, h):
    gs = 20
    r, c = h//gs+1, w//gs+1
    dm = np.zeros((r,c))
    for i in keep:
        if classes[i] == "WEED":
            x1,y1,x2,y2 = boxes[i]
            dm[max(0,y1//gs):min(r,y2//gs+1), max(0,x1//gs):min(c,x2//gs+1)] += scores[i]
    return dm


def get_ai_advice(wc, cc, wd, ac):
    a = {"weed_info": WEED_DATABASE["genel"], "crop_info": CROP_DATABASE["genel"],
         "seasonal": SEASONAL_ADVICE.get(datetime.now().month, {}), "actions": []}
    if wd > 30:
        a.update({"risk_level":"KRITIK","risk_color":"#FF0000","risk_icon":"🚨",
                  "summary":"Acil mudahale gerekli!",
                  "actions":["🚨 Acil herbisit","📞 Danismana haber","🔄 2 gun icinde tekrar tarama"]})
    elif wd > 20:
        a.update({"risk_level":"YUKSEK","risk_color":"#FF4757","risk_icon":"🔴",
                  "summary":"Herbisit uygulamasi onerilir.",
                  "actions":["💊 Secici herbisit","⛏️ Mekanik capa","📅 1 haftada kontrol"]})
    elif wd > 10:
        a.update({"risk_level":"ORTA","risk_color":"#FFA502","risk_icon":"🟡",
                  "summary":"Hedefli mudahale yeterli.",
                  "actions":["⛏️ Elle temizlik","🎯 Noktasal herbisit","📅 2 haftada kontrol"]})
    elif wc > 0:
        a.update({"risk_level":"DUSUK","risk_color":"#2ED573","risk_icon":"🟢",
                  "summary":"Koruyucu onlemler yeterli.",
                  "actions":["👁️ Gozlem","✋ Elle cekme","🌾 Malclama"]})
    else:
        a.update({"risk_level":"TEMIZ","risk_color":"#00D2D3","risk_icon":"✅",
                  "summary":"Tarla saglikli!",
                  "actions":["✅ Bakim devam","🌱 Gubreleme","💧 Sulama"]})
    return a


try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model hatasi: {e}"); st.stop()


# SIDEBAR
with st.sidebar:
    st.markdown("## 🎛️ Ayarlar")
    threshold = st.slider("🎯 Confidence", 0.1, 0.9, 0.60, 0.05)
    size_threshold = st.slider("📏 Boyut (%)", 5, 50, 42, 1)
    nms_iou = st.slider("🔗 NMS IoU", 0.1, 0.7, 0.20, 0.05)
    st.markdown("---")
    st.markdown("### 🗺️ GPS")
    gps_lat = st.number_input("📍 Enlem", value=39.9334, format="%.6f")
    gps_lon = st.number_input("📍 Boylam", value=32.8597, format="%.6f")
    st.markdown("---")
    st.markdown("### 🌡️ Hava Durumu API")
    weather_api_key = st.text_input("🔑 OpenWeather API Key", type="password",
                                     help="openweathermap.org'dan ucretsiz alin")
    st.markdown("---")
    st.markdown("### 📜 Gecmis")
    if st.session_state.scan_history:
        for s in st.session_state.scan_history[-5:][::-1]:
            st.markdown(f"• {s['time']} | 🌿{s['weeds']} 🌾{s['crops']} 📊{s['density']:.1f}%")
    if st.button("🗑️ Temizle", use_container_width=True):
        st.session_state.scan_history = []; st.rerun()


# SEKMELER
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📸 Fotograf", "🌡️ Hava & Ilacalama", "🧠 AI Danismanlik",
    "🎥 Video", "📊 Analitik", "🗺️ GPS"
])


# TAB 1: FOTOGRAF
with tab1:
    uploaded = st.file_uploader("📁 Tarla Fotografi", type=["jpg","png","jpeg"], key="photo")
    if uploaded:
        original_img = Image.open(uploaded).convert("RGB")
        w, h = original_img.size
        ta = w * h
        with st.spinner("🔍 Analiz..."):
            ba, sa, ca, aa, ki = process_image(original_img, interpreter, input_details,
                                                output_details, threshold, size_threshold, nms_iou)
        result_img = original_img.copy()
        if ki: result_img = draw_detections(result_img, ba, sa, ca, ki)

        wc = sum(1 for i in ki if ca[i]=="WEED")
        cc = sum(1 for i in ki if ca[i]=="CROP")
        wa = sum(aa[i] for i in ki if ca[i]=="WEED")
        wd = (wa/ta)*100 if ta > 0 else 0
        ac = float(np.mean([sa[i] for i in ki])) if ki else 0

        st.session_state.scan_history.append({"time":datetime.now().strftime("%H:%M:%S"),
            "weeds":wc,"crops":cc,"density":float(wd),"confidence":ac})

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌿</div><div class='metric-value' style='color:#FF4757'>{wc}</div><div class='metric-label'>WEEDS</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌾</div><div class='metric-value' style='color:#2ED573'>{cc}</div><div class='metric-label'>CROPS</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='glass-card'><div class='metric-icon'>📊</div><div class='metric-value' style='color:#FFA502'>{wd:.1f}%</div><div class='metric-label'>DENSITY</div></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='glass-card'><div class='metric-icon'>🎯</div><div class='metric-value' style='color:#3742FA'>{ac:.0%}</div><div class='metric-label'>CONF</div></div>", unsafe_allow_html=True)

        i1,i2 = st.columns(2)
        with i1: st.image(original_img, caption="📸 Orijinal", use_container_width=True)
        with i2: st.image(result_img, caption="🎯 Sonuc", use_container_width=True)

        if wc > 0:
            st.markdown("### 📈 Heatmap")
            dm = generate_heatmap(ba, sa, ca, ki, w, h)
            fig = px.imshow(dm, color_continuous_scale="RdYlGn_r")
            fig.update_layout(height=400, margin=dict(r=0,t=10,l=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💾 Indir")
        d1,d2 = st.columns(2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with d1:
            buf = io.BytesIO(); result_img.save(buf, format="PNG")
            st.download_button("🖼️ PNG", buf.getvalue(), f"agri_{ts}.png", "image/png", use_container_width=True)
        with d2:
            jr = {"timestamp":ts,"weeds":wc,"crops":cc,"density":round(wd,2)}
            st.download_button("📋 JSON", json.dumps(jr,indent=2), f"agri_{ts}.json", "application/json", use_container_width=True)


# TAB 2: HAVA DURUMU & ILACALAMA
with tab2:
    st.markdown("<div class='tab-header'>🌡️ Hava Durumu & Akilli Ilacalama Zamanlama</div>", unsafe_allow_html=True)

    if weather_api_key:
        with st.spinner("🌤️ Hava durumu aliniyor..."):
            current, forecast = get_weather(gps_lat, gps_lon, weather_api_key)

        if current and "main" in current:
            spray = analyze_spray_conditions(current, forecast)

            if spray:
                # Ilacalama Skoru
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {spray["overall_color"]}44, {spray["overall_color"]}22);
                            padding: 2rem; border-radius: 20px; text-align: center;
                            border: 2px solid {spray["overall_color"]}; margin-bottom: 1.5rem;'>
                    <h2 style='margin:0;'>💊 Ilacalama Uygunluk Skoru</h2>
                    <div class='spray-score' style='color:{spray["overall_color"]};'>{spray["score"]}/100</div>
                    <h3 style='margin:0;'>{spray["overall"]}</h3>
                    <p style='font-size:1.1em;'>{spray["overall_advice"]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Hava Durumu Kartlari
                st.markdown("### 🌤️ Guncel Hava Durumu")
                w1, w2, w3, w4 = st.columns(4)

                with w1:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div class='metric-icon'>🌡️</div>
                        <div class='metric-value'>{spray["temp"]:.1f}°C</div>
                        <div class='metric-label'>{spray["temp_status"]}</div>
                    </div>""", unsafe_allow_html=True)

                with w2:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div class='metric-icon'>💨</div>
                        <div class='metric-value'>{spray["wind_speed"]} km/h</div>
                        <div class='metric-label'>{spray["wind_status"]}</div>
                    </div>""", unsafe_allow_html=True)

                with w3:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div class='metric-icon'>💧</div>
                        <div class='metric-value'>{spray["humidity"]}%</div>
                        <div class='metric-label'>{spray["humidity_status"]}</div>
                    </div>""", unsafe_allow_html=True)

                with w4:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div class='metric-icon'>🌧️</div>
                        <div class='metric-value'>{"Var" if spray["rain_now"] else "Yok"}</div>
                        <div class='metric-label'>{spray["rain_status"]}</div>
                    </div>""", unsafe_allow_html=True)

                # Detayli Oneriler
                st.markdown("### 📋 Detayli Kosul Analizi")

                st.markdown(f"""
                <div class='weather-card' style='border-color: {"#FF4757" if spray["temp"]>35 or spray["temp"]<5 else "#2ED573"};'>
                    🌡️ <b>Sicaklik:</b> {spray["temp_advice"]}
                </div>
                <div class='weather-card' style='border-color: {"#FF4757" if spray["wind_speed"]>15 else "#2ED573"};'>
                    💨 <b>Ruzgar:</b> {spray["wind_advice"]}
                </div>
                <div class='weather-card' style='border-color: {"#FF4757" if spray["humidity"]>90 else "#2ED573"};'>
                    💧 <b>Nem:</b> {spray["humidity_advice"]}
                </div>
                <div class='weather-card' style='border-color: {"#FF4757" if spray["rain_now"] or spray["rain_tomorrow"] else "#2ED573"};'>
                    🌧️ <b>Yagis:</b> {spray["rain_advice"]}
                </div>
                """, unsafe_allow_html=True)

                # En Iyi Zamanlar
                if spray["best_times"]:
                    st.markdown("### ⏰ En Iyi Ilacalama Zamanlari (48 saat)")
                    for idx, bt in enumerate(spray["best_times"]):
                        st.markdown(f"""
                        <div class='weather-card' style='border-color: #2ED573;'>
                            ✅ <b>Oneri {idx+1}:</b> {bt} - Kosullar uygun!
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ 48 saat icinde ideal ilacalama zamani bulunamadi.")

                # 5 Gunluk Tahmin Grafigi
                if forecast and "list" in forecast:
                    st.markdown("### 📈 5 Gunluk Hava Tahmini")
                    times_f = [datetime.fromtimestamp(i["dt"]).strftime("%d/%m %H:%M") for i in forecast["list"][:20]]
                    temps_f = [i["main"]["temp"] for i in forecast["list"][:20]]
                    winds_f = [i["wind"]["speed"]*3.6 for i in forecast["list"][:20]]
                    humids_f = [i["main"]["humidity"] for i in forecast["list"][:20]]

                    fig_w = go.Figure()
                    fig_w.add_trace(go.Scatter(x=times_f, y=temps_f, name="🌡️ Sicaklik (C)",
                                              line=dict(color="#FF6348", width=2)))
                    fig_w.add_trace(go.Scatter(x=times_f, y=winds_f, name="💨 Ruzgar (km/h)",
                                              line=dict(color="#1E90FF", width=2)))
                    fig_w.add_trace(go.Scatter(x=times_f, y=humids_f, name="💧 Nem (%)",
                                              line=dict(color="#2ED573", width=2)))
                    fig_w.add_hline(y=15, line_dash="dash", line_color="red",
                                   annotation_text="Ruzgar Limiti (15 km/h)")
                    fig_w.update_layout(height=400, xaxis_title="Tarih/Saat")
                    st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.error("❌ Hava durumu alinamadi. API key'inizi kontrol edin.")
    else:
        st.markdown("""
        ### 🔑 API Key Gerekli
        
        1. **https://openweathermap.org** adresine gidin
        2. Ucretsiz hesap olusturun
        3. API Keys bolumunden key kopyalayin
        4. Sol panelde **API Key** alanina yapistring
        
        > Ucretsiz plan: Gunluk 1000 sorgu yeterli!
        """)


# TAB 3: AI DANISMANLIK
with tab3:
    st.markdown("<div class='tab-header'>🧠 AI Tarim Danismanligi</div>", unsafe_allow_html=True)
    if st.session_state.scan_history:
        last = st.session_state.scan_history[-1]
        adv = get_ai_advice(last["weeds"], last["crops"], last["density"], last["confidence"])

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {adv["risk_color"]}44, {adv["risk_color"]}22);
                    padding: 1.5rem; border-radius: 20px; border-left: 6px solid {adv["risk_color"]};'>
            <h2>{adv["risk_icon"]} Risk: {adv["risk_level"]}</h2>
            <p style='font-size:1.1em;'>{adv["summary"]}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Aksiyon Plani")
        for idx, act in enumerate(adv["actions"]):
            st.markdown(f"<div class='advice-card' style='border-color:{adv[\"risk_color\"]};'><b>Adim {idx+1}:</b> {act}</div>", unsafe_allow_html=True)

        sea = adv["seasonal"]
        if sea:
            st.markdown(f"### {sea.get('icon','')} Mevsimsel ({sea.get('season','')})")
            st.markdown(f"<div class='advice-card' style='border-color:#667eea;'>{sea.get('advice','')}</div>", unsafe_allow_html=True)

        if last["weeds"] > 0:
            st.markdown("### 🦠 Hastaliklar")
            for d in adv["weed_info"]["diseases"]:
                rc = "#FF4757" if d["risk"]=="Yuksek" else "#FFA502" if d["risk"]=="Orta" else "#2ED573"
                st.markdown(f"<div class='disease-card'><h4>{d['icon']} {d['name']} <span style='background:{rc};color:white;padding:2px 8px;border-radius:5px;'>{d['risk']}</span></h4><p>{d['detail']}</p></div>", unsafe_allow_html=True)

            st.markdown("### 💊 Tedavi")
            for t in adv["weed_info"]["treatment"]:
                st.markdown(f"<div class='treatment-card'><h4>{t['icon']} {t['method']}</h4><p>{t['detail']}</p><small>⏰ {t['timing']}</small></div>", unsafe_allow_html=True)

        if last["crops"] > 0:
            st.markdown("### 🌾 Mahsul Bakimi")
            for t in adv["crop_info"]["health_tips"]:
                st.markdown(f"<div class='treatment-card'><h4>{t['icon']} {t['tip']}</h4><p>{t['detail']}</p></div>", unsafe_allow_html=True)
    else:
        st.info("📸 Once fotograf sekmesinden tarama yapin!")


# TAB 4: VIDEO
with tab4:
    st.markdown("<div class='tab-header'>🎥 Video</div>", unsafe_allow_html=True)
    vf = st.file_uploader("🎥 Video", type=["mp4","avi","mov"], key="vid")
    if vf:
        tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf2.write(vf.read()); tf2.close()
        cap = cv2.VideoCapture(tf2.name)
        totf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        st.info(f"📹 {totf} kare | {fps} FPS")
        fskip = st.slider("⏭️ Aralik", 5, 60, 30, 5)
        if st.button("🚀 Baslat", use_container_width=True):
            prog = st.progress(0); stat = st.empty(); prev = st.empty()
            res = []; fi = 0; pc = 0
            while cap.isOpened():
                ret, fr = cap.read()
                if not ret: break
                if fi % fskip == 0:
                    pf = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    b,s,cl,a,k = process_image(pf, interpreter, input_details, output_details, threshold, size_threshold, nms_iou)
                    wc2 = sum(1 for i in k if cl[i]=="WEED")
                    res.append({"time":fi/fps,"weeds":wc2,"crops":sum(1 for i in k if cl[i]=="CROP")})
                    if k:
                        rf = pf.copy(); rf = draw_detections(rf,b,s,cl,k)
                        prev.image(rf, use_container_width=True)
                    pc += 1; prog.progress(min(fi/max(totf,1),1.0))
                    stat.text(f"Kare: {pc}")
                fi += 1
            cap.release(); prog.progress(1.0); stat.success(f"✅ {pc} kare")
            if res:
                fv = go.Figure()
                fv.add_trace(go.Scatter(x=[r["time"] for r in res], y=[r["weeds"] for r in res],
                                        mode="lines+markers", name="🌿", line=dict(color="#FF4757",width=3)))
                fv.update_layout(height=400)
                st.plotly_chart(fv, use_container_width=True)
    else: st.info("🎥 Video yukleyin")


# TAB 5: ANALITIK
with tab5:
    st.markdown("<div class='tab-header'>📊 Analitik</div>", unsafe_allow_html=True)
    if st.session_state.scan_history:
        hist = st.session_state.scan_history
        x = list(range(1, len(hist)+1))
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=x, y=[h["weeds"] for h in hist], name="🌿", marker_color="#FF4757"))
        fig1.add_trace(go.Bar(x=x, y=[h["crops"] for h in hist], name="🌾", marker_color="#2ED573"))
        fig1.update_layout(barmode="group", height=380)
        st.plotly_chart(fig1, use_container_width=True)
    else: st.info("📸 Tarama yapin")


# TAB 6: GPS
with tab6:
    st.markdown("<div class='tab-header'>🗺️ GPS</div>", unsafe_allow_html=True)
    st.map(pd.DataFrame({"lat":[gps_lat],"lon":[gps_lon]}), zoom=14)


# FOOTER
st.markdown("""
<div style='text-align:center; padding:2rem; margin-top:3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius:20px; color:white;'>
    <h3>🌾 AgriVision AI Pro v3.0</h3>
    <p>📸 Detection | 🌡️ Weather | 🧠 AI Advisor | 🎥 Video | 📊 Analytics | 🗺️ GPS</p>
</div>
""", unsafe_allow_html=True)
