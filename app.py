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
# VARSAYILAN API KEY
# ═══════════════════════════════════════════════════════════════
DEFAULT_WEATHER_API_KEY = "ff84a945e17df9578ea8b2c85d6acaf3"

# ═══════════════════════════════════════════════════════════════
# ÇOK DİLLİ DESTEK
# ═══════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "tr": {
        "title": "AgriVision AI Pro",
        "subtitle": "Detection | AI Advisor | Weather | Disease Control | GPS",
        "settings": "Ayarlar",
        "confidence": "Confidence",
        "size_threshold": "Boyut (%)",
        "nms_iou": "NMS IoU",
        "gps": "GPS",
        "latitude": "Enlem",
        "longitude": "Boylam",
        "weather_api": "Hava Durumu API",
        "use_default_key": "Varsayılan API Key kullan",
        "api_key_active": "Varsayılan API Key aktif",
        "custom_api_key": "Özel API Key girildi",
        "test_api": "API Key Test Et",
        "history": "Geçmiş",
        "clear": "Temizle",
        "theme": "Tema",
        "language": "Dil",
        "dark_mode": "Karanlık Mod",
        "light_mode": "Aydınlık Mod",
        "alert_settings": "Uyarı Ayarları",
        "density_threshold": "Yoğunluk Eşiği (%)",
        "enable_alerts": "Uyarıları Aç",
        "photo_tab": "📸 Fotoğraf",
        "weather_tab": "🌡️ Hava & İlaçlama",
        "ai_tab": "🧠 AI Danışmanlık",
        "video_tab": "🎥 Video",
        "analytics_tab": "📊 Analitik",
        "gps_tab": "🗺️ GPS",
        "multi_photo_tab": "📸 Çoklu Fotoğraf",
        "upload_photos": " Yabancı Ot Fotoğrafları (Max 10)",
        "upload_photo": " Yabancı Ot Fotoğrafı",
        "analyzing": "Analiz ediliyor...",
        "weeds": "YABANCI OT",
        "crops": "MAHSUL",
        "density": "YOĞUNLUK",
        "conf": "GÜVEN",
        "original": "Orijinal",
        "result": "Sonuç",
        "heatmap": "Isı Haritası",
        "download": "İndir",
        "detection_table": "Detection Tablosu",
        "image": "Görsel",
        "type": "Tür",
        "confidence_score": "Güven Skoru",
        "area": "Alan (px²)",
        "position": "Pozisyon",
        "critical_alert": "KRİTİK UYARI",
        "high_weed_density": "Yüksek yabancı ot yoğunluğu tespit edildi!",
        "immediate_action": "Acil müdahale gerekli!",
    },
    "en": {
        "title": "AgriVision AI Pro",
        "subtitle": "Detection | AI Advisor | Weather | Disease Control | GPS",
        "settings": "Settings",
        "confidence": "Confidence",
        "size_threshold": "Size (%)",
        "nms_iou": "NMS IoU",
        "gps": "GPS",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "weather_api": "Weather API",
        "use_default_key": "Use default API Key",
        "api_key_active": "Default API Key active",
        "custom_api_key": "Custom API Key entered",
        "test_api": "Test API Key",
        "history": "History",
        "clear": "Clear",
        "theme": "Theme",
        "language": "Language",
        "dark_mode": "Dark Mode",
        "light_mode": "Light Mode",
        "alert_settings": "Alert Settings",
        "density_threshold": "Density Threshold (%)",
        "enable_alerts": "Enable Alerts",
        "photo_tab": "📸 Photo",
        "weather_tab": "🌡️ Weather & Spraying",
        "ai_tab": "🧠 AI Advisor",
        "video_tab": "🎥 Video",
        "analytics_tab": "📊 Analytics",
        "gps_tab": "🗺️ GPS",
        "multi_photo_tab": "📸 Multi Photo",
        "upload_photos": "Field Photos (Max 10)",
        "upload_photo": "Field Photo",
        "analyzing": "Analyzing...",
        "weeds": "WEEDS",
        "crops": "CROPS",
        "density": "DENSITY",
        "conf": "CONF",
        "original": "Original",
        "result": "Result",
        "heatmap": "Heatmap",
        "download": "Download",
        "detection_table": "Detection Table",
        "image": "Image",
        "type": "Type",
        "confidence_score": "Confidence Score",
        "area": "Area (px²)",
        "position": "Position",
        "critical_alert": "CRITICAL ALERT",
        "high_weed_density": "High weed density detected!",
        "immediate_action": "Immediate action required!",
    },
    "ar": {
        "title": "AgriVision AI Pro",
        "subtitle": "الكشف | مستشار الذكاء الاصطناعي | الطقس | مكافحة الأمراض | GPS",
        "settings": "الإعدادات",
        "confidence": "الثقة",
        "size_threshold": "الحجم (%)",
        "nms_iou": "NMS IoU",
        "gps": "GPS",
        "latitude": "خط العرض",
        "longitude": "خط الطول",
        "weather_api": "API الطقس",
        "use_default_key": "استخدم المفتاح الافتراضي",
        "api_key_active": "المفتاح الافتراضي نشط",
        "custom_api_key": "تم إدخال مفتاح مخصص",
        "test_api": "اختبار المفتاح",
        "history": "السجل",
        "clear": "مسح",
        "theme": "المظهر",
        "language": "اللغة",
        "dark_mode": "الوضع الداكن",
        "light_mode": "الوضع الفاتح",
        "alert_settings": "إعدادات التنبيه",
        "density_threshold": "عتبة الكثافة (%)",
        "enable_alerts": "تفعيل التنبيهات",
        "photo_tab": "📸 صورة",
        "weather_tab": "🌡️ الطقس والرش",
        "ai_tab": "🧠 مستشار الذكاء",
        "video_tab": "🎥 فيديو",
        "analytics_tab": "📊 التحليلات",
        "gps_tab": "🗺️ GPS",
        "multi_photo_tab": "📸 صور متعددة",
        "upload_photos": "صور الحقل (حد أقصى 10)",
        "upload_photo": "صورة الحقل",
        "analyzing": "جاري التحليل...",
        "weeds": "الأعشاب",
        "crops": "المحاصيل",
        "density": "الكثافة",
        "conf": "الثقة",
        "original": "الأصلي",
        "result": "النتيجة",
        "heatmap": "خريطة الحرارة",
        "download": "تحميل",
        "detection_table": "جدول الكشف",
        "image": "الصورة",
        "type": "النوع",
        "confidence_score": "درجة الثقة",
        "area": "المساحة (px²)",
        "position": "الموضع",
        "critical_alert": "تنبيه حرج",
        "high_weed_density": "تم اكتشاف كثافة عالية من الأعشاب!",
        "immediate_action": "مطلوب إجراء فوري!",
    }
}

def t(key):
    """Get translation for current language"""
    lang = st.session_state.get('language', 'tr')
    return TRANSLATIONS.get(lang, TRANSLATIONS['tr']).get(key, key)

# ═══════════════════════════════════════════════════════════════
# TEMA YÖNETİMİ
# ═══════════════════════════════════════════════════════════════
def get_theme_css():
    """Get CSS based on selected theme"""
    theme = st.session_state.get('theme', 'dark')
    
    if theme == 'dark':
        return """
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
        .metric-value { font-size: 2em; font-weight: 700; margin: 0; color: white; }
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
        .weather-card, .advice-card {
            background: rgba(255,255,255,0.08);
            border-radius: 15px;
            padding: 1.2rem;
            margin: 0.5rem 0;
            border-left: 4px solid;
            color: white;
        }
        .api-status {
            background: rgba(46,213,115,0.15);
            border: 1px solid #2ED573;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            text-align: center;
            margin-top: 0.5rem;
            color: white;
        }
        .alert-box {
            background: linear-gradient(135deg, #FF4757 0%, #FF6348 100%);
            border: 2px solid #FF0000;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
            color: white;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        </style>
        """
    else:  # light theme
        return """
        <style>
        .glass-card {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            padding: 1.5rem;
            text-align: center;
        }
        .metric-icon { font-size: 2.2em; }
        .metric-value { font-size: 2em; font-weight: 700; margin: 0; color: #333; }
        .metric-label { color: rgba(0,0,0,0.7); margin: 0; font-size: 0.9em; }
        .tab-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0.8rem 1.5rem;
            border-radius: 15px;
            color: white;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        .weather-card, .advice-card {
            background: rgba(0,0,0,0.03);
            border-radius: 15px;
            padding: 1.2rem;
            margin: 0.5rem 0;
            border-left: 4px solid;
            color: #333;
        }
        .api-status {
            background: rgba(46,213,115,0.1);
            border: 1px solid #2ED573;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            text-align: center;
            margin-top: 0.5rem;
            color: #2ED573;
        }
        .alert-box {
            background: linear-gradient(135deg, #FF4757 0%, #FF6348 100%);
            border: 2px solid #FF0000;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
            color: white;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        </style>
        """

# ═══════════════════════════════════════════════════════════════
# HAVA DURUMU FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════
def get_weather(lat, lon, api_key):
    """OpenWeatherMap'den hava durumu al"""
    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr"
        current = requests.get(current_url, timeout=5).json()

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
    wind_speed = weather_current["wind"]["speed"] * 3.6
    description = weather_current["weather"][0]["description"]
    icon_code = weather_current["weather"][0]["icon"]

    rain_now = "rain" in weather_current or "Rain" in description or "yagmur" in description.lower()

    rain_tomorrow = False
    rain_hours = []
    if weather_forecast and "list" in weather_forecast:
        for item in weather_forecast["list"][:8]:
            if "rain" in item or any("rain" in w["main"].lower() for w in item["weather"]):
                rain_tomorrow = True
                dt = datetime.fromtimestamp(item["dt"])
                rain_hours.append(dt.strftime("%H:%M"))

    score = 100

    if wind_speed > 25:
        score -= 50
        wind_status = "🔴 TEHLIKELI"
    elif wind_speed > 15:
        score -= 30
        wind_status = "🟡 RISKLI"
    elif wind_speed > 8:
        score -= 10
        wind_status = "🟢 DIKKATLI"
    else:
        wind_status = "✅ IDEAL"

    if temp > 35:
        score -= 30
        temp_status = "🔴 COK SICAK"
    elif temp > 28:
        score -= 10
        temp_status = "🟡 SICAK"
    elif temp < 5:
        score -= 30
        temp_status = "🔴 COK SOGUK"
    elif temp < 12:
        score -= 10
        temp_status = "🟡 SERIN"
    else:
        temp_status = "✅ IDEAL"

    if humidity > 90:
        score -= 20
        humidity_status = "🟡 COK NEMLI"
    elif humidity < 40:
        score -= 15
        humidity_status = "🟡 KURU"
    else:
        humidity_status = "✅ IDEAL"

    if rain_now:
        score -= 40
        rain_status = "🔴 YAGMURLU"
    elif rain_tomorrow:
        score -= 20
        rain_status = "🟡 YAGMUR BEKLENIYOR"
    else:
        rain_status = "✅ YAGMUR YOK"

    score = max(0, min(100, score))

    if score >= 80:
        overall = "✅ MUKEMMEL"
        overall_color = "#2ED573"
    elif score >= 60:
        overall = "🟢 UYGUN"
        overall_color = "#7BED9F"
    elif score >= 40:
        overall = "🟡 DIKKATLI"
        overall_color = "#FFA502"
    elif score >= 20:
        overall = "🟠 RISKLI"
        overall_color = "#FF6348"
    else:
        overall = "🔴 UYGUN DEGIL"
        overall_color = "#FF4757"

    return {
        "score": score,
        "overall": overall,
        "overall_color": overall_color,
        "temp": temp,
        "temp_status": temp_status,
        "humidity": humidity,
        "humidity_status": humidity_status,
        "wind_speed": round(wind_speed, 1),
        "wind_status": wind_status,
        "rain_status": rain_status,
    }

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "language" not in st.session_state:
    st.session_state.language = "tr"
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "alert_threshold" not in st.session_state:
    st.session_state.alert_threshold = 25.0
if "enable_alerts" not in st.session_state:
    st.session_state.enable_alerts = True

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriVision AI Pro",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# Apply theme
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Header
st.markdown(f"""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🌱 {t('title')}</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
        {t('subtitle')}
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp

# ═══════════════════════════════════════════════════════════════
# DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0

def containment(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
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
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cls = classes[i]
        if cls == "WEED":
            color, bg, em = "#FF4757", "#FF6B7A", "🌿"
        else:
            color, bg, em = "#2ED573", "#51CF66", "🌾"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{em} {cls} {scores[i]:.0%}"
        bb = draw.textbbox((0, 0), label, font=font)
        lw, lh = bb[2] - bb[0] + 12, bb[3] - bb[1] + 8
        draw.rectangle([x1, max(0, y1 - lh), x1 + lw, max(0, y1 - lh) + lh], fill=bg)
        draw.text((x1 + 6, max(0, y1 - lh) + 4), label, fill="white", font=font)
    return img
    
def process_image(img, interp, inp, out, thresh, size_t, nms_iou):
    w, h = img.size
    ta = w * h

    arr = np.expand_dims(
        np.array(img.resize((640, 640)), dtype=np.float32) / 255.0, 0
    )
    interp.set_tensor(inp[0]["index"], arr)
    interp.invoke()

    raw = interp.get_tensor(out[0]["index"])[0]
    preds = np.transpose(raw, (1, 0))

    # ══════ DEBUG BİLGİ ══════
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 DEBUG")
    st.sidebar.write(f"Raw shape: {raw.shape}")
    st.sidebar.write(f"Preds shape: {preds.shape}")
    st.sidebar.write(f"Sınıf sayısı: {preds.shape[1] - 4}")

    # İlk yüksek skorlu satırları göster
    high_score_count = 0
    sample_rows = []
    for row in preds:
        class_scores = row[4:]
        best = float(np.max(class_scores))
        if best > 0.05:
            high_score_count += 1
            if len(sample_rows) < 5:
                sample_rows.append({
                    "score": f"{best:.3f}",
                    "class_id": int(np.argmax(class_scores)),
                    "all_scores": [f"{s:.3f}" for s in class_scores],
                    "box": [f"{v:.2f}" for v in row[:4]]
                })

    st.sidebar.write(f"Score > 0.05: **{high_score_count}** adet")
    st.sidebar.write(f"Score > thresh({thresh}): kontrol ediliyor...")
    
    for i, sr in enumerate(sample_rows):
        st.sidebar.write(f"#{i}: cls={sr['class_id']}, "
                        f"score={sr['score']}, "
                        f"scores={sr['all_scores']}")
    # ══════ DEBUG BİTİŞ ══════

    CLASS_NAMES = {0: "CROP", 1: "WEED"}

    ba, sa, ca, aa = [], [], [], []

    for row in preds:
        class_scores = row[4:]
        best_score = float(np.max(class_scores))
        class_id = int(np.argmax(class_scores))

        if best_score < thresh:
            continue

        x, y, bw, bh = row[0], row[1], row[2], row[3]
        x1 = max(0, int((x - bw / 2) * w))
        y1 = max(0, int((y - bh / 2) * h))
        x2 = min(w, int((x + bw / 2) * w))
        y2 = min(h, int((y + bh / 2) * h))

        if (x2 - x1) > 10 and (y2 - y1) > 10:
            a = (x2 - x1) * (y2 - y1)
            ba.append([x1, y1, x2, y2])
            sa.append(best_score)
            ca.append(CLASS_NAMES.get(class_id, "WEED"))
            aa.append(a)

    # DEBUG: Filtre öncesi sayı
    st.sidebar.write(f"Filtre öncesi: **{len(ba)}** tespit")
    if ca:
        weed_n = sum(1 for c in ca if c == "WEED")
        crop_n = sum(1 for c in ca if c == "CROP")
        st.sidebar.write(f"WEED: {weed_n} | CROP: {crop_n}")

    keep = class_aware_nms(ba, sa, ca, nms_iou, 0.6)

    # DEBUG: NMS sonrası
    st.sidebar.write(f"NMS sonrası: **{len(keep)}** tespit")

    return ba, sa, ca, aa, keep
    
def generate_heatmap(boxes, scores, classes, keep, w, h):
    gs = 20
    r, c = h // gs + 1, w // gs + 1
    dm = np.zeros((r, c))
    for i in keep:
        if classes[i] == "WEED":
            x1, y1, x2, y2 = boxes[i]
            dm[max(0, y1 // gs):min(r, y2 // gs + 1), max(0, x1 // gs):min(c, x2 // gs + 1)] += scores[i]
    return dm

def create_detection_table(boxes, scores, classes, keep, img_name=""):
    """Create detailed detection table"""
    data = []
    for idx, i in enumerate(keep):
        x1, y1, x2, y2 = boxes[i]
        data.append({
            t('image'): img_name or f"Detection #{idx+1}",
            t('type'): t('weeds') if classes[i] == "WEED" else t('crops'),
            t('confidence_score'): f"{scores[i]:.2%}",
            t('area'): f"{(x2-x1)*(y2-y1):.0f}",
            t('position'): f"({x1:.0f}, {y1:.0f})",
        })
    return pd.DataFrame(data)

def show_alert(density):
    """Show alert if density exceeds threshold"""
    if st.session_state.enable_alerts and density > st.session_state.alert_threshold:
        st.markdown(f"""
        <div class='alert-box'>
            <h2>🚨 {t('critical_alert')} 🚨</h2>
            <h3>{t('high_weed_density')}</h3>
            <p style='font-size:1.5em;'>📊 {density:.1f}%</p>
            <p>{t('immediate_action')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.audio("https://www.soundjay.com/buttons/sounds/beep-07.mp3", autoplay=True)

# Load model
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
    # Language & Theme
    st.markdown(f"## ⚙️ {t('settings')}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.language = st.selectbox(
            f"🌍 {t('language')}",
            options=["tr", "en", "ar"],
            format_func=lambda x: {"tr": "🇹🇷 Türkçe", "en": "🇬🇧 English", "ar": "🇸🇦 العربية"}[x],
            key="lang_selector"
        )
    
    with col2:
        st.session_state.theme = st.selectbox(
            f"🎨 {t('theme')}",
            options=["dark", "light"],
            format_func=lambda x: t('dark_mode') if x == "dark" else t('light_mode'),
            key="theme_selector"
        )
    
    st.markdown("---")
    
    # Detection Settings
    threshold = st.slider(f"🎯 {t('confidence')}", 0.1, 0.9, 0.60, 0.05)
    size_threshold = st.slider(f"📏 {t('size_threshold')}", 5, 50, 42, 1)
    nms_iou = st.slider(f"🔗 {t('nms_iou')}", 0.1, 0.7, 0.20, 0.05)
    
    st.markdown("---")
    
    # Alert Settings
    st.markdown(f"### 🔔 {t('alert_settings')}")
    st.session_state.enable_alerts = st.checkbox(t('enable_alerts'), value=True)
    st.session_state.alert_threshold = st.slider(
        f"📊 {t('density_threshold')}",
        5.0, 50.0, 25.0, 5.0,
        disabled=not st.session_state.enable_alerts
    )
    
    st.markdown("---")
    
    # GPS
    st.markdown(f"### 🗺️ {t('gps')}")
    gps_lat = st.number_input(f"📍 {t('latitude')}", value=39.9334, format="%.6f")
    gps_lon = st.number_input(f"📍 {t('longitude')}", value=32.8597, format="%.6f")
    
    st.markdown("---")
    
    # Weather API
    st.markdown(f"### 🌡️ {t('weather_api')}")
    use_default_key = st.checkbox(t('use_default_key'), value=True)
    
    if use_default_key:
        weather_api_key = DEFAULT_WEATHER_API_KEY
        st.markdown(f"""
        <div class='api-status'>
            ✅ {t('api_key_active')}
        </div>
        """, unsafe_allow_html=True)
    else:
        weather_api_key = st.text_input("🔑 API Key", type="password")
        if weather_api_key:
            st.markdown(f"""
            <div class='api-status'>
                ✅ {t('custom_api_key')}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # History
    st.markdown(f"### 📜 {t('history')}")
    if st.session_state.scan_history:
        for s in st.session_state.scan_history[-5:][::-1]:
            st.markdown(f"• {s['time']} | 🌿{s['weeds']} 🌾{s['crops']} 📊{s['density']:.1f}%")
    if st.button(f"🗑️ {t('clear')}", use_container_width=True):
        st.session_state.scan_history = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    t('photo_tab'),
    t('multi_photo_tab'),
    t('weather_tab'),
    t('ai_tab'),
    t('video_tab'),
    t('analytics_tab'),
    t('gps_tab')
])

# TAB 1: Single Photo
with tab1:
    uploaded = st.file_uploader(f"📁 {t('upload_photo')}", type=["jpg", "png", "jpeg"], key="photo")
    if uploaded:
        original_img = Image.open(uploaded).convert("RGB")
        w, h = original_img.size
        ta = w * h
        
        with st.spinner(f"🔍 {t('analyzing')}"):
            ba, sa, ca, aa, ki = process_image(original_img, interpreter, input_details,
                                                output_details, threshold, size_threshold, nms_iou)
        
        result_img = original_img.copy()
        if ki:
            result_img = draw_detections(result_img, ba, sa, ca, ki)
        
        wc = sum(1 for i in ki if ca[i] == "WEED")
        cc = sum(1 for i in ki if ca[i] == "CROP")
        wa = sum(aa[i] for i in ki if ca[i] == "WEED")
        wd = (wa / ta) * 100 if ta > 0 else 0
        ac = float(np.mean([sa[i] for i in ki])) if ki else 0
        
        # Save to history
        st.session_state.scan_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "weeds": wc,
            "crops": cc,
            "density": float(wd),
            "confidence": ac
        })
        
        # Show alert
        show_alert(wd)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌿</div><div class='metric-value' style='color:#FF4757'>{wc}</div><div class='metric-label'>{t('weeds')}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌾</div><div class='metric-value' style='color:#2ED573'>{cc}</div><div class='metric-label'>{t('crops')}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>📊</div><div class='metric-value' style='color:#FFA502'>{wd:.1f}%</div><div class='metric-label'>{t('density')}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🎯</div><div class='metric-value' style='color:#3 742FA'>{ac:.0%}</div><div class='metric-label'>{t('conf')}</div></div>", unsafe_allow_html=True)
        
        # Images
        i1, i2 = st.columns(2)
        with i1:
            st.image(original_img, caption=f"📸 {t('original')}", use_container_width=True)
        with i2:
            st.image(result_img, caption=f"🎯 {t('result')}", use_container_width=True)
        
        # Detection Table
        if ki:
            st.markdown(f"### 📋 {t('detection_table')}")
            df_det = create_detection_table(ba, sa, ca, ki, uploaded.name)
            
            # Filters
            ft1, ft2 = st.columns(2)
            with ft1:
                type_filter = st.multiselect(
                    f"🔍 {t('type')}",
                    options=[t('weeds'), t('crops')],
                    default=[t('weeds'), t('crops')]
                )
            with ft2:
                conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, key="conf_filter_single")
            
            # Apply filters
            df_filtered = df_det[df_det[t('type')].isin(type_filter)]
            conf_values = df_filtered[t('confidence_score')].str.rstrip('%').astype(float) / 100
            df_filtered = df_filtered[conf_values >= conf_min]
            
            st.dataframe(
                df_filtered,
                use_container_width=True,
                height=min(400, len(df_filtered) * 40 + 40)
            )
            
            # Stats
            st.markdown("#### 📊 Özet İstatistikler")
            stat1, stat2, stat3 = st.columns(3)
            with stat1:
                st.metric(t('weeds'), wc, delta=None)
            with stat2:
                st.metric(t('crops'), cc, delta=None)
            with stat3:
                ratio = f"{wc/(wc+cc)*100:.1f}%" if (wc+cc) > 0 else "0%"
                st.metric("Weed Ratio", ratio)
        
        # Heatmap
        if wc > 0:
            st.markdown(f"### 📈 {t('heatmap')}")
            dm = generate_heatmap(ba, sa, ca, ki, w, h)
            fig = px.imshow(dm, color_continuous_scale="RdYlGn_r")
            fig.update_layout(height=400, margin=dict(r=0, t=10, l=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        # Download
        st.markdown(f"### 💾 {t('download')}")
        d1, d2, d3 = st.columns(3)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with d1:
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button("🖼️ PNG", buf.getvalue(), f"agri_{ts}.png", "image/png", use_container_width=True)
        with d2:
            jr = {"timestamp": ts, "weeds": wc, "crops": cc, "density": round(wd, 2)}
            st.download_button("📋 JSON", json.dumps(jr, indent=2), f"agri_{ts}.json", "application/json", use_container_width=True)
        with d3:
            if ki:
                csv_buf = df_det.to_csv(index=False)
                st.download_button("📊 CSV", csv_buf, f"agri_{ts}.csv", "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: ÇOKLU FOTOĞRAF
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<div class='tab-header'>📸 {t('multi_photo_tab')}</div>", unsafe_allow_html=True)
    
    multi_uploaded = st.file_uploader(
        f"📁 {t('upload_photos')}",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="multi_photo"
    )
    
    if multi_uploaded:
        # Limit to 10
        if len(multi_uploaded) > 10:
            st.warning("⚠️ Max 10 fotoğraf! İlk 10 tanesi işlenecek.")
            multi_uploaded = multi_uploaded[:10]
        
        st.info(f"📷 {len(multi_uploaded)} fotoğraf yüklendi")
        
        # Process all button
        if st.button("🚀 Tümünü Analiz Et", use_container_width=True, key="analyze_all"):
            all_results = []
            all_tables = []
            progress = st.progress(0)
            
            for idx, file in enumerate(multi_uploaded):
                with st.spinner(f"🔍 {t('analyzing')} ({idx+1}/{len(multi_uploaded)}) - {file.name}"):
                    img = Image.open(file).convert("RGB")
                    w_img, h_img = img.size
                    ta_img = w_img * h_img
                    
                    ba_m, sa_m, ca_m, aa_m, ki_m = process_image(
                        img, interpreter, input_details, output_details,
                        threshold, size_threshold, nms_iou
                    )
                    
                    result_img_m = img.copy()
                    if ki_m:
                        result_img_m = draw_detections(result_img_m, ba_m, sa_m, ca_m, ki_m)
                    
                    wc_m = sum(1 for i in ki_m if ca_m[i] == "WEED")
                    cc_m = sum(1 for i in ki_m if ca_m[i] == "CROP")
                    wa_m = sum(aa_m[i] for i in ki_m if ca_m[i] == "WEED")
                    wd_m = (wa_m / ta_img) * 100 if ta_img > 0 else 0
                    ac_m = float(np.mean([sa_m[i] for i in ki_m])) if ki_m else 0
                    
                    all_results.append({
                        "name": file.name,
                        "original": img,
                        "result": result_img_m,
                        "weeds": wc_m,
                        "crops": cc_m,
                        "density": wd_m,
                        "confidence": ac_m,
                        "boxes": ba_m,
                        "scores": sa_m,
                        "classes": ca_m,
                        "keep": ki_m
                    })
                    
                    # Add to detection table
                    if ki_m:
                        df_m = create_detection_table(ba_m, sa_m, ca_m, ki_m, file.name)
                        all_tables.append(df_m)
                    
                    # Save to history
                    st.session_state.scan_history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "weeds": wc_m,
                        "crops": cc_m,
                        "density": float(wd_m),
                        "confidence": ac_m
                    })
                
                progress.progress((idx + 1) / len(multi_uploaded))
            
            progress.progress(1.0)
            st.success(f"✅ {len(multi_uploaded)} fotoğraf analiz edildi!")
            
            # ─── SUMMARY CARDS ───
            st.markdown("### 📊 Toplu Özet")
            
            total_weeds = sum(r["weeds"] for r in all_results)
            total_crops = sum(r["crops"] for r in all_results)
            avg_density = np.mean([r["density"] for r in all_results])
            avg_conf = np.mean([r["confidence"] for r in all_results if r["confidence"] > 0])
            max_density = max(r["density"] for r in all_results)
            max_density_img = [r["name"] for r in all_results if r["density"] == max_density][0]
            
            # Alert for overall
            if st.session_state.enable_alerts and avg_density > st.session_state.alert_threshold:
                st.markdown(f"""
                <div class='alert-box'>
                    <h2>🚨 {t('critical_alert')} 🚨</h2>
                    <h3>{t('high_weed_density')}</h3>
                    <p>Ortalama yoğunluk: <b>{avg_density:.1f}%</b> | En yüksek: <b>{max_density:.1f}%</b> ({max_density_img})</p>
                    <p>{t('immediate_action')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                st.markdown(f"<div class='glass-card'><div class='metric-icon'>📷</div><div class='metric-value'>{len(all_results)}</div><div class='metric-label'>FOTO</div></div>", unsafe_allow_html=True)
            with s2:
                st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌿</div><div class='metric-value' style='color:#FF4757'>{total_weeds}</div><div class='metric-label'>{t('weeds')}</div></div>", unsafe_allow_html=True)
            with s3:
                st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌾</div><div class='metric-value' style='color:#2ED573'>{total_crops}</div><div class='metric-label'>{t('crops')}</div></div>", unsafe_allow_html=True)
            with s4:
                st.markdown(f"<div class='glass-card'><div class='metric-icon'>📊</div><div class='metric-value' style='color:#FFA502'>{avg_density:.1f}%</div><div class='metric-label'>AVG {t('density')}</div></div>", unsafe_allow_html=True)
            with s5:
                st.markdown(f"<div class='glass-card'><div class='metric-icon'>🎯</div><div class='metric-value' style='color:#3742FA'>{avg_conf:.0%}</div><div class='metric-label'>AVG {t('conf')}</div></div>", unsafe_allow_html=True)
            
            # ─── COMPARISON CHART ───
            st.markdown("### 📈 Fotoğraf Karşılaştırma")
            
            fig_comp = go.Figure()
            names = [r["name"][:15] for r in all_results]
            fig_comp.add_trace(go.Bar(
                x=names,
                y=[r["weeds"] for r in all_results],
                name=f"🌿 {t('weeds')}",
                marker_color="#FF4757"
            ))
            fig_comp.add_trace(go.Bar(
                x=names,
                y=[r["crops"] for r in all_results],
                name=f"🌾 {t('crops')}",
                marker_color="#2ED573"
            ))
            fig_comp.update_layout(
                barmode="group",
                height=400,
                xaxis_title=t('image'),
                yaxis_title="Count",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Density comparison
            fig_dens = go.Figure()
            colors = ["#FF4757" if r["density"] > st.session_state.alert_threshold else "#2ED573" for r in all_results]
            fig_dens.add_trace(go.Bar(
                x=names,
                y=[r["density"] for r in all_results],
                marker_color=colors,
                text=[f"{r['density']:.1f}%" for r in all_results],
                textposition="outside"
            ))
            fig_dens.add_hline(
                y=st.session_state.alert_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"⚠️ Eşik ({st.session_state.alert_threshold}%)"
            )
            fig_dens.update_layout(
                height=400,
                xaxis_title=t('image'),
                yaxis_title=f"{t('density')} (%)",
                xaxis_tickangle=-45,
                title=f"📊 {t('density')} Karşılaştırma"
            )
            st.plotly_chart(fig_dens, use_container_width=True)
            
            # ─── INDIVIDUAL RESULTS ───
            st.markdown("### 🖼️ Detaylı Sonuçlar")
            
            # 2 columns per row
            for i in range(0, len(all_results), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(all_results):
                        r = all_results[i + j]
                        with cols[j]:
                            # Alert badge
                            alert_badge = ""
                            if r["density"] > st.session_state.alert_threshold:
                                alert_badge = " 🚨"
                            
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.08); border-radius:15px; 
                                        padding:1rem; margin-bottom:1rem;
                                        border-left: 4px solid {"#FF4757" if r["density"] > 15 else "#2ED573"};'>
                                <h4>📷 {r["name"]}{alert_badge}</h4>
                                <p>🌿 {r["weeds"]} | 🌾 {r["crops"]} | 📊 {r["density"]:.1f}% | 🎯 {r["confidence"]:.0%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            view_mode = st.radio(
                                "Görünüm",
                                [t('original'), t('result')],
                                horizontal=True,
                                key=f"view_{i+j}"
                            )
                            
                            if view_mode == t('original'):
                                st.image(r["original"], use_container_width=True)
                            else:
                                st.image(r["result"], use_container_width=True)
            
            # ─── COMBINED DETECTION TABLE ───
            if all_tables:
                st.markdown(f"### 📋 {t('detection_table')} (Tüm Fotoğraflar)")
                
                combined_df = pd.concat(all_tables, ignore_index=True)
                
                # Filters
                tf1, tf2, tf3 = st.columns(3)
                with tf1:
                    img_filter = st.multiselect(
                        f"📷 {t('image')}",
                        options=combined_df[t('image')].unique().tolist(),
                        default=combined_df[t('image')].unique().tolist(),
                        key="img_filter_multi"
                    )
                with tf2:
                    type_filter_m = st.multiselect(
                        f"🔍 {t('type')}",
                        options=[t('weeds'), t('crops')],
                        default=[t('weeds'), t('crops')],
                        key="type_filter_multi"
                    )
                with tf3:
                    sort_col = st.selectbox(
                        "↕️ Sort By",
                        options=combined_df.columns.tolist(),
                        key="sort_multi"
                    )
                
                # Apply filters
                filtered_df = combined_df[
                    (combined_df[t('image')].isin(img_filter)) &
                    (combined_df[t('type')].isin(type_filter_m))
                ].sort_values(sort_col, ascending=False)
                
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=min(600, len(filtered_df) * 40 + 40)
                )
                
                st.markdown(f"**Toplam: {len(filtered_df)} detection**")
                
                # Download combined
                csv_all = combined_df.to_csv(index=False)
                st.download_button(
                    "📊 Tüm Tabloyu İndir (CSV)",
                    csv_all,
                    f"agri_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; border:2px dashed rgba(255,255,255,0.3); border-radius:20px;'>
            <h2>📸 Çoklu Fotoğraf Yükleyin</h2>
            <p>Aynı anda <b>10 fotoğrafa kadar</b> yükleyip karşılaştırabilirsiniz</p>
            <p>🌿 Her fotoğraf için ayrı analiz + toplu karşılaştırma</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: HAVA DURUMU & İLAÇLAMA
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<div class='tab-header'>🌡️ {t('weather_tab')}</div>", unsafe_allow_html=True)
    
    if weather_api_key:
        with st.spinner("🌤️ Hava durumu alınıyor..."):
            current, forecast = get_weather(gps_lat, gps_lon, weather_api_key)
        
        if current and "main" in current:
            spray = analyze_spray_conditions(current, forecast)
            
            if spray:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {spray["overall_color"]}44, {spray["overall_color"]}22);
                            padding: 2rem; border-radius: 20px; text-align: center;
                            border: 2px solid {spray["overall_color"]}; margin-bottom: 1.5rem;'>
                    <h2 style='margin:0;'>💊 İlaçlama Uygunluk Skoru</h2>
                    <div style='font-size:3em; font-weight:800; color:{spray["overall_color"]};'>{spray["score"]}/100</div>
                    <h3 style='margin:0;'>{spray["overall"]}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌡️</div><div class='metric-value'>{spray['temp']:.1f}°C</div><div class='metric-label'>{spray['temp_status']}</div></div>", unsafe_allow_html=True)
                with w2:
                    st.markdown(f"<div class='glass-card'><div class='metric-icon'>💨</div><div class='metric-value'>{spray['wind_speed']} km/h</div><div class='metric-label'>{spray['wind_status']}</div></div>", unsafe_allow_html=True)
                with w3:
                    st.markdown(f"<div class='glass-card'><div class='metric-icon'>💧</div><div class='metric-value'>{spray['humidity']}%</div><div class='metric-label'>{spray['humidity_status']}</div></div>", unsafe_allow_html=True)
                with w4:
                    st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌧️</div><div class='metric-value'>{spray['rain_status']}</div><div class='metric-label'>Yağış</div></div>", unsafe_allow_html=True)
                
                # Forecast chart
                if forecast and "list" in forecast:
                    st.markdown("### 📈 5 Günlük Tahmin")
                    times_f = [datetime.fromtimestamp(i["dt"]).strftime("%d/%m %H:%M") for i in forecast["list"][:20]]
                    temps_f = [i["main"]["temp"] for i in forecast["list"][:20]]
                    winds_f = [i["wind"]["speed"] * 3.6 for i in forecast["list"][:20]]
                    humids_f = [i["main"]["humidity"] for i in forecast["list"][:20]]
                    
                    fig_w = go.Figure()
                    fig_w.add_trace(go.Scatter(x=times_f, y=temps_f, name="🌡️ Sıcaklık", line=dict(color="#FF6348", width=2)))
                    fig_w.add_trace(go.Scatter(x=times_f, y=winds_f, name="💨 Rüzgar", line=dict(color="#1E90FF", width=2)))
                    fig_w.add_trace(go.Scatter(x=times_f, y=humids_f, name="💧 Nem", line=dict(color="#2ED573", width=2)))
                    fig_w.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Rüzgar Limiti")
                    fig_w.update_layout(height=400)
                    st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.error("❌ Hava durumu alınamadı.")
    else:
        st.info("🔑 Sol panelden API Key ayarlayın.")


# ═══════════════════════════════════════════════════════════════
# TAB 4: AI DANIŞMANLIK
# ═══════════════════════════════════════════════════════════════
WEED_DATABASE = {
    "genel": {
        "diseases": [
            {"name": "Kurutucu Mantarı", "risk": "Yüksek", "icon": "🍄", "detail": "Nemli ortamlarda yayılır."},
            {"name": "Yaprak Leke Hastalığı", "risk": "Orta", "icon": "🦠", "detail": "Sarı-kahverengi lekeler."},
            {"name": "Kök Çürümesi", "risk": "Düşük", "icon": "🪱", "detail": "Aşırı sulamada görülür."}
        ],
        "treatment": [
            {"method": "Herbisit", "icon": "💊", "detail": "Seçici herbisit uygulayın.", "timing": "Ot 5-10 cm boyundayken"},
            {"method": "Mekanik Çapa", "icon": "⛏️", "detail": "Elle veya makineli çapalama.", "timing": "Haftada 1-2 kez"},
            {"method": "Malçlama", "icon": "🌾", "detail": "Organik malç ile örtün.", "timing": "Ekim sonrası"}
        ]
    }
}

CROP_DATABASE = {
    "genel": {
        "health_tips": [
            {"tip": "Toprak Analizi", "icon": "🧪", "detail": "Yılda 1 kez toprak analizi yaptırın."},
            {"tip": "Gübreleme", "icon": "🌱", "detail": "Dengeli gübreleme yapın."},
            {"tip": "Sulama", "icon": "💧", "detail": "Damla sulama en verimli."},
            {"tip": "Hastalık Takibi", "icon": "🔍", "detail": "Haftada 1 tarla gezisi yapın."}
        ]
    }
}

SEASONAL_ADVICE = {
    1: {"season": "Kış", "icon": "❄️", "advice": "Toprak hazırlığı."},
    2: {"season": "Kış", "icon": "❄️", "advice": "Sera üretimi."},
    3: {"season": "İlkbahar", "icon": "🌸", "advice": "Erken ekim."},
    4: {"season": "İlkbahar", "icon": "🌸", "advice": "Ana ekim."},
    5: {"season": "İlkbahar", "icon": "🌸", "advice": "Ot mücadelesi kritik!"},
    6: {"season": "Yaz", "icon": "☀️", "advice": "Sulama kritik."},
    7: {"season": "Yaz", "icon": "☀️", "advice": "Sıcaklık stresi."},
    8: {"season": "Yaz", "icon": "☀️", "advice": "Erken hasat."},
    9: {"season": "Sonbahar", "icon": "🍂", "advice": "Ana hasat."},
    10: {"season": "Sonbahar", "icon": "🍂", "advice": "Sonbahar ekimi."},
    11: {"season": "Sonbahar", "icon": "🍂", "advice": "Tarla temizliği."},
    12: {"season": "Kış", "icon": "❄️", "advice": "Yıllık değerlendirme."}
}

def get_ai_advice(wc, cc, wd, ac):
    a = {"weed_info": WEED_DATABASE["genel"], "crop_info": CROP_DATABASE["genel"],
         "seasonal": SEASONAL_ADVICE.get(datetime.now().month, {}), "actions": []}
    if wd > 30:
        a.update({"risk_level": "KRİTİK", "risk_color": "#FF0000", "risk_icon": "🚨",
                  "summary": "Acil müdahale gerekli!",
                  "actions": ["🚨 Acil herbisit", "📞 Danışmana haber", "🔄 2 gün içinde tekrar"]})
    elif wd > 20:
        a.update({"risk_level": "YÜKSEK", "risk_color": "#FF4757", "risk_icon": "🔴",
                  "summary": "Herbisit uygulaması önerilir.",
                  "actions": ["💊 Seçici herbisit", "⛏️ Mekanik çapa", "📅 1 haftada kontrol"]})
    elif wd > 10:
        a.update({"risk_level": "ORTA", "risk_color": "#FFA502", "risk_icon": "🟡",
                  "summary": "Hedefli müdahale yeterli.",
                  "actions": ["⛏️ Elle temizlik", "🎯 Noktasal herbisit", "📅 2 haftada kontrol"]})
    elif wc > 0:
        a.update({"risk_level": "DÜŞÜK", "risk_color": "#2ED573", "risk_icon": "🟢",
                  "summary": "Koruyucu önlemler yeterli.",
                  "actions": ["👁️ Gözlem", "✋ Elle çekme", "🌾 Malçlama"]})
    else:
        a.update({"risk_level": "TEMİZ", "risk_color": "#00D2D3", "risk_icon": "✅",
                  "summary": "Tarla sağlıklı!",
                  "actions": ["✅ Bakım devam", "🌱 Gübreleme", "💧 Sulama"]})
    return a


with tab4:
    st.markdown(f"<div class='tab-header'>🧠 {t('ai_tab')}</div>", unsafe_allow_html=True)
    if st.session_state.scan_history:
        last = st.session_state.scan_history[-1]
        adv = get_ai_advice(last["weeds"], last["crops"], last["density"], last["confidence"])
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {adv["risk_color"]}44, {adv["risk_color"]}22);
                    padding: 1.5rem; border-radius: 20px; border-left: 6px solid {adv["risk_color"]};'>
            <h2>{adv["risk_icon"]} Risk: {adv["risk_level"]}</h2>
            <p style='font-size:1.1em;'>{adv["summary"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Aksiyon Planı")
        for idx, act in enumerate(adv["actions"]):
            st.markdown(f"""
            <div class='advice-card' style='border-color:{adv["risk_color"]};'>
                <b>Adım {idx+1}:</b> {act}
            </div>
            """, unsafe_allow_html=True)
        
        sea = adv["seasonal"]
        if sea:
            st.markdown(f"### {sea.get('icon','')} Mevsimsel ({sea.get('season','')})")
            st.markdown(f"<div class='advice-card' style='border-color:#667eea;'>{sea.get('advice','')}</div>", unsafe_allow_html=True)
        
        if last["weeds"] > 0:
            st.markdown("### 🦠 Hastalıklar")
            for d in adv["weed_info"]["diseases"]:
                rc = "#FF4757" if d["risk"] == "Yüksek" else "#FFA502" if d["risk"] == "Orta" else "#2ED573"
                st.markdown(f"""
                <div style='background: rgba(255,0,0,0.08); border-radius:12px; padding:1rem; margin:0.5rem 0; border-left:4px solid {rc};'>
                    <h4>{d['icon']} {d['name']} <span style='background:{rc};color:white;padding:2px 8px;border-radius:5px;'>{d['risk']}</span></h4>
                    <p>{d['detail']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 💊 Tedavi")
            for tr in adv["weed_info"]["treatment"]:
                st.markdown(f"""
                <div style='background: rgba(0,255,0,0.08); border-radius:12px; padding:1rem; margin:0.5rem 0; border-left:4px solid #2ED573;'>
                    <h4>{tr['icon']} {tr['method']}</h4>
                    <p>{tr['detail']}</p>
                    <small>⏰ {tr['timing']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        if last["crops"] > 0:
            st.markdown("### 🌾 Mahsul Bakımı")
            for tip in adv["crop_info"]["health_tips"]:
                st.markdown(f"""
                <div style='background: rgba(0,255,0,0.08); border-radius:12px; padding:1rem; margin:0.5rem 0; border-left:4px solid #2ED573;'>
                    <h4>{tip['icon']} {tip['tip']}</h4>
                    <p>{tip['detail']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📸 Önce fotoğraf sekmesinden tarama yapın!")


# ═══════════════════════════════════════════════════════════════
# TAB 5: VIDEO
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<div class='tab-header'>🎥 {t('video_tab')}</div>", unsafe_allow_html=True)
    vf = st.file_uploader("🎥 Video", type=["mp4", "avi", "mov"], key="vid")
    if vf:
        tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf2.write(vf.read())
        tf2.close()
        cap = cv2.VideoCapture(tf2.name)
        totf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        st.info(f"📹 {totf} kare | {fps} FPS")
        fskip = st.slider("⏭️ Aralık", 5, 60, 30, 5)
        if st.button("🚀 Başlat", use_container_width=True):
            prog = st.progress(0)
            stat = st.empty()
            prev = st.empty()
            res = []
            fi = 0
            pc = 0
            while cap.isOpened():
                ret, fr = cap.read()
                if not ret: break
                if fi % fskip == 0:
                    pf = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    b, s, cl, a, k = process_image(pf, interpreter, input_details, output_details, threshold, size_threshold, nms_iou)
                    wc2 = sum(1 for i in k if cl[i] == "WEED")
                    res.append({"time": fi / fps, "weeds": wc2, "crops": sum(1 for i in k if cl[i] == "CROP")})
                    if k:
                        rf = pf.copy()
                        rf = draw_detections(rf, b, s, cl, k)
                        prev.image(rf, use_container_width=True)
                        pc += 1
                    prog.progress(min(fi / max(totf, 1), 1.0))
                    stat.text(f"Kare: {pc}")
                fi += 1
            cap.release()
            prog.progress(1.0)
            stat.success(f"✅ {pc} kare işlendi")
            
            if res:
                # Video detection table
                st.markdown(f"### 📋 Video {t('detection_table')}")
                video_df = pd.DataFrame(res)
                video_df.columns = ["Zaman (s)", t('weeds'), t('crops')]
                st.dataframe(video_df, use_container_width=True)
                
                # Alert check for video
                max_weeds_frame = max(res, key=lambda x: x["weeds"])
                if st.session_state.enable_alerts and max_weeds_frame["weeds"] > 5:
                    st.markdown(f"""
                    <div class='alert-box'>
                        <h2>🚨 {t('critical_alert')} 🚨</h2>
                        <p>Video karesinde yüksek yabancı ot: <b>{max_weeds_frame['weeds']}</b> adet 
                        (t={max_weeds_frame['time']:.1f}s)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Timeline chart
                fv = go.Figure()
                fv.add_trace(go.Scatter(
                    x=[r["time"] for r in res],
                    y=[r["weeds"] for r in res],
                    mode="lines+markers",
                    name=f"🌿 {t('weeds')}",
                    line=dict(color="#FF4757", width=3)
                ))
                fv.add_trace(go.Scatter(
                    x=[r["time"] for r in res],
                    y=[r["crops"] for r in res],
                    mode="lines+markers",
                    name=f"🌾 {t('crops')}",
                    line=dict(color="#2ED573", width=3)
                ))
                fv.update_layout(
                    height=400,
                    xaxis_title="Zaman (s)",
                    yaxis_title="Tespit Sayısı",
                    title="📈 Video Zaman Çizelgesi"
                )
                st.plotly_chart(fv, use_container_width=True)
                
                # Download video results
                csv_vid = video_df.to_csv(index=False)
                st.download_button(
                    "📊 Video Sonuçlarını İndir (CSV)",
                    csv_vid,
                    f"video_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        st.info("🎥 Video yükleyin")


# ═══════════════════════════════════════════════════════════════
# TAB 6: ANALİTİK
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"<div class='tab-header'>📊 {t('analytics_tab')}</div>", unsafe_allow_html=True)
    
    if st.session_state.scan_history:
        hist = st.session_state.scan_history
        
        # Summary metrics
        st.markdown("### 📈 Genel İstatistikler")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            total_scans = len(hist)
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>📷</div><div class='metric-value'>{total_scans}</div><div class='metric-label'>Toplam Tarama</div></div>", unsafe_allow_html=True)
        with m2:
            total_w = sum(h["weeds"] for h in hist)
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌿</div><div class='metric-value' style='color:#FF4757'>{total_w}</div><div class='metric-label'>Toplam Yabancı Ot</div></div>", unsafe_allow_html=True)
        with m3:
            total_c = sum(h["crops"] for h in hist)
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>🌾</div><div class='metric-value' style='color:#2ED573'>{total_c}</div><div class='metric-label'>Toplam Mahsul</div></div>", unsafe_allow_html=True)
        with m4:
            avg_d = np.mean([h["density"] for h in hist])
            st.markdown(f"<div class='glass-card'><div class='metric-icon'>📊</div><div class='metric-value' style='color:#FFA502'>{avg_d:.1f}%</div><div class='metric-label'>Ort. Yoğunluk</div></div>", unsafe_allow_html=True)
        
        # Bar chart
        st.markdown("### 📊 Tarama Bazlı Karşılaştırma")
        x = list(range(1, len(hist) + 1))
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=x,
            y=[h["weeds"] for h in hist],
            name=f"🌿 {t('weeds')}",
            marker_color="#FF4757"
        ))
        fig1.add_trace(go.Bar(
            x=x,
            y=[h["crops"] for h in hist],
            name=f"🌾 {t('crops')}",
            marker_color="#2ED573"
        ))
        fig1.update_layout(barmode="group", height=380, xaxis_title="Tarama #", yaxis_title="Sayı")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Density trend
        st.markdown("### 📈 Yoğunluk Trendi")
        fig2 = go.Figure()
        densities = [h["density"] for h in hist]
        colors_d = ["#FF4757" if d > st.session_state.alert_threshold else "#2ED573" for d in densities]
        fig2.add_trace(go.Scatter(
            x=x,
            y=densities,
            mode="lines+markers",
            name=t('density'),
            line=dict(color="#FFA502", width=3),
            marker=dict(color=colors_d, size=10)
        ))
        fig2.add_hline(
            y=st.session_state.alert_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"⚠️ Eşik ({st.session_state.alert_threshold}%)"
        )
        fig2.update_layout(height=380, xaxis_title="Tarama #", yaxis_title=f"{t('density')} (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Confidence trend
        st.markdown("### 🎯 Güven Skoru Trendi")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=x,
            y=[h["confidence"] * 100 for h in hist],
            mode="lines+markers",
            name=t('conf'),
            line=dict(color="#3742FA", width=3),
            fill="tozeroy",
            fillcolor="rgba(55,66,250,0.1)"
        ))
        fig3.update_layout(height=380, xaxis_title="Tarama #", yaxis_title=f"{t('conf')} (%)")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Pie chart
        st.markdown("### 🥧 Dağılım")
        p1, p2 = st.columns(2)
        with p1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=[t('weeds'), t('crops')],
                values=[total_w, total_c],
                marker_colors=["#FF4757", "#2ED573"],
                hole=0.4
            )])
            fig_pie.update_layout(height=350, title="Tespit Dağılımı")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with p2:
            # Alert count
            alert_count = sum(1 for h in hist if h["density"] > st.session_state.alert_threshold)
            safe_count = len(hist) - alert_count
            fig_pie2 = go.Figure(data=[go.Pie(
                labels=["⚠️ Uyarı", "✅ Güvenli"],
                values=[alert_count, safe_count],
                marker_colors=["#FF4757", "#2ED573"],
                hole=0.4
            )])
            fig_pie2.update_layout(height=350, title="Uyarı Dağılımı")
            st.plotly_chart(fig_pie2, use_container_width=True)
        
        # Detailed History Table
        st.markdown(f"### 📋 Detaylı Geçmiş Tablosu")
        hist_df = pd.DataFrame(hist)
        hist_df.columns = ["Zaman", t('weeds'), t('crops'), t('density'), t('conf')]
        hist_df.index = range(1, len(hist_df) + 1)
        hist_df.index.name = "#"
        
        # Sortable table
        sort_by = st.selectbox(
            "↕️ Sırala",
            options=hist_df.columns.tolist(),
            key="hist_sort"
        )
        sort_order = st.radio("Sıralama", ["Azalan", "Artan"], horizontal=True, key="hist_order")
        
        sorted_df = hist_df.sort_values(
            sort_by,
            ascending=(sort_order == "Artan")
        )
        
        st.dataframe(sorted_df, use_container_width=True, height=min(500, len(sorted_df) * 40 + 40))
        
        # Download history
        csv_hist = hist_df.to_csv()
        st.download_button(
            "📊 Geçmişi İndir (CSV)",
            csv_hist,
            f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("📸 Tarama yapın, veriler burada görünecek.")


# ═══════════════════════════════════════════════════════════════
# TAB 7: GPS
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.markdown(f"<div class='tab-header'>🗺️ {t('gps_tab')}</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='glass-card' style='margin-bottom:1rem;'>
        <h3>📍 Konum Bilgisi</h3>
        <p><b>{t('latitude')}:</b> {gps_lat:.6f} | <b>{t('longitude')}:</b> {gps_lon:.6f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.map(pd.DataFrame({"lat": [gps_lat], "lon": [gps_lon]}), zoom=14)
    
    # Scan history on map (if multiple locations exist)
    if len(st.session_state.scan_history) > 1:
        st.markdown("### 📊 Tarama Noktaları")
        st.info("GPS konumunuzu değiştirerek farklı noktalar ekleyebilirsiniz.")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
lang_flag = {"tr": "🇹🇷", "en": "🇬🇧", "ar": "🇸🇦"}.get(st.session_state.language, "🌍")
theme_icon = "🌙" if st.session_state.theme == "dark" else "☀️"

st.markdown(f"""
<div style='text-align:center; padding:2rem; margin-top:3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius:20px; color:white;'>
    <h3>🌾 AgriVision AI Pro v4.0</h3>
    <p>📸 Detection | 📸 Multi-Photo | 🌡️ Weather | 🧠 AI Advisor | 🎥 Video | 📊 Analytics | 🗺️ GPS</p>
    <p>{lang_flag} {st.session_state.language.upper()} | {theme_icon} {st.session_state.theme.title()} | 🔔 Alerts: {"ON" if st.session_state.enable_alerts else "OFF"}</p>
</div>
""", unsafe_allow_html=True)
