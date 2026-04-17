# agrivision-ai
<div align="center">

# 🌱 AgriVision AI Pro v4.0

### AI-Powered Smart Agriculture Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://agrivision-ai09.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

[🔗 Live Demo](https://agrivision-ai09.streamlit.app/) · [📄 Report Bug](https://github.com/KULLANICI_ADIN/agrivision-ai/issues) · [✨ Request Feature](https://github.com/KULLANICI_ADIN/agrivision-ai/issues)

<img src="screenshots/banner.png" alt="AgriVision AI Pro" width="800">

</div>

---

## 📌 About

AgriVision AI Pro is a comprehensive smart agriculture platform that uses 
artificial intelligence to detect weeds and crops in real-time, provide 
weather-based spraying recommendations, and offer AI-powered farming advisory.

During development, I discovered that the training dataset used large bounding 
boxes covering entire weed areas. Rather than retraining, I implemented an 
OpenCV post-processing pipeline to detect individual plants — turning a 
dataset limitation into a hybrid AI solution.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 **Weed & Crop Detection** | Real-time detection using TensorFlow Lite |
| 📸 **Camera Integration** | Take photos directly from browser |
| 🔄 **Before/After Slider** | Interactive comparison view |
| 📸 **Multi-Photo Analysis** | Batch process up to 10 images |
| 🌡️ **Weather Integration** | OpenWeatherMap API with spray scheduling |
| 🌡️ **Soil Temperature** | Estimated soil temp at surface/10cm/30cm |
| 🧠 **AI Advisory** | Disease diagnosis & treatment recommendations |
| 🔔 **Smart Alerts** | Customizable density threshold alarms |
| 📊 **Analytics Dashboard** | Interactive charts and heatmaps |
| 📄 **PDF Reports** | Automated professional reports |
| 📱 **QR Code Sharing** | Quick app sharing via QR |
| 💬 **Field Notes** | Save notes with detection data |
| 🗺️ **GPS Tracking** | Location-based field mapping |
| 🌍 **Multi-Language** | Turkish / English / Arabic |
| 🌙 **Dark/Light Theme** | User preference |
| 👥 **Visitor Counter** | Track app usage |
| 🎥 **Video Analysis** | Frame-by-frame weed detection |

---

## 🖼️ Screenshots

<div align="center">

| Detection | Weather & Spraying | Analytics |
|:---------:|:---------:|:---------:|
| <img src="screenshots/detection.png" width="280"> | <img src="screenshots/weather.png" width="280"> | <img src="screenshots/analytics.png" width="280"> |

| Multi-Photo | AI Advisory | Before/After |
|:---------:|:---------:|:---------:|
| <img src="screenshots/multi.png" width="280"> | <img src="screenshots/ai.png" width="280"> | <img src="screenshots/beforeafter.png" width="280"> |

</div>

---

## 🛠️ Tech Stack

