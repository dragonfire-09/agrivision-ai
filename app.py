from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = str(Path(__file__).parent / "best_float32.tflite")

st.set_page_config(page_title="AgriVision AI", layout="wide")
st.title("🌱 AgriVision AI — Weed Detection")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
threshold = st.slider("Confidence", 0.1, 0.9, 0.5)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    w,h = img.size

    resized = img.resize((640,640))
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.expand_dims(arr,0)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    preds = np.transpose(output[0], (1,0))

    draw = ImageDraw.Draw(img)
    weed_count = 0

    for row in preds:
        x,y,bw,bh,ws,cs = row

        if ws < threshold:
            continue

        x1 = int((x-bw/2)*w)
        y1 = int((y-bh/2)*h)
        x2 = int((x+bw/2)*w)
        y2 = int((y+bh/2)*h)

        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        weed_count += 1

    st.image(img, caption="Detection Result", use_container_width=True)
    st.metric("Weeds", weed_count)
