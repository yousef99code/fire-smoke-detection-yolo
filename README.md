# 🔥 Fire & Smoke Detection

A YOLO-powered computer-vision web app that detects **fire** and **smoke** in images and videos.
Upload a file, press **Analyze**, and the model tells you whether it contains fire or smoke and shows exactly where.

**Developed by Eng. Youssef Madin**

![App preview](preview.png)

---

## ✨ Features

- Upload an **image** (JPG, PNG, BMP, WEBP) or a **video** (MP4, MOV, AVI, MKV)
- A clear verdict: **FIRE DETECTED**, **SMOKE DETECTED** or **NO FIRE OR SMOKE**
- Bounding boxes with the class name and confidence on every detection
- Detection table, counters and downloadable result (image or video)
- Adjustable settings: confidence, IoU, image size and video frame skipping
- Animated, modern interface

---

## 🧠 How the website works

The app has **3 steps**. They are real buttons at the top of the page, so you can move between them.

| Step | What you do | What happens |
|---|---|---|
| **1 · Upload** | Choose an image or a video | The file is stored in memory and a preview is shown |
| **2 · Analyze** | Press **Analyze now** | The YOLO model scans the file |
| **3 · Result** | Look at the output | Verdict, boxes, confidence, counters and download button |

### What happens behind the scenes

1. **Model loading.** `best (1).onnx` is loaded once with Ultralytics and kept in memory (`st.cache_resource`), so the page stays fast.
2. **Upload.** The uploaded file is saved in `st.session_state`, so it is not lost when you click between steps.
3. **Image analysis.** The image is passed to the model with your confidence, IoU and image-size settings.
   Every detected box is drawn on the image (orange-red for fire, grey-blue for smoke).
4. **Video analysis.** The video is read frame by frame with OpenCV.
   Only every N-th frame is analyzed (set in the sidebar) to make it faster.
   The annotated frames are written to a new video, then converted to H.264 with `ffmpeg` so any browser can play it.
5. **Verdict.**
   - fire found → **FIRE DETECTED**
   - no fire but smoke found → **SMOKE DETECTED**
   - nothing found → **NO FIRE OR SMOKE**
6. **Result.** Everything is shown on the Result step and can be downloaded.

### Sidebar settings

| Setting | Meaning |
|---|---|
| **Confidence threshold** | Minimum confidence to keep a detection. Higher = fewer false alarms but real fires may be missed. Lower = more detections but more false alarms. |
| **IoU threshold (NMS)** | How overlapping boxes on the same object are merged. |
| **Image size** | Size the image is resized to before the model. Keep it equal to the size used when the ONNX file was exported (usually 640). |
| **Video: process every N-th frame** | Higher = faster video analysis, lower = more detailed. |

---

## 📁 Project structure

```
your-repo/
├── app.py              # the Streamlit app
├── best (1).onnx       # the trained YOLO model (ONNX format)
├── requirements.txt    # Python libraries
├── packages.txt        # system packages for Streamlit Cloud (libgl1, ffmpeg)
└── README.md
```

> The model file must be in the **same folder as `app.py`**.
> The app looks for `best (1).onnx`; if the name is different, it uses the first `.onnx` file it finds.

---

## ▶️ Run the website on your computer

**1. Install Python 3.10 or newer**, then open a terminal in the project folder.

**2. (Recommended) create a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

**3. Install the libraries**

```bash
pip install -r requirements.txt
```

**4. (Optional) install ffmpeg** so the result video plays inside the browser.
Download it from <https://ffmpeg.org> and add it to your PATH.
Without it the app still works, but the result video may not play in some browsers.

**5. Run the app**

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Upload these files to a **GitHub repository**: `app.py`, `best (1).onnx`, `requirements.txt`, `packages.txt`, `README.md`.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose the repository and the branch, and set **Main file path** to `app.py`.
4. Press **Deploy**. The first start takes a few minutes because `ultralytics` is large.

---

## 🐍 Run the model WITHOUT Streamlit

You do not need the website to use the model. You can run it directly from the terminal or from your own Python code.

Install only what you need:

```bash
pip install ultralytics onnxruntime opencv-python
```

### Option 1: one command (Ultralytics CLI)

On an **image**:

```bash
yolo predict model="best (1).onnx" source="test.jpg" conf=0.35 imgsz=640 save=True
```

On a **video**:

```bash
yolo predict model="best (1).onnx" source="test.mp4" conf=0.35 imgsz=640 save=True
```

On the **webcam** (live window):

```bash
yolo predict model="best (1).onnx" source=0 conf=0.35 show=True
```

The results are saved in `runs/detect/predict/`.

### Option 2: a Python script for an image

```python
from ultralytics import YOLO

model = YOLO("best (1).onnx", task="detect")

results = model.predict("test.jpg", conf=0.35, iou=0.5, imgsz=640, save=True)

for r in results:
    if len(r.boxes) == 0:
        print("No fire or smoke detected")
    for box in r.boxes:
        name = r.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        print(f"{name}  {conf:.0%}  box=({x1}, {y1}, {x2}, {y2})")
```

### Option 3: a Python script for a video

```python
import cv2
from ultralytics import YOLO

model = YOLO("best (1).onnx", task="detect")

cap = cv2.VideoCapture("test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

found_fire = found_smoke = False
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, conf=0.35, imgsz=640, verbose=False)[0]
    names = [r.names[int(c)] for c in r.boxes.cls]
    found_fire |= "fire" in names
    found_smoke |= "smoke" in names
    out.write(r.plot())          # frame with the boxes drawn

cap.release()
out.release()

print("Fire detected :", found_fire)
print("Smoke detected:", found_smoke)
print("Saved: result.mp4")
```

### Option 4: use it in Google Colab

```python
!pip install -q ultralytics onnxruntime
from ultralytics import YOLO

model = YOLO("/content/best (1).onnx", task="detect")   # upload the file to Colab first
model.predict("/content/test.jpg", conf=0.35, save=True)
```

The result image is saved in `runs/detect/predict/`.

> **Tip:** if you have the original PyTorch file (`best.pt`), you can use it the same way:
> `YOLO("best.pt")`. The `.onnx` file is the same model exported for faster and lighter inference.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| "Could not find the model" | Put `best (1).onnx` in the same folder as `app.py`. |
| Nothing is detected | Lower the **Confidence threshold** (try 0.15) and test an image that clearly shows fire. |
| Wrong or empty results | Set **Image size** to the size used when exporting the ONNX model (usually 640). |
| Result video does not play | Install **ffmpeg** (on Streamlit Cloud, keep `packages.txt` in the repo). |
| Video analysis is slow | Increase **process every N-th frame** in the sidebar. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again. |
| First start is slow on Streamlit Cloud | Normal: `ultralytics` is a large package. |

---

## 🧪 Model

- **Task:** object detection (YOLO)
- **Classes:** `fire`, `smoke`
- **Format used by the app:** ONNX (`best (1).onnx`)
- **Training:** Ultralytics YOLO on a Roboflow fire-and-smoke dataset, trained in Google Colab

| Metric | Value |
|---|---|
| Precision | ___ |
| Recall | ___ |
| mAP@0.5 | ___ |
| mAP@0.5:0.95 | ___ |

*(Replace the `___` with the results from your training.)*

---

## ⚠️ Limitations

- It can give **false alarms** with fire-colored objects (sunsets, red lights, orange clothes) or fog.
- Small, far or thin smoke and low-light or blurry images can be missed.
- It only knows the situations that exist in its training data.
- Videos are analyzed frame by frame, with no memory of previous frames.
- This is an educational project. **It is not a replacement for a real fire alarm or safety system.**

---

## 👤 Author

**Eng. Youssef Madin**
Built with Python, Ultralytics YOLO, OpenCV and Streamlit.
