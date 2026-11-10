"""
Fire & Smoke Detection - Streamlit app
Author: Eng. Youssef Madin

How it works (3 real steps, click them in the bar):
  1. Upload  -> choose an image or a video
  2. Analyze -> the YOLO model (ONNX) scans it
  3. Result  -> prediction (fire / smoke), boxes, confidence, download

Model: best (1).onnx  (put it in the same folder as this file)
Run  : streamlit run app.py
"""

import io
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fire & Smoke Detection | Eng. Youssef Madin",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_NAME = "best (1).onnx"

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

# BGR colors used to draw the boxes
BOX_COLORS = {
    "fire": (30, 90, 255),
    "smoke": (200, 170, 140),
}
DEFAULT_COLOR = (80, 220, 120)

# ----------------------------------------------------------------------------
# Styling (CSS + animations)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#080a11; --bg2:#10131d; --card:#151927; --line:rgba(255,255,255,.08);
  --fire:#ff5a1f; --fire2:#ffb300; --smoke:#8ea2c0; --ok:#22c55e; --text:#e8ecf4; --muted:#9aa4b8;
}
html, body, [class*="css"]{font-family:'Poppins',sans-serif;}
.stApp{
  background:
    radial-gradient(1000px 520px at 85% -10%, rgba(255,90,31,.18), transparent 60%),
    radial-gradient(800px 460px at -5% 15%, rgba(142,162,192,.13), transparent 60%),
    var(--bg);
  color:var(--text);
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem; max-width:1250px; position:relative; z-index:2;}

/* ---------- floating embers (background) ---------- */
.embers{position:fixed; inset:0; overflow:hidden; pointer-events:none; z-index:1;}
.embers span{
  position:absolute; bottom:-14px; width:6px; height:6px; border-radius:50%;
  background:radial-gradient(circle, #ffd27a, #ff5a1f 60%, transparent 70%);
  filter:blur(.4px); opacity:0; animation:rise linear infinite;
}

/* ---------- top brand bar ---------- */
.topbar{
  display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;
  padding:12px 22px; margin-bottom:18px; border-radius:16px;
  background:rgba(21,25,39,.72); backdrop-filter:blur(10px); border:1px solid var(--line);
  animation:dropIn .8s cubic-bezier(.2,.9,.3,1.2) both;
}
.brand{font-weight:700; letter-spacing:.04em; display:flex; align-items:center; gap:10px;}
.brand .dot{width:10px; height:10px; border-radius:50%; background:var(--fire); animation:ping 1.6s ease-out infinite;}
.author{
  font-weight:800; font-size:1.15rem; letter-spacing:.14em; text-transform:uppercase;
  background:linear-gradient(90deg,#ff5a1f,#ffd27a,#ff5a1f,#ffd27a); background-size:300% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:shine 5s linear infinite; position:relative;
}
.author::after{
  content:""; position:absolute; left:0; bottom:-5px; height:2px; width:100%;
  background:linear-gradient(90deg,transparent,#ffb300,transparent);
  animation:sweep 2.8s ease-in-out infinite;
}

/* ---------- hero ---------- */
.hero{
  position:relative; overflow:hidden; text-align:center;
  padding:46px 24px 38px; margin-bottom:22px;
  border-radius:26px; border:1px solid rgba(255,90,31,.25);
  background:linear-gradient(135deg, rgba(255,90,31,.13), rgba(21,25,39,.92) 45%, rgba(142,162,192,.10));
  box-shadow:0 20px 70px rgba(0,0,0,.5), 0 0 60px rgba(255,90,31,.08) inset;
  animation:fadeUp .9s ease both;
}
.hero::before{
  content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(560px 280px at 50% 0%, rgba(255,90,31,.26), transparent 70%);
  animation:glow 4s ease-in-out infinite;
}
.hero > *{position:relative; z-index:1;}
.badge{
  display:inline-block; padding:6px 18px; border-radius:999px; font-size:.76rem; letter-spacing:.22em;
  color:var(--fire2); border:1px solid rgba(255,179,0,.4); background:rgba(255,179,0,.08);
  animation:float 3.5s ease-in-out infinite;
}
.hero h1{
  font-size:3.3rem; font-weight:800; line-height:1.1; margin:8px 0 12px;
  background:linear-gradient(90deg,#ff5a1f,#ffb300,#ffffff,#ffb300,#ff5a1f); background-size:300% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:shine 6s linear infinite, zoomIn 1s ease both;
}
.flame{display:block; font-size:3.4rem; line-height:1; margin:16px auto 0; width:max-content; animation:flicker 1.4s ease-in-out infinite; transform-origin:50% 90%; filter:drop-shadow(0 0 16px rgba(255,90,31,.85));}
.typer{
  display:block; margin:0 auto; overflow:hidden; white-space:nowrap; width:0; max-width:100%;
  font-family:'JetBrains Mono',monospace; font-size:1.05rem; color:var(--fire2);
  border-right:3px solid var(--fire2);
  animation:typing 3.6s steps(48) .6s forwards, caret .75s step-end infinite;
}
.hero p{color:var(--muted); max-width:720px; margin:16px auto 0; line-height:1.7; animation:fadeUp 1.2s ease .4s both;}

/* ---------- marquee ---------- */
.marquee{overflow:hidden; max-width:100%; border-radius:12px; border:1px solid var(--line); background:rgba(255,255,255,.02); margin-bottom:22px;}
.marquee .track{display:flex; width:max-content; animation:scroll 26s linear infinite;}
.marquee span{padding:10px 26px; color:var(--muted); font-size:.85rem; letter-spacing:.2em; white-space:nowrap;}
.marquee span b{color:var(--fire2);}

/* ---------- section title ---------- */
.section-title{font-size:1.15rem; font-weight:600; margin:8px 0 10px; display:flex; align-items:center; gap:10px; animation:fadeUp .6s ease both;}
.section-title::before{content:""; width:5px; height:22px; border-radius:4px; background:linear-gradient(180deg,#ff5a1f,#ffb300); animation:grow 1.6s ease-in-out infinite;}
.hint{color:var(--muted); font-size:.92rem; margin:-4px 0 14px;}

/* ---------- cards ---------- */
.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:16px; margin:10px 0 22px;}
.card{
  background:linear-gradient(160deg, var(--card), var(--bg2)); border:1px solid var(--line);
  border-radius:18px; padding:18px 20px; animation:fadeUp .7s ease both;
  transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.card:nth-child(2){animation-delay:.12s;} .card:nth-child(3){animation-delay:.24s;}
.card:hover{transform:translateY(-6px); border-color:rgba(255,90,31,.55); box-shadow:0 14px 34px rgba(255,90,31,.2);}
.card .label{color:var(--muted); font-size:.8rem; letter-spacing:.1em; text-transform:uppercase;}
.card .value{font-size:2.4rem; font-weight:800; margin-top:4px; animation:pop .8s cubic-bezier(.2,1.4,.4,1) both;}
.card.fire .value{color:var(--fire);} .card.smoke .value{color:var(--smoke);} .card.ok .value{color:var(--ok);}

/* ---------- verdict ---------- */
.verdict{
  position:relative; text-align:center; border-radius:24px; padding:30px 20px; margin:6px 0 22px; overflow:hidden;
  animation:zoomIn .7s cubic-bezier(.2,1.2,.4,1) both;
}
.verdict .big{font-size:2.3rem; font-weight:800; letter-spacing:.04em;}
.verdict .sub{margin-top:6px; opacity:.85;}
.verdict .ico{font-size:3rem; display:inline-block; animation:flicker 1.2s ease-in-out infinite;}
.verdict.fire{background:linear-gradient(135deg,rgba(255,60,30,.25),rgba(120,20,10,.25)); border:1px solid rgba(255,70,40,.7); color:#ffd0c2; animation:zoomIn .7s ease both, alarm 1.2s ease-in-out infinite .7s;}
.verdict.smoke{background:linear-gradient(135deg,rgba(255,179,0,.18),rgba(120,90,0,.2)); border:1px solid rgba(255,179,0,.6); color:#ffe19a;}
.verdict.safe{background:linear-gradient(135deg,rgba(34,197,94,.16),rgba(10,70,40,.2)); border:1px solid rgba(34,197,94,.55); color:#a7f3c4;}
.verdict.safe .ico{animation:float 2.5s ease-in-out infinite;}

/* ---------- analyzing animation ---------- */
.scan{
  position:relative; height:190px; border-radius:20px; overflow:hidden; border:1px solid rgba(255,90,31,.4);
  background:repeating-linear-gradient(0deg, rgba(255,90,31,.05) 0 2px, transparent 2px 22px), #0d1019;
  display:grid; place-items:center; margin:10px 0 18px;
}
.scan-line{position:absolute; left:0; right:0; height:70px; background:linear-gradient(180deg,transparent,rgba(255,90,31,.45),transparent); animation:scan 1.6s ease-in-out infinite;}
.scan-text{position:relative; font-family:'JetBrains Mono',monospace; letter-spacing:.2em; color:var(--fire2);}
.dots::after{content:''; animation:dots 1.2s steps(4) infinite;}

/* ---------- buttons (real, clickable) ---------- */
.stButton>button, .stDownloadButton>button{
  width:100%; position:relative; overflow:hidden; border-radius:14px; padding:.75rem 1.4rem;
  font-weight:700; letter-spacing:.03em; transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
/* active step / main action */
.stButton button[kind="primary"], .stButton [data-testid="stBaseButton-primary"], .stDownloadButton>button{
  background:linear-gradient(90deg,#ff5a1f,#ff8a00); color:#fff; border:0; animation:btnGlow 2.4s ease-in-out infinite;
}
/* inactive step */
.stButton button[kind="secondary"], .stButton [data-testid="stBaseButton-secondary"]{
  background:var(--card); color:var(--muted); border:1px solid var(--line);
}
.stButton>button:hover, .stDownloadButton>button:hover{transform:translateY(-3px) scale(1.02); box-shadow:0 12px 30px rgba(255,90,31,.45); color:#fff; border-color:rgba(255,90,31,.7);}

[data-testid="stFileUploader"] section{
  background:rgba(255,255,255,.03); border:2px dashed rgba(255,90,31,.5); border-radius:20px; padding:26px;
  transition:all .3s ease; animation:borderPulse 2.6s ease-in-out infinite;
}
[data-testid="stFileUploader"] section:hover{background:rgba(255,90,31,.08); border-color:var(--fire); transform:scale(1.01);}

[data-testid="stImage"] img, [data-testid="stVideo"] video{
  border-radius:18px; border:1px solid rgba(255,90,31,.35); box-shadow:0 14px 40px rgba(0,0,0,.5);
  animation:reveal .9s ease both;
}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e1120,#080a11); border-right:1px solid var(--line);}
.side-title{font-weight:700; font-size:1.1rem; margin-bottom:4px;}
.side-sub{color:var(--muted); font-size:.85rem; margin-bottom:12px;}
.chips{display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 4px;}
.chip{padding:5px 14px; border-radius:999px; font-size:.82rem; font-weight:600; border:1px solid rgba(255,179,0,.5); color:#ffd27a; background:rgba(255,179,0,.08); animation:float 3s ease-in-out infinite;}
.chip:nth-child(2){animation-delay:.5s;} .chip:nth-child(3){animation-delay:1s;}

.footer{text-align:center; color:var(--muted); font-size:.85rem; margin-top:40px; padding:18px; border-top:1px solid var(--line);}
.footer b{color:var(--fire2);}

/* ---------- keyframes ---------- */
@keyframes fadeUp{from{opacity:0; transform:translateY(22px);} to{opacity:1; transform:none;}}
@keyframes dropIn{from{opacity:0; transform:translateY(-30px);} to{opacity:1; transform:none;}}
@keyframes zoomIn{from{opacity:0; transform:scale(.85);} to{opacity:1; transform:none;}}
@keyframes pop{0%{transform:scale(.3); opacity:0;} 100%{transform:scale(1); opacity:1;}}
@keyframes reveal{from{opacity:0; filter:blur(10px); transform:scale(.97);} to{opacity:1; filter:none; transform:none;}}
@keyframes shine{to{background-position:300% center;}}
@keyframes sweep{0%,100%{transform:scaleX(.1); opacity:.3;} 50%{transform:scaleX(1); opacity:1;}}
@keyframes glow{0%,100%{opacity:.5;} 50%{opacity:1;}}
@keyframes flicker{0%,100%{transform:scale(1) rotate(-3deg);} 50%{transform:scale(1.18) rotate(4deg);}}
@keyframes float{0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);}}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(255,90,31,.7);} 100%{box-shadow:0 0 0 14px rgba(255,90,31,0);}}
@keyframes typing{to{width:49ch;}}
@keyframes caret{50%{border-color:transparent;}}
@keyframes scroll{to{transform:translateX(-50%);}}
@keyframes grow{0%,100%{transform:scaleY(.6);} 50%{transform:scaleY(1.15);}}
@keyframes scan{0%{top:-70px;} 100%{top:190px;}}
@keyframes dots{0%{content:'';} 25%{content:'.';} 50%{content:'..';} 75%{content:'...';}}
@keyframes alarm{0%,100%{box-shadow:0 0 0 0 rgba(255,60,30,.6);} 50%{box-shadow:0 0 0 18px rgba(255,60,30,0);}}
@keyframes btnGlow{0%,100%{box-shadow:0 0 0 rgba(255,90,31,0);} 50%{box-shadow:0 0 22px rgba(255,90,31,.45);}}
@keyframes borderPulse{0%,100%{border-color:rgba(255,90,31,.35);} 50%{border-color:rgba(255,179,0,.9);}}
@keyframes rise{0%{transform:translateY(0) scale(1); opacity:0;} 10%{opacity:.9;} 100%{transform:translateY(-105vh) translateX(40px) scale(.3); opacity:0;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def embers_html(n=26):
    rnd = random.Random(7)
    spans = "".join(
        f'<span style="left:{rnd.randint(1, 98)}%;animation-duration:{rnd.randint(8, 18)}s;'
        f'animation-delay:{rnd.uniform(0, 12):.1f}s;width:{rnd.randint(3, 8)}px;height:{rnd.randint(3, 8)}px"></span>'
        for _ in range(n)
    )
    return f'<div class="embers">{spans}</div>'


st.markdown(embers_html(), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def stretch(fn, *args, **kwargs):
    """Call a Streamlit element so it fills its column, whatever the Streamlit version."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return fn(*args, **kwargs, **extra)
        except Exception:
            if not extra:
                raise


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    """Load the ONNX model once and keep it in memory."""
    path = BASE_DIR / MODEL_NAME
    if not path.exists():
        found = sorted(BASE_DIR.glob("*.onnx"))
        if not found:
            return None, None
        path = found[0]
    return YOLO(str(path), task="detect"), path.name


def draw_detections(frame_bgr, result):
    """Draw boxes + labels on a BGR frame. Returns (annotated_bgr, counts, rows)."""
    out = frame_bgr.copy()
    names = result.names
    counts, rows = {}, []
    thickness = max(2, int(round(min(out.shape[:2]) / 250)))
    font_scale = max(0.5, min(out.shape[:2]) / 900)

    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(names[cls_id])
        key = name.lower()
        color = BOX_COLORS.get(key, DEFAULT_COLOR)

        counts[key] = counts.get(key, 0) + 1
        rows.append({"Class": name.capitalize(), "Confidence %": round(conf * 100, 1),
                     "Box (x1, y1, x2, y2)": f"{x1}, {y1}, {x2}, {y2}"})

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{name.capitalize()} {conf * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        top = max(y1 - th - base - 6, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 10, top + th + base + 6), color, -1)
        cv2.putText(out, label, (x1 + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out, counts, rows


def detect(model, frame_bgr, conf, iou, imgsz):
    result = model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    return draw_detections(frame_bgr, result)


def verdict_card(fire, smoke, extra=""):
    if fire > 0:
        html = ('<div class="verdict fire"><div class="ico">🔥</div><div class="big">FIRE DETECTED</div>'
                f'<div class="sub">The model found fire. Take action immediately! {extra}</div></div>')
    elif smoke > 0:
        html = ('<div class="verdict smoke"><div class="ico">💨</div><div class="big">SMOKE DETECTED</div>'
                f'<div class="sub">The model found smoke - possible fire risk, check the area. {extra}</div></div>')
    else:
        html = ('<div class="verdict safe"><div class="ico">✅</div><div class="big">NO FIRE OR SMOKE</div>'
                f'<div class="sub">The model did not detect fire or smoke. {extra}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def metric_cards(items):
    """items: list of (css_class, label, value)"""
    inner = "".join(
        f'<div class="card {c}"><div class="label">{l}</div><div class="value">{v}</div></div>'
        for c, l, v in items
    )
    st.markdown(f'<div class="cards">{inner}</div>', unsafe_allow_html=True)


def analyzing_box(text="Analyzing"):
    return (f'<div class="scan"><div class="scan-line"></div>'
            f'<div class="scan-text">{text.upper()}<span class="dots"></span></div></div>')


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def to_png_bytes(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def show_media_preview(media):
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if media["kind"] == "image":
            stretch(st.image, Image.open(io.BytesIO(media["bytes"])).convert("RGB"))
        else:
            st.video(media["bytes"])


# ----------------------------------------------------------------------------
# Analysis (image / video) -> returns a result dict stored in session_state
# ----------------------------------------------------------------------------
def analyze_image(model, media, conf, iou, imgsz):
    scan_ph = st.empty()
    scan_ph.markdown(analyzing_box("Analyzing image"), unsafe_allow_html=True)

    pil = Image.open(io.BytesIO(media["bytes"])).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    t0 = time.time()
    annotated, counts, rows = detect(model, frame_bgr, conf, iou, imgsz)
    elapsed = (time.time() - t0) * 1000
    time.sleep(0.8)  # let the scan animation play briefly
    scan_ph.empty()

    return {
        "kind": "image",
        "original": np.array(pil),
        "annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "fire": counts.get("fire", 0),
        "smoke": counts.get("smoke", 0),
        "rows": rows,
        "elapsed": elapsed,
    }


def analyze_video(model, media, conf, iou, imgsz, skip):
    suffix = "." + media["ext"] if media.get("ext") else ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_in.write(media["bytes"])
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        st.error("The video could not be opened. Try another file.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), max(fps / skip, 1), (w, h))

    scan_ph = st.empty()
    scan_ph.markdown(analyzing_box("Analyzing video"), unsafe_allow_html=True)
    frame_ph = st.empty()
    progress = st.progress(0, text="Processing...")

    frames_fire = frames_smoke = processed = 0
    max_fire = max_smoke = 0
    best_conf = 0.0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            annotated, counts, rows = detect(model, frame, conf, iou, imgsz)
            fire, smoke = counts.get("fire", 0), counts.get("smoke", 0)
            processed += 1
            frames_fire += fire > 0
            frames_smoke += smoke > 0
            max_fire, max_smoke = max(max_fire, fire), max(max_smoke, smoke)
            best_conf = max([best_conf] + [r["Confidence %"] for r in rows])
            writer.write(annotated)

            stretch(frame_ph.image, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            progress.progress(min(idx / total, 1.0), text=f"Processing frame {idx}/{total}")
        idx += 1

    cap.release()
    writer.release()
    progress.empty()
    scan_ph.empty()
    frame_ph.empty()

    # re-encode to H.264 so the browser can play the result
    final_out = raw_out
    if shutil.which("ffmpeg"):
        h264 = raw_out.replace(".mp4", "_h264.mp4")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_out,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", h264],
            capture_output=True,
        )
        if proc.returncode == 0 and Path(h264).exists():
            final_out = h264

    return {
        "kind": "video",
        "video_bytes": Path(final_out).read_bytes(),
        "frames_fire": int(frames_fire),
        "frames_smoke": int(frames_smoke),
        "processed": processed,
        "max_fire": max_fire,
        "max_smoke": max_smoke,
        "best_conf": best_conf,
    }


# ----------------------------------------------------------------------------
# Result rendering
# ----------------------------------------------------------------------------
def render_result(res):
    if res["kind"] == "image":
        best = max((r["Confidence %"] for r in res["rows"]), default=0)
        verdict_card(res["fire"], res["smoke"], f"(top confidence {best:.0f}%)" if res["rows"] else "")
        metric_cards([
            ("fire", "🔥 Fire", res["fire"]),
            ("smoke", "💨 Smoke", res["smoke"]),
            ("ok", "Total detections", len(res["rows"])),
        ])
        c1, c2 = st.columns(2, gap="large")
        with c1:
            section("Original")
            stretch(st.image, res["original"])
        with c2:
            section("Model prediction")
            stretch(st.image, res["annotated"])
        st.caption(f"Inference time: {res['elapsed']:.0f} ms")
        if res["rows"]:
            section("Detections")
            stretch(st.dataframe, pd.DataFrame(res["rows"]), hide_index=True)
        st.download_button("⬇️ Download result image", to_png_bytes(res["annotated"]),
                           file_name="fire_smoke_result.png", mime="image/png")
    else:
        best = res["best_conf"]
        verdict_card(res["max_fire"] and 1 or 0, res["max_smoke"] and 1 or 0,
                     f"(top confidence {best:.0f}%)" if best else "")
        metric_cards([
            ("fire", "🔥 Frames with fire", res["frames_fire"]),
            ("smoke", "💨 Frames with smoke", res["frames_smoke"]),
            ("ok", "Frames processed", res["processed"]),
        ])
        section("Model prediction (video)")
        st.video(res["video_bytes"])
        st.download_button("⬇️ Download result video", res["video_bytes"],
                           file_name="fire_smoke_result.mp4", mime="video/mp4")


# ----------------------------------------------------------------------------
# Top bar + hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
      <div class="brand"><span class="dot"></span> FIRE &amp; SMOKE AI</div>
      <div class="author">Eng. Youssef Madin</div>
    </div>

    <div class="hero">
      <div class="badge">AI SAFETY VISION</div>
      <div class="flame">🔥</div>
      <h1>Fire &amp; Smoke Detection</h1>
      <span class="typer">Upload an image or video. AI finds fire &amp; smoke.</span>
      <p>A YOLO-powered computer-vision system. Drop any image or video and the model predicts
      whether it contains <b>fire</b> or <b>smoke</b>, and shows exactly where.</p>
    </div>

    <div class="marquee"><div class="track">
      <span><b>●</b> FIRE DETECTION</span><span><b>●</b> SMOKE DETECTION</span><span><b>●</b> IMAGES &amp; VIDEOS</span>
      <span><b>●</b> YOLO + ONNX</span><span><b>●</b> REAL-TIME ANALYSIS</span><span><b>●</b> EARLY WARNING</span>
      <span><b>●</b> FIRE DETECTION</span><span><b>●</b> SMOKE DETECTION</span><span><b>●</b> IMAGES &amp; VIDEOS</span>
      <span><b>●</b> YOLO + ONNX</span><span><b>●</b> REAL-TIME ANALYSIS</span><span><b>●</b> EARLY WARNING</span>
    </div></div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
model, model_file = load_model()

with st.sidebar:
    st.markdown('<div class="side-title">⚙️ Detection settings</div>'
                '<div class="side-sub">Tune the model to your scene</div>', unsafe_allow_html=True)
    conf_thr = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                         help="Higher = fewer false alarms, but real fires may be missed.")
    iou_thr = st.slider("IoU threshold (NMS)", 0.10, 0.95, 0.50, 0.05,
                        help="Controls how overlapping boxes are merged.")
    imgsz = st.select_slider("Image size", options=[320, 416, 512, 640, 800], value=640,
                             help="Should match the size used when the ONNX model was exported (usually 640).")
    skip = st.slider("Video: process every N-th frame", 1, 10, 3, help="Higher = faster video processing.")
    st.divider()
    if model is not None:
        st.success(f"Model loaded: `{model_file}`")
        chips = "".join(f'<span class="chip">{str(v).upper()}</span>' for v in model.names.values())
        st.markdown('<div class="side-sub">The model is trained to detect:</div>'
                    f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    else:
        st.error("Model file not found")
    st.caption("Built with YOLO + Streamlit")

if model is None:
    st.error(
        f"Could not find the model. Put **{MODEL_NAME}** in the same folder as `app.py` "
        "(next to this file) and reload the page."
    )
    st.stop()

# ----------------------------------------------------------------------------
# State + step navigation (these are REAL buttons)
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("step", 1)
ss.setdefault("media", None)
ss.setdefault("result", None)
ss.setdefault("uploader_key", 0)


def go(n):
    ss["step"] = n


def reset_all():
    ss["step"] = 1
    ss["media"] = None
    ss["result"] = None
    ss["uploader_key"] += 1


nav_cols = st.columns(3)
for i, (col, label) in enumerate(zip(nav_cols, ["1 · Upload", "2 · Analyze", "3 · Result"]), start=1):
    with col:
        st.button(label, key=f"nav_{i}", on_click=go, args=(i,),
                  type="primary" if ss["step"] == i else "secondary")
st.write("")

media = ss["media"]

# ============================== STEP 1: UPLOAD ===============================
if ss["step"] == 1:
    section("Step 1 · Upload an image or a video")
    st.markdown('<div class="hint">Choose a file from your device. Images: JPG, PNG, BMP, WEBP · '
                'Videos: MP4, MOV, AVI, MKV</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload", type=IMAGE_TYPES + VIDEO_TYPES, key=f"uploader_{ss['uploader_key']}",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        ext = Path(uploaded.name).suffix.lower().lstrip(".")
        cur = ss["media"]
        if cur is None or cur["name"] != uploaded.name or cur["size"] != len(data):
            ss["media"] = {
                "name": uploaded.name, "bytes": data, "ext": ext, "size": len(data),
                "kind": "image" if ext in IMAGE_TYPES else "video",
            }
            ss["result"] = None
        media = ss["media"]

    if media:
        st.success(f"File ready: **{media['name']}** ({media['kind']})")
        show_media_preview(media)
        st.button("Next: Analyze  →", key="to_analyze", on_click=go, args=(2,))
    else:
        st.info("👆 Drop an image or a video above to begin.")

# ============================== STEP 2: ANALYZE ==============================
elif ss["step"] == 2:
    section("Step 2 · Analyze with the YOLO model")
    if not media:
        st.warning("No file yet. Please upload an image or a video first.")
        st.button("← Go to Upload", key="back_upload", on_click=go, args=(1,))
    else:
        st.markdown(f'<div class="hint">File: <b>{media["name"]}</b> · confidence ≥ {conf_thr:.2f} · '
                    f'IoU {iou_thr:.2f} · size {imgsz}</div>', unsafe_allow_html=True)
        show_media_preview(media)

        if st.button("🔍 Analyze now", key="run_analysis"):
            if media["kind"] == "image":
                res = analyze_image(model, media, conf_thr, iou_thr, imgsz)
            else:
                res = analyze_video(model, media, conf_thr, iou_thr, imgsz, skip)
            if res is not None:
                ss["result"] = res
                ss["step"] = 3
                st.rerun()

# ============================== STEP 3: RESULT ===============================
else:
    section("Step 3 · Result")
    res = ss["result"]
    if not res:
        st.warning("No result yet. Upload a file, then press **Analyze now**.")
        c1, c2 = st.columns(2)
        with c1:
            st.button("← Go to Upload", key="r_upload", on_click=go, args=(1,))
        with c2:
            if media:
                st.button("Go to Analyze  →", key="r_analyze", on_click=go, args=(2,))
    else:
        render_result(res)
        st.write("")
        st.button("🔄 Analyze another file", key="again", on_click=reset_all)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="footer">Developed by <b>Eng. Youssef Madin</b> · Fire &amp; Smoke Detection with YOLO</div>',
    unsafe_allow_html=True,
)
