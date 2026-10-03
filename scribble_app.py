import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from diffusion import diffusion_process

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
import onnxruntime as ort

# streamlit-drawable-canvas 0.9.3 still calls streamlit.elements.image.image_to_url(image, width, ...)
# which newer Streamlit moved to elements.lib.image_utils with a LayoutConfig argument.
# Install a positional-compatible shim only when the old attribute is missing (upload path).
import streamlit.elements.image as _st_image
if not hasattr(_st_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        return _image_to_url(image, _LayoutConfig(width=width), clamp, channels, output_format, image_id)

    _st_image.image_to_url = _image_to_url_compat

st.set_page_config(page_title="Scribble Segmentation", page_icon="✏️", layout="wide")

# Streamlit's default main-container padding (96px top / 160px bottom) leaves a large empty
# vertical band; trim it so the app uses the full viewport height.
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"], .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Keep the sidebar (and app) full viewport height so no white gap shows below the
       sidebar, including during Streamlit reruns when content briefly collapses. */
    [data-testid="stSidebar"], [data-testid="stAppViewContainer"] {
        min-height: 100vh !important;
    }
    /* Hidden proxies for the canvas toolbar's trash icon - see _wire_canvas_reset_buttons() */
    .st-key-canvas_clear_hooks { display: none; }
    /* Composite action buttons: shrink to content with minimal side padding */
    .st-key-btn_clear button, .st-key-btn_generate button {
        width: auto !important;
        min-width: 0 !important;
        padding-left: 5px !important;
        padding-right: 5px !important;
    }
    /* Center the composite preview image (and its legend) under the button row */
    #composite-preview { text-align: center; }
    #composite-preview .composite-legend {
        font-size: 0.8rem;
        color: rgba(49, 51, 63, 0.6);
        margin-top: 0.25rem;
    }
    /* Center the two action buttons as a group on the column */
    .st-key-composite_btnrow [data-testid="stHorizontalBlock"] {
        justify-content: center;
        gap: 0.4rem;
    }
    .st-key-composite_btnrow [data-testid="stColumn"] {
        width: auto !important;
        flex: 0 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

base_location = os.path.dirname(os.path.abspath(__file__))
BUCKET_ID = "mtaal/scribblegen"
REMOTE_MODEL_PATH = "scribble_model_small.onnx"
LOCAL_MODEL_DIR = Path(base_location) / "model"
MODEL_FILENAME = "scribble_model_small.onnx"
MODEL_PATH = LOCAL_MODEL_DIR / MODEL_FILENAME
MODEL_REVISION_PATH = LOCAL_MODEL_DIR / ".scribble_model_revision.json"


# create tmp folder in base_location if it does not exist
try:
    Path(f"{base_location}/tmp/").mkdir()
except FileExistsError:
    pass

def _read_local_model_revision():
    if not MODEL_REVISION_PATH.exists():
        return None

    try:
        with MODEL_REVISION_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("revision")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _write_local_model_revision(revision):
    payload = {
        "revision": revision,
        "downloaded_at": time.time(),
    }
    with MODEL_REVISION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _is_model_dir_empty():
    if not LOCAL_MODEL_DIR.exists():
        return True

    return not any(LOCAL_MODEL_DIR.iterdir())


def _download_latest_model():
    from huggingface_hub import download_bucket_files, list_bucket_tree

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    remote_model_file = None
    for bucket_file in list_bucket_tree(BUCKET_ID, recursive=True):
        if bucket_file.path == REMOTE_MODEL_PATH:
            remote_model_file = bucket_file
            break

    if remote_model_file is None:
        raise FileNotFoundError(f"Remote model not found in bucket: {BUCKET_ID}/{REMOTE_MODEL_PATH}")

    remote_revision = remote_model_file.xet_hash
    if remote_revision is None:
        remote_revision = f"{remote_model_file.size}:{remote_model_file.uploaded_at}"

    local_revision = _read_local_model_revision()

    if not MODEL_PATH.exists() or _is_model_dir_empty() or local_revision != remote_revision:
        download_bucket_files(
            BUCKET_ID,
            files=[(remote_model_file, MODEL_PATH)],
            raise_on_missing_files=True,
            token=False,
        )
        _write_local_model_revision(remote_revision)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")

    return MODEL_PATH


@st.cache_resource
def load_onnx_session():
    model_path = _download_latest_model()

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# put the ort_session in the streamlit session state to avoid reloading it every time
if "ort_session" not in st.session_state:
    st.session_state["ort_session"] = None

# Streamlit Cloud may run the app in a fresh worker; keep initialization tolerant to missing runtimes.
if "onnx_init_error" not in st.session_state:
    st.session_state["onnx_init_error"] = None

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None  # kept for backwards-compat; not actively used

if 'canvas_read_only' not in st.session_state:
    st.session_state.canvas_read_only = False

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

if 'canvas_gt_bg' not in st.session_state:
    st.session_state.canvas_gt_bg = None

if 'canvas_pred_bg' not in st.session_state:
    st.session_state.canvas_pred_bg = None

# One version per canvas: bumping a version remounts that canvas with a fresh (empty) drawing.
# They are bumped together on upload / global clear, and individually by the per-canvas delete.
if 'canvas_gt_version' not in st.session_state:
    st.session_state.canvas_gt_version = 0

if 'canvas_pred_version' not in st.session_state:
    st.session_state.canvas_pred_version = 0

if 'uploaded_source' not in st.session_state:
    st.session_state.uploaded_source = None

if 'generate_requested' not in st.session_state:
    st.session_state.generate_requested = False

def _apply_uploaded_image():
    """on_change callback of the file uploader: split the uploaded image into the GT (green) and
    prediction (red) canvas backgrounds. Runs before the script body of the triggered rerun, so the
    canvases pick up the new backgrounds in that same run (no extra click, no st.rerun())."""
    uploaded_file = st.session_state.get("upload_file")
    if uploaded_file is None:
        return
    img = Image.open(uploaded_file)
    # Resize to canvas size with nearest neighbor for sharp edges
    img = img.resize((256, 256), Image.Resampling.NEAREST)
    img_array = np.array(img.convert("RGBA"))

    # Threshold R (prediction) and G (ground truth) to 0/255, drop blue, fully opaque
    img_array[:, :, 0] = np.where(img_array[:, :, 0] > 0, 255, 0)
    img_array[:, :, 1] = np.where(img_array[:, :, 1] > 0, 255, 0)
    img_array[:, :, 2] = 0
    img_array[:, :, 3] = 255
    img_array = img_array.astype(np.uint8)

    # Per-channel background images for the two canvases (RGB, black background)
    gt_bg = np.zeros((256, 256, 3), dtype=np.uint8)
    gt_bg[:, :, 1] = img_array[:, :, 1]
    st.session_state.canvas_gt_bg = Image.fromarray(gt_bg, mode='RGB')

    pred_bg = np.zeros((256, 256, 3), dtype=np.uint8)
    pred_bg[:, :, 0] = img_array[:, :, 0]
    st.session_state.canvas_pred_bg = Image.fromarray(pred_bg, mode='RGB')

    st.session_state.canvas_read_only = True
    # force fresh canvases
    st.session_state.canvas_gt_version += 1
    st.session_state.canvas_pred_version += 1
    st.session_state.uploaded_source = getattr(uploaded_file, "name", None)


def _keep_canvas_backgrounds_alive():
    """Re-register both canvas background images with Streamlit's media file manager at the start
    of every run, before any element is created.

    Streamlit clears a session's media references when a run starts and deletes unreferenced files
    when it ends - also for runs aborted by a RerunException. After an upload both canvases remount
    and each triggers a rerun; if one of those aborts the current run between the GT and the
    prediction canvas, the prediction background file is deleted before the iframe fetches it and
    that canvas stays blank until a later run happens to complete first. Registering the same bytes
    here (same file id, same URL) keeps both files referenced for the whole run."""
    from streamlit_drawable_canvas import _resize_img

    for name, bg in (("gt", st.session_state.canvas_gt_bg), ("pred", st.session_state.canvas_pred_bg)):
        if bg is not None:
            _st_image.image_to_url(_resize_img(bg, 256, 256), 256, True, "RGB", "PNG", f"scribble-canvas-bg-{name}")

if st.session_state["ort_session"] is None:
    try:
        print("Creating ONNX session from synced model")
        st.session_state["ort_session"] = load_onnx_session()
        torch.manual_seed(42)
        st.session_state["onnx_init_error"] = None
    except Exception as exc:
        st.session_state["onnx_init_error"] = str(exc)
        st.session_state["ort_session"] = None
else:
    print("Using existing ONNX session")

ort_session = st.session_state["ort_session"]

if st.session_state["onnx_init_error"]:
    st.warning(f"Model initialization issue: {st.session_state['onnx_init_error']}")

def decoder(x, t, conditionGt, conditionPred, scribble_class=0):
    print(f"scribble_class: {scribble_class}")
    if ort_session is None:
        st.error("ONNX session is not initialized.")
        return None
    inputs = {
        "x": x,
        "t": t,
        "conditionGt": conditionGt,
        "conditionPred": conditionPred,
        "scribbleClass": np.array([scribble_class], dtype=np.int64)
    }

    result = ort_session.run(None, inputs)
    return result

def main():
    full_app()


def about():
    st.markdown(
        """
    Welcome to the Scribble Test Generation Tool. With this tool you can create ground truth and prediction 
    areas and then test generating a scribble.
    """
    )

def composite_canvases(gt_data, pred_data):
    """Combine GT (green channel) and Prediction (red channel) canvas data into a single RGBA array.
    GT only  -> green  (#00FF00)
    Pred only -> red   (#FF0000)
    Both      -> yellow (#FFFF00)
    """
    result = np.zeros((256, 256, 4), dtype=np.uint8)
    if gt_data is not None:
        result[:, :, 1] = gt_data[:, :, 1]  # Green channel from GT canvas
    if pred_data is not None:
        result[:, :, 0] = pred_data[:, :, 0]  # Red channel from Pred canvas
    result[:, :, 3] = 255  # Fully opaque
    return result


def make_preview_image(gt_data, pred_data):
    """Build a composite RGB preview: GT=green, Pred=red, overlap=yellow."""
    h, w = 256, 256
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    gt_mask = np.zeros((h, w), dtype=bool)
    pred_mask = np.zeros((h, w), dtype=bool)

    if gt_data is not None:
        arr = gt_data.astype(np.uint8)
        mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
        gt_img = Image.fromarray(arr, mode=mode).resize((w, h), Image.Resampling.NEAREST).convert("RGBA")
        gt_mask = np.array(gt_img)[:, :, 1] > 0     # any green drawn

    if pred_data is not None:
        arr = pred_data.astype(np.uint8)
        mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
        pred_img = Image.fromarray(arr, mode=mode).resize((w, h), Image.Resampling.NEAREST).convert("RGBA")
        pred_mask = np.array(pred_img)[:, :, 0] > 0  # any red drawn

    overlap = gt_mask & pred_mask

    # GT only – green
    canvas[gt_mask & ~overlap] = [0, 255, 0]
    # Pred only – red
    canvas[pred_mask & ~overlap] = [255, 0, 0]
    # Overlap – yellow
    canvas[overlap] = [255, 255, 0]

    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


def _wire_canvas_reset_buttons():
    """Bridge the two canvas toolbars to Streamlit, from the parent document.

    The toolbar is all-or-nothing (display_toolbar) and lives inside the component iframe, which is
    served from the app's own origin - so it can be reached into from here. Two things need fixing:

    * "Send to Streamlit" (the download icon) is hidden; the app has its own download link.
    * "Reset canvas & history" only clears the strokes drawn inside the component. An uploaded image
      is passed as background_image and is painted on a separate canvas the reset never touches, so
      with an upload loaded the icon looks dead. Clicking it also clicks the matching hidden
      Streamlit button, which drops that background and remounts the canvas.

    Each canvas sits in its own keyed container, so the icons are matched to the right canvas by
    wrapper rather than by iframe order.
    """
    components.html(
        """
        <script>
        (function () {
          const CANVASES = [
            { wrapper: ".st-key-canvas_gt_wrap", clearButton: ".st-key-btn_clear_gt button" },
            { wrapper: ".st-key-canvas_pred_wrap", clearButton: ".st-key-btn_clear_pred button" },
          ];

          const wire = () => {
            const parentDoc = window.parent.document;
            CANVASES.forEach(({ wrapper, clearButton }) => {
              const host = parentDoc.querySelector(wrapper);
              if (!host) return;
              const frame = host.querySelector('iframe[src*="streamlit_drawable_canvas"]');
              if (!frame) return;
              let doc;
              try {
                doc = frame.contentDocument;
              } catch (e) {
                return; /* not reachable yet */
              }
              if (!doc) return;

              doc.querySelectorAll('img[alt="Send to Streamlit"]').forEach((img) => {
                img.style.display = "none";
              });

              doc.querySelectorAll('img[alt="Reset canvas & history"]').forEach((img) => {
                if (img.dataset.scribbleClearBound) return;
                img.dataset.scribbleClearBound = "1";
                img.addEventListener("click", () => {
                  const btn = parentDoc.querySelector(clearButton);
                  if (btn) btn.click();
                });
              });
            });
          };

          wire();
          setInterval(wire, 500);
        })();
        </script>
        """,
        height=0,
    )


def full_app():
    _keep_canvas_backgrounds_alive()

    st.sidebar.header("Configuration")

    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    else:
        point_display_radius = 0

    scribble_class_label = st.sidebar.radio(
        "Scribble Class",
        ["Background", "Foreground"],
        index=0,
    )
    scribble_class = 0 if scribble_class_label == "Background" else 1

    num_steps = st.sidebar.number_input(
        "Diffusion steps",
        min_value=10,
        max_value=50,
        value=50,
        step=1,
        help="Fewer steps sample faster by skipping timesteps (t walks from 50 down to 0).",
    )

    display_toolbar = st.sidebar.checkbox("Display toolbar", True)

    # File uploader; the image is applied in the on_change callback (see _apply_uploaded_image)
    st.file_uploader(
        "Upload a GT/prediction image (green = ground truth, red = prediction)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="upload_file",
        on_change=_apply_uploaded_image,
    )

    # Track active canvas in session state
    if 'active_canvas' not in st.session_state:
        st.session_state.active_canvas = "Ground Truth (Green)"

    # Determine draw mode per canvas — inactive one is locked to transform
    active_draw_mode: str = drawing_mode or "freedraw"
    canvas_mode_gt = (
        "transform" if st.session_state.canvas_read_only
        else (active_draw_mode if st.session_state.active_canvas == "Ground Truth (Green)" else "transform")
    )
    canvas_mode_pred = (
        "transform" if st.session_state.canvas_read_only
        else (active_draw_mode if st.session_state.active_canvas == "Prediction (Red)" else "transform")
    )

    # Hidden buttons clicked from JS when the canvas toolbar's trash icon is used; see
    # _wire_canvas_reset_buttons(). They must be rendered (and handled) before the canvases so a
    # clear takes effect in the same run that reports it.
    with st.container(key="canvas_clear_hooks"):
        clear_gt_clicked = st.button("Clear GT canvas", key="btn_clear_gt")
        clear_pred_clicked = st.button("Clear Prediction canvas", key="btn_clear_pred")

    if clear_gt_clicked or clear_pred_clicked:
        # The toolbar's own reset only wipes the strokes drawn inside the component; the uploaded
        # image lives in background_image and survives it. Drop that background too and remount the
        # canvas, so the drawing area really ends up empty.
        if clear_gt_clicked:
            st.session_state.canvas_gt_bg = None
            st.session_state.canvas_gt_version += 1
        if clear_pred_clicked:
            st.session_state.canvas_pred_bg = None
            st.session_state.canvas_pred_version += 1
        # An upload locks both canvases; a cleared canvas is only useful if it can be drawn on.
        st.session_state.canvas_read_only = False
        st.rerun()

    # --- Drawing row: GT | Pred | Composite ---
    col_gt, col_pred, col_preview = st.columns(3)

    with col_gt:
        active_gt = st.session_state.active_canvas == "Ground Truth (Green)"
        st.markdown("#### Ground Truth (green)")
        if st.button("✏️ Drawing here" if active_gt else "Draw here",
                     key="btn_gt", type="primary" if active_gt else "secondary"):
            st.session_state.active_canvas = "Ground Truth (Green)"
            st.rerun()
        with st.container(key="canvas_gt_wrap"):
            canvas_gt = st_canvas(
                fill_color="#00FF00",
                stroke_width=stroke_width,
                stroke_color="#00FF00",
                background_image=st.session_state.canvas_gt_bg,
                update_streamlit=True,
                height=256,
                width=256,
                drawing_mode=canvas_mode_gt,
                point_display_radius=point_display_radius if active_draw_mode == "point" else 0,
                display_toolbar=display_toolbar,
                key=f"canvas_gt_{st.session_state.canvas_gt_version}",
            )

    with col_pred:
        active_pred = st.session_state.active_canvas == "Prediction (Red)"
        st.markdown("#### Prediction (red)")
        if st.button("✏️ Drawing here" if active_pred else "Draw here",
                     key="btn_pred", type="primary" if active_pred else "secondary"):
            st.session_state.active_canvas = "Prediction (Red)"
            st.rerun()
        with st.container(key="canvas_pred_wrap"):
            canvas_pred = st_canvas(
                fill_color="#FF0000",
                stroke_width=stroke_width,
                stroke_color="#FF0000",
                background_image=st.session_state.canvas_pred_bg,
                update_streamlit=True,
                height=256,
                width=256,
                drawing_mode=canvas_mode_pred,
                point_display_radius=point_display_radius if active_draw_mode == "point" else 0,
                display_toolbar=display_toolbar,
                key=f"canvas_pred_{st.session_state.canvas_pred_version}",
            )

    _wire_canvas_reset_buttons()

    with col_preview:
        st.markdown("#### Composite")
        # Action buttons in one centered row directly under the title, so the composite image
        # below lines up with the GT/Prediction canvases and is centered on the button row.
        with st.container(key="composite_btnrow"):
            clear_col, gen_col = st.columns(2)
        with clear_col:
            clear_clicked = st.button("🗑️ Clear and Draw Again", key="btn_clear")
        with gen_col:
            if st.button("Generate Scribble", key="btn_generate"):
                # Remember the request in session state rather than acting on the button value
                # directly: any widget event during the diffusion (canvas update, radio change,
                # second click) raises RerunException at the next st.* call and aborts this run.
                # The flag survives the abort, so the next run simply retries the generation.
                st.session_state.generate_requested = True

        gt_data = canvas_gt.image_data
        pred_data = canvas_pred.image_data
        # After an upload the canvas key changed so image_data is None on first render;
        # fall back to the stored background images so the composite shows immediately.
        # Also fall back if the canvas returned an all-zero array (blank fresh canvas).
        if st.session_state.canvas_gt_bg is not None and (gt_data is None or not np.any(gt_data[:, :, 1])):
            gt_data = np.array(st.session_state.canvas_gt_bg)
        if st.session_state.canvas_pred_bg is not None and (pred_data is None or not np.any(pred_data[:, :, 0])):
            pred_data = np.array(st.session_state.canvas_pred_bg)
        preview_img = make_preview_image(gt_data, pred_data)
        # Encode as base64 so we can set exact CSS height/margin to align with the canvas
        buf = BytesIO()
        preview_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f'<div id="composite-preview">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:256px; height:256px; display:inline-block;">'
            f'<div class="composite-legend">'
            f'<span style="color:#09ab3b">green</span> = GT &middot; '
            f'<span style="color:#ff2b2b">red</span> = Prediction &middot; '
            f'<span style="color:#e0a800">yellow</span> = overlap'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if clear_clicked:
            st.session_state.canvas_read_only = False
            st.session_state.generated_image = None
            st.session_state.canvas_gt_bg = None
            st.session_state.canvas_pred_bg = None
            st.session_state.uploaded_source = None
            st.session_state.canvas_gt_version += 1
            st.session_state.canvas_pred_version += 1
            st.rerun()

    st.markdown("---")

    if st.session_state.get('generate_requested'):
        # Build input: use canvas data, falling back to stored bg images (set on upload)
        gt_src = canvas_gt.image_data
        if st.session_state.canvas_gt_bg is not None and (gt_src is None or not np.any(gt_src[:, :, 1])):
            gt_src = np.array(st.session_state.canvas_gt_bg)
        pred_src = canvas_pred.image_data
        if st.session_state.canvas_pred_bg is not None and (pred_src is None or not np.any(pred_src[:, :, 0])):
            pred_src = np.array(st.session_state.canvas_pred_bg)

        if gt_src is not None or pred_src is not None:
            composited = composite_canvases(gt_src, pred_src)
            input_img = Image.fromarray(composited).convert('RGB')
        else:
            st.error("No image available. Please draw something or upload a file.")
            input_img = None
            st.session_state.generate_requested = False

        if input_img is not None:
            progress_bar = st.progress(0, text="Generating scribble...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text=f"Diffusion step {current}/{total}")

            try:
                result_img = diffusion_process(
                    input_img,
                    decoder,
                    scribble_class=scribble_class,
                    num_steps=int(num_steps),
                    progress_callback=update_progress,
                )
            except Exception as e:
                progress_bar.empty()
                st.error(f"Diffusion failed: {e}")
                st.session_state.generate_requested = False
            else:
                progress_bar.empty()
                if result_img is not None:
                    # Keep the previous result until a new one exists; store as numpy array
                    st.session_state.generated_image = np.array(result_img)
                st.session_state.generate_requested = False

    # --- Results row ---
    if st.session_state.generated_image is not None:
        st.subheader("Generated Scribble — Individual Channels")

        result_array = st.session_state.generated_image
        col_out, col_g, col_r, col_b = st.columns(4)

        # Base filename for downloads, derived from the uploaded image if any
        uploaded_source = st.session_state.get("uploaded_source")
        name_base = Path(uploaded_source).stem if uploaded_source else "scribble"

        def _png_bytes(arr):
            buf = BytesIO()
            Image.fromarray(arr.astype(np.uint8), mode="RGB").save(buf, format="PNG")
            return buf.getvalue()

        def _download(label, arr, suffix):
            st.download_button(
                ":material/download:",  # icon-only (standard Material download icon)
                data=_png_bytes(arr),
                file_name=f"{name_base}_{suffix}.png",
                mime="image/png",
                key=f"dl_{suffix}",
                help=f"Download {label} as PNG",
            )

        with col_out:
            st.markdown("**Output**")
            st.image(result_array.astype(np.uint8), width=256)  # display 256x256; download stays 128x128
            _download("Output", result_array, "output")

        with col_g:
            st.markdown("**Green — Ground Truth**")
            green_rgb = np.zeros_like(result_array)
            green_rgb[:, :, 1] = result_array[:, :, 1]
            st.image(green_rgb.astype(np.uint8), width=256)
            _download("Green", green_rgb, "ground_truth")

        with col_r:
            st.markdown("**Red — Prediction**")
            red_rgb = np.zeros_like(result_array)
            red_rgb[:, :, 0] = result_array[:, :, 0]
            st.image(red_rgb.astype(np.uint8), width=256)
            _download("Red", red_rgb, "prediction")

        with col_b:
            st.markdown("**Blue — Scribble**")
            blue_rgb = np.zeros_like(result_array)
            blue_rgb[:, :, 2] = result_array[:, :, 2]
            st.image(blue_rgb.astype(np.uint8), width=256)
            _download("Blue", blue_rgb, "scribble")

        if not np.any(result_array[:, :, 2]):
            st.warning(
                "The model returned an empty scribble for this input / Scribble Class combination. "
                "Try regenerating or the other Scribble Class."
            )
        else:
            st.success("Image generated successfully!")

        uploaded_source = st.session_state.get('uploaded_source', None)
        if uploaded_source:
            st.markdown("**Uploaded filename:**")
            st.write(Path(uploaded_source).name if (os.path.sep in uploaded_source or '/' in uploaded_source) else uploaded_source)

def png_export():
    st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("tmp/").glob("*.png"):
        st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    if st.session_state["button_id"] == "":
        button_id = ''.join(c for c in str(uuid.uuid4()).replace("-", "") if not c.isdigit())
        st.session_state["button_id"] = button_id

    button_id = st.session_state["button_id"]
    file_path = f"tmp/{button_id}.png"

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data).decode()
        except AttributeError:
            b64 = base64.b64encode(img_data.encode()).decode() # type: ignore

        dl_link = (
            custom_css
            + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
        )
        st.markdown(dl_link, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Scribble Tester", page_icon=":pencil2:", layout="wide"
    )
    st.title("Scribble Tester")
    st.sidebar.subheader("Configuration")
    main()
