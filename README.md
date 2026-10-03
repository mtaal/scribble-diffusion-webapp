# Scribble Diffusion Web App

A Streamlit web interface for [scribble-diffusion](https://github.com/mtaal/scribble-diffusion),
a conditional diffusion model (UNet + ControlNet) that generates interactive scribbles for
segmentation correction.

You draw (or upload) a ground-truth mask and a prediction mask. The app then runs the
diffusion sampler with ONNX Runtime and generates a scribble that marks where the
prediction should be corrected.

## Installation

Tested with Python 3.12.

```bash
git clone https://github.com/mtaal/scribble-diffusion-webapp.git
cd scribble-diffusion-webapp

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

## Running

```bash
python -m streamlit run scribble_app.py
```

Then open http://localhost:8501 (the port and theme are set in `.streamlit/config.toml`).

## Usage

1. **Provide the masks**, using either option:
   - Draw them on the **Ground Truth (green)** and **Prediction (red)** canvases. Click
     *Draw here* to choose which canvas is active.
   - Upload an image in which green is the ground truth and red is the prediction.
2. **Check the Composite preview.** Green is GT only, red is prediction only, and yellow is
   where they overlap.
3. **Set the options in the sidebar:**
   - **Drawing tool:** freedraw, line, rect, circle, transform, polygon or point
   - **Stroke width:** brush size for drawing
   - **Scribble Class:** generate a *Background* scribble (false positive region) or a
     *Foreground* scribble (false negative region)
   - **Diffusion steps:** 10 to 50. Fewer steps sample faster by skipping timesteps.
4. **Click Generate Scribble.** The output appears together with its individual channels
   (green = ground truth, red = prediction, blue = generated scribble).
5. **Click Clear and Draw Again** to start over.

## Project Structure

```
.
├── scribble_app.py        # Streamlit UI (canvases, upload, model sync, generation)
├── diffusion.py           # DDPM sampler running the ONNX denoiser
├── app.py                 # Entry point wrapper around scribble_app.main()
├── .streamlit/config.toml # Server and theme settings
├── requirements.txt       # Python dependencies
└── LICENSE                # MIT License
```

## Model

On startup the app downloads the latest ONNX model from the public Hugging Face bucket
`mtaal/scribblegen` into a local `model/` directory next to the app. If the directory is
missing, empty, or the remote model has changed, the cached ONNX file is refreshed before the
UI loads. No Hugging Face account or token is needed.

To train your own model and export it to ONNX, see
[scribble-diffusion](https://github.com/mtaal/scribble-diffusion).

## Acknowledgments

- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) for the drawing canvas

## License

[MIT](LICENSE)
