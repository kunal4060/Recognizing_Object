# Recognizing Object

Real-time object detection from your webcam using a pretrained YOLOv8 model. Open the script, point your camera at something, and bounding boxes are drawn live on the video feed.

The repo also contains `face.py`, a stub reserved for a future face-detection experiment (for example, an attendance-style detector).

## What it does

`main.py` loads the pretrained `yolov8n.pt` weights, captures frames from the default webcam with OpenCV, runs detection on each frame, and displays the annotated video in a window. Press `Esc` to quit.

## Requirements

- Python 3.8 or newer
- A webcam
- The Python packages `ultralytics` and `opencv-python`

## Install

### 1. Download the project

```bash
git clone https://github.com/kunal4060/Recognizing_Object.git
cd Recognizing_Object
```

### 2. Install the Python packages

```bash
pip install ultralytics opencv-python
```

## Usage

Run the detector:

```bash
python main.py
```

A window titled "Detection" opens with live bounding boxes. Press `Esc` to close it. On first run, the YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically by the `ultralytics` package.

## Troubleshooting

### No window appears / camera not found

Make sure no other application is using the webcam, and that your OS has granted camera permission to the terminal or editor running the script.

### `ModuleNotFoundError`

Install the dependencies with the same Python you use to run the script:

```bash
python -m pip install ultralytics opencv-python
```

## License

Provided for learning purposes. No warranty is provided.
