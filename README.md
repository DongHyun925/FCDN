# FCDN
<p align="center">
  <img src="images/example_foggy_day.png" width="45%">
  <img src="images/example_clear_night.png" width="45%">
</p>

<p align="center">
  <em>Example inference results of FCDN (Foggy/Clear and Day/Night)</em>
</p>

**FCDN (Foggy/Clear & Day/Night Detector)**  
A video-based classification project that automatically determines **day/night** conditions and **fog presence** from real-world surveillance videos.

---

## Overview
FCDN is designed to analyze input videos and classify environmental conditions using a lightweight and interpretable pipeline.  
The system first distinguishes **day vs. night** based on global brightness, applies **gamma correction** for night scenes, and then classifies **foggy vs. clear** conditions using a trained MLP model based on handcrafted visual features.

---

## Pipeline
1. **Day / Night Classification**
   - Based on average frame brightness
   - Threshold-based decision (default: 100)

2. **Night-time Enhancement**
   - Gamma correction applied when classified as NIGHT (gamma = 2.0)

3. **Foggy / Clear Classification**
   - Extracts 7 numerical visual features
   - Uses a trained **MLP binary classifier**

---

## Project Structure
```text
FCDN/
├─ images/
│  ├─ example_foggy_day.png
│  └─ example_clear_night.png
├─ .gitignore
├─ README.md
├─ main.py
├─ day_night_detector.py
├─ foggy_clear_detector.py
├─ tetra_brighten.py
├─ best_mlp_strong_focal.pth
├─ clear_day_tetra.mp4
├─ clear_night_tetra.mp4
├─ foggy_day_tetra.mp4
└─ foggy_night_tetra.mp4
```

---

## Visual Features for Fog Classification (7)
- `haze_strength` (currently fixed to 0)
- `sharpness`
- `brightness`
- `contrast`
- `edge_density`
- `dark_mean`
- `saturation`

---

## Requirements
```bash
pip install opencv-python numpy torch
```

## Usage
```bash
python main.py
```
Press `q` to quit.

## Configuration
- **Day/Night threshold**: `day_night_detector.py`
- **Fog classification threshold**: `foggy_clear_detector.py`

## Applications
- CCTV environment perception
- Smart city monitoring
- Port and coastal surveillance
- Preprocessing for vision-based systems

## Author
- GitHub: https://github.com/DongHyun925
