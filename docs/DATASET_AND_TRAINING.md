# Dataset generation & model training

The runtime detector is **YOLO26-seg fine-tuned on an AirSim auto-labeled dataset** of the target
drone. No manual labeling is involved: AirSim's ground-truth instance **segmentation** buffer
provides perfect masks, which are converted to YOLO bounding boxes + polygons automatically.

Final weights live at:

```
runs/train/airsim_drone/weights/best.pt          # single class: 'drone'
```

---

## 1. How the AirSim dataset is generated

Script: `datasets/generate_airsim_dataset.py`

```powershell
python datasets\generate_airsim_dataset.py --num 700 --out datasets\airsim_drone
```

What it does:

1. Connects to a **freshly launched** AirSim (Ego at spawn `(0,0,0)` facing +X). It does **not**
   `reset()` or move the Ego — doing so corrupts the segmentation buffer (see `GOTCHAS.md`).
2. Sets all objects to segmentation id 0, then sets `Target.*` to a distinct id (`TARGET_ID = 25`).
3. **Teleports only the Target** to many random viewpoints inside the Ego camera's FOV cone
   (distance 4–25 m, within ~±22° horizontally and within the vertical FOV), at random yaw. A
   fraction (`--neg-frac`, default 0.12) are **negatives** — Target moved far out of view.
4. For each viewpoint, grabs **Scene + Segmentation in one `simGetImages` call** so both come from
   the same rendered frame.
5. Extracts the target mask as the **non-background region**: the most common color in the seg image
   is treated as background (sky/ground), everything sufficiently different is the target. This is
   robust to AirSim's seg **colormap changing between sessions** — only the Target has a distinct id,
   so we detect "not background" rather than a fixed color (see `GOTCHAS.md`).
6. Finds the largest contour, simplifies it with `approxPolyDP`, and writes a **normalized polygon**
   label line `0 x1 y1 x2 y2 …` (YOLO-seg format). Targets smaller than `--min-area` (default 20 px)
   are skipped; negatives get an empty label file.
7. Splits into train/val (`--val-frac`, default 0.15) and writes `data.yaml`.

Output layout:

```
datasets/airsim_drone/
  images/train/airsim_00000.jpg ...
  images/val/...
  labels/train/airsim_00000.txt      # "0 x1 y1 x2 y2 ..." normalized polygon, or empty (negative)
  labels/val/...
  data.yaml                          # nc: 1, names: ['drone']
```

Key flags: `--num` (target image count), `--out`, `--val-frac`, `--neg-frac`, `--min-area`, `--seed`.

> The same idea generalizes: AirSim's segmentation GT → bbox + polygon means every captured frame is
> labeled for free, with perfect masks, for both the detection and segmentation heads.

---

## 2. (Optional) external UAV datasets

For broader, real-world variety the project also supports public air-to-air datasets, converted to
YOLO and optionally combined with the AirSim set.

* `datasets/download.py` — prints acquisition steps (most are gated behind registration / Drive) and
  expects raw data under `datasets/raw/<name>/`:
  * **Det-Fly** (VOC boxes, ~13k air-to-air images),
  * **ARD100** (MOT, extremely small targets),
  * **Drone-vs-Bird** (MOT, drones vs birds),
  * **UAVDB** (COCO with point-guided masks → seg head).
* `datasets/to_yolo.py` — converts VOC / COCO / MOT → YOLO; all UAVs map to one class `uav`.

```powershell
python datasets\to_yolo.py --name det-fly --format voc
python datasets\to_yolo.py --name uavdb   --format coco --seg     # keep polygons for seg
python datasets\to_yolo.py --combine det-fly ard100 --out datasets\uav_det.yaml
```

* `datasets/sam_automask.py` — SAM 2 auto-masking to add segmentation labels to box-only datasets
  (heavy; install on demand).

---

## 3. Training YOLO26-seg

Script: `perception/train_yolo26.py`

```powershell
# Instance segmentation on the AirSim drone dataset (what produced best.pt):
python perception\train_yolo26.py ^
  --data datasets\airsim_drone\data.yaml ^
  --task segment ^
  --model yolo26s-seg.pt ^
  --name airsim_drone ^
  --epochs 100 --imgsz 1280

# Detection-only variant:
python perception\train_yolo26.py --data datasets\uav_det.yaml --task detect --epochs 100
```

Defaults are tuned for **small air-to-air targets on a 12 GB GPU** (RTX 5070 Ti):

| Setting | Default | Why |
|---------|---------|-----|
| `--imgsz` | 1280 | Large input helps tiny targets. |
| `--batch` | `-1` | Auto-fit to ~60% VRAM. |
| `--epochs` | 100 | — |
| `--patience` | 30 | Early-stop. |
| mosaic | 1.0, `close_mosaic=10` | Augment, then disable for the last 10 epochs to sharpen small-box fit. |
| `scale` | 0.5 | Scale augmentation. |
| `cos_lr` | true | Cosine LR schedule. |
| optimizer | AdamW (auto) | — |

Outputs land under `runs/train/<name>/`; the best checkpoint is
`runs/train/<name>/weights/best.pt`. The script also prints a quick val summary (mAP50, mAP50-95).

> Default pretrained model name is `yolo26s-seg.pt` (segment) / `yolo26s.pt` (detect). The detect
> CLI falls back to `yolo11n*` if YOLO26 weights are unavailable.

---

## 4. Evaluation

### Detection / segmentation accuracy — `perception/eval/eval_detection.py`

```powershell
python perception\eval\eval_detection.py ^
  --data datasets\airsim_drone\data.yaml ^
  --model runs\train\airsim_drone\weights\best.pt ^
  --task segment
```

Reports box mAP50 / mAP50-95 (and mask mAP when `--task segment`). Default `--imgsz 1280`,
`--split val`.

### Tracking accuracy — `perception/eval/eval_tracking.py`

```powershell
python perception\eval\eval_tracking.py --gt gt\seq01\gt.txt --pred runs\track\exp\tracks.jsonl
```

Computes MOTA / IDF1 / ID-switches with `motmetrics`. Ground truth is MOT-Challenge format; the
prediction can be the `tracks.jsonl` emitted by `detect_track.py` or a MOT-format txt.

---

## 5. Using the model

The fine-tuned model is wired into every runtime entrypoint by default:

```python
MODEL = "runs/train/airsim_drone/weights/best.pt"     # webui/app.py, sim/airsim/*.py
model.track(frame, tracker="perception/trackers/bytetrack_uav.yaml",
            persist=True, imgsz=960, conf=0.3)
```

Note runtime inference uses `imgsz=960` for speed in the live loop, while training uses `imgsz=1280`
for accuracy on tiny targets.

To retrain with a bigger / more varied dataset, regenerate or combine datasets (sections 1–2),
re-run training (section 3) to a new `--name`, then point `MODEL` at the new `best.pt`.
