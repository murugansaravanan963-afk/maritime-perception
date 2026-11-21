"""Pre- and post-processing for the 3 scene models, shared by the Pi runner and the export check.

No torch here: only numpy + OpenCV, so the same file runs on the laptop and on the Pi.

  water    : 640x384 letterboxed RGB -> logits (3, 48, 80) at 1/8 size: obstacle, water, sky
  horizon  : 640x384 letterboxed RGB -> (2,) y of the water edge at x=0 and x=639, divided by 384
  seastate : 224x224 RGB water patch -> (8,) Beaufort 1-8 logits; calm=1-2, moderate=3-4, rough=5-8
All three take ImageNet-normalised input. On the Hailo the normalisation runs on chip, so the Pi
feeds plain 0-255 uint8 RGB; with ONNX Runtime the Pi normalises first (to_onnx_input).
"""
import numpy as np
import cv2

W, H = 640, 384                       # scene canvas used in training (scene_lars_v1)
PAD = 114
SEG_CLASSES = ['obstacle', 'water', 'sky']
SEA_CLASSES = ['calm', 'moderate', 'rough']
SEA_SIZE = 224                        # sea-state input
SEA_PATCH = 331                       # UNIRI patch size; the model was trained on its centre 224 crop
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(img_bgr):
    """Keep aspect, resize into 640x384, centre-pad with 114 (same as build_scene_data.py).
    Returns RGB uint8, scale r, pad (left, top)."""
    h, w = img_bgr.shape[:2]
    r = min(W / w, H / h)
    nw, nh = round(w * r), round(h * r)
    left, top = (W - nw) // 2, (H - nh) // 2
    out = np.full((H, W, 3), PAD, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), r, (left, top)


def to_onnx_input(rgb_uint8):
    """HxWx3 uint8 RGB -> (1,3,H,W) float32, ImageNet-normalised (what the ONNX models expect)."""
    x = (rgb_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def channels_first(t, channels):
    """Hailo gives NHWC, ONNX gives NCHW: return (channels, h, w) either way."""
    t = np.asarray(t, np.float32).squeeze()
    return t if t.shape[0] == channels else np.moveaxis(t, -1, 0)


def water_mask(seg_logits):
    """(3, 48, 80) logits -> (384, 640) uint8 class map (0 obstacle, 1 water, 2 sky).
    Bilinear upsampling of the logits, like the training model's last layer."""
    logits = channels_first(seg_logits, len(SEG_CLASSES))
    up = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in logits])
    return up.argmax(0).astype(np.uint8)


def horizon_line(out2, r, pad):
    """(2,) model output -> (y_left, y_right) on the 640x384 canvas and a function y(x) in original pixels."""
    y_left, y_right = (np.asarray(out2, np.float32).reshape(-1)[:2] * H).tolist()

    def y_at(x_orig):
        x_canvas = x_orig * r + pad[0]
        y_canvas = y_left + (y_right - y_left) * x_canvas / (W - 1)
        return (y_canvas - pad[1]) / r
    return (y_left, y_right), y_at


def sea_state(logits8):
    """(8,) Beaufort logits -> calm/moderate/rough probabilities and the expected Beaufort level."""
    z = np.asarray(logits8, np.float64).reshape(-1)[:8]
    p = np.exp(z - z.max())
    p /= p.sum()
    probs = np.array([p[0:2].sum(), p[2:4].sum(), p[4:8].sum()])
    return probs, float((p * np.arange(1, 9)).sum())


def sea_patch(img_bgr, mask, r, pad, y_at, min_water=0.8):
    """Pick a 331x331 patch of the ORIGINAL frame that is mostly water and below the horizon,
    and return its centre 224x224 crop as RGB uint8 (the same view the model was trained on).
    Returns (patch or None, (x, y) top-left in original pixels, water fraction)."""
    h, w = img_bgr.shape[:2]
    side = min(SEA_PATCH, h, w)
    if side < SEA_SIZE:
        return None, None, 0.0
    water = (mask == 1).astype(np.uint8)
    best = (None, None, 0.0)
    for fx in (0.5, 0.3, 0.7, 0.1, 0.9):                      # centre first
        x0 = int(round((w - side) * fx))
        top = max(0.0, y_at(x0 + side / 2)) + 0.05 * h        # start a little below the horizon
        for y0 in (top, top + side * 0.5, h - side):
            y0 = int(min(max(y0, 0), h - side))
            # the same window on the 640x384 mask
            mx0, my0 = int(x0 * r + pad[0]), int(y0 * r + pad[1])
            ms = max(1, int(side * r))
            frac = float(water[my0:my0 + ms, mx0:mx0 + ms].mean()) if ms > 0 else 0.0
            if frac > best[2]:
                c = (side - SEA_SIZE) // 2
                crop = img_bgr[y0 + c:y0 + c + SEA_SIZE, x0 + c:x0 + c + SEA_SIZE]
                best = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x0, y0), frac)
    return best if best[2] >= min_water else (None, best[1], best[2])


def gate_detections(boxes, y_at, margin_px=20):
    """Sea-aware gate: keep a box only if its bottom edge is not far above the horizon line.
    boxes (N,4) xyxy in original pixels -> boolean keep mask."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    return np.array([b[3] >= y_at(x) - margin_px for b, x in zip(boxes, cx)])


def overlay(img_bgr, mask, r, pad, y_at, sea=None, patch_xy=None, boxes=(), labels=(), dropped=()):
    """Picture for the demo: water tinted blue, sky tinted grey, obstacles red, horizon yellow."""
    h, w = img_bgr.shape[:2]
    nh, nw = round(h * r), round(w * r)
    m = cv2.resize(mask[pad[1]:pad[1] + nh, pad[0]:pad[0] + nw], (w, h), interpolation=cv2.INTER_NEAREST)
    colours = np.array([[40, 40, 220], [200, 120, 30], [170, 170, 170]], np.uint8)   # BGR
    view = cv2.addWeighted(img_bgr, 0.65, colours[m], 0.35, 0)
    cv2.line(view, (0, int(y_at(0))), (w - 1, int(y_at(w - 1))), (0, 255, 255), 3)
    for (x1, y1, x2, y2), text in zip(boxes, labels):
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(view, text, (int(x1), max(int(y1) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for x1, y1, x2, y2 in dropped:
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    if patch_xy is not None:
        x0, y0 = patch_xy
        cv2.rectangle(view, (x0, y0), (x0 + SEA_PATCH, y0 + SEA_PATCH), (255, 255, 255), 1)
    if sea is not None:
        probs, level = sea
        text = f'sea: {SEA_CLASSES[int(np.argmax(probs))]} ({probs.max():.2f})  Beaufort ~{level:.1f}'
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return view
