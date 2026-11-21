#!/usr/bin/env python3
"""
Golden reference model for the YOLOv8 DFL + NMS FPGA block (Hailo-8L + FPGA post-processing).

Works for any YOLOv8 detection model: input size and class count are parameters
(powerline model = 416x416, 20 classes; e.g. another project = 1280x1280, N classes).

Two models live here:
  float_ref()  - same math as powerline_inference.py (the deployed Pi CPU code)
  fixed_ref()  - bit-accurate integer model. The RTL must match THIS one exactly.

Usage
  # real vectors dumped on the Pi with dump_vectors_pi.py (size/classes read from meta.json)
  python3 golden_model.py --vectors fpga_vectors --out rtl_vectors

  # synthetic vectors (no Pi needed; ASSUMED quant params) for early RTL bring-up
  python3 golden_model.py --synthetic 10 --imgsz 416 --classes 20 --out rtl_vectors_synth
  python3 golden_model.py --synthetic 3 --imgsz 1280 --classes 20 --out rtl_vectors_synth_1280
  python3 golden_model.py --synthetic 3 --imgsz 1280 736 --classes 5 --out synth_1280x736   # W H

Outputs (in --out)
  rtl_params.vh             localparams for the RTL (sizes, bit widths, thresholds, quant)
  dfl_exp_lut.mem           exp() LUT for the DFL softmax ($readmemh)
  frame_XXXX/in_bbox.hex    stimulus: raw bbox logits   (quantized, from Hailo)
  frame_XXXX/in_score.hex   stimulus: raw class scores  (quantized, sigmoid already applied on Hailo)
  frame_XXXX/exp_dfl.hex    expected: DFL offsets per anchor (checkpoint 1)
  frame_XXXX/exp_decode.hex expected: boxes/class/score/pass per anchor (checkpoint 2)
  frame_XXXX/exp_nms.hex    expected: final detections (checkpoint 3)
  frame_XXXX/exp_nms.txt    same detections in readable form + float-reference comparison
  summary.txt               fixed-point vs float-reference agreement for all frames
"""
import argparse
import glob
import json
import math
import os
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------- fixed by YOLOv8
DFL_SIZE = 16
STRIDES = [8, 16, 32]

# ---------------------------------------------------------------- fixed-point choices
LUT_FRAC = 15      # exp LUT values are Q0.15, LUT[0] = 32768 (fits in 16 bits unsigned)
OFF_FRAC = 8       # DFL offset (grid units, 0..15) is Q4.8 -> 12 bits
COORD_FRAC = 4     # box coordinates in pixels, 1/16 px resolution
DEFAULT_CONF = 0.3   # powerline_inference.py defaults
DEFAULT_IOU = 0.45


class Model:
    """Model geometry. All derived sizes and RTL bit widths come from (width, height, classes)."""

    def __init__(self, width, height, classes, max_cand, max_det):
        for v in (width, height):
            if v % 32:
                raise SystemExit(f"Input size {v} is not a multiple of 32 (YOLOv8 max stride).")
        self.W, self.H, self.NC = width, height, classes
        self.max_cand, self.max_det = max_cand, max_det
        self.grids = [(width // s, height // s) for s in STRIDES]          # (gw, gh) per stride
        self.n_per_stride = [gw * gh for gw, gh in self.grids]
        self.N = sum(self.n_per_stride)
        rows = []
        for (gw, gh), s in zip(self.grids, STRIDES):
            for y in range(gh):
                for x in range(gw):
                    rows.append((x, y, s, int(math.log2(s))))
        self.anchors = np.array(rows, dtype=np.int64)                      # x, y, stride, log2(stride)
        # RTL bit widths
        hi = max(width, height) << COORD_FRAC
        self.coord_bits = hi.bit_length()                                   # unsigned, after clip
        self.coord_s_bits = ((max(width, height) + 15 * 32) << COORD_FRAC).bit_length() + 1  # signed, before clip
        self.area_bits = (hi * hi).bit_length()
        self.anchor_bits = max(1, (self.N - 1).bit_length())
        self.class_bits = max(1, (classes - 1).bit_length())
        self.hx_coord = max(4, -(-self.coord_bits // 4))                    # hex digits
        self.hx_anchor = max(4, -(-self.anchor_bits // 4))
        self.hx_class = max(2, -(-self.class_bits // 4))


# ================================================================ float reference
# Copied from powerline_inference.py. Changes: inputs are dequantized here, geometry is
# parameterized (x clipped to W, y to H), and NMS ties are broken deterministically (see _nms).
def _float_anchors(m):
    a = []
    for (gw, gh), stride in zip(m.grids, STRIDES):
        for y in range(gh):
            for x in range(gw):
                a.append([(x + 0.5) * stride, (y + 0.5) * stride, stride])
    return np.array(a, dtype=np.float32)


def _dfl_decode(dfl_output):
    n = dfl_output.shape[0]
    dfl = dfl_output.reshape(n, 4, DFL_SIZE)
    dfl_exp = np.exp(dfl - np.max(dfl, axis=2, keepdims=True))
    dfl_softmax = dfl_exp / np.sum(dfl_exp, axis=2, keepdims=True)
    bins = np.arange(DFL_SIZE, dtype=np.float32)
    return np.sum(dfl_softmax * bins[None, None, :], axis=2)


def _nms(boxes, scores, class_ids, iou_threshold):
    keep = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes, cls_scores = boxes[cls_mask], scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        # Original: cls_scores.argsort()[::-1]  (unstable -> ties between equal quantized scores
        # resolve arbitrarily). Here: score desc, then lower anchor index first, same as the RTL rule.
        order = np.lexsort((cls_indices, -cls_scores))
        cls_boxes, cls_scores, cls_indices = cls_boxes[order], cls_scores[order], cls_indices[order]
        while len(cls_boxes) > 0:
            keep.append(cls_indices[0])
            if len(cls_boxes) == 1:
                break
            xx1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
            yy1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
            xx2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
            yy2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area1 = (cls_boxes[0, 2] - cls_boxes[0, 0]) * (cls_boxes[0, 3] - cls_boxes[0, 1])
            area2 = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            remaining = np.where(iou < iou_threshold)[0]
            cls_boxes = cls_boxes[remaining + 1]
            cls_scores = cls_scores[remaining + 1]
            cls_indices = cls_indices[remaining + 1]
    return np.array(keep, dtype=int)


def float_ref(m, bbox_q, score_q, q, conf, iou):
    """Deployed CPU pipeline on dequantized Hailo outputs. Returns list of (x1,y1,x2,y2,score,cls) floats."""
    bbox = (np.float32(q["bbox_scale"]) * (bbox_q.astype(np.float32) - np.float32(q["bbox_zp"]))).reshape(-1, 64)
    score = (np.float32(q["score_scale"]) * (score_q.astype(np.float32) - np.float32(q["score_zp"]))).reshape(-1, m.NC)
    anchors = _float_anchors(m)
    off = _dfl_decode(bbox)
    cx, cy, st = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    boxes = np.stack([cx - off[:, 0] * st, cy - off[:, 1] * st, cx + off[:, 2] * st, cy + off[:, 3] * st], axis=1)
    cls = np.argmax(score, axis=1)
    sc = np.max(score, axis=1)
    msk = sc > conf
    boxes, sc, cls = boxes[msk], sc[msk], cls[msk]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, m.W)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, m.H)
    if len(boxes) == 0:
        return []
    k = _nms(boxes, sc, cls, iou)
    return [(*boxes[i].tolist(), float(sc[i]), int(cls[i])) for i in k]


# ================================================================ fixed-point model
def build_lut(bbox_scale, qbits):
    """LUT[d] = round(exp(-scale*d) * 2^LUT_FRAC), d = qmax - q. Truncated after the first zero entry."""
    lut = []
    for d in range(1 << qbits):
        v = int(round(math.exp(-bbox_scale * d) * (1 << LUT_FRAC)))
        lut.append(v)
        if v == 0:
            break
    return np.array(lut, dtype=np.int64)


def score_threshold_q(score_scale, score_zp, conf, qbits):
    """Smallest quantized score q with float32(scale*(q-zp)) > conf  ->  pass = (q >= Q_THR)."""
    s, z, c = np.float32(score_scale), np.float32(score_zp), np.float32(conf)
    for qv in range(1 << qbits):
        if s * (np.float32(qv) - z) > c:
            return qv
    return 1 << qbits  # nothing can pass


def iou_fraction(iou):
    f = Fraction(str(iou)).limit_denominator(256)
    return f.numerator, f.denominator


def fixed_ref(m, bbox_q, score_q, p):
    """Bit-accurate model. p = params dict from make_params()."""
    lut = p["lut"]
    depth = len(lut)
    b = bbox_q.reshape(m.N, 4, DFL_SIZE).astype(np.int64)
    s = score_q.reshape(m.N, m.NC).astype(np.int64)

    # ---- DFL: softmax expectation over 16 bins, per side (l, t, r, b)
    qmax = b.max(axis=2, keepdims=True)
    d = np.minimum(qmax - b, depth - 1)           # clamp; LUT[last] is 0 (or d never reaches it)
    e = lut[d]                                     # Q0.15
    S = e.sum(axis=2)                              # <= 16*32768  -> 20 bits
    N = (e * np.arange(DFL_SIZE)).sum(axis=2)      # <= 15*16*32768 -> 23 bits
    off = (N << OFF_FRAC) // S                     # Q4.8, floor division, 0..3840 -> 12 bits

    # ---- box decode (pixels, Q.COORD_FRAC)
    x, y, lg = m.anchors[:, 0], m.anchors[:, 1], m.anchors[:, 3]
    cx = ((2 * x + 1) << lg) << (COORD_FRAC - 1)  # (x+0.5)*stride
    cy = ((2 * y + 1) << lg) << (COORD_FRAC - 1)
    off_px = (off << lg[:, None]) >> (OFF_FRAC - COORD_FRAC)   # offset*stride, floor
    hx, hy = m.W << COORD_FRAC, m.H << COORD_FRAC
    x1 = np.clip(cx - off_px[:, 0], 0, hx)
    y1 = np.clip(cy - off_px[:, 1], 0, hy)
    x2 = np.clip(cx + off_px[:, 2], 0, hx)
    y2 = np.clip(cy + off_px[:, 3], 0, hy)

    # ---- class + threshold (on quantized score; sigmoid already done on Hailo)
    cls = s.argmax(axis=1)                         # first max wins on ties
    sq = s.max(axis=1)
    passed = sq >= p["q_thr"]

    # ---- candidate list (anchor order), capped
    cand = np.nonzero(passed)[0]
    overflow = len(cand) > m.max_cand
    cand = cand[:m.max_cand]

    # ---- class-aware greedy NMS. Order: class asc, score desc, anchor index asc.
    num, den = p["iou_num"], p["iou_den"]
    area = (x2 - x1) * (y2 - y1)
    dets = []
    for c in range(m.NC):
        idx = [i for i in cand if cls[i] == c]
        idx.sort(key=lambda i: (-sq[i], i))
        alive = [True] * len(idx)
        for a in range(len(idx)):
            if not alive[a]:
                continue
            i = idx[a]
            dets.append(i)
            for bb in range(a + 1, len(idx)):
                if not alive[bb]:
                    continue
                j = idx[bb]
                iw = min(x2[i], x2[j]) - max(x1[i], x1[j])
                ih = min(y2[i], y2[j]) - max(y1[i], y1[j])
                inter = max(iw, 0) * max(ih, 0)
                union = area[i] + area[j] - inter
                # suppress if IoU >= num/den, division-free: den*inter >= num*union
                if inter > 0 and den * inter >= num * union:
                    alive[bb] = False
    dets = dets[:m.max_det]
    return dict(off=off, x1=x1, y1=y1, x2=x2, y2=y2, cls=cls, sq=sq, passed=passed,
                n_cand=int(passed.sum()), overflow=bool(overflow), dets=dets)


# ================================================================ params + file writers
def make_params(q, conf, iou):
    qbits = q["qbits"]
    sbits = q.get("sbits", qbits)
    num, den = iou_fraction(iou)
    return dict(lut=build_lut(q["bbox_scale"], qbits),
                q_thr=score_threshold_q(q["score_scale"], q["score_zp"], conf, sbits),
                iou_num=num, iou_den=den, qbits=qbits, sbits=sbits, conf=conf, iou=iou, q=q)


def write_params_vh(path, m, p):
    q = p["q"]
    (g8w, g8h), (g16w, g16h), (g32w, g32h) = m.grids
    L = [
        "// Auto-generated by golden_model.py - do not edit by hand",
        f"// Model: {m.W}x{m.H} input, {m.NC} classes",
        f"// HEF quant: bbox scale={q['bbox_scale']!r} zp={q['bbox_zp']!r} | score scale={q['score_scale']!r} zp={q['score_zp']!r}",
        f"// conf={p['conf']}  iou={p['iou']}",
        "// --- geometry",
        f"localparam integer IMG_W         = {m.W};",
        f"localparam integer IMG_H         = {m.H};",
        f"localparam integer NUM_CLASSES   = {m.NC};",
        f"localparam integer DFL_BINS      = {DFL_SIZE};",
        f"localparam integer GRID8_W       = {g8w};   localparam integer GRID8_H  = {g8h};",
        f"localparam integer GRID16_W      = {g16w};   localparam integer GRID16_H = {g16h};",
        f"localparam integer GRID32_W      = {g32w};   localparam integer GRID32_H = {g32h};",
        f"localparam integer N_ANCHORS_S8  = {m.n_per_stride[0]};",
        f"localparam integer N_ANCHORS_S16 = {m.n_per_stride[1]};",
        f"localparam integer N_ANCHORS_S32 = {m.n_per_stride[2]};",
        f"localparam integer N_ANCHORS     = {m.N};",
        "// --- input format",
        f"localparam integer Q_BITS        = {p['qbits']};   // bbox value width from Hailo",
        f"localparam integer SCORE_Q_BITS  = {p['sbits']};   // score value width from Hailo",
        "// --- fixed point",
        f"localparam integer LUT_FRAC      = {LUT_FRAC};",
        f"localparam integer LUT_DEPTH     = {len(p['lut'])};",
        f"localparam integer OFF_FRAC      = {OFF_FRAC};",
        f"localparam integer COORD_FRAC    = {COORD_FRAC};",
        f"localparam integer COORD_BITS    = {m.coord_bits};   // unsigned, after clip",
        f"localparam integer COORD_S_BITS  = {m.coord_s_bits};   // signed, before clip",
        f"localparam integer AREA_BITS     = {m.area_bits};",
        f"localparam integer ANCHOR_BITS   = {m.anchor_bits};",
        f"localparam integer CLASS_BITS    = {m.class_bits};",
        "// --- thresholds / caps",
        f"localparam integer SCORE_Q_THR   = {p['q_thr']};   // pass if score_q >= this",
        f"localparam integer IOU_NUM       = {p['iou_num']};",
        f"localparam integer IOU_DEN       = {p['iou_den']};   // suppress if IOU_DEN*inter >= IOU_NUM*union (and inter>0)",
        f"localparam integer MAX_CAND      = {m.max_cand};",
        f"localparam integer MAX_DET       = {m.max_det};",
    ]
    open(path, "w").write("\n".join(L) + "\n")


def write_lut(path, lut):
    open(path, "w").write("".join(f"{int(v):04x}\n" for v in lut))


def write_frame(d, m, bbox_q, score_q, r, p, fref):
    os.makedirs(d, exist_ok=True)
    w = 2 if p["qbits"] == 8 else 4
    ws = 2 if p["sbits"] == 8 else 4
    hc, ha, hk = m.hx_coord, m.hx_anchor, m.hx_class
    with open(os.path.join(d, "in_bbox.hex"), "w") as f:     # anchor, side(l,t,r,b), bin
        f.write("".join(f"{int(v):0{w}x}\n" for v in bbox_q.reshape(-1)))
    with open(os.path.join(d, "in_score.hex"), "w") as f:    # anchor, class
        f.write("".join(f"{int(v):0{ws}x}\n" for v in score_q.reshape(-1)))
    with open(os.path.join(d, "exp_dfl.hex"), "w") as f:     # per anchor: off_l off_t off_r off_b (Q4.8)
        for o in r["off"]:
            f.write(" ".join(f"{int(v):03x}" for v in o) + "\n")
    with open(os.path.join(d, "exp_decode.hex"), "w") as f:  # per anchor: x1 y1 x2 y2 cls score pass
        for a in range(m.N):
            f.write(f"{r['x1'][a]:0{hc}x} {r['y1'][a]:0{hc}x} {r['x2'][a]:0{hc}x} {r['y2'][a]:0{hc}x} "
                    f"{r['cls'][a]:0{hk}x} {r['sq'][a]:0{ws}x} {int(r['passed'][a]):1x}\n")
    with open(os.path.join(d, "exp_nms.hex"), "w") as f:     # line0: n_det overflow; then x1 y1 x2 y2 cls score anchor
        f.write(f"{len(r['dets']):04x} {int(r['overflow']):1x}\n")
        for a in r["dets"]:
            f.write(f"{r['x1'][a]:0{hc}x} {r['y1'][a]:0{hc}x} {r['x2'][a]:0{hc}x} {r['y2'][a]:0{hc}x} "
                    f"{r['cls'][a]:0{hk}x} {r['sq'][a]:0{ws}x} {a:0{ha}x}\n")
    ok, msg = compare(r, fref)
    if r["overflow"] or len(r["dets"]) >= m.max_det:
        ok, msg = False, (f"CAP HIT (candidates={r['n_cand']}, MAX_CAND={m.max_cand}, MAX_DET={m.max_det}) - "
                          f"RTL output is still valid vs exp_*.hex, but differs from the uncapped float code. "
                          f"Raise --max-cand/--max-det if this happens on real data.")
    with open(os.path.join(d, "exp_nms.txt"), "w") as f:
        f.write(f"model {m.W}x{m.H}, {m.NC} classes, {m.N} anchors\n")
        f.write(f"candidates passing threshold: {r['n_cand']}  overflow: {r['overflow']}\n")
        f.write(f"fixed-point detections (pixels in {m.W}x{m.H} space):\n")
        for a in r["dets"]:
            sc = p["q"]["score_scale"] * (r["sq"][a] - p["q"]["score_zp"])
            f.write(f"  anchor {a:5d} cls {r['cls'][a]:2d} score {sc:.3f}  "
                    f"[{r['x1'][a]/16:.2f}, {r['y1'][a]/16:.2f}, {r['x2'][a]/16:.2f}, {r['y2'][a]/16:.2f}]\n")
        f.write("float reference (powerline_inference.py math):\n")
        for x1, y1, x2, y2, sc, c in fref:
            f.write(f"  cls {c:2d} score {sc:.3f}  [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]\n")
        f.write(f"agreement: {msg}\n")
    return ok, msg


def compare(r, fref, tol_px=1.0):
    """Match fixed-point detections to float-reference detections (same class, coords within tol)."""
    fx = [(r["x1"][a] / 16, r["y1"][a] / 16, r["x2"][a] / 16, r["y2"][a] / 16, int(r["cls"][a])) for a in r["dets"]]
    used = [False] * len(fref)
    worst = 0.0
    for b in fx:
        best, bi = None, -1
        for k, fr in enumerate(fref):
            if used[k] or fr[5] != b[4]:
                continue
            e = max(abs(b[0] - fr[0]), abs(b[1] - fr[1]), abs(b[2] - fr[2]), abs(b[3] - fr[3]))
            if best is None or e < best:
                best, bi = e, k
        if bi < 0 or best > tol_px:
            return False, f"MISMATCH: fixed={len(fx)} dets, float={len(fref)} dets (unmatched fixed box {b})"
        used[bi] = True
        worst = max(worst, best)
    if not all(used):
        return False, f"MISMATCH: fixed={len(fx)} dets, float={len(fref)} dets"
    return True, f"OK: {len(fx)} dets match, max coord error {worst:.3f} px"


# ================================================================ inputs
def load_real(vec_dir):
    meta = json.load(open(os.path.join(vec_dir, "meta.json")))
    q = dict(bbox_scale=meta["bbox"]["qp_scale"], bbox_zp=meta["bbox"]["qp_zp"],
             score_scale=meta["score"]["qp_scale"], score_zp=meta["score"]["qp_zp"],
             qbits=meta["bbox"]["bits"], sbits=meta["score"]["bits"])
    geom = (meta["input_w"], meta["input_h"], meta["num_classes"])
    frames = []
    for fpath in sorted(glob.glob(os.path.join(vec_dir, "frame_*.npz"))):
        z = np.load(fpath)
        frames.append((os.path.splitext(os.path.basename(fpath))[0], z["bbox_q"], z["score_q"]))
    return q, geom, frames


def make_synthetic(m, n, seed=0):
    """Plausible-looking quantized tensors with ASSUMED quant params. For RTL bring-up only."""
    rng = np.random.default_rng(seed)
    q = dict(bbox_scale=0.08, bbox_zp=96.0, score_scale=1.0 / 255, score_zp=0.0, qbits=8, sbits=8)
    acx = (m.anchors[:, 0] + 0.5) * m.anchors[:, 2]
    acy = (m.anchors[:, 1] + 0.5) * m.anchors[:, 2]
    bins = np.arange(DFL_SIZE)
    frames = []
    for fi in range(n):
        logits = rng.normal(-2.0, 0.7, (m.N, 4, DFL_SIZE))
        scores = np.abs(rng.normal(0, 0.03, (m.N, m.NC)))
        smax = 200.0 * min(m.W, m.H) / 416
        for _ in range(rng.integers(2, 8)):                # objects
            c = rng.integers(0, m.NC)
            w, h = rng.uniform(20, smax, 2)
            ox, oy = rng.uniform(w / 2, m.W - w / 2), rng.uniform(h / 2, m.H - h / 2)
            sel = np.nonzero((np.abs(acx - ox) <= w / 4) & (np.abs(acy - oy) <= h / 4))[0]
            for a in sel:
                ax, ay, st = acx[a], acy[a], m.anchors[a, 2]
                tgt = np.array([ax - (ox - w / 2), ay - (oy - h / 2), (ox + w / 2) - ax, (oy + h / 2) - ay]) / st
                if tgt.max() > 14.5 or tgt.min() < 0.5:
                    continue
                logits[a] = 6.0 - 1.5 * (bins[None, :] - tgt[:, None]) ** 2 + rng.normal(0, 0.3, (4, DFL_SIZE))
                scores[a, c] = rng.uniform(0.35, 0.95)
        bq = np.clip(np.round(logits / q["bbox_scale"] + q["bbox_zp"]), 0, 255).astype(np.uint8)
        sq = np.clip(np.round(np.clip(scores, 0, 1) / q["score_scale"] + q["score_zp"]), 0, 255).astype(np.uint8)
        frames.append((f"frame_{fi:04d}", bq.reshape(m.N, 64), sq))
    return q, frames


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vectors", help="folder written by dump_vectors_pi.py")
    g.add_argument("--synthetic", type=int, help="generate N synthetic frames (assumed quant params)")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[416], help="synthetic only: S or W H (multiples of 32)")
    ap.add_argument("--classes", type=int, default=20, help="synthetic only: number of classes")
    ap.add_argument("--out", default="rtl_vectors")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--max-cand", type=int, default=None, help="default: 1024 for <=640px, 4096 above")
    ap.add_argument("--max-det", type=int, default=100)
    a = ap.parse_args()

    if a.vectors:
        q, (W, H, NC), frames = load_real(a.vectors)
    else:
        W, H = (a.imgsz[0], a.imgsz[0]) if len(a.imgsz) == 1 else a.imgsz[:2]
        NC = a.classes
    max_cand = a.max_cand or (1024 if max(W, H) <= 640 else 4096)
    m = Model(W, H, NC, max_cand, a.max_det)
    if not a.vectors:
        q, frames = make_synthetic(m, a.synthetic)

    p = make_params(q, a.conf, a.iou)
    os.makedirs(a.out, exist_ok=True)
    write_params_vh(os.path.join(a.out, "rtl_params.vh"), m, p)
    write_lut(os.path.join(a.out, "dfl_exp_lut.mem"), p["lut"])

    summary = [f"source: {'synthetic (ASSUMED quant params)' if a.synthetic else a.vectors}",
               f"model: {W}x{H}, {NC} classes, {m.N} anchors, grids {m.grids}",
               f"quant: {q}", f"SCORE_Q_THR={p['q_thr']}  IOU={p['iou_num']}/{p['iou_den']}  LUT_DEPTH={len(p['lut'])}", ""]
    n_ok = 0
    for name, bq, sq in frames:
        r = fixed_ref(m, bq, sq, p)
        fref = float_ref(m, bq, sq, q, a.conf, a.iou)
        ok, msg = write_frame(os.path.join(a.out, name), m, bq, sq, r, p, fref)
        n_ok += ok
        summary.append(f"{name}: cand={r['n_cand']:5d} dets={len(r['dets']):3d}  {msg}")
    summary.append(f"\n{n_ok}/{len(frames)} frames: fixed-point matches float reference")
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
