#!/usr/bin/env python3
"""body sway — track moving points in tree/leaf footage and draw their paths.

    python trace_wind.py videos/*.mp4                # default: sparse LK tracking
    python trace_wind.py clip.mov --method dis       # denser, better on soft/blurry foliage
    python trace_wind.py clip.mov --stabilize        # handheld footage
    python trace_wind.py clip.mov --preview          # also render an animated trails video

Each video produces out/<name>/ with:
    still.png     accumulated wind lines on a dark background
    tracks.json   every trajectory + per-track stats (for the viewer)
    footage.mp4   the processed (resized / stabilized) frames, for the viewer underlay
    preview.mp4   optional fading-trails animation
and out/index.json lists all results for index.html.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

BG = (16, 13, 11) 

# shared with index.html; edit palettes.json to change colors in both
PALETTES = json.loads((Path(__file__).parent / "palettes.json").read_text())



class FrameWriter:
    """H.264 via ffmpeg when available (plays in browsers), else OpenCV mp4v."""

    def __init__(self, path, w, h, fps, crf=24):
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg:
            self.proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                 "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                 "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.ffmpeg:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.vw.write(frame)

    def close(self):
        if self.ffmpeg:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.vw.release()


def read_frames(path, max_width, start, duration):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    limit = int(duration * fps) if duration else None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w) if max_width else 1.0
    # even dimensions keep yuv420p encoders happy
    size = (int(w*scale)//2*2, int(h*scale)//2*2)

    def gen():
        n = 0
        while limit is None or n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield frame
            n += 1
        cap.release()

    return gen(), fps, size



class Stabilizer:
    """Warp every frame onto frame 0 using a RANSAC similarity transform.

    Wind motion is local and oscillating, camera shake is global, so RANSAC over
    many features mostly locks onto the camera. Very gusty shots can fool it.
    """

    def __init__(self):
        self.prev = None
        self.to_ref = np.eye(3)

    def __call__(self, frame, gray):
        h, w = gray.shape
        if self.prev is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev, 600, 0.01, 12)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, winSize=(31, 31), maxLevel=4)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.0)
                    if m is not None:
                        self.to_ref = self.to_ref @ np.linalg.inv(np.vstack([m, [0, 0, 1]]))
        self.prev = gray
        m = self.to_ref[:2]
        frame = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT)
        gray = cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (w, h))
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
        return frame, gray, valid


LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def step_lk(prev, cur, p0):
    """Pyramidal Lucas-Kanade with a forward-backward check.

    Foliage is full of repeated texture; a point that doesn't come back to where
    it started when tracked in reverse has latched onto the wrong leaf.
    """
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), None, **LK_PARAMS)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **LK_PARAMS)
    p1 = p1.reshape(-1, 2)
    fb = np.linalg.norm(p0 - p0r.reshape(-1, 2), axis=1)
    return p1, fb, (st.ravel() == 1) & (st2.ravel() == 1)


def sample_flow(flow, pts):
    mx = pts[:, 0].reshape(1, -1).astype(np.float32)
    my = pts[:, 1].reshape(1, -1).astype(np.float32)
    return cv2.remap(flow, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)


def make_step_dis():
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def step(prev, cur, p0):
        fwd = dis.calc(prev, cur, None)
        bwd = dis.calc(cur, prev, None)
        p1 = p0 + sample_flow(fwd, p0)
        fb = np.linalg.norm(p1 + sample_flow(bwd, p1) - p0, axis=1)
        return p1, fb, np.ones(len(p0), bool)

    return step


#  tracking

def seed_points(gray, existing, budget, min_dist, quality, mask):
    m = None
    if mask is None:
        m = np.full(gray.shape, 255, np.uint8)
    else:
        m = mask.copy()
    for x, y in existing.astype(int):
        cv2.circle(m, (int(x), int(y)), min_dist, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, budget, quality, min_dist, mask=m, blockSize=7)
    return np.empty((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)


def track(frames, fps, size, args, footage_path):
    w, h = size
    step = step_lk if args.method == "lk" else make_step_dis()
    quality = 0.01 if args.method == "lk" else 0.003
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    stab = Stabilizer() if args.stabilize else None
    user_mask = None
    if args.mask:
        user_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        user_mask = cv2.resize(user_mask, size, interpolation=cv2.INTER_NEAREST)
        user_mask = np.where(user_mask > 127, 255, 0).astype(np.uint8)
    max_jump = args.max_jump * w / 1000  # relative to a 1000 px wide frame

    writer = FrameWriter(footage_path, w, h, fps, crf=26)
    tracks = []  # [start_frame, [(x, y), ...]]
    ids = np.empty(0, int)
    pos = np.empty((0, 2), np.float32)
    prev = None
    n = 0
    for n, frame in enumerate(frames):
        gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        valid = None
        if stab:
            frame, gray, valid = stab(frame, gray)
        writer.write(frame)
        mask = user_mask
        if valid is not None:
            mask = valid if mask is None else cv2.bitwise_and(mask, valid)

        if prev is not None and len(ids):
            p1, fb, ok = step(prev, gray, pos)
            ok &= fb < args.fb_thresh
            ok &= np.linalg.norm(p1 - pos, axis=1) < max_jump
            ok &= (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
            if mask is not None:
                xi = np.clip(p1[:, 0].astype(int), 0, w - 1)
                yi = np.clip(p1[:, 1].astype(int), 0, h - 1)
                ok &= mask[yi, xi] > 0
            for i, p in zip(ids[ok], p1[ok]):
                tracks[i][1].append((float(p[0]), float(p[1])))
            ids, pos = ids[ok], p1[ok].astype(np.float32)

        if n % args.reseed == 0 and len(ids) < args.max_points:
            new = seed_points(gray, pos, args.max_points - len(ids), args.min_dist, quality, mask)
            if len(new):
                new_ids = np.arange(len(tracks), len(tracks) + len(new))
                tracks.extend([n, [(float(x), float(y))]] for x, y in new)
                ids = np.concatenate([ids, new_ids])
                pos = np.vstack([pos, new]).astype(np.float32)

        prev = gray
        if n % 30 == 0:
            print(f"  frame {n:5d}  active {len(ids):5d}  total {len(tracks):6d}", end="\r")
    writer.close()
    print()
    return tracks, n + 1


#   analysis

def smooth(P, k):
    if k <= 1 or len(P) < k:
        return P
    pad = k // 2
    Pp = np.pad(P, ((pad, k - 1 - pad), (0, 0)), mode="edge")
    kern = np.ones(k) / k
    return np.stack([np.convolve(Pp[:, d], kern, mode="valid") for d in range(2)], 1)


def dominant_freq(P, fps):
    """Strongest oscillation frequency (Hz) of a trajectory, after removing drift."""
    n = len(P)
    t = np.arange(n)
    win = np.hanning(n)
    power = 0
    for d in range(2):
        x = P[:, d] - np.polyval(np.polyfit(t, P[:, d], 1), t)
        power = power + np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / fps)
    return float(freqs[1 + np.argmax(power[1:])])


def analyze(raw_tracks, fps, w, args):
    min_amp = args.min_amp * w / 1000
    out = []
    for start, pts in raw_tracks:
        if len(pts) < args.min_len:
            continue
        P = smooth(np.asarray(pts), args.smooth)
        c = P - P.mean(0)
        amp = float(np.sqrt((c ** 2).sum(1).mean()))  # rms distance from rest position
        if amp < min_amp:
            continue  # trunk, ground, sky edges: not wind
        seg = np.diff(P, axis=0)
        # principal sway axis, 0..180 degrees
        evals, evecs = np.linalg.eigh(np.cov(c.T))
        axis = float(np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1])) % 180)
        out.append({
            "s": int(start),
            "p": np.round(P, 1).ravel().tolist(),
            "f": round(dominant_freq(P, fps), 3) if len(P) >= 24 else None,
            "a": round(amp, 2),
            "v": round(float(np.linalg.norm(seg, axis=1).mean() * fps), 1),
            "d": round(axis, 1),
        })
    return out


#   rendering

def hex_to_bgr(h):
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32) / 255


def palette_lookup(name, t):
    stops = np.stack([hex_to_bgr(c) for c in PALETTES[name]])
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def track_colors(tracks, mode, palette):
    n = len(tracks)
    if mode == "direction":
        hue = np.array([t["d"] for t in tracks]) % 180  # OpenCV hue is 0..180
        hsv = np.stack([hue, np.full(n, 150), np.full(n, 255)], 1).astype(np.uint8)[None]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32) / 255
    key = {"speed": "v", "frequency": "f", "amplitude": "a"}.get(mode)
    if key is None:
        return palette_lookup(palette, np.zeros(n) + 0.85)
    vals = np.array([t[key] if t[key] is not None else np.nan for t in tracks], float)
    if key == "f":
        vals = np.log(np.maximum(vals, 0.05))
    lo, hi = np.nanpercentile(vals, [5, 95]) if np.isfinite(vals).any() else (0, 1)
    norm = np.nan_to_num((vals - lo) / max(hi - lo, 1e-6), nan=0.5)
    return palette_lookup(palette, norm)


def tonemap(acc, exposure, bloom):
    if bloom > 0:
        acc = acc + bloom * cv2.GaussianBlur(acc, (0, 0), sigmaX=acc.shape[1] / 300)
    lit = 1 - np.exp(-acc * exposure)
    bg = np.array(BG, np.float32) / 255
    return (np.clip(bg + lit * (1 - bg), 0, 1) * 255).astype(np.uint8)


def render_still(tracks, size, args):
    s = args.scale
    w, h = size
    acc = np.zeros((h * s, w * s, 3), np.float32)
    colors = track_colors(tracks, args.color, args.palette)
    thickness = max(1, round(s * args.width))
    # draw in batches so overlapping strokes add up (dense gusts glow brighter)
    for b in range(0, len(tracks), 48):
        layer = np.zeros_like(acc, dtype=np.uint8)
        for t, col in zip(tracks[b:b + 48], colors[b:b + 48]):
            pts = (np.asarray(t["p"]).reshape(-1, 2) * s * 16).astype(np.int32)  # 4-bit subpixel
            cv2.polylines(layer, [pts], False, (col * 255).tolist(), thickness, cv2.LINE_AA, shift=4)
        acc += layer.astype(np.float32) / 255 * args.alpha
    return tonemap(acc, args.exposure, args.bloom)


def render_preview(tracks, size, n_frames, fps, path, args):
    w, h = size
    colors = track_colors(tracks, args.color, args.palette)
    starts = np.array([t["s"] for t in tracks])
    arrays = [np.asarray(t["p"]).reshape(-1, 2) for t in tracks]
    decay = 0.01 ** (1 / max(args.trail, 1))
    acc = np.zeros((h, w, 3), np.float32)
    writer = FrameWriter(path, w, h, fps, crf=20)
    for k in range(n_frames):
        acc *= decay
        layer = np.zeros((h, w, 3), np.uint8)
        for i in np.nonzero((starts < k) & (starts + np.array([len(a) for a in arrays]) > k))[0]:
            j = k - starts[i]
            seg = (arrays[i][j - 1:j + 1] * 16).astype(np.int32)
            cv2.polylines(layer, [seg], False, (colors[i] * 255).tolist(), 1, cv2.LINE_AA, shift=4)
        acc += layer.astype(np.float32) / 255 * 0.6
        writer.write(tonemap(acc, 1.6, 0))
    writer.close()


#   main

def write_manifest(out_root):
    entries = []
    for f in sorted(out_root.glob("*/tracks.json")):
        with open(f) as fh:
            meta = json.load(fh)["meta"]
        entries.append({"name": meta["name"], "tracks": f"{f.parent.name}/tracks.json",
                        "still": f"{f.parent.name}/still.png", "frames": meta["frames"],
                        "fps": meta["fps"], "count": meta["count"]})
    (out_root / "index.json").write_text(json.dumps(entries, indent=1))


def process(video, args):
    video = Path(video)
    out = Path(args.out) / video.stem
    print(f"{video.name}")
    t0 = time.time()
    frames, fps, size = read_frames(video, args.max_width, args.start, args.duration)
    out.mkdir(parents=True, exist_ok=True)
    raw, n_frames = track(frames, fps, size, args, out / "footage.mp4")
    tracks = analyze(raw, fps, size[0], args)
    print(f"  {len(raw)} raw tracks -> {len(tracks)} kept   ({time.time() - t0:.1f}s)")
    if not tracks:
        print("  nothing moved enough; try lowering --min-amp or --min-len")
        return

    meta = {"name": video.stem, "source": video.name, "width": size[0], "height": size[1],
            "fps": round(fps, 3), "frames": n_frames, "count": len(tracks), "footage": "footage.mp4",
            "params": {k: v for k, v in vars(args).items() if k not in ("videos", "out")}}
    with open(out / "tracks.json", "w") as fh:
        json.dump({"meta": meta, "tracks": tracks}, fh, separators=(",", ":"))
    cv2.imwrite(str(out / "still.png"), render_still(tracks, size, args))
    if args.preview:
        render_preview(tracks, size, n_frames, fps, out / "preview.mp4", args)
    print(f"  -> {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="out")
    g = ap.add_argument_group("tracking")
    g.add_argument("--method", choices=["lk", "dis"], default="lk",
                   help="lk: sparse corners, crisp on sharp edges; dis: dense flow, more coverage on soft texture")
    g.add_argument("--max-width", type=int, default=1280, help="downscale wider videos (speed)")
    g.add_argument("--start", type=float, default=0, help="seconds")
    g.add_argument("--duration", type=float, default=0, help="seconds, 0 = whole video")
    g.add_argument("--stabilize", action="store_true", help="remove camera shake (handheld footage)")
    g.add_argument("--mask", help="b/w image, white = where to track (e.g. paint out sky or ground)")
    g.add_argument("--max-points", type=int, default=2500, help="max simultaneously tracked points")
    g.add_argument("--min-dist", type=int, default=6, help="px between seeded points")
    g.add_argument("--reseed", type=int, default=5, help="add new points every N frames")
    g.add_argument("--fb-thresh", type=float, default=0.6, help="forward-backward error limit, px")
    g.add_argument("--max-jump", type=float, default=25, help="max per-frame move, px per 1000 px width")
    g = ap.add_argument_group("filtering")
    g.add_argument("--min-len", type=int, default=12, help="drop tracks shorter than this many frames")
    g.add_argument("--min-amp", type=float, default=0.6, help="drop near-static tracks, px per 1000 px width")
    g.add_argument("--smooth", type=int, default=3, help="moving-average window, frames")
    g = ap.add_argument_group("rendering")
    g.add_argument("--color", choices=["mono", "direction", "speed", "frequency", "amplitude"], default="mono")
    g.add_argument("--palette", choices=list(PALETTES), help="default: mono for --color mono, else ember")
    g.add_argument("--scale", type=int, default=2, help="still image resolution multiplier")
    g.add_argument("--width", type=float, default=0.6, help="line width, px at 1x")
    g.add_argument("--alpha", type=float, default=0.35, help="per-line opacity before tonemapping")
    g.add_argument("--exposure", type=float, default=1.4)
    g.add_argument("--bloom", type=float, default=0.6)
    g.add_argument("--preview", action="store_true", help="also render preview.mp4 with fading trails")
    g.add_argument("--trail", type=int, default=20, help="preview trail length, frames")
    args = ap.parse_args()
    args.palette = args.palette or ("mono" if args.color == "mono" else "ember")

    for v in args.videos:
        try:
            process(v, args)
        except RuntimeError as e:
            print(f"  skipped: {e}")
            continue
        write_manifest(Path(args.out))  # after each video, so the viewer sees results as they finish
    write_manifest(Path(args.out))


if __name__ == "__main__":
    main()
