# Body Sway

Track branches and leaves in tree footage and draw how they move, as light lines on a dark background. Then view the results interactively.

```
pip install -r requirements.txt          # opencv-python, numpy (+ ffmpeg on PATH for browser-playable video)

python trace_wind.py videos/*.mp4        # -> out/<name>/still.png, tracks.json, footage.mp4
python -m http.server                    # then open http://localhost:8000/
```

## Pipeline

1. **Read and downscale** frames to `--max-width` (1280 by default).
2. **Optional stabilization** (`--stabilize`). This estimates a RANSAC similarity transform between frames and warps every frame back onto frame 0. Use it for handheld footage. Otherwise camera shake gets drawn as "wind".
3. **Contrast normalization** with CLAHE, so features in shaded parts of the canopy can still be tracked.
4. **Seeding.** Shi-Tomasi corners (`goodFeaturesToTrack`) are placed at least `--min-dist` apart. New points are added every `--reseed` frames to replace tracks that died.
5. **Tracking**, with a choice of two methods:
   - `--method lk` (default): pyramidal Lucas-Kanade. Precise on sharp edges such as branch silhouettes against the sky.
   - `--method dis`: dense DIS optical flow that is sampled at each point. Covers more of soft, blurry or low-contrast foliage.
6. **Rejecting bad tracks.** This is the most important step for foliage. Each point is tracked forward and then backward, and is dropped if it doesn't return to within `--fb-thresh` px. Points are also dropped if they jump more than `--max-jump` in one frame or leave `--mask`.
7. **Analysis.** Tracks are lightly smoothed. Tracks that are too short (`--min-len`) or almost static (`--min-amp`) are dropped; this removes the trunk, ground and sky. Each remaining track gets:
   - `f`: the dominant oscillation frequency (FFT of the drift-removed trajectory)
   - `a`: RMS amplitude
   - `v`: mean speed
   - `d`: principal sway axis
8. **Rendering.** Lines are drawn additively in batches, tonemapped and given a light bloom. Dense, gusty regions glow brighter. `--preview` also renders an animated video with fading trails.

Useful variations:

```
python trace_wind.py clip.mov --color direction                  # hue = sway axis
python trace_wind.py clip.mov --color frequency --palette ember  # slow branches vs fast leaves
python trace_wind.py clip.mov --mask sky_mask.png                # white = track here
python trace_wind.py clip.mov --start 12 --duration 8 --preview
```

## Viewer (`index.html`)

- **Trace** mode builds up the whole drawing over time. **Trails** mode shows comet trails that fade, like watching the wind live.
- **Color by**: direction of motion, speed, sway frequency, amplitude, or time of appearance.
- **Filter motion**:
  - **Sway** keeps paths below about 1 Hz, which tend to be branches.
  - **Flutter** keeps paths above about 2 Hz, which tend to be leaves.
  - Sliders for frequency, amplitude and density.
- **Hover** a line to highlight that one path and see its stats.
- **Footage** opacity overlays the processed source video, kept in sync with the drawing.
- **Save PNG** exports the current frame. Space toggles play.
- It loads `out/index.json` automatically. You can also drop a `tracks.json` onto the page, but then the footage underlay isn't available.

## Notes on difficult foliage footage

Tree footage is close to a worst case for classical tracking. Its texture repeats, leaves flip and change appearance, sunlight flickers, and the canopy occludes itself constantly. Some suggestions:

- **Filming matters more than the algorithm.**
  - Use a tripod.
  - A high shutter speed avoids motion blur on fluttering leaves.
  - 60 fps or higher keeps per-frame motion small. Lucas-Kanade assumes motion of only a few pixels per frame.
  - Overcast light avoids flickering dappled shadows.
  - Silhouettes against a bright sky are the easiest case of all: a backlit branch is a high-contrast edge that tracks beautifully.
- **Short tracks are normal.** In dense foliage most tracks live under a second. They still draw well. Increase `--min-len` if the drawing looks noisy, or lower it if it looks sparse.
- **Mask out the sky** if clouds are moving, and mask the ground if there is grass or shadow motion.
- **Telling branches from leaves by motion works better than by appearance.** Branches sway slowly with large amplitude; leaves flutter quickly with small amplitude. The frequency filter in the viewer uses this difference, and it needs no segmentation model. It is a heuristic, not a classifier.

## Research directions, if the classical pipeline isn't enough

- **Learned point tracking (TAP models).** These follow points through occlusion and appearance changes. That is exactly where Lucas-Kanade fails on leaves.
  - CoTracker / CoTracker3 (Meta)
  - TAPIR / BootsTAP (Google DeepMind)
  - The tracks they produce could be written into the same `tracks.json` format and reuse this renderer and viewer unchanged.
- **Learned dense optical flow:** RAFT and its successors. These give better flow on fine texture than DIS, but run much slower without a GPU.
- **Segmentation:** SAM 2 (Meta) segments and tracks objects through a video from a click or a box. Prompting it on individual branches or leaf clusters would give actual "this branch" or "this leaf" identity. Points inside each mask could then be tracked and colored by object.
- **Motion analysis of trees specifically:**
  - *Eulerian Video Magnification* and *Phase-Based Video Motion Processing* (MIT CSAIL) amplify tiny sway motion, so they are useful as preprocessing for gentle breezes.
  - Abe Davis's *Visual Vibrometry* and *Image-Space Modal Bases* work recovers vibration modes of objects, including plants, from video. That is the frequency idea here, done rigorously.
  - *Generative Image Dynamics* (Li et al., CVPR 2024) models oscillating natural motion, such as trees and flowers in wind, as per-pixel motion spectra. It is close in spirit to "draw the wind" and could inspire spectral visualizations.
