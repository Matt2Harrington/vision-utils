```
 __   _(_)___(_) ___  _ __    _   _| |_(_) |___
 \ \ / / / __| |/ _ \| '_ \  | | | | __| | / __|
  \ V /| \__ \ | (_) | | | | | |_| | |_| | \__ \
   \_/ |_|___/_|\___/|_| |_|  \__,_|\__|_|_|___/
```
# vision-utils

This repo contains various projects related to the Vision Pro & visionOS.

## Content

- [Convert a 2D Photo to Spatial Photo](./spatialconverter/)
- [Convert a 2D Video to Spatial Video](./spatialconverter/)
- [CLI to generate a stereoscopic image](./picCombiner)
- [visionOS Icons](./icons)

## 2D to Spatial Content
Convert any photos to spatial photos and videos viewable in the Apple Vision Pro! There is a mini swift cli executable that works on M1 apple computers to attach png files together.

See [Photo Blog Post](https://blog.studiolanes.com/posts/2d-to-spatial-photos) and [Video Blog Post](https://blog.studiolanes.com/posts/converting-spatial-videos) for more info.

### Dependencies
We borrow the executable and iPhone args from [Mike Swanson](https://blog.mikeswanson.com/spatial) for converting over under videos to spatial videos.

Requires [poetry](https://github.com/python-poetry/poetry) on PATH and
**Python 3.13** (3.10–3.13 work; avoid Homebrew's `python@3.14` — its bundled
`pyexpat` is broken on macOS and crashes poetry with `Symbol not found:
_XML_SetAllocTrackerActivationThreshold`).

```bash
brew install python@3.13   # if not already installed

cd spatialconverter
poetry env use python3.13
poetry install

# transformers can't be resolved by poetry from source, so install it into the
# project venv directly. Required the first time only.
poetry run pip install -q "git+https://github.com/huggingface/transformers.git"
```

### Subsequent runs

```bash
cd spatialconverter/spatialconverter
poetry run python main.py --photo /path/to/photo.png
# poetry run python main.py --video /path/to/video.mp4
```

The video pipeline accepts a number of flags to trade speed for quality and to
control the spatial-output metadata that Vision Pro reads. Run
`poetry run python main.py --help` for the full list, or here are the most
useful ones:

```bash
# Faster preview (smaller depth model, half the source fps)
poetry run python main.py --video clip.mp4 --model-size small --target-fps 24

# Wider lens — fixes "too zoomed in" playback on Vision Pro
poetry run python main.py --video clip.mp4 --hfov 90

# Stronger 3D effect
poetry run python main.py --video clip.mp4 --shift-left 15 --shift-right 70
```

Available flags: `--model-size {small,base,large}` (default `large`),
`--target-fps F` (default = source fps), `--shift-left N` / `--shift-right N`
(default 10 / 50), `--hfov F` (default 63.4°), `--cdist F` (default 19.24),
`--hadjust F` (default 0.02), `--projection {rect,fisheye,half_equirect}`
(default `rect`), `--spatial-extra "<...>"` (free-form flags appended to the
`./spatial make` call).