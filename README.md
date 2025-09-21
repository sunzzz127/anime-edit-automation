# Anime Edit Automator

A beat-synced AMV generator that auto-cuts your source video to the music and sprinkles in common AMV effects (punch-in zooms, white flashes, micro shake, vignette, light color grade, and optional letterbox).

## Features
- **Beat/Onset detection** (librosa) -> cuts land on musical moments
- **Velocity ramps** (slow→fast / fast→slow) for energy
- **Punch-in zooms** on segments for emphasis
- **White flashes** every few cuts for impact
- **Micro shake** for handheld vibe
- **Vignette + color grade** to taste
- **Letterbox** cinematic bars (toggleable)
- Exports **1080p60** by default

## Install
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> You need `ffmpeg` in your PATH. Install via:
> - macOS: `brew install ffmpeg`
> - Windows: [gyan.dev FFmpeg builds] or `winget install ffmpeg`
> - Linux: `sudo apt-get install ffmpeg`

## Usage
```
python anime_edit_automator.py --video path/to/source.mp4 --audio path/to/music.mp3 --out edit.mp4
```

### Useful flags
- `--fps 60` change target fps
- `--width 1920 --height 1080` change output resolution
- `--minseg 0.25 --maxseg 1.25` segment length range (sec)
- `--no-shake` disable micro shake
- `--no-letterbox` remove black bars

### Tips
- Use a **clean high-bitrate** source video and a **WAV/320kbps MP3** for best beat detection.
- If cuts feel too dense/sparse, tweak `--minseg / --maxseg`.
- Want heavier zooms? Increase `max_zoom` in `punch_in()`.
- You can customize the grade in the code: `colorx(...), lum_contrast(...)`.

## License
MIT
