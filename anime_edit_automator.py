#!/usr/bin/env python3
"""
Anime Edit Automator
--------------------

Beat-synced edit generator for anime clips.

What it does
- Detects beats/onsets from the music track (librosa)
- Slices the video at musically meaningful points
- Applies common AMV-style effects:
  * velocity ramps (slow/fast sections)
  * punch-in zooms on beats
  * film flash/white flashes
  * screen shake (micro jitter)
  * vignette & slight color grade
  * optional letterbox bars
- Exports to 1080p60 by default (configurable)

Inputs
- One source video (.mp4, .mkv, .mov, etc.)
- One music track (.mp3, .wav, .m4a, etc.)

Author: Vub
License: MIT
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np

# Audio + CV/video stack
import librosa
from moviepy.editor import (AudioFileClip, CompositeAudioClip, CompositeVideoClip,
                            VideoFileClip, vfx, ColorClip)
from moviepy.video.fx.all import lum_contrast, colorx, crop

# ---------------
# Helper effects
# ---------------

def flash_white(duration=0.06, size=(1920,1080)):
    """Quick white flash overlay for transitions/impacts."""
    return ColorClip(size, color=(255,255,255)).set_duration(duration).set_opacity(0.8)

def vignette_mask(w=1920, h=1080, strength=0.8, falloff=0.6):
    """
    Create a simple radial vignette (0..1 alpha) as an array.
    """
    y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
    r = np.sqrt(x*x + y*y)
    vign = 1 - np.clip((r - falloff) / (1 - falloff), 0, 1) * strength
    vign = np.clip(vign, 0, 1)
    return (vign * 255).astype('uint8')

def add_vignette(clip, strength=0.6, falloff=0.55):
    """Apply vignette by multiplying frames with a radial mask."""
    w, h = clip.w, clip.h
    mask = vignette_mask(w, h, strength, falloff)
    def apply(frame):
        # frame is HxWx3 uint8
        out = (frame.astype(np.float32) * (mask[..., None]/255.0)).clip(0,255)
        return out.astype('uint8')
    return clip.fl_image(apply)

def micro_shake(clip, amp=8, freq=20):
    """
    Subtle handheld shake via jittery cropping.
    amp: pixel amplitude
    freq: shakes per second (randomized)
    """
    w, h = clip.w, clip.h
    crop_w, crop_h = w-amp, h-amp
    def make_pos(t):
        # random but smooth-ish jitter
        rng = np.random.RandomState(int(t*freq)*13 + 7)
        dx = rng.randint(0, amp)
        dy = rng.randint(0, amp)
        return dx, dy, dx+crop_w, dy+crop_h
    return clip.fl(lambda gf, t: crop(gf(t), *make_pos(t)))

def punch_in(clip, max_zoom=1.08, ease_ms=120):
    """
    Quick zoom-in effect at start of the clip, easing back.
    """
    ease = ease_ms/1000.0
    def zoom(t):
        if t < ease:
            # ease-in
            return 1 + (max_zoom-1) * (t/ease)
        else:
            # ease-out back toward 1 over same duration
            t2 = min(t-ease, ease)
            return max_zoom - (max_zoom-1) * (t2/ease)
    return clip.fx(vfx.resize, lambda t: zoom(t))

def velocity_ramp(clip, slow_factor=0.75, fast_factor=1.15, pattern="slow-into-fast"):
    """
    Speed curve across a small segment.
    pattern: "slow-into-fast" or "fast-into-slow"
    """
    dur = clip.duration
    mid = dur/2
    if pattern == "slow-into-fast":
        def speed(t):
            return slow_factor if t < mid else fast_factor
    else:
        def speed(t):
            return fast_factor if t < mid else slow_factor
    # MoviePy doesn't support time-varying speed directly; approximate by two halves.
    first = clip.subclip(0, mid).fx(vfx.speedx, slow_factor if pattern=="slow-into-fast" else fast_factor)
    second = clip.subclip(mid, dur).fx(vfx.speedx, fast_factor if pattern=="slow-into-fast" else slow_factor)
    # Adjust to same total duration
    first = first.set_end(first.start + first.duration)
    second = second.set_start(first.end).set_end(first.end + second.duration)
    return CompositeVideoClip([first, second]).set_duration(first.duration + second.duration)

# ---------------
# Beat detection
# ---------------

def detect_beats(audio_path, sr=44100, onset_backtrack=True, hop_length=512, tightness=2.0):
    """
    Return a list of beat times (in seconds) using onset detection.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    # Onset strength envelope
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Onset times
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr,
                                        hop_length=hop_length, backtrack=onset_backtrack,
                                        units='time')
    # Optional: filter onsets for "stronger" hits
    thresh = np.percentile(oenv, 75)
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop_length)
    strong = times[oenv >= thresh]
    # Combine: keep onsets, but ensure minimum spacing
    beats = []
    min_gap = 0.12  # 120 ms
    last = -999
    for t in np.sort(np.concatenate([onsets, strong])):
        if t - last >= min_gap:
            beats.append(float(t))
            last = t
    # Deduplicate/sort
    beats = sorted(list(set([round(b, 3) for b in beats])))
    return beats

# ---------------
# Main assembly
# ---------------

def build_edit(video_path, audio_path, out_path,
               target_fps=60, target_res=(1920,1080),
               clip_min=0.25, clip_max=1.25,
               zoom_every_n=3, flash_every_n=4,
               shake=True, letterbox=True):
    """
    Create the final edit.
    - Slices the video at beat times into short segments
    - Randomizes segment lengths within [clip_min, clip_max] seconds
    - Adds periodic punch-ins and flashes
    """
    video = VideoFileClip(str(video_path)).resize(newsize=target_res).set_fps(target_fps)
    music = AudioFileClip(str(audio_path))
    beats = detect_beats(audio_path)

    # Limit edit length to music duration
    total_dur = min(video.duration, music.duration)
    timeline = []
    t = 0.0
    seg_idx = 0

    # Map beats into segments
    beat_idx = 0
    while t < total_dur - clip_min and beat_idx < len(beats):
        start = beats[beat_idx]
        if start < t:
            beat_idx += 1
            continue
        # choose a random end near next beats
        length = random.uniform(clip_min, clip_max)
        end = min(start + length, total_dur)
        if end - start < 0.12:  # too short
            beat_idx += 1
            continue
        sub = video.subclip(start, end)

        # Occasional velocity ramp for variety
        if seg_idx % 5 == 2 and (end - start) >= 0.6:
            sub = velocity_ramp(sub, pattern=random.choice(["slow-into-fast","fast-into-slow"]))

        # Punch-in zoom on every Nth segment
        if seg_idx % zoom_every_n == 0:
            sub = punch_in(sub, max_zoom=random.uniform(1.06, 1.12))

        # Subtle shake
        if shake and (seg_idx % 2 == 1):
            sub = micro_shake(sub, amp=6, freq=18)

        # Slight grade
        sub = colorx(sub, 1.05)
        sub = lum_contrast(sub, lum=8, contrast=20)

        # Append to timeline
        if timeline:
            sub = sub.set_start(timeline[-1].end)
        timeline.append(sub)
        seg_idx += 1
        t = timeline[-1].end
        beat_idx += int(max(1, math.ceil(length / 0.35)))  # skip some beats so cuts aren't too dense

        # Flash overlay sometimes at cut points
        if seg_idx % flash_every_n == 0:
            fl = flash_white(size=target_res).set_start(timeline[-1].start)
            timeline.append(fl)

    # Stitch video
    comp = CompositeVideoClip(timeline, size=target_res).set_duration(min(t, total_dur))

    # Letterbox
    if letterbox:
        bar_h = int(0.09 * target_res[1])
        top = ColorClip((target_res[0], bar_h), color=(0,0,0)).set_opacity(1).set_position(("center","top")).set_duration(comp.duration)
        bot = ColorClip((target_res[0], bar_h), color=(0,0,0)).set_opacity(1).set_position(("center","bottom")).set_duration(comp.duration)
        comp = CompositeVideoClip([comp, top, bot], size=target_res).set_duration(comp.duration)

    # Set audio and write
    comp = comp.set_audio(music.subclip(0, comp.duration))
    comp.write_videofile(str(out_path),
                         fps=target_fps,
                         codec="libx264",
                         audio_codec="aac",
                         threads=4,
                         preset="medium",
                         bitrate="8M")

def main():
    p = argparse.ArgumentParser(description="Anime edit automation: beat-synced cuts + effects.")
    p.add_argument("--video", required=True, help="Input video file")
    p.add_argument("--audio", required=True, help="Music file (mp3/wav)")
    p.add_argument("--out", default="anime_edit_output.mp4", help="Output video path")
    p.add_argument("--fps", type=int, default=60, help="Target FPS (default 60)")
    p.add_argument("--width", type=int, default=1920, help="Output width (default 1920)")
    p.add_argument("--height", type=int, default=1080, help="Output height (default 1080)")
    p.add_argument("--minseg", type=float, default=0.25, help="Min segment length (sec)")
    p.add_argument("--maxseg", type=float, default=1.25, help="Max segment length (sec)")
    p.add_argument("--no-shake", action="store_true", help="Disable micro shake")
    p.add_argument("--no-letterbox", action="store_true", help="Disable letterbox bars")
    args = p.parse_args()

    build_edit(
        video_path=args.video,
        audio_path=args.audio,
        out_path=args.out,
        target_fps=args.fps,
        target_res=(args.width, args.height),
        clip_min=args.minseg,
        clip_max=args.maxseg,
        shake=not args.no_shake,
        letterbox=not args.no_letterbox
    )

if __name__ == "__main__":
    main()
