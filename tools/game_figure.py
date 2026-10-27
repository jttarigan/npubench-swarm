"""Compose the study 3 game figure (paper/study3/figures/fig_game.png): a 2 x 4 grid of frames from
a hand-played screen recording "paper/NPUSwarm Gameplay.mp4" (POCO X6 Pro, 1220x2712, XL model on
the NPU path, recorded 2026-08-01, not in the repository).
Frames are extracted with tools/extract_frames.swift (AVFoundation; the Mac has no ffmpeg):

  swift tools/extract_frames.swift "paper/NPUSwarm Gameplay.mp4" <framedir> 6 14 20 26 36 40 48 52
  python3 tools/game_figure.py <framedir>

Panels (a)-(h) = 6, 14, 20, 26, 36, 40, 48, 52 s = waves 1, 3, 4, 5, 7, 8, 9, 10 (8 to 80 enemies).
Each frame is cropped to the band around the ship (rows 300-2150 of 2712: the status overlay, the
empty sky and the empty floor go) so the action fills the panel; nothing else is edited. The player
casts with touch gestures (the game's menu: O = bomb, / = beam, W = shield): the bright lines are
beams, the shaded discs are bombs (damage zones), the thin cyan curves are strokes being drawn. The
study's unattended games have none of these; the caption says so.
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "study3", "figures", "fig_game.png")
FRAMES = ["frame_006.0s.png", "frame_014.0s.png", "frame_020.0s.png", "frame_026.0s.png",
          "frame_036.0s.png", "frame_040.0s.png", "frame_048.0s.png", "frame_052.0s.png"]
CROP = (0, 300, 1220, 2150)   # left, top, right, bottom in source pixels
COLS = 4
GAP = 20                      # white gap between panels, source pixels (user: half of 40)
OUT_WIDTH = 3000              # 500 dpi at the manuscript's 6 in figure width


def main(framedir):
    panels = [Image.open(os.path.join(framedir, f)).convert("RGB").crop(CROP) for f in FRAMES]
    w, h = panels[0].size
    rows = (len(panels) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * w + (COLS - 1) * GAP, rows * h + (rows - 1) * GAP), "white")
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    for i, p in enumerate(panels):
        x, y = (i % COLS) * (w + GAP), (i // COLS) * (h + GAP)
        sheet.paste(p, (x, y))
        ImageDraw.Draw(sheet).text((x + 40, y + 28), f"({'abcdefgh'[i]})", fill="white", font=font)
    sheet = sheet.resize((OUT_WIDTH, round(OUT_WIDTH * sheet.height / sheet.width)), Image.LANCZOS)
    sheet.save(OUT, optimize=True)
    print("wrote", os.path.normpath(OUT), sheet.size)


if __name__ == "__main__":
    main(sys.argv[1])
