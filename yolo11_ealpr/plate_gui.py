"""
A simple desktop app: open a car photo and read its licence plate.

    python plate_gui.py                # then click "Open photo..."
    python plate_gui.py car.jpg        # open a photo straight away

Built with tkinter (part of Python, nothing to install) on top of
plate_reader.py, so it uses exactly the same models and post-processing as
the evaluation scripts.

How it stays responsive: loading the models and reading a photo run in a
background thread. tkinter widgets may only be touched from the main thread,
so the worker puts its result in a queue and the window checks that queue
every 100 ms.
"""

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from plate_reader import draw, plate_pieces

PHOTO_BOX = (760, 640)        # max size of the photo shown on the left
CROP_WIDTH = 320              # max width of the main plate crop on the right


def text_image(pieces, size=44):
    """Render the plate text as an image, piece by piece, left to right (so
    the digits and letters keep the same layout as on the real plate)."""
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    gap = size // 3
    widths = [probe.textlength(s, font=font) for s in pieces]
    w = int(sum(widths) + gap * (len(pieces) - 1)) + 24
    img = Image.new("RGB", (w, size + 24), "white")
    d = ImageDraw.Draw(img)
    x = 12
    for s, sw in zip(pieces, widths):
        d.text((x, 8), s, font=font, fill="black")
        x += sw + gap
    return img


def to_tk(bgr_or_pil, max_w, max_h):
    """Convert an OpenCV (BGR) or PIL image to a Tk image that fits a box."""
    img = bgr_or_pil
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.copy()
    img.thumbnail((max_w, max_h))           # keeps the aspect ratio
    return ImageTk.PhotoImage(img)


class App(tk.Tk):
    def __init__(self, first_photo=None):
        super().__init__()
        self.title("Egyptian Licence Plate Reader")
        self.geometry("1180x740")
        self.minsize(900, 600)

        self.reader = None
        self.results = queue.Queue()        # messages from worker threads
        self.plates = []
        self._tk_images = []                # keep references (Tk forgets)

        # ---- top bar ------------------------------------------------------
        bar = tk.Frame(self, padx=8, pady=8)
        bar.pack(side="top", fill="x")
        self.open_btn = tk.Button(bar, text="Open photo...", width=16,
                                  command=self.open_photo, state="disabled")
        self.open_btn.pack(side="left")
        self.copy_btn = tk.Button(bar, text="Copy plate text", width=16,
                                  command=self.copy_text, state="disabled")
        self.copy_btn.pack(side="left", padx=8)
        self.status = tk.Label(bar, text="Loading models...", anchor="w",
                               fg="#555")
        self.status.pack(side="left", fill="x", expand=True, padx=8)

        # ---- photo on the left, results on the right ----------------------
        body = tk.Frame(self)
        body.pack(fill="both", expand=True)
        # A fixed-size frame (in pixels) holds the photo; pack_propagate(False)
        # stops it from shrinking or growing to fit its contents.
        holder = tk.Frame(body, width=PHOTO_BOX[0], height=PHOTO_BOX[1],
                          bg="#222")
        holder.pack_propagate(False)
        holder.pack(side="left", padx=8, pady=(0, 8))
        self.photo_label = tk.Label(holder, bg="#222", fg="#aaa",
                                    font=("Segoe UI", 13),
                                    text="Open a car photo to read its plate")
        self.photo_label.pack(fill="both", expand=True)
        self.side = tk.Frame(body, padx=12)
        self.side.pack(side="left", fill="both", expand=True)

        self.first_photo = first_photo
        threading.Thread(target=self._load_models, daemon=True).start()
        self.after(100, self._poll)

    # ---- background work ---------------------------------------------------
    def _load_models(self):
        try:
            from plate_reader import PlateReader
            reader = PlateReader()
            # warm-up run so the first real photo is fast
            reader.read(np.zeros((480, 640, 3), np.uint8))
            self.results.put(("ready", reader))
        except Exception as exc:                       # e.g. missing weights
            self.results.put(("error", f"Could not load the models:\n{exc}"))

    def _read(self, path):
        try:
            img = cv2.imdecode(np.fromfile(str(path), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("not an image file")
            t = time.perf_counter()
            plates = self.reader.read(img)
            ms = (time.perf_counter() - t) * 1000
            self.results.put(("done", (path, img, plates, ms)))
        except Exception as exc:
            self.results.put(("error", f"Could not read {Path(path).name}:\n{exc}"))

    def _poll(self):
        """Runs on the main thread every 100 ms; handles worker messages."""
        try:
            while True:
                kind, data = self.results.get_nowait()
                if kind == "ready":
                    self.reader = data
                    self.open_btn.config(state="normal")
                    self.status.config(text="Ready - open a car photo.")
                    if self.first_photo:
                        self.start_reading(self.first_photo)
                elif kind == "done":
                    self.show(*data)
                elif kind == "error":
                    self.open_btn.config(state="normal" if self.reader else "disabled")
                    self.status.config(text="Error - see message.")
                    messagebox.showerror("Plate reader", data)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ---- user actions ------------------------------------------------------
    def open_photo(self):
        path = filedialog.askopenfilename(
            title="Choose a car photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")])
        if path:
            self.start_reading(path)

    def start_reading(self, path):
        self.open_btn.config(state="disabled")
        self.status.config(text=f"Reading {Path(path).name} ...")
        threading.Thread(target=self._read, args=(path,), daemon=True).start()

    def copy_text(self):
        if self.plates:
            self.clipboard_clear()
            self.clipboard_append(self.plates[0]["text"])
            self.status.config(text="Plate text copied to the clipboard.")

    # ---- display -----------------------------------------------------------
    def show(self, path, img, plates, ms):
        self.plates = plates
        self._tk_images = []
        self.open_btn.config(state="normal")
        self.copy_btn.config(state="normal" if plates else "disabled")
        self.status.config(text=f"{Path(path).name}  -  {len(plates)} plate(s) "
                                f"found in {ms:.0f} ms")

        annotated = to_tk(draw(img, plates), *PHOTO_BOX)
        self._tk_images.append(annotated)
        self.photo_label.config(image=annotated, text="")

        for w in self.side.winfo_children():
            w.destroy()
        if not plates:
            tk.Label(self.side, text="No plate found in this photo.",
                     font=("Segoe UI", 14), fg="#a00").pack(anchor="w", pady=8)
            return

        # The main (most confident) plate is shown large; up to 2 others
        # (often background cars) are shown compact so everything fits.
        for i, p in enumerate(plates[:3]):
            main = i == 0
            box = tk.LabelFrame(self.side, padx=8, pady=6,
                                text=f"Plate {i + 1}" + (" (main)" if main else ""),
                                font=("Segoe UI", 10, "bold"))
            box.pack(anchor="w", fill="x", pady=(0, 10))

            # enlarge the crop to fit a box, keeping its shape
            crop = Image.fromarray(cv2.cvtColor(p["crop"], cv2.COLOR_BGR2RGB))
            max_w, max_h = (CROP_WIDTH, 130) if main else (200, 60)
            s = min(max_w / crop.width, max_h / crop.height)
            crop = crop.resize((max(1, round(crop.width * s)),
                                max(1, round(crop.height * s))), Image.LANCZOS)
            tk_crop = ImageTk.PhotoImage(crop)
            tk_text = ImageTk.PhotoImage(
                text_image(plate_pieces(p), size=44 if main else 26))
            self._tk_images += [tk_crop, tk_text]
            row = tk.Frame(box)
            row.pack(anchor="w")
            tk.Label(row, image=tk_crop).pack(side="top" if main else "left",
                                              anchor="w")
            tk.Label(row, image=tk_text, bd=1, relief="solid").pack(
                side="top" if main else "left", anchor="w", pady=6, padx=(0 if main else 8, 0))

            if p["valid"]:
                verdict, colour = "Valid plate format", "#080"
            elif not p["digits"] and not p["letters"]:
                verdict, colour = "Could not read - too small or blurry", "#a00"
            else:
                verdict, colour = "Uncertain - format looks wrong", "#c60"
            tk.Label(box, text=verdict, fg=colour,
                     font=("Segoe UI", 11 if main else 9, "bold")).pack(anchor="w")
            if not main:
                continue
            details = (f"Digits: {len(p['digits'])}   Letters: {len(p['letters'])}\n"
                       f"Plate detection confidence: {p['plate_conf']:.2f}\n"
                       f"Weakest character confidence: {p['conf']:.2f}")
            if p["fixes"]:
                details += "\nAuto-corrected: " + "; ".join(p["fixes"])
            tk.Label(box, text=details, justify="left", fg="#333",
                     font=("Segoe UI", 10)).pack(anchor="w")


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
