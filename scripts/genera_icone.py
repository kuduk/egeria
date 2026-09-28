"""Icone dell'interfaccia ricavate dal logo (src/egeria/web/img/logo.png).

    .venv/bin/python scripts/genera_icone.py

Dal logo (emblema + scritta su fondo bianco) ritaglia l'emblema, toglie il fondo
("color to alpha": i bordi sfumati restano puliti) e lo appoggia su un quadrato chiaro
arrotondato. Il volto è disegnato con il bianco del fondo, quindi l'emblema va sempre su
un fondo chiaro: sullo scuro diventerebbe un negativo. Scrive in src/egeria/web/img/:
- emblema.png (96 px): nella barra in alto, uguale nei due temi;
- favicon.png (64 px) e apple-touch-icon.png (180 px);
- logo-1280x640.png: il logo intero centrato, per l'anteprima social di GitHub;
- logo-1280x640-orizzontale.png: emblema a sinistra e scritta a destra, stesso formato.
Da rilanciare se cambia il logo.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG = Path(__file__).resolve().parent.parent / "src" / "egeria" / "web" / "img"


def color_to_alpha(rgb: np.ndarray, background: np.ndarray) -> np.ndarray:
    """RGB su fondo uniforme -> RGBA: l'alfa minima che spiega ogni pixel come colore sopra il fondo."""
    p, b = rgb / 255.0, background / 255.0
    alpha = np.max(np.where(p < b, (b - p) / np.maximum(b, 1e-6), (p - b) / np.maximum(1 - b, 1e-6)), axis=-1)
    alpha = np.clip(alpha, 0, 1)
    safe = np.maximum(alpha, 1e-6)[..., None]
    color = np.clip((p - (1 - alpha[..., None]) * b) / safe, 0, 1)
    return np.dstack([color, alpha])


def emblem_box(rgb: np.ndarray, background: np.ndarray, margin: int = 16) -> tuple[int, int, int, int]:
    """Riquadro quadrato attorno all'emblema: la parte disegnata sopra la scritta."""
    ink = np.abs(rgb.astype(int) - background).max(-1) > 40
    rows = np.nonzero(ink.any(1))[0]
    gaps = np.nonzero(np.diff(rows) > 20)[0]  # la prima riga vuota lunga separa emblema e scritta
    bottom = rows[gaps[0]] if len(gaps) else rows[-1]
    ys, xs = np.nonzero(ink[: bottom + 1])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + margin
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def to_image(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray((rgba * 255).round().astype(np.uint8), "RGBA")


def resized(image: Image.Image, side: int) -> Image.Image:
    # Ridimensionamento con alfa premoltiplicata: niente aloni scuri sui bordi.
    return image.convert("RGBa").resize((side, side), Image.LANCZOS).convert("RGBA")


def tile(emblem: Image.Image, side: int, background: tuple[int, int, int]) -> Image.Image:
    scale = 4  # si disegna grande e si riduce, per bordi arrotondati puliti
    big = Image.new("RGBA", (side * scale, side * scale), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle([0, 0, side * scale - 1, side * scale - 1], radius=int(side * scale * 0.22),
                                          fill=(*background, 255))
    inner = int(side * scale * 0.84)
    offset = (side * scale - inner) // 2
    big.alpha_composite(resized(emblem, inner), (offset, offset))
    return resized(big, side)


def ink_box(rgb: np.ndarray, background: np.ndarray, rows: slice = slice(None)) -> tuple[int, int, int, int]:
    """Riquadro stretto attorno a ciò che è disegnato (nelle righe indicate)."""
    ink = np.abs(rgb.astype(int) - background).max(-1) > 40
    ink[:rows.start or 0] = False
    if rows.stop is not None:
        ink[rows.stop:] = False
    ys, xs = np.nonzero(ink)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def social_previews(rgb: np.ndarray, background: np.ndarray, size=(1280, 640)) -> None:
    """Anteprima social di GitHub (1280×640, con almeno 40 px di margine libero)."""
    fill = tuple(int(v) for v in background)
    logo = Image.fromarray(rgb)
    # Logo intero centrato: si toglie il bianco attorno e si lascia un margine ampio.
    centered = Image.new("RGB", size, fill)
    content = fit(logo.crop(ink_box(rgb, background)), size[0] - 2 * 120, size[1] - 2 * 70)
    centered.paste(content, ((size[0] - content.width) // 2, (size[1] - content.height) // 2))
    centered.save(IMG / "logo-1280x640.png", optimize=True)
    # Orizzontale: emblema a sinistra, scritta a destra, centrati in verticale.
    ink = np.abs(rgb.astype(int) - background).max(-1) > 40
    rows = np.nonzero(ink.any(1))[0]
    split = rows[np.nonzero(np.diff(rows) > 20)[0][0]] + 1  # fine dell'emblema, inizio della scritta
    emblem = fit(logo.crop(ink_box(rgb, background, slice(0, split))), 10_000, 400)
    word = fit(logo.crop(ink_box(rgb, background, slice(split, None))), 10_000, 92)
    gap = 64
    width = emblem.width + gap + word.width
    horizontal = Image.new("RGB", size, fill)
    x = (size[0] - width) // 2
    horizontal.paste(emblem, (x, (size[1] - emblem.height) // 2))
    horizontal.paste(word, (x + emblem.width + gap, (size[1] - word.height) // 2))
    horizontal.save(IMG / "logo-1280x640-orizzontale.png", optimize=True)


def main() -> None:
    rgb = np.asarray(Image.open(IMG / "logo.png").convert("RGB"))
    background = np.median(np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)]), axis=0)
    original = background.copy()  # il fondo vero del logo, per le immagini che lo contengono
    if background.min() > 245:
        # Fondo quasi bianco: si usa il bianco puro. Con 254 un pixel a 255 avrebbe alfa 1 (puntini bianchi).
        background = np.full(3, 255.0)
    box = emblem_box(rgb, background)
    rgba = color_to_alpha(rgb[box[1]: box[3], box[0]: box[2]].astype(float), background)

    emblem = to_image(rgba)
    fill = tuple(int(v) for v in background)
    tile(emblem, 96, fill).save(IMG / "emblema.png", optimize=True)
    tile(emblem, 64, fill).save(IMG / "favicon.png", optimize=True)
    tile(emblem, 180, fill).save(IMG / "apple-touch-icon.png", optimize=True)
    social_previews(rgb, original)
    for name in ("emblema.png", "favicon.png", "apple-touch-icon.png", "logo-1280x640.png", "logo-1280x640-orizzontale.png"):
        print(f"{name}: {(IMG / name).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
