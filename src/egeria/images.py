"""Caricamento delle immagini dello stato: path, url o base64."""

from __future__ import annotations

import base64
import io
import urllib.request

from .schema import RequestError

USER_AGENT = "Egeria/0.1 (decision model research)"


def load_images(state: list, max_side: int | None = None) -> list:
    """Immagini dello stato, nell'ordine in cui compaiono, convertite in RGB ed eventualmente ridotte."""
    from PIL import Image

    images = []
    for number, part in enumerate(state):
        if part.get("type") != "image":
            continue
        try:
            if "path" in part:
                image = Image.open(part["path"])
            elif "url" in part:
                request = urllib.request.Request(part["url"], headers={"User-Agent": USER_AGENT})
                image = Image.open(io.BytesIO(urllib.request.urlopen(request, timeout=30).read()))
            else:
                image = Image.open(io.BytesIO(base64.b64decode(part["base64"])))
            image = image.convert("RGB")
        except (OSError, ValueError) as error:
            raise RequestError(f"state[{number}]: immagine non leggibile ({error})") from error
        if max_side:
            image.thumbnail((max_side, max_side))
        images.append(image)
    return images
