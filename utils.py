def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    """
    1) Scan query_img columns for any colored candle pixels.
    2) First column with NO candle pixels becomes x_split.
    3) Paste forecast to right of x_split, draw red divider there.
    4) Trace blue forecast line, then crop margins.
    """
    import numpy as _np
    from PIL import ImageDraw, Image

    w, h = query_img.size

    # 1) Detect candle pixels (<240) and sum per column
    arr = _np.array(query_img.convert("RGB"))
    candle_mask = _np.any(arr < 240, axis=2)
    col_counts = candle_mask.sum(axis=0)

    # 2) Find first fully blank column
    blank_cols = _np.where(col_counts == 0)[0]
    x_split = int(blank_cols[0]) if blank_cols.size else (w // 2)

    # 3) Prepare overlay canvas
    overlay = query_img.convert("RGBA")
    m = match_img.resize((w, h)).convert("RGBA")

    # 4) Paste forecast region
    fc = m.crop((x_split, 0, w, h))
    mask_fc = fc.split()[-1].point(lambda p: 128)
    overlay.paste(fc, (x_split, 0), mask_fc)

    # 5) Draw red divider
    draw = ImageDraw.Draw(overlay)
    draw.line([(x_split, 0), (x_split, h)], fill=(255, 0, 0), width=2)

    # 6) Trace blue line on the pasted region
    left  = overlay.crop((0, 0, x_split, h))
    right = overlay.crop((x_split, 0, w, h))
    ar = _np.array(right)
    rh, rw = ar.shape[:2]
    y0, y1 = int(0.1*rh), int(0.9*rh)
    gray = _np.dot(ar[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    mp = int(0.01*(y1-y0))
    coords = [
        (x, int(_np.median(_np.where(dark[:,x])[0])) + y0)
        for x in range(rw)
        if len(_np.where(dark[:,x])[0]) >= mp
    ]
    al = _np.array(left)
    grayl = _np.dot(al[y0:y1], [0.299, 0.587, 0.114])
    darkl = grayl < 200
    ys = _np.where(darkl[:, -2])[0]
    ymed = int(_np.median(ys)) + y0 if ys.size >= mp else (y0+y1)//2
    full = [(0, ymed)] + coords

    blank = Image.new("RGB", (rw, rh), (255,255,255))
    d2 = ImageDraw.Draw(blank)
    if full:
        d2.line(full, fill=(0,0,255), width=3)

    # 7) Reassemble and crop 10% margins
    combo = Image.new("RGB", (w, h))
    combo.paste(left.convert("RGB"), (0, 0))
    combo.paste(blank, (x_split, 0))
    cx, cy = int(0.1*w), int(0.1*h)
    return combo.crop((cx, cy, w-cx, h-cy))
