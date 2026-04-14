#!/usr/bin/env python3
"""
Minecraft Skin Ancestry Tree Generator (Optimized with NumPy)
==============================================================
Generates a nested folder tree where each level is a feature choice:
  suit_color / tie_color / hair_color / eye_color / accent_color

Every combination exists. Each folder has a skin.png showing
all choices made up to that level.

Browse like: variants/navy_suit/gold_tie/blonde_hair/blue_eyes/cyan_accent/skin.png

Tuned for ~400K total skins, targeting under 10 minutes.
"""

import numpy as np
from PIL import Image
import os
import shutil
import time

ORIGINAL = "c1e9862df153554f.png"
OUTPUT_DIR = "variants"

# ============================================================
# REGIONS
# ============================================================

REGIONS = {
    'head_top': (8, 0, 16, 8), 'head_front': (8, 8, 16, 16),
    'head_right': (0, 8, 8, 16), 'head_left': (16, 8, 24, 16),
    'head_back': (24, 8, 32, 16), 'head_bottom': (16, 0, 24, 8),
    'body_front': (20, 20, 28, 32), 'body_back': (32, 20, 40, 32),
    'body_right': (16, 20, 20, 32), 'body_left': (28, 20, 32, 32),
    'body_top': (20, 16, 28, 20), 'body_bottom': (28, 16, 36, 20),
    'right_arm_front': (44, 20, 48, 32), 'right_arm_back': (52, 20, 56, 32),
    'right_arm_outer': (40, 20, 44, 32), 'right_arm_inner': (48, 20, 52, 32),
    'right_arm_top': (44, 16, 48, 20), 'right_arm_bottom': (48, 16, 52, 20),
    'left_arm_front': (36, 52, 40, 64), 'left_arm_back': (44, 52, 48, 64),
    'left_arm_outer': (32, 52, 36, 64), 'left_arm_inner': (40, 52, 44, 64),
    'left_arm_top': (36, 48, 40, 52), 'left_arm_bottom': (40, 48, 44, 52),
    'right_leg_front': (4, 20, 8, 32), 'right_leg_back': (12, 20, 16, 32),
    'right_leg_outer': (0, 20, 4, 32), 'right_leg_inner': (8, 20, 12, 32),
    'right_leg_top': (4, 16, 8, 20), 'right_leg_bottom': (8, 16, 12, 20),
    'left_leg_front': (20, 52, 24, 64), 'left_leg_back': (28, 52, 32, 64),
    'left_leg_outer': (16, 52, 20, 64), 'left_leg_inner': (24, 52, 28, 64),
    'left_leg_top': (20, 48, 24, 52), 'left_leg_bottom': (24, 48, 28, 52),
    'head_overlay_top': (40, 0, 48, 8), 'head_overlay_front': (40, 8, 48, 16),
    'head_overlay_right': (32, 8, 40, 16), 'head_overlay_left': (48, 8, 56, 16),
    'head_overlay_back': (56, 8, 64, 16), 'head_overlay_bottom': (48, 0, 56, 8),
    'body_overlay_front': (20, 36, 28, 48), 'body_overlay_back': (32, 36, 40, 48),
    'body_overlay_right': (16, 36, 20, 48), 'body_overlay_left': (28, 36, 32, 48),
    'body_overlay_top': (20, 32, 28, 36), 'body_overlay_bottom': (28, 32, 36, 36),
    'right_arm_overlay_front': (44, 36, 48, 48), 'right_arm_overlay_back': (52, 36, 56, 48),
    'right_arm_overlay_outer': (40, 36, 44, 48), 'right_arm_overlay_inner': (48, 36, 52, 48),
    'right_arm_overlay_top': (44, 32, 48, 36), 'right_arm_overlay_bottom': (48, 32, 52, 36),
    'left_arm_overlay_front': (52, 52, 56, 64), 'left_arm_overlay_back': (60, 52, 64, 64),
    'left_arm_overlay_outer': (48, 52, 52, 64), 'left_arm_overlay_inner': (56, 52, 60, 64),
    'left_arm_overlay_top': (52, 48, 56, 52), 'left_arm_overlay_bottom': (56, 48, 60, 52),
    'right_leg_overlay_front': (4, 36, 8, 48), 'right_leg_overlay_back': (12, 36, 16, 48),
    'right_leg_overlay_outer': (0, 36, 4, 48), 'right_leg_overlay_inner': (8, 36, 12, 48),
    'right_leg_overlay_top': (4, 32, 8, 36), 'right_leg_overlay_bottom': (8, 32, 12, 36),
    'left_leg_overlay_front': (4, 52, 8, 64), 'left_leg_overlay_back': (12, 52, 16, 64),
    'left_leg_overlay_outer': (0, 52, 4, 64), 'left_leg_overlay_inner': (8, 52, 12, 64),
    'left_leg_overlay_top': (4, 48, 8, 52), 'left_leg_overlay_bottom': (8, 48, 12, 52),
}

# ============================================================
# MASKS (pre-computed once)
# ============================================================

def build_region_mask(region_names):
    mask = np.zeros((64, 64), dtype=bool)
    for name in region_names:
        if name in REGIONS:
            x1, y1, x2, y2 = REGIONS[name]
            mask[y1:y2, x1:x2] = True
    return mask

def classify_image(img_array):
    r, g, b, a = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2], img_array[:,:,3]
    opaque = a > 0
    ri, gi, bi = r.astype(int), g.astype(int), b.astype(int)
    is_grey = opaque & (np.abs(ri - gi) <= 5) & (np.abs(gi - bi) <= 5) & (np.abs(ri - bi) <= 5)
    not_grey = opaque & ~is_grey
    brightness = (ri + gi + bi) / 3.0
    
    tie = opaque & (r > 140) & (g < 80) & (b < 80) & not_grey
    gold = not_grey & (r > 140) & (g > 80) & (gi - bi > 50) & ~tie
    cyan = not_grey & (b > 140) & (r < 80) & (g > 100)
    shirt = is_grey & (brightness >= 180)
    suit_light = is_grey & (brightness >= 60) & (brightness < 180)
    suit_dark = is_grey & (brightness < 60)
    
    return {
        'tie': tie, 'gold': gold, 'cyan': cyan,
        'shirt': shirt, 'suit_light': suit_light, 'suit_dark': suit_dark,
    }

# ============================================================
# FAST COLOR TRANSFORMS (vectorized, operate on pre-extracted pixel data)
# ============================================================

def rgb_to_hsv_vec(rgb_f):
    """(N,3) float [0,1] -> (N,3) HSV."""
    r, g, b = rgb_f[:,0], rgb_f[:,1], rgb_f[:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    diff = maxc - minc
    s = np.where(maxc > 0, diff / maxc, 0.0)
    h = np.zeros_like(r)
    mask = diff > 0
    m = mask & (maxc == r); h[m] = ((g[m] - b[m]) / diff[m]) % 6.0
    m = mask & (maxc == g); h[m] = (b[m] - r[m]) / diff[m] + 2.0
    m = mask & (maxc == b); h[m] = (r[m] - g[m]) / diff[m] + 4.0
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=1)

def hsv_to_rgb_vec(hsv):
    """(N,3) HSV -> (N,3) float [0,1]."""
    h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]
    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    rgb = np.zeros((len(h), 3))
    for idx in range(6):
        m = i == idx
        if not np.any(m): continue
        if idx == 0: rgb[m] = np.stack([v[m], t[m], p[m]], axis=1)
        elif idx == 1: rgb[m] = np.stack([q[m], v[m], p[m]], axis=1)
        elif idx == 2: rgb[m] = np.stack([p[m], v[m], t[m]], axis=1)
        elif idx == 3: rgb[m] = np.stack([p[m], q[m], v[m]], axis=1)
        elif idx == 4: rgb[m] = np.stack([t[m], p[m], v[m]], axis=1)
        elif idx == 5: rgb[m] = np.stack([v[m], p[m], q[m]], axis=1)
    return np.clip(rgb, 0, 1)

# Pre-compute extraction indices for each mask to avoid recomputing
class MaskData:
    """Pre-computed data for a pixel mask to speed up transforms."""
    def __init__(self, mask, base_array):
        self.ys, self.xs = np.where(mask)
        self.n = len(self.ys)
        if self.n > 0:
            self.base_rgb_f = base_array[self.ys, self.xs, :3].astype(np.float32) / 255.0
            self.base_v = rgb_to_hsv_vec(self.base_rgb_f)[:, 2].copy()

def apply_colorize(img_arr, mask_data, target_hue, target_sat, bright_adj):
    """Apply colorization to pre-computed mask locations. Returns new array."""
    if mask_data.n == 0:
        return img_arr.copy()
    result = img_arr.copy()
    v = np.clip(mask_data.base_v * bright_adj, 0, 1)
    
    # Rapidly drop saturation for bright pixels (details like buttons/seams)
    # so they remain natural white or grey instead of brightly colored dots.
    # A base_v of 0.2 stays full sat. 0.4 becomes 0.6 sat. 0.5+ becomes 0 sat.
    sat_factor = np.clip(1.0 - (mask_data.base_v * 2.0)**4, 0.0, 1.0)
    
    hsv = np.column_stack([
        np.full(mask_data.n, target_hue, dtype=np.float32),
        np.full(mask_data.n, target_sat, dtype=np.float32) * sat_factor,
        v
    ])
    new_rgb = (hsv_to_rgb_vec(hsv) * 255).astype(np.uint8)
    result[mask_data.ys, mask_data.xs, :3] = new_rgb
    return result

def apply_colorize_inplace(img_arr, mask_data, target_hue, target_sat, bright_adj):
    """Apply colorization in-place (faster, modifies img_arr directly)."""
    if mask_data.n == 0:
        return
    v = np.clip(mask_data.base_v * bright_adj, 0, 1)
    
    # Rapidly drop saturation for bright pixels (details like buttons/seams)
    sat_factor = np.clip(1.0 - (mask_data.base_v * 2.0)**4, 0.0, 1.0)
    
    hsv = np.column_stack([
        np.full(mask_data.n, target_hue, dtype=np.float32),
        np.full(mask_data.n, target_sat, dtype=np.float32) * sat_factor,
        v
    ])
    new_rgb = (hsv_to_rgb_vec(hsv) * 255).astype(np.uint8)
    img_arr[mask_data.ys, mask_data.xs, :3] = new_rgb

def apply_recolor_tie(img_arr, mask_data, base_array, target_hue, target_sat, target_vmult):
    """Recolor tie pixels. Uses base_array's brightness. Modifies img_arr in-place."""
    if mask_data.n == 0:
        return
    # Re-extract from current image (tie brightness may differ from base after suit recolor)
    pixels = base_array[mask_data.ys, mask_data.xs, :3].astype(np.float32) / 255.0
    hsv = rgb_to_hsv_vec(pixels)
    hsv[:, 0] = target_hue
    hsv[:, 1] = target_sat
    hsv[:, 2] = np.clip(hsv[:, 2] * target_vmult, 0, 1)
    new_rgb = (hsv_to_rgb_vec(hsv) * 255).astype(np.uint8)
    img_arr[mask_data.ys, mask_data.xs, :3] = new_rgb

def apply_eyes(img_arr, main_color, dark_color):
    """Set eye pixels. Modifies in-place."""
    for x, y in [(11,12),(13,12),(11,13),(13,13)]:
        img_arr[y, x, :3] = main_color; img_arr[y, x, 3] = 255
    for x, y in [(10,12),(14,12),(10,13),(14,13)]:
        img_arr[y, x, :3] = dark_color; img_arr[y, x, 3] = 255

def apply_accents(img_arr, mask_data, base_array, target_hue, target_sat, target_vmult):
    """Recolor accent pixels. Modifies in-place."""
    if mask_data.n == 0:
        return
    pixels = base_array[mask_data.ys, mask_data.xs, :3].astype(np.float32) / 255.0
    hsv = rgb_to_hsv_vec(pixels)
    hsv[:, 0] = target_hue
    hsv[:, 1] = target_sat
    hsv[:, 2] = np.clip(hsv[:, 2] * target_vmult, 0, 1)
    new_rgb = (hsv_to_rgb_vec(hsv) * 255).astype(np.uint8)
    img_arr[mask_data.ys, mask_data.xs, :3] = new_rgb

# ============================================================
# FEATURE OPTIONS (tuned for ~400K total, great variety)
# ============================================================

SUIT_COLORS = [
    ("black_suit", 0.0, 0.0, 1.0),
    ("charcoal_suit", 0.0, 0.0, 1.2),
    ("grey_suit", 0.0, 0.0, 1.5),
    ("light_grey_suit", 0.0, 0.0, 1.8),
    ("navy_suit", 0.62, 0.65, 1.0),
    ("royal_blue_suit", 0.60, 0.70, 1.1),
    ("midnight_suit", 0.65, 0.55, 0.85),
    ("burgundy_suit", 0.97, 0.65, 1.0),
    ("wine_suit", 0.95, 0.55, 0.95),
    ("crimson_suit", 0.0, 0.60, 1.05),
    ("forest_suit", 0.38, 0.55, 1.0),
    ("emerald_suit", 0.42, 0.65, 1.1),
    ("olive_suit", 0.22, 0.45, 1.05),
    ("purple_suit", 0.78, 0.60, 1.0),
    ("plum_suit", 0.83, 0.45, 1.0),
    ("teal_suit", 0.50, 0.55, 1.0),
    ("brown_suit", 0.07, 0.50, 1.0),
    ("tan_suit", 0.10, 0.30, 1.4),
    ("cream_suit", 0.12, 0.15, 2.0),
    ("white_suit", 0.12, 0.08, 2.2),
    ("rust_suit", 0.04, 0.65, 1.0),
    ("copper_suit", 0.05, 0.60, 1.2),
]

TIE_COLORS = [
    ("red_tie", 0.0, 0.75, 1.0),
    ("burgundy_tie", 0.97, 0.60, 0.8),
    ("blue_tie", 0.60, 0.70, 1.0),
    ("navy_tie", 0.63, 0.65, 0.8),
    ("green_tie", 0.38, 0.65, 1.0),
    ("gold_tie", 0.13, 0.75, 1.0),
    ("orange_tie", 0.08, 0.75, 1.0),
    ("purple_tie", 0.78, 0.65, 1.0),
    ("pink_tie", 0.93, 0.50, 1.0),
    ("teal_tie", 0.48, 0.65, 1.0),
    ("cyan_tie", 0.52, 0.70, 1.0),
    ("silver_tie", 0.0, 0.05, 1.0),
    ("black_tie", 0.0, 0.0, 0.25),
    ("white_tie", 0.12, 0.08, 1.4),
    ("yellow_tie", 0.16, 0.75, 1.2),
    ("coral_tie", 0.02, 0.55, 1.0),
]

HAIR_COLORS = [
    ("black_hair", 0.0, 0.0, 1.0),
    ("dark_brown_hair", 0.06, 0.40, 1.1),
    ("brown_hair", 0.07, 0.45, 1.3),
    ("blonde_hair", 0.12, 0.55, 2.5),
    ("platinum_hair", 0.14, 0.15, 4.0),
    ("ginger_hair", 0.05, 0.70, 1.6),
    ("auburn_hair", 0.03, 0.60, 1.5),
    ("silver_hair", 0.0, 0.0, 4.0),
    ("white_hair", 0.0, 0.0, 5.5),
    ("blue_hair", 0.58, 0.80, 1.8),
    ("purple_hair", 0.78, 0.55, 1.2),
    ("pink_hair", 0.92, 0.65, 1.6),
    ("green_hair", 0.42, 0.60, 1.5),
    ("teal_hair", 0.50, 0.60, 1.5),
    ("red_hair", 0.0, 0.55, 1.2),
    ("orange_hair", 0.08, 0.70, 1.7),
]

EYE_COLORS = [
    ("dark_eyes", [20,20,20], [10,10,10]),
    ("brown_eyes", [100,60,30], [60,35,15]),
    ("green_eyes", [50,140,60], [30,100,40]),
    ("blue_eyes", [40,80,180], [20,50,140]),
    ("light_blue_eyes", [80,140,220], [50,100,180]),
    ("amber_eyes", [180,140,30], [140,100,10]),
    ("grey_eyes", [120,125,130], [80,85,90]),
    ("violet_eyes", [120,50,180], [80,30,140]),
    ("red_eyes", [180,20,20], [140,10,10]),
    ("cyan_eyes", [40,180,200], [20,140,160]),
]

SPARK_COLORS = [
    ("gold_spark", 0.13, 0.75, 1.0),
    ("red_spark", 0.0, 0.70, 1.0),
    ("orange_spark", 0.08, 0.70, 1.0),
    ("green_spark", 0.38, 0.65, 1.0),
    ("cyan_spark", 0.52, 0.75, 1.0),
    ("blue_spark", 0.60, 0.70, 1.0),
    ("purple_spark", 0.78, 0.65, 1.0),
    ("pink_spark", 0.93, 0.55, 1.0),
]

CROWN_COLORS = [
    ("gold_crown", 0.13, 0.75, 1.0),
    ("red_crown", 0.0, 0.70, 1.0),
    ("orange_crown", 0.08, 0.70, 1.0),
    ("green_crown", 0.38, 0.65, 1.0),
    ("cyan_crown", 0.52, 0.75, 1.0),
    ("blue_crown", 0.60, 0.70, 1.0),
    ("purple_crown", 0.78, 0.65, 1.0),
    ("pink_crown", 0.93, 0.55, 1.0),
]

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  MINECRAFT SKIN ANCESTRY TREE GENERATOR")
    print("=" * 60)
    
    base_img = Image.open(ORIGINAL).convert("RGBA")
    base = np.array(base_img)
    
    # UNIFY SHADING: The original creator shaded the left side of the body slightly darker.
    # To prevent the left arm looking like a different color after saturation adjustment,
    # map the left-side shades exactly to the right-side shades.
    r,g,b = base[:,:,0], base[:,:,1], base[:,:,2]
    m = (r==17)&(g==17)&(b==17); base[m,:3] = 20
    m = (r==28)&(g==28)&(b==28); base[m,:3] = 33
    m = (r==39)&(g==39)&(b==39); base[m,:3] = 46
    m = (r==189)&(g==189)&(b==189); base[m,:3] = 223
    m = (r==21)&(g==21)&(b==21); base[m,:3] = 20
    m = (r==36)&(g==36)&(b==36); base[m,:3] = 33
    m = (r==23)&(g==23)&(b==23); base[m,:3] = 25
    m = (r==198)&(g==198)&(b==198); base[m,:3] = 233
    m = (r==209)&(g==209)&(b==209); base[m,:3] = 246
    
    print("\nPre-computing masks...")
    classes = classify_image(base)
    
    # Base body regions (where actual shirt pixels live)
    suit_base_rnames = [k for k in REGIONS if ('body' in k or 'arm' in k or 'leg' in k) and 'overlay' not in k]
    # Overlay body regions (dark+mid greyscale = suit detail; bright greyscale = subtle highlights, leave alone)
    suit_overlay_rnames = [k for k in REGIONS if ('body' in k or 'arm' in k or 'leg' in k) and 'overlay' in k]
    head_rnames = [k for k in REGIONS if 'head' in k]
    head_base_rnames = [k for k in REGIONS if 'head' in k and 'overlay' not in k]
    head_overlay_rnames = [k for k in REGIONS if 'head_overlay' in k]
    
    suit_base_region = build_region_mask(suit_base_rnames)
    suit_overlay_region = build_region_mask(suit_overlay_rnames)
    head_region = build_region_mask(head_rnames)
    head_base_region = build_region_mask(head_base_rnames)
    head_overlay_region = build_region_mask(head_overlay_rnames)
    
    # On base layers: suit_dark + suit_light (leave white shirt pixels alone)
    # On overlay layers: suit_dark + suit_light only (bright highlights stay grey for subtle detail)
    suit_mask = (classes['suit_dark'] | classes['suit_light']) & (suit_base_region | suit_overlay_region)
    # The original tie is red, making the tie mask catch a red pixel on the crown. Restrict it to the body.
    tie_mask = classes['tie'] & (suit_base_region | suit_overlay_region)
    
    # Bounding box for the white mask's dark eyes/features so they don't get colored as hair
    mask_box = np.zeros((64, 64), dtype=bool)
    mask_box[8:16, 44:48] = True
    
    # Hair includes base head layer, plus head overlay (sideburns, bands) except for the mask box
    hair_mask = (classes['suit_dark'] | classes['suit_light']) & (head_base_region | (head_overlay_region & ~mask_box))
    
    # Spark is on the body, Crown is on the head overlay
    base_accents = (classes['gold'] | classes['cyan'] | (classes['tie'] & head_overlay_region))
    spark_mask = base_accents & (suit_base_region | suit_overlay_region)
    crown_mask = base_accents & head_overlay_region
    
    # Pre-compute mask data for fast transforms
    suit_md = MaskData(suit_mask, base)
    tie_md = MaskData(tie_mask, base)
    hair_md = MaskData(hair_mask, base)
    spark_md = MaskData(spark_mask, base)
    crown_md = MaskData(crown_mask, base)
    
    ns = len(SUIT_COLORS)
    nt = len(TIE_COLORS)
    nh = len(HAIR_COLORS)
    ne = len(EYE_COLORS)
    nsp = len(SPARK_COLORS)
    ncr = len(CROWN_COLORS)
    
    total_leaves = ns * nt * nh * ne * nsp * ncr
    
    print(f"\nTree: {ns} suits × {nt} ties × {nh} hair × {ne} eyes × {nsp} sparks × {ncr} crowns")
    print(f"Total full combos: {total_leaves:,}")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Save original
    os.makedirs(os.path.join(OUTPUT_DIR, "00_original"), exist_ok=True)
    Image.fromarray(base, 'RGBA').save(os.path.join(OUTPUT_DIR, "00_original", "skin.png"))
    
    start = time.time()
    count = 0
    
    for si, (sname, shue, ssat, sbright) in enumerate(SUIT_COLORS):
        t0 = time.time()
        
        # Level 1: Apply suit color
        suit_img = apply_colorize(base, suit_md, shue, ssat, sbright)
        suit_path = os.path.join(OUTPUT_DIR, sname)
        os.makedirs(suit_path, exist_ok=True)
        Image.fromarray(suit_img, 'RGBA').save(os.path.join(suit_path, f"{sname}-skin.png"))
        count += 1
        
        for ti, (tname, thue, tsat, tbright) in enumerate(TIE_COLORS):
            # Level 2: Apply tie
            tie_img = suit_img.copy()
            apply_recolor_tie(tie_img, tie_md, base, thue, tsat, tbright)
            tie_path = os.path.join(suit_path, tname)
            os.makedirs(tie_path, exist_ok=True)
            Image.fromarray(tie_img, 'RGBA').save(os.path.join(tie_path, f"{sname}-{tname}-skin.png"))
            count += 1
            
            for hi, (hname, hhue, hsat, hbright) in enumerate(HAIR_COLORS):
                # Level 3: Apply hair
                hair_img = tie_img.copy()
                # Need to recalc hair brightness from base since hair region
                # might overlap with suit region classification
                hair_v = np.clip(hair_md.base_v * hbright, 0, 1)
                if hair_md.n > 0:
                    hsv = np.column_stack([
                        np.full(hair_md.n, hhue, dtype=np.float32),
                        np.full(hair_md.n, hsat, dtype=np.float32),
                        hair_v
                    ])
                    new_rgb = (hsv_to_rgb_vec(hsv) * 255).astype(np.uint8)
                    hair_img[hair_md.ys, hair_md.xs, :3] = new_rgb
                
                hair_path = os.path.join(tie_path, hname)
                os.makedirs(hair_path, exist_ok=True)
                Image.fromarray(hair_img, 'RGBA').save(os.path.join(hair_path, f"{sname}-{tname}-{hname}-skin.png"))
                count += 1
                
                for ei, (ename, emain, edark) in enumerate(EYE_COLORS):
                    # Level 4: Apply eyes
                    eye_img = hair_img.copy()
                    apply_eyes(eye_img, emain, edark)
                    eye_path = os.path.join(hair_path, ename)
                    os.makedirs(eye_path, exist_ok=True)
                    Image.fromarray(eye_img, 'RGBA').save(os.path.join(eye_path, f"{sname}-{tname}-{hname}-{ename}-skin.png"))
                    count += 1
                    
                    for sp_i, (spname, sphue, spsat, spbright) in enumerate(SPARK_COLORS):
                        # Level 5: Apply spark
                        spark_img = eye_img.copy()
                        apply_accents(spark_img, spark_md, base, sphue, spsat, spbright)
                        spark_path = os.path.join(eye_path, spname)
                        os.makedirs(spark_path, exist_ok=True)
                        
                        for cr_i, (crname, crhue, crsat, crbright) in enumerate(CROWN_COLORS):
                            # Level 6: Apply crown
                            crown_img = spark_img.copy()
                            apply_accents(crown_img, crown_md, base, crhue, crsat, crbright)
                            crown_path = os.path.join(spark_path, crname)
                            os.makedirs(crown_path, exist_ok=True)
                            Image.fromarray(crown_img, 'RGBA').save(os.path.join(crown_path, f"{sname}-{tname}-{hname}-{ename}-{spname}-{crname}-skin.png"))
                            count += 1
        
        elapsed = time.time() - start
        suit_time = time.time() - t0
        rate = count / elapsed
        remaining = total_all - count
        eta = remaining / rate if rate > 0 else 0
        pct = count / total_all * 100
        print(f"  [{pct:5.1f}%] {sname:22s} | {count:>8,}/{total_all:,} | {rate:,.0f}/sec | ETA {eta/60:.1f}min | suit took {suit_time:.1f}s")
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  DONE! {count:,} skins in {elapsed:.1f}s ({count/elapsed:,.0f}/sec)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 60}")
    print(f"\n  Example paths:")
    print(f"    {OUTPUT_DIR}/navy_suit/navy_suit-skin.png")
    print(f"    {OUTPUT_DIR}/navy_suit/gold_tie/navy_suit-gold_tie-skin.png")
    print(f"    {OUTPUT_DIR}/navy_suit/gold_tie/blonde_hair/navy_suit-gold_tie-blonde_hair-skin.png")
    print(f"    {OUTPUT_DIR}/navy_suit/gold_tie/blonde_hair/blue_eyes/navy_suit-gold_tie-blonde_hair-blue_eyes-skin.png")
    print(f"    {OUTPUT_DIR}/navy_suit/gold_tie/blonde_hair/blue_eyes/cyan_accent/navy_suit-gold_tie-blonde_hair-blue_eyes-cyan_accent-skin.png")

if __name__ == "__main__":
    main()
