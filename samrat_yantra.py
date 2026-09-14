"""
Samrat Yantra — Ujjain (lat 23.18 N)
Parametric printable model.

Coordinate system used throughout:
  origin = south tip of the gnomon base, on the midplane
  +X east, +Y north, +Z up
  base plate top surface = z = 0
"""
import math, numpy as np, trimesh
from shapely.geometry import Polygon

# ---------------- parameters ----------------
PHI_DEG   = 23.18          # latitude of Ujjain
OBLIQ_DEG = 23.44          # max solar declination

L         = 221.0          # hypotenuse length, mm
T_GN      = 20.0           # gnomon wall thickness
R         = 45.0           # dial cylinder radius
A1        = None           # band start along axis (derived)
BAND_W    = 35.0           # band width along axis
CLEAR     = 20.0           # lowest band point above base plate

PLATE_T   = 6.0
PLATE_MX  = 68.0           # widened for the quadrant rim
TENON     = 3.0            # gnomon tenon depth into plate
GROOVE_W  = 20.5           # plate groove width (0.5 clearance)
SPLIT_Y   = 100.0          # base plate joint

PIN_D     = 5.0
PIN_DEPTH = 5.0

HOUR_W    = 1.0            # hour groove width, mm on the surface
HOUR_D    = 1.0            # hour groove depth, mm
TEN_W     = 0.6
TEN_D     = 0.7
TEN_FRAC  = 0.40           # 10-min ticks run this fraction of band width

R_LIP     = 16.0           # lip radius, tangent to the dial at H=+/-90
LIP_OUT   = 3.0            # how far the lip travels outward before the flat
TOP_W     = 7.0            # width of the blunt flat top
NLIP      = 14
NH        = 1801           # angular samples per quadrant

phi = math.radians(PHI_DEG)
S, C = math.sin(phi), math.cos(phi)

A1 = (CLEAR + R * C) / S
A2 = A1 + BAND_W
BASE_RUN = L * C
H_GN = L * S
RESERVE = L - A2
NEEDED = R * math.tan(math.radians(OBLIQ_DEG))

print(f"phi            {PHI_DEG} deg   sin={S:.6f} cos={C:.6f}")
print(f"L              {L:.2f} mm")
print(f"base run       {BASE_RUN:.2f} mm")
print(f"gnomon height  {H_GN:.2f} mm")
print(f"A1             {A1:.3f} mm")
print(f"A2             {A2:.3f} mm")
print(f"tip height     {(A1+A2)/2*S:.2f} mm")
print(f"lowest band z  {A1*S - R*C:.3f} mm")
print(f"summer reserve {RESERVE:.2f} mm  (needs {NEEDED:.2f})  spare {RESERVE-NEEDED:.2f}")

A_BREAK = A1 + TEN_FRAC * BAND_W
hour_ang = math.degrees(HOUR_W / R)
ten_ang  = math.degrees(TEN_W / R)

HOURS = [k * 15.0 for k in range(-6, 7)]
TENS  = [k * 2.5 for k in range(-36, 37)]


def groove_depth(hdeg, a):
    d = 0.0
    for k in HOURS:
        if abs(hdeg - k) <= hour_ang / 2:
            d = max(d, HOUR_D)
    if a <= A_BREAK:
        for k in TENS:
            if abs(hdeg - k) <= ten_ang / 2:
                d = max(d, TEN_D)
    return d


def quadrant(side):
    """side = -1 west (H from -90..0), +1 east (H from 0..+90)."""
    x0 = side * T_GN / 2.0
    th_max = math.degrees(math.acos(1 - LIP_OUT / R_LIP))
    lip = []
    for th in np.linspace(th_max, 0.0, NLIP)[:-1]:
        t = math.radians(th)
        lip.append((R + R_LIP * (1 - math.cos(t)), R_LIP * math.sin(t)))
    lip = [(R + LIP_OUT + TOP_W, R_LIP * math.sin(math.radians(th_max)))] + lip
    if side < 0:
        Hs = list(np.linspace(-90.0, 0.0, NH))
        cols = [(x0 - d, 0.0, None, dz) for d, dz in lip] + \
               [(x0 + R*math.sin(math.radians(h)),
                 math.cos(math.radians(h)), h, 0.0) for h in Hs]
    else:
        Hs = list(np.linspace(0.0, 90.0, NH))
        cols = [(x0 + R*math.sin(math.radians(h)),
                 math.cos(math.radians(h)), h, 0.0) for h in Hs] + \
               [(x0 + d, 0.0, None, dz) for d, dz in reversed(lip)]
    As = np.array([A1, A_BREAK - 0.05, A_BREAK + 0.05, A2])
    nH, na = len(cols), len(As)

    top = np.zeros((nH, na, 3))
    bot = np.zeros((nH, na, 3))
    for i, (xi, ch, hd, dz) in enumerate(cols):
        for j, a in enumerate(As):
            y = a * C + R * S * ch
            z = a * S - R * C * ch + dz
            top[i, j] = (xi, y, z - (0.0 if hd is None else groove_depth(hd, a)))
            bot[i, j] = (xi, y, 0.0)

    verts = np.vstack([top.reshape(-1, 3), bot.reshape(-1, 3)])
    N = nH * na

    def ti(i, j):   return i * na + j
    def bi(i, j):   return N + i * na + j

    faces = []
    for i in range(nH - 1):
        for j in range(na - 1):
            faces += [[ti(i, j), ti(i + 1, j), ti(i + 1, j + 1)],
                      [ti(i, j), ti(i + 1, j + 1), ti(i, j + 1)]]
            faces += [[bi(i, j), bi(i + 1, j + 1), bi(i + 1, j)],
                      [bi(i, j), bi(i, j + 1), bi(i + 1, j + 1)]]
    for i in range(nH - 1):                      # a = A1 and a = A2 edges
        for j in (0, na - 1):
            faces += [[ti(i, j), bi(i, j), bi(i + 1, j)],
                      [ti(i, j), bi(i + 1, j), ti(i + 1, j)]]
    for j in range(na - 1):                      # H end caps
        for i in (0, nH - 1):
            faces += [[ti(i, j), bi(i, j), bi(i, j + 1)],
                      [ti(i, j), bi(i, j + 1), ti(i, j + 1)]]

    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    m.fix_normals()

    holes = []
    for hd in (-72.0, -18.0):
        hd = hd if side < 0 else -hd
        h = math.radians(hd)
        am = (A1 + A2) / 2
        px = x0 + R * math.sin(h)
        py = am * C + R * S * math.cos(h)
        cyl = trimesh.creation.cylinder(radius=PIN_D / 2 + 0.15, height=PIN_DEPTH * 2)
        cyl.apply_translation((px, py, 0.0))
        holes.append(cyl)
    return trimesh.boolean.difference([m] + holes)


def gnomon():
    poly = Polygon([(0, -TENON), (BASE_RUN, -TENON), (BASE_RUN, H_GN), (0, 0)])
    m = trimesh.creation.extrude_polygon(poly, T_GN)
    M = np.eye(4)
    M[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    m.apply_transform(M)
    m.apply_translation((-T_GN / 2, 0, 0))
    return m


def plate(half):
    y0, y1 = (-3.0, SPLIT_Y) if half == "south" else (SPLIT_Y, BASE_RUN + 3.0)
    box = trimesh.creation.box(extents=(2 * PLATE_MX, y1 - y0, PLATE_T))
    box.apply_translation((0, (y0 + y1) / 2, -PLATE_T / 2))

    cuts = []
    g = trimesh.creation.box(extents=(GROOVE_W, y1 - y0 + 4, TENON * 2))
    g.apply_translation((0, (y0 + y1) / 2, 0))
    cuts.append(g)

    if half == "north":
        for side in (-1, 1):
            x0 = side * T_GN / 2
            for hd in (-72.0, -18.0):
                hd = hd if side < 0 else -hd
                h = math.radians(hd)
                am = (A1 + A2) / 2
                px = x0 + R * math.sin(h)
                py = am * C + R * S * math.cos(h)
                cyl = trimesh.creation.cylinder(radius=PIN_D / 2 + 0.15,
                                                height=PIN_DEPTH * 2)
                cyl.apply_translation((px, py, -PIN_DEPTH))
                cuts.append(cyl)
    return trimesh.boolean.difference([box] + cuts)


def pin():
    return trimesh.creation.cylinder(radius=PIN_D / 2 - 0.1, height=PIN_DEPTH * 2 - 0.6)


parts = {
    "quadrant_west": quadrant(-1),
    "quadrant_east": quadrant(+1),
    "baseplate_south": plate("south"),
    "baseplate_north": plate("north"),
    "pin": pin(),
}

import os
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
for name, m in parts.items():
    print(f"{name:18s} watertight={str(m.is_watertight):5s} "
          f"vol={m.volume/1000:8.2f} cm3  tris={len(m.faces):6d}  "
          f"bbox={np.round(m.extents,1)}")
    m.export(f"/mnt/user-data/outputs/{name}.stl")

asm = trimesh.util.concatenate([parts[k] for k in
      ("quadrant_west", "quadrant_east",
       "baseplate_south", "baseplate_north")] +
      [trimesh.load("/mnt/user-data/outputs/gnomon_half_west.stl"),
       trimesh.load("/mnt/user-data/outputs/gnomon_half_east.stl")])
asm.export("/mnt/user-data/outputs/assembled_preview.stl")
print("assembly bbox", np.round(asm.extents, 1))

print("\nWest quadrant hour lines, lower endpoints (x, y, z):")
for k in range(-6, 1):
    hd = k * 15.0
    h = math.radians(hd)
    x = -T_GN / 2 + R * math.sin(h)
    y = A1 * C + R * S * math.cos(h)
    z = A1 * S - R * C * math.cos(h)
    print(f"  H={hd:+7.1f}  {12+k:2d}:00   {x:8.2f} {y:8.2f} {z:8.2f}")
print(f"upper offset = (0, {BAND_W*C:.2f}, {BAND_W*S:.2f})")
