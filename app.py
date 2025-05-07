"""
app.py — crash-proof preview + accurate quote backend
=====================================================
• Shell volume uses nozzle diameter, not layer thickness.
• Pricing logic updated to match your calculate_cost() exactly.
"""
import io, math, logging, tempfile, os
from typing import Optional

import numpy as np
import trimesh
from flask import Flask, request, jsonify, render_template, Response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB
log = app.logger; log.setLevel(logging.INFO)

# ── Constants ────────────────────────────────────────────────────
PRICE_PER_KG       = 80.0         # SAR per kg plastic
DEFAULT_FILL       = 0.10         # 10 % infill
DEFAULT_LAYER_H    = 0.2          # mm
NOZZLE_DIAMETER    = 0.4          # mm
RATE_CM3_H         = 15.0         # cm³ printed per hour
PLA_RHO            = 1.24         # g/cm³
FACE_CAP           = 25_000       # preview decimation target
OPEN3D_SIZE_LIMIT  = 120 * 1024**2  # skip Open3D on >120 MB
SUPPORT_EXTRA_TIME = 0.20         # +20 % time for supports
IRONING_TIME_MULT  = 0.10         # +10 % time for ironing

# ── Helpers ──────────────────────────────────────────────────────
def flatten(obj) -> Optional[trimesh.Trimesh]:
    try:
        while True:
            if isinstance(obj, trimesh.Trimesh): return obj
            if isinstance(obj, trimesh.Scene):
                obj = obj.dump(); continue
            if isinstance(obj, (list, tuple)):
                obj = [o for o in obj if o]
                if not obj: return None
                if len(obj) == 1: obj = obj[0]; continue
                obj = trimesh.util.concatenate(obj); continue
            return None
    except Exception:
        return None

def stride(mesh: trimesh.Trimesh, limit: int = FACE_CAP) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    tgt   = max(1, limit // len(parts))
    out   = []
    for p in parts:
        step = max(1, int(p.faces.shape[0] / tgt))
        mask = np.arange(p.faces.shape[0]) % step == 0
        out.append(p.submesh([mask], only_watertight=False))
    return flatten(trimesh.util.concatenate(out))

def fast_bbox(blob: bytes) -> np.ndarray:
    try:
        v = np.frombuffer(blob, np.float32).reshape(-1,3)
        return v.max(0) - v.min(0)
    except Exception:
        return np.array([1.0,1.0,1.0])

def quick_stats(blob: bytes, ftyp: str) -> dict:
    mesh = flatten(trimesh.load(io.BytesIO(blob), file_type=ftyp, force='mesh'))
    if mesh is None:
        size = fast_bbox(blob)
        vol  = size.prod() * 1e6 * 0.42
    else:
        if not mesh.is_watertight: mesh = mesh.fill_holes()
        vol = mesh.volume / 1e3
    weight = vol * PLA_RHO
    dims   = [round(d,2) for d in (mesh.extents if mesh else fast_bbox(blob))]
    return dict(volume_cm3=round(vol,2),
                solid_weight_g=round(weight,2),
                dims_mm=dims)

# ── /api/preview ─────────────────────────────────────────────────
@app.route("/api/preview", methods=["POST"])
def preview():
    blob  = request.files["file"].read()
    name  = request.files["file"].filename.lower()
    ftyp  = "obj" if name.endswith(".obj") else "stl"
    big   = len(blob) > OPEN3D_SIZE_LIMIT

    mesh = None if big else flatten(
        trimesh.load(io.BytesIO(blob), file_type=ftyp, force='mesh'))

    # A) huge or unparseable → voxel boxes
    if big or mesh is None:
        size  = fast_bbox(blob); pitch = max(size) / 80
        vox   = trimesh.load(io.BytesIO(blob), file_type=ftyp,
                             force='mesh', skip_mass_properties=True
                            ).voxelized(pitch)
        mesh  = stride(vox.as_boxes().merge_vertices())
        log.info("Preview: voxel boxes (big)")
        return Response(mesh.export(file_type=ftyp),
                        headers={"Content-Type":"application/octet-stream"})

    # B) Open3D decimate (≤120 MB)
    if mesh.faces.shape[0] > FACE_CAP:
        try:
            import open3d as o3d
            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=".stl" if ftyp=="stl" else ".obj")
            tmp.write(blob); tmp.close()
            o_mesh = o3d.io.read_triangle_mesh(tmp.name,
                                               enable_post_processing=False)
            os.unlink(tmp.name)
            if len(o_mesh.triangles) > 250_000:
                vs     = max(o_mesh.get_max_bound()-o_mesh.get_min_bound()) / 200
                o_mesh = o_mesh.simplify_vertex_clustering(vs)
            o_mesh = o_mesh.simplify_quadric_decimation(FACE_CAP)
            o_mesh.remove_degenerate_triangles()
            o_mesh.remove_unreferenced_vertices()
            verts = np.asarray(o_mesh.vertices)
            faces = np.asarray(o_mesh.triangles)
            mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            log.info("Preview: Open3D")
        except Exception as e:
            log.warning("Open3D skipped: %s", e)
            mesh = None

    # C) fallback voxel boxes
    if mesh is None or mesh.faces.size == 0:
        size  = fast_bbox(blob); pitch = max(size) / 80
        vox   = trimesh.load(io.BytesIO(blob), file_type=ftyp,
                             force='mesh', skip_mass_properties=True
                            ).voxelized(pitch)
        mesh  = stride(vox.as_boxes().merge_vertices())
        log.info("Preview: voxel boxes fallback")

    return Response(mesh.export(file_type=ftyp),
                    headers={"Content-Type":"application/octet-stream"})

# ── /api/quote ───────────────────────────────────────────────────
@app.route("/api/quote", methods=["POST"])
def quote():
    blob       = request.files["file"].read()
    name       = request.files["file"].filename.lower()
    ftyp       = "obj" if name.endswith(".obj") else "stl"

    # parse form
    fill_ratio = float(request.form.get("fill_ratio",  DEFAULT_FILL))
    layer_h    = float(request.form.get("layer_height",DEFAULT_LAYER_H))
    supports   = request.form.get("supports") == "1"
    wall_count = int(request.form.get("wall_count", 2))
    ironing    = request.form.get("ironing") == "1"

    # reload mesh to get accurate geometry
    mesh = flatten(trimesh.load(io.BytesIO(blob),
                                file_type=ftyp, force='mesh'))
    if mesh is None:
        stats = quick_stats(blob, ftyp)
        vol   = stats["volume_cm3"]
        solid = stats["solid_weight_g"]
        area  = fast_bbox(blob).prod() * 6
    else:
        if not mesh.is_watertight: mesh = mesh.fill_holes()
        vol   = mesh.volume / 1e3
        solid = vol * PLA_RHO
        area  = mesh.area

    # infill
    infill_vol = vol * fill_ratio

    # shell
    shell_vol_cm3 = (area * NOZZLE_DIAMETER * wall_count) / 1000.0

    # supports
    support_vol = shell_vol_cm3 * 0.1 if supports else 0

    total_vol = infill_vol + shell_vol_cm3 + support_vol
    weight_g  = total_vol * PLA_RHO

    # print time
    time_h = total_vol / RATE_CM3_H
    time_h *= (DEFAULT_LAYER_H / layer_h)
    if supports: time_h *= (1 + SUPPORT_EXTRA_TIME)
    if ironing:  time_h *= (1 + IRONING_TIME_MULT)

    # ── PRICING LOGIC from your calculate_cost() ────────────────
    # round up hours to integer
    t = math.ceil(time_h)
    if t <= 0:
        time_cost = 0
    elif t == 1:
        time_cost = 5
    else:
        # exactly: 5*(t-1) + (t*2)
        time_cost = 5*(t-1) + (t*2)

    # material cost
    material_cost = PRICE_PER_KG * (weight_g / 1000.0)

    # flat fee
    flat_fee = 15

    total_cost = round(material_cost + time_cost + flat_fee, 2)

    return jsonify(
        volume_cm3       = round(vol,2),
        solid_weight_g   = round(solid,2),
        dims_mm          = [round(d,2) for d in mesh.extents] if mesh else stats["dims_mm"],
        fill_ratio       = fill_ratio,
        weight_g         = round(weight_g,2),
        time_h           = round(time_h,2),
        total            = total_cost
    )

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
