import trimesh
import numpy as np
import pandas as pd
import os
import re

from scipy.signal import savgol_filter, argrelextrema
from scipy.sparse import coo_matrix, identity

# =========================================================
# PATHS
# =========================================================

MESH_DIR = "../meshes/"
ANTHRO_FILE = "../anthropometry.xlsx"

RESULT_DIR = "../results/"
OBJ_DIR = os.path.join(RESULT_DIR, "colored_meshes")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OBJ_DIR, exist_ok=True)

# =========================================================
# LOAD ANTHROPOMETRY
# =========================================================

anthro = pd.read_excel(ANTHRO_FILE)

# =========================================================
# FUNCTIONS
# =========================================================

def laplacian_smoothing(mesh, lamb=0.3, iterations=10):

    V = mesh.vertices.copy()
    F = mesh.faces
    n = len(V)

    I, J = [], []

    for tri in F:
        for i in range(3):
            I.append(tri[i])
            J.append(tri[(i+1) % 3])

            I.append(tri[i])
            J.append(tri[(i+2) % 3])

    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    A = A.maximum(A.T)

    D = np.array(A.sum(axis=1)).flatten()

    D_inv = 1.0 / D
    D_inv[np.isinf(D_inv)] = 0

    L = identity(n) - coo_matrix(
        (D_inv, (range(n), range(n))),
        shape=(n, n)
    ) @ A

    V_smooth = V.copy()

    for _ in range(iterations):
        V_smooth = V_smooth - lamb * (L @ V_smooth)

    mesh.vertices = V_smooth

    return mesh


def extract_longest_loop_3d(section):

    if section is None:
        return None

    curves = []

    for e in section.entities:
        if hasattr(e, "points"):
            curves.append(section.vertices[e.points])

    if len(curves) == 0:
        return None

    lengths = [
        np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))
        for c in curves
    ]

    return curves[np.argmax(lengths)]


def color_vertices(mesh, curve_points, color=[255,0,0,255]):

    verts = mesh.vertices

    if mesh.visual.vertex_colors is None:
        mesh.visual.vertex_colors = np.tile(
            [200,200,200,255],
            (len(verts),1)
        )

    for p in curve_points:

        d = np.linalg.norm(verts - p, axis=1)

        idx = np.argmin(d)

        mesh.visual.vertex_colors[idx] = color

    return mesh


# =========================================================
# MAIN LOOP
# =========================================================

results = []
correction_factor = 4.0

mesh_files = [
    f for f in os.listdir(MESH_DIR)
    if f.endswith(".obj")
]

for filename in mesh_files:

    print("Processing:", filename)

    try:

        # =================================================
        # ID
        # =================================================

        #id_model = re.search(r"(B\d+)", filename).group(1)
        id_model = filename.replace("_clean.obj", "")

        row = anthro[anthro["ID"] == id_model]

        if len(row) == 0:
            print("Anthropometry not found")
            continue

        height_real = float(row["height"].values[0])
        waist_real  = float(row["waist"].values[0])
        hip_real    = float(row["hip"].values[0])

        # =================================================
        # LOAD MESH
        # =================================================

        mesh = trimesh.load(
            os.path.join(MESH_DIR, filename),
            process=False
        )

        mesh = laplacian_smoothing(mesh)

        vertices = mesh.vertices

        # =================================================
        # SCALE TO REAL HEIGHT
        # =================================================

        ranges = vertices.max(axis=0) - vertices.min(axis=0)

        long_axis = np.argmax(ranges)

        v = vertices[:, long_axis]

        mesh_height = v.max() - v.min()

        #corrected_height = height_real * 0.985
        scale = height_real / mesh_height

        mesh.apply_scale(scale)

        vertices = mesh.vertices

        # =================================================
        # SLICES
        # =================================================

        plane_normal = np.zeros(3)
        plane_normal[long_axis] = 1.0

        center = mesh.bounding_box.centroid

        s = vertices[:, long_axis]

        s_min, s_max = s.min(), s.max()

        num_slices = 500 #300

        slices = np.linspace(s_min, s_max, num_slices)

        def get_section(h):

            origin = center.copy()
            origin[long_axis] = h

            return mesh.section(
                plane_origin=origin,
                plane_normal=plane_normal
            )

        perimeters = []

        for h in slices:

            sec = get_section(h)

            if sec is None:
                perimeters.append(np.nan)
                continue

            planar, _ = sec.to_2D()

            perimeters.append(planar.length)

        perimeters = np.array(perimeters)

        # =================================================
        # SMOOTH
        # =================================================

        p_smooth = savgol_filter(
            perimeters,
            window_length=21,
            polyorder=3
        )

        min_idx = argrelextrema(
            p_smooth,
            np.less
        )[0]

        max_idx = argrelextrema(
            p_smooth,
            np.greater
        )[0]

        # torso

        i0 = int(0.35 * num_slices)
        i1 = int(0.65 * num_slices)

        min_idx = [i for i in min_idx if i0 < i < i1]
        max_idx = [i for i in max_idx if i0 < i < i1]

        center_idx = int(0.5 * num_slices)

        # =================================================
        # HIP DETECTION
        # =================================================

        search_start = int(0.48 * num_slices)
        search_end   = int(0.58 * num_slices)

        hip_candidates = [
            i for i in max_idx
            if search_start < i < search_end
        ]

        # -------------------------------------------------
        # CASE 1:
        # local maxima exist
        # -------------------------------------------------

        if len(hip_candidates) > 0:

            hip_idx = max(
                hip_candidates,
                key=lambda i: p_smooth[i]
            )

        # -------------------------------------------------
        # CASE 2:
        # use maximum perimeter in region
        # -------------------------------------------------

        else:

            local_region = p_smooth[search_start:search_end]

            hip_idx = (
                search_start +
                np.argmax(local_region)
            )


        # =================================================
        # WAIST DETECTION
        # =================================================

        # buscar ARRIBA de hip
        search_start = hip_idx + int(0.08*num_slices)
        search_end   = hip_idx + int(0.22*num_slices)

        # limitar rango válido
        search_end = min(search_end, num_slices-1)

        # mínimos locales en la región
        waist_candidates = [
            i for i in min_idx
            if search_start < i < search_end
        ]

        # -------------------------------------------------
        # CASE 1:
        # local minima exists
        # -------------------------------------------------

        if len(waist_candidates) > 0:

            waist_idx = min(
                waist_candidates,
                key=lambda i: p_smooth[i]
            )

        # -------------------------------------------------
        # CASE 2:
        # use minimum of region
        # -------------------------------------------------

        else:

            local_region = p_smooth[search_start:search_end]

            waist_idx = (
                search_start +
                np.argmin(local_region)
            )

        # -------------------------------------------------

        # =================================================
        # HEIGHT POSITIONS
        # =================================================

        waist_h = slices[waist_idx]
        hip_h   = slices[hip_idx]

        # -------------------------------------------------
        # MOVE WAIST - correction factor
        # -------------------------------------------------

        waist_h = waist_h - correction_factor

        # =================================================
        # RECALCULATE WAIST PERIMETER
        # =================================================

        waist_section = get_section(waist_h)

        if waist_section is not None:
            waist_planar, _ = waist_section.to_2D()
            waist_3d = waist_planar.length

        else:
            waist_3d = np.nan

        # =================================================
        # HIP PERIMETER
        # =================================================

        hip_3d = perimeters[hip_idx]

        # =================
        # correction factor
        # =================

        waist_3d = waist_3d + correction_factor
        hip_3d = hip_3d + correction_factor

        # =================================================
        # ERRORS
        # =================================================

        waist_error = waist_3d - waist_real
        hip_error   = hip_3d - hip_real

        waist_abs_error = abs(waist_error)
        hip_abs_error   = abs(hip_error)

        waist_pct_error = (
            waist_abs_error / waist_real
        ) * 100

        hip_pct_error = (
            hip_abs_error / hip_real
        ) * 100

        # =================================================
        # RED CURVES
        # =================================================

        waist_curve = extract_longest_loop_3d(
            get_section(waist_h)
        )

        hip_curve = extract_longest_loop_3d(
            get_section(hip_h)
        )

        mesh = color_vertices(mesh, waist_curve, color=[255,0,0,255]) # waist red
        mesh = color_vertices(mesh, hip_curve, color=[0,0,255,255]) # hip blue

        out_obj = os.path.join(
            OBJ_DIR,
            f"{id_model}_colored.obj"
        )

        mesh.export(out_obj)

        # =================================================
        # SAVE RESULTS
        # =================================================

        results.append({

            "ID": id_model,

            "height_real": height_real,

            "waist_real": waist_real,
            "waist_3d": waist_3d,
            "waist_error": waist_error,
            "waist_abs_error": waist_abs_error,
            "waist_pct_error": waist_pct_error,

            "hip_real": hip_real,
            "hip_3d": hip_3d,
            "hip_error": hip_error,
            "hip_abs_error": hip_abs_error,
            "hip_pct_error": hip_pct_error
        })

    except Exception as e:

        print("ERROR:", filename)
        print(e)

# =========================================================
# EXPORT EXCEL
# =========================================================

df = pd.DataFrame(results)

df.to_excel(
    os.path.join(
        RESULT_DIR,
        "body2vec_measurements.xlsx"
    ),
    index=False
)

print("DONE")
