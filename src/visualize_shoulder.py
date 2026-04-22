import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. LOAD TRC FILE
# ============================================================

def load_trc(trc_path: str):
    """
    Load a Vicon .TRC file and return:
    - dataframe with numeric marker data
    - marker_names in order
    - metadata dict
    """
    if not os.path.exists(trc_path):
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    with open(trc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("TRC file seems too short or malformed.")

    # Metadata
    meta_header = lines[1].strip().split("\t")
    meta_values = lines[2].strip().split("\t")
    metadata = {}
    for k, v in zip(meta_header, meta_values):
        metadata[k] = v

    # Marker names are on line 4 (index 3)
    marker_line = lines[3].rstrip("\n").split("\t")
    xyz_line = lines[4].rstrip("\n").split("\t")

    # First two columns are Frame# and Time
    raw_marker_names = marker_line[2:]

    marker_names = []
    for name in raw_marker_names:
        if name.strip() != "":
            marker_names.append(name.strip())

    # Clean marker names: e.g. "get:LSHO" -> "LSHO"
    cleaned_marker_names = [name.split(":")[-1] for name in marker_names]

    # Data starts from line 6 (index 5)
    df = pd.read_csv(
        trc_path,
        sep="\t",
        skiprows=5,
        header=None,
        engine="python"
    )

    # Drop completely empty trailing columns
    df = df.dropna(axis=1, how="all")

    expected_cols = 2 + 3 * len(cleaned_marker_names)
    if df.shape[1] < expected_cols:
        raise ValueError(
            f"Unexpected number of columns. Got {df.shape[1]}, expected at least {expected_cols}."
        )

    # Build column names
    columns = ["Frame", "Time"]
    for marker in cleaned_marker_names:
        columns.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

    df = df.iloc[:, :len(columns)]
    df.columns = columns

    # Convert all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, cleaned_marker_names, metadata


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_marker(df: pd.DataFrame, frame_idx: int, marker: str):
    """
    Return np.array([x, y, z]) for a given marker at a frame.
    """
    row = df.iloc[frame_idx]
    return np.array([
        row[f"{marker}_X"],
        row[f"{marker}_Y"],
        row[f"{marker}_Z"]
    ], dtype=float)


def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    Return angle in degrees between vectors v1 and v2.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return np.nan
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_right_arm_metrics(df: pd.DataFrame):
    """
    Compute simple shoulder and elbow angles for the right arm.
    Uses:
    - trunk approx from C7 -> STRN
    - upper arm from RSHO -> RELB
    - forearm from RELB -> RWRA
    """
    required = ["C7", "STRN", "RSHO", "RELB", "RWRA"]
    for marker in required:
        for axis in ["X", "Y", "Z"]:
            col = f"{marker}_{axis}"
            if col not in df.columns:
                raise ValueError(f"Missing required marker column: {col}")

    shoulder_angles = []
    elbow_angles = []

    for i in range(len(df)):
        c7 = get_marker(df, i, "C7")
        strn = get_marker(df, i, "STRN")
        rsho = get_marker(df, i, "RSHO")
        relb = get_marker(df, i, "RELB")
        rwra = get_marker(df, i, "RWRA")

        trunk = c7 - strn
        upper_arm = relb - rsho
        forearm = rwra - relb

        shoulder_angle = angle_between(trunk, upper_arm)
        elbow_angle = angle_between(upper_arm, forearm)

        shoulder_angles.append(shoulder_angle)
        elbow_angles.append(elbow_angle)

    metrics = pd.DataFrame({
        "Frame": df["Frame"],
        "Time": df["Time"],
        "RightShoulderAngle_deg": shoulder_angles,
        "RightElbowAngle_deg": elbow_angles
    })

    return metrics


# ============================================================
# 3. STATIC PLOT
# ============================================================

def plot_frame(df: pd.DataFrame, frame_idx: int = 0):
    """
    Plot one 3D frame of upper-body markers.
    """
    markers_to_show = [
        "C7", "T10", "CLAV", "STRN",
        "LSHO", "LELB", "LWRA",
        "RSHO", "RELB", "RWRA",
        "LASI", "RASI"
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = [], [], []

    for marker in markers_to_show:
        if f"{marker}_X" in df.columns:
            p = get_marker(df, frame_idx, marker)
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
            ax.scatter(p[0], p[1], p[2], s=40)
            ax.text(p[0], p[1], p[2], marker, fontsize=8)

    # Simple segments
    segments = [
        ("CLAV", "C7"),
        ("CLAV", "STRN"),
        ("RSHO", "RELB"),
        ("RELB", "RWRA"),
        ("LSHO", "LELB"),
        ("LELB", "LWRA"),
        ("LASI", "RASI")
    ]

    for a, b in segments:
        if f"{a}_X" in df.columns and f"{b}_X" in df.columns:
            pa = get_marker(df, frame_idx, a)
            pb = get_marker(df, frame_idx, b)
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]]
            )

    ax.set_title(f"Upper-body markers - Frame {frame_idx}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Make axes roughly equal
    if xs and ys and zs:
        xmid = (max(xs) + min(xs)) / 2
        ymid = (max(ys) + min(ys)) / 2
        zmid = (max(zs) + min(zs)) / 2
        max_range = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs)
        ) / 2 + 50

        ax.set_xlim(xmid - max_range, xmid + max_range)
        ax.set_ylim(ymid - max_range, ymid + max_range)
        ax.set_zlim(zmid - max_range, zmid + max_range)

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. ANIMATION
# ============================================================

def animate_upper_body(df: pd.DataFrame, step: int = 10):
    """
    Animate upper-body motion.
    step=10 means show every 10th frame to make it lighter.
    """
    markers_to_show = [
        "C7", "T10", "CLAV", "STRN",
        "LSHO", "LELB", "LWRA",
        "RSHO", "RELB", "RWRA",
        "LASI", "RASI"
    ]

    segments = [
        ("CLAV", "C7"),
        ("CLAV", "STRN"),
        ("RSHO", "RELB"),
        ("RELB", "RWRA"),
        ("LSHO", "LELB"),
        ("LELB", "LWRA"),
        ("LASI", "RASI")
    ]

    frame_indices = list(range(0, len(df), step))

    # Precompute bounds
    all_x, all_y, all_z = [], [], []
    for marker in markers_to_show:
        if f"{marker}_X" in df.columns:
            all_x.extend(df[f"{marker}_X"].dropna().values.tolist())
            all_y.extend(df[f"{marker}_Y"].dropna().values.tolist())
            all_z.extend(df[f"{marker}_Z"].dropna().values.tolist())

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_num):
        ax.clear()
        idx = frame_indices[frame_num]

        # Plot markers
        for marker in markers_to_show:
            if f"{marker}_X" in df.columns:
                p = get_marker(df, idx, marker)
                ax.scatter(p[0], p[1], p[2], s=40)
                ax.text(p[0], p[1], p[2], marker, fontsize=8)

        # Plot segments
        for a, b in segments:
            if f"{a}_X" in df.columns and f"{b}_X" in df.columns:
                pa = get_marker(df, idx, a)
                pb = get_marker(df, idx, b)
                ax.plot(
                    [pa[0], pb[0]],
                    [pa[1], pb[1]],
                    [pa[2], pb[2]]
                )

        ax.set_title(f"Upper-body motion - Frame {idx}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

        xmid = (max(all_x) + min(all_x)) / 2
        ymid = (max(all_y) + min(all_y)) / 2
        zmid = (max(all_z) + min(all_z)) / 2
        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z)
        ) / 2 + 50

        ax.set_xlim(xmid - max_range, xmid + max_range)
        ax.set_ylim(ymid - max_range, ymid + max_range)
        ax.set_zlim(zmid - max_range, zmid + max_range)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=50,
        repeat=True
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. PLOT ANGLES
# ============================================================

def plot_metrics(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics["Time"], metrics["RightShoulderAngle_deg"], label="Right shoulder angle")
    ax.plot(metrics["Time"], metrics["RightElbowAngle_deg"], label="Right elbow angle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Right arm kinematic angles")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. EXPORT SIMPLE CSV FOR YOUR PROJECT
# ============================================================

def export_project_csv(df: pd.DataFrame, output_csv: str):
    """
    Export a simplified CSV with only the most useful markers for your project.
    """
    wanted_markers = ["C7", "STRN", "RSHO", "RELB", "RWRA", "LSHO", "LELB", "LWRA"]

    cols = ["Frame", "Time"]
    for marker in wanted_markers:
        for axis in ["X", "Y", "Z"]:
            col = f"{marker}_{axis}"
            if col in df.columns:
                cols.append(col)

    out = df[cols].copy()
    out.to_csv(output_csv, index=False)
    print(f"Saved simplified project CSV to: {output_csv}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    trc_path = os.path.join("data", "viconRecord.TRC")

    print("Loading TRC file...")
    df, markers, metadata = load_trc(trc_path)

    print("\nMetadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    print("\nMarkers found:")
    print(markers)

    print(f"\nNumber of frames: {len(df)}")
    print(f"Time duration: {df['Time'].iloc[-1]:.2f} s")

    # Show one static frame
    plot_frame(df, frame_idx=0)

    # Compute angles
    metrics = compute_right_arm_metrics(df)
    print("\nAngle summary:")
    print(metrics[["RightShoulderAngle_deg", "RightElbowAngle_deg"]].describe())

    # Plot angles
    plot_metrics(metrics)

    # Export simple CSV
    export_project_csv(df, "project_upperbody_markers.csv")

    # Animate
    animate_upper_body(df, step=10)


if __name__ == "__main__":
    main()