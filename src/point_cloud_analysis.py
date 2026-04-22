import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD TRC
# ============================================================

def load_trc(trc_path):
    with open(trc_path, "r") as f:
        lines = f.readlines()

    marker_line = lines[3].split("\t")[2:]
    marker_names = [m.strip().split(":")[-1] for m in marker_line if m.strip() != ""]

    df = pd.read_csv(trc_path, sep="\t", skiprows=5, header=None)
    df = df.dropna(axis=1, how="all")

    columns = ["Frame", "Time"]
    for m in marker_names:
        columns += [f"{m}_X", f"{m}_Y", f"{m}_Z"]

    df = df.iloc[:, :len(columns)]
    df.columns = columns

    return df, marker_names


# ============================================================
# METRICS
# ============================================================

def compute_metrics(df, marker):
    X = df[f"{marker}_X"].values
    Y = df[f"{marker}_Y"].values
    Z = df[f"{marker}_Z"].values

    coords = np.vstack((X, Y, Z)).T

    # displacement
    displacement = np.linalg.norm(coords[-1] - coords[0])

    # path length
    diffs = np.diff(coords, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    # spatial spread
    spread = np.mean(np.linalg.norm(coords - coords.mean(axis=0), axis=1))

    return displacement, path_length, spread


# ============================================================
# ANALYSIS
# ============================================================

def analyze_markers(df, markers):
    results = []

    for m in markers:
        if f"{m}_X" in df.columns:
            disp, path, spread = compute_metrics(df, m)
            results.append([m, disp, path, spread])

    res_df = pd.DataFrame(results, columns=["Marker", "Displacement", "PathLength", "Spread"])

    # normalize and compute involvement score
    for col in ["Displacement", "PathLength", "Spread"]:
        res_df[col] = res_df[col] / res_df[col].max()

    res_df["InvolvementScore"] = res_df[["Displacement", "PathLength", "Spread"]].mean(axis=1)

    return res_df.sort_values(by="InvolvementScore", ascending=False)


# ============================================================
# PLOT POINT CLOUD
# ============================================================

def plot_point_cloud(df, markers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for m in markers:
        if f"{m}_X" in df.columns:
            ax.scatter(df[f"{m}_X"], df[f"{m}_Y"], df[f"{m}_Z"], s=1, label=m)

    ax.set_title("Nuage de points du mouvement")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.legend()
    plt.show()


# ============================================================
# BAR PLOT
# ============================================================

def plot_involvement(res_df):
    plt.figure(figsize=(10,5))
    plt.bar(res_df["Marker"], res_df["InvolvementScore"])
    plt.xticks(rotation=90)
    plt.title("Score d'implication des marqueurs")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    path = os.path.join("data", "viconRecord.TRC")

    df, markers = load_trc(path)

    print("Analyzing markers...")
    res_df = analyze_markers(df, markers)

    print(res_df.head())

    res_df.to_csv("marker_analysis.csv", index=False)

    # Select key markers (clean visualization)
    selected = ["RSHO","RELB","RWRA","LSHO","C7","T10","CLAV","STRN","LASI","RASI"]

    plot_point_cloud(df, selected)
    plot_involvement(res_df)


if __name__ == "__main__":
    main()