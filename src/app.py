
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Analyse cinématique de l'épaule",
    page_icon="🫀",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 1rem;
}
.small-muted {
    opacity: 0.75;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_trc_from_bytes(file_bytes: bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    if len(lines) < 6:
        raise ValueError("Fichier TRC invalide ou incomplet.")

    meta_header = lines[1].split("\t")
    meta_values = lines[2].split("\t")
    metadata = {}
    for k, v in zip(meta_header, meta_values):
        metadata[k.strip()] = v.strip()

    marker_line = lines[3].split("\t")[2:]
    marker_names = [m.strip().split(":")[-1] for m in marker_line if m.strip()]

    df = pd.read_csv(io.StringIO(text), sep="\t", skiprows=5, header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    columns = ["Frame", "Time"]
    for m in marker_names:
        columns.extend([f"{m}_X", f"{m}_Y", f"{m}_Z"])

    df = df.iloc[:, :len(columns)].copy()
    df.columns = columns

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, marker_names, metadata


def get_marker_xyz(df: pd.DataFrame, marker: str):
    return df[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].to_numpy(dtype=float)


def compute_marker_metrics(df: pd.DataFrame, marker: str):
    coords = get_marker_xyz(df, marker)
    coords = coords[~np.isnan(coords).any(axis=1)]
    if len(coords) < 2:
        return {
            "Marker": marker,
            "Displacement_mm": np.nan,
            "PathLength_mm": np.nan,
            "Spread_mm": np.nan,
        }

    displacement = float(np.linalg.norm(coords[-1] - coords[0]))
    diffs = np.diff(coords, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    center = np.mean(coords, axis=0)
    spread = float(np.mean(np.linalg.norm(coords - center, axis=1)))

    return {
        "Marker": marker,
        "Displacement_mm": displacement,
        "PathLength_mm": path_length,
        "Spread_mm": spread,
    }


def build_metrics_table(df: pd.DataFrame, markers: list[str]):
    rows = [compute_marker_metrics(df, m) for m in markers if f"{m}_X" in df.columns]
    out = pd.DataFrame(rows)
    score_cols = ["Displacement_mm", "PathLength_mm", "Spread_mm"]
    for col in score_cols:
        maxv = out[col].max()
        out[f"{col}_norm"] = out[col] / maxv if pd.notna(maxv) and maxv > 0 else 0.0
    out["InvolvementScore"] = out[[f"{c}_norm" for c in score_cols]].mean(axis=1)
    out = out.sort_values("InvolvementScore", ascending=False).reset_index(drop=True)
    return out


def simulate_disabled(df: pd.DataFrame, factor: float = 0.45):
    out = df.copy()
    required = ["RSHO", "RELB", "RWRA"]
    if not all(f"{m}_X" in out.columns for m in required):
        return out

    for axis in ["X", "Y", "Z"]:
        shoulder = out[f"RSHO_{axis}"]
        elbow = out[f"RELB_{axis}"]
        wrist = out[f"RWRA_{axis}"]

        out[f"RELB_{axis}"] = shoulder + factor * (elbow - shoulder)
        out[f"RWRA_{axis}"] = shoulder + factor * (wrist - shoulder)

    return out


def make_point_cloud_figure(df: pd.DataFrame, markers: list[str], sample_step: int = 5, point_size: int = 3, title: str = "Nuage de points"):
    fig = go.Figure()
    sub = df.iloc[::sample_step]
    for marker in markers:
        x_col, y_col, z_col = f"{marker}_X", f"{marker}_Y", f"{marker}_Z"
        if x_col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=sub[x_col],
                y=sub[y_col],
                z=sub[z_col],
                mode="markers",
                name=marker,
                marker=dict(size=point_size),
                hovertemplate=f"<b>{marker}</b><br>X=%{{x:.1f}} mm<br>Y=%{{y:.1f}} mm<br>Z=%{{z:.1f}} mm<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=45, b=10),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(orientation="v"),
    )
    return fig


def make_involvement_bar(metrics_df: pd.DataFrame, top_n: int = 12):
    sub = metrics_df.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=sub["InvolvementScore"],
            y=sub["Marker"],
            orientation="h",
            hovertemplate="<b>%{y}</b><br>Score=%{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} des marqueurs les plus impliqués",
        xaxis_title="Score d'implication",
        yaxis_title="Marqueur",
        margin=dict(l=10, r=10, t=45, b=10),
        height=420,
    )
    return fig


def make_comparison_figure(df_normal: pd.DataFrame, df_disabled: pd.DataFrame, markers: list[str], sample_step: int = 5):
    fig = go.Figure()
    normal = df_normal.iloc[::sample_step]
    limited = df_disabled.iloc[::sample_step]

    for marker in markers:
        x_col, y_col, z_col = f"{marker}_X", f"{marker}_Y", f"{marker}_Z"
        if x_col not in df_normal.columns:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=normal[x_col],
                y=normal[y_col],
                z=normal[z_col],
                mode="lines",
                name=f"{marker} - normal",
                hovertemplate=f"<b>{marker} normal</b><extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=limited[x_col],
                y=limited[y_col],
                z=limited[z_col],
                mode="lines",
                name=f"{marker} - limité",
                line=dict(dash="dot"),
                hovertemplate=f"<b>{marker} limité</b><extra></extra>",
            )
        )

    fig.update_layout(
        title="Comparaison : mouvement normal vs mouvement limité simulé",
        margin=dict(l=10, r=10, t=45, b=10),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        height=700,
    )
    return fig


st.sidebar.title("Paramètres")
uploaded = st.sidebar.file_uploader("Importer un fichier .TRC", type=["trc"])

preset = st.sidebar.selectbox(
    "Sélection rapide de marqueurs",
    [
        "Haut du corps",
        "Bras droit + tronc",
        "Bras gauche + tronc",
        "Tous les marqueurs",
        "Personnalisé",
    ],
)

sample_step = st.sidebar.slider("Sous-échantillonnage des points", 1, 20, 5, 1)
point_size = st.sidebar.slider("Taille des points", 2, 8, 3, 1)
show_disabled = st.sidebar.checkbox("Afficher une simulation de limitation", value=False)
disabled_factor = st.sidebar.slider("Facteur de limitation", 0.2, 0.9, 0.45, 0.05)
top_n = st.sidebar.slider("Nombre de marqueurs dans le classement", 5, 20, 12, 1)

st.title("Interface d'analyse cinématique de l'épaule")
st.caption("Upload d'un fichier TRC, visualisation du nuage de points, classement des marqueurs impliqués et comparaison avec une mobilité limitée simulée.")

if uploaded is None:
    st.info("Commence par importer un fichier `.TRC` dans la barre latérale.")
    st.stop()

try:
    file_bytes = uploaded.read()
    df, markers, metadata = load_trc_from_bytes(file_bytes)
except Exception as e:
    st.error(f"Impossible de lire le fichier : {e}")
    st.stop()

preset_map = {
    "Haut du corps": ["C7", "T10", "CLAV", "STRN", "LSHO", "LELB", "LWRA", "RSHO", "RELB", "RWRA", "LASI", "RASI"],
    "Bras droit + tronc": ["C7", "T10", "CLAV", "STRN", "RSHO", "RELB", "RWRA", "RFIN", "LASI", "RASI"],
    "Bras gauche + tronc": ["C7", "T10", "CLAV", "STRN", "LSHO", "LELB", "LWRA", "LFIN", "LASI", "RASI"],
    "Tous les marqueurs": markers,
}
default_markers = preset_map.get(preset, ["RSHO", "RELB", "RWRA", "C7", "CLAV", "STRN"])

selected_markers = st.sidebar.multiselect(
    "Marqueurs affichés",
    options=markers,
    default=[m for m in default_markers if m in markers] if preset != "Personnalisé" else markers[:10],
)

if not selected_markers:
    st.warning("Sélectionne au moins un marqueur.")
    st.stop()

metrics_df = build_metrics_table(df, markers)
df_disabled = simulate_disabled(df, factor=disabled_factor) if show_disabled else None

col1, col2, col3, col4 = st.columns(4)
duration = float(df["Time"].iloc[-1]) if "Time" in df.columns and len(df) else 0.0
top_marker = metrics_df.iloc[0]["Marker"] if len(metrics_df) else "N/A"
low_marker = metrics_df.iloc[-1]["Marker"] if len(metrics_df) else "N/A"

col1.metric("Frames", f"{len(df)}")
col2.metric("Durée", f"{duration:.2f} s")
col3.metric("Marqueur le plus impliqué", top_marker)
col4.metric("Marqueur le moins impliqué", low_marker)

with st.expander("Métadonnées du fichier"):
    st.json(metadata)

left, right = st.columns([1.45, 1])

with left:
    fig = make_point_cloud_figure(
        df if not show_disabled else df_disabled,
        selected_markers,
        sample_step=sample_step,
        point_size=point_size,
        title="Nuage de points du mouvement" if not show_disabled else "Nuage de points — mouvement limité simulé",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.plotly_chart(make_involvement_bar(metrics_df, top_n=top_n), use_container_width=True)

st.subheader("Tableau des indicateurs par marqueur")
display_df = metrics_df[["Marker", "Displacement_mm", "PathLength_mm", "Spread_mm", "InvolvementScore"]].copy()
display_df = display_df.rename(columns={
    "Marker": "Marqueur",
    "Displacement_mm": "Déplacement total (mm)",
    "PathLength_mm": "Longueur de trajectoire (mm)",
    "Spread_mm": "Dispersion spatiale (mm)",
    "InvolvementScore": "Score d'implication",
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Télécharger le tableau des résultats (CSV)",
    data=csv_bytes,
    file_name="marker_analysis.csv",
    mime="text/csv",
)

if show_disabled:
    st.subheader("Comparaison normal vs limité simulé")
    compare_markers = [m for m in ["RSHO", "RELB", "RWRA", "C7", "CLAV", "STRN"] if m in markers]
    comp_fig = make_comparison_figure(df, df_disabled, compare_markers, sample_step=sample_step)
    st.plotly_chart(comp_fig, use_container_width=True)

    st.markdown("""
    <div class="small-muted">
    La simulation rapproche artificiellement le coude et le poignet droits de l'épaule droite afin d'imiter une limitation d'amplitude.
    Cette comparaison sert de démonstrateur visuel et ne constitue pas un modèle clinique validé.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Conseil de démo** : importe un fichier TRC, affiche le haut du corps, puis active la simulation pour montrer comment l'interface peut comparer deux profils de mouvement.")
