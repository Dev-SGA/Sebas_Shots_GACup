import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Shot Map Analysis")

st.title("Shot Map Analysis - Multiple Matches")
st.caption("Click on the icons on the pitch to play the corresponding video analysis and see goal placement.")

# ==========================
# GOAL DIMENSIONS
# ==========================
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44

# ==========================
# Data Setup (agora com goal_x / goal_y)
# type, x, y, xg, goal_x, goal_y, video
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("A gol", 117.35, 26.20, 0.18, 0.35, 0.12, None),
        ("A gol", 108.87, 48.81, 0.12, 2.34, 0.55, None),

        ("Bloqueado", 104.22, 26.87, 0.06, None, None, None),
        ("Bloqueado", 98.23, 22.88, 0.05, None, None, None),

        ("Fora", 113.19, 43.82, 0.08, None, None, None),
    ],
    "Vs Slavia Praha": [
        ("Fora", 108.04, 35.68, 0.07, None, None, None),
        ("Fora", 96.07, 7.09, 0.03, None, None, None),
    ],
    "Vs Sockers": [
        ("Trave", 104.05, 56.79, 0.15, -0.00, 0.14, None),
        ("A gol", 109.70, 41.83, 0.11, 4.40, 0.50, None),
        ("Gol", 112.36, 46.32, 0.35, 1.03, 0.15, None),
    ],
}

# Create DataFrames for each match and combined
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(
        events,
        columns=["type", "x", "y", "xg", "goal_x", "goal_y", "video"]
    )

# All games combined
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All shots": df_all}
full_data.update(dfs_by_match)

# ==========================
# Style (parecido com seu shotmap)
# ==========================
def get_style(result_type: str, has_video: bool):
    t = (result_type or "").strip().upper()
    alpha = 0.95 if has_video else 0.85

    if t == "GOL":
        return "*", (239/255, 71/255, 111/255, alpha), 1.5  # #EF476F
    if t in ("A GOL", "NO ALVO"):
        return "h", (6/255, 214/255, 160/255, alpha), 1.5   # #06D6A0
    if t == "FORA":
        return "o", (255/255, 209/255, 102/255, alpha), 1.5 # #FFD166
    if t == "BLOQUEADO":
        return "s", (17/255, 138/255, 178/255, alpha), 1.5  # #118AB2
    if t == "TRAVE":
        return "D", (160/255, 114/255, 255/255, alpha), 1.5 # roxo

    return "o", (0.6, 0.6, 0.6, alpha), 1.2

def size_from_xg(xg: float, scale: float = 1400.0):
    return (float(xg) * scale) + 60

# ==========================
# Goal chart (mesmo visual do seu código)
# ==========================
def draw_goal(selected_event: pd.Series | None):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    # goal frame
    ax.plot([0, GOAL_WIDTH], [GOAL_HEIGHT, GOAL_HEIGHT], color="white", lw=3)
    ax.plot([0, 0], [0, GOAL_HEIGHT], color="white", lw=3)
    ax.plot([GOAL_WIDTH, GOAL_WIDTH], [0, GOAL_HEIGHT], color="white", lw=3)

    # 3x3 grid
    x1 = GOAL_WIDTH / 3
    x2 = 2 * GOAL_WIDTH / 3
    y1 = GOAL_HEIGHT / 3
    y2 = 2 * GOAL_HEIGHT / 3
    ax.plot([x1, x1], [0, GOAL_HEIGHT], color="white", alpha=0.2)
    ax.plot([x2, x2], [0, GOAL_HEIGHT], color="white", alpha=0.2)
    ax.plot([0, GOAL_WIDTH], [y1, y1], color="white", alpha=0.2)
    ax.plot([0, GOAL_WIDTH], [y2, y2], color="white", alpha=0.2)

    # limits/axes
    ax.set_xlim(-0.5, GOAL_WIDTH + 0.5)
    ax.set_ylim(0, GOAL_HEIGHT + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Goal View (Shot Placement)", color="white")

    # plot selected event shot placement
    if selected_event is not None:
        gx = selected_event.get("goal_x")
        gy = selected_event.get("goal_y")

        if pd.notna(gx) and pd.notna(gy):
            t = (selected_event.get("type") or "").strip().upper()
            if t == "GOL":
                c, m = "#EF476F", "*"
            elif t in ("A GOL", "NO ALVO"):
                c, m = "#06D6A0", "h"
            elif t == "FORA":
                c, m = "#FFD166", "o"
            elif t == "BLOQUEADO":
                c, m = "#118AB2", "s"
            elif t == "TRAVE":
                c, m = "#A072FF", "D"
            else:
                c, m = "#999999", "o"

            ax.scatter(
                gx, gy,
                color=c,
                marker=m,
                s=120,
                edgecolors="white",
                linewidth=1.5,
                zorder=3
            )

    plt.tight_layout()
    return fig

# ==========================
# Sidebar Configuration
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

df_base = full_data[selected_match].copy()
result_options = sorted(df_base["type"].unique().tolist())
selected_results = st.sidebar.multiselect("Shot Result", result_options, default=result
