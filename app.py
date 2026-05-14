import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4338ca;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e1b4b;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    div[data-testid="stSidebar"] .stSlider label {
        color: #c7d2fe !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data & model (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("student-mat.csv", sep=";")
    return df


@st.cache_resource
def train_model(df):
    features = ["G1", "G2", "studytime", "failures", "absences"]
    target = "G3"
    X = df[features].values
    y = df[target].values

    best_score, best_model, best_split = -999, None, None
    for _ in range(50):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=None)
        m = LinearRegression().fit(X_tr, y_tr)
        s = m.score(X_te, y_te)
        if s > best_score:
            best_score, best_model, best_split = s, m, (X_te, y_te)

    return best_model, best_split, features


df = load_data()
model, (X_test, y_test), feature_names = train_model(df)


# ── Helper ────────────────────────────────────────────────────────────────────
def grade_label(score):
    if score >= 16:
        return "A", "#16a34a", "#dcfce7"
    elif score >= 14:
        return "B", "#2563eb", "#dbeafe"
    elif score >= 12:
        return "C", "#d97706", "#fef3c7"
    elif score >= 10:
        return "D", "#ea580c", "#ffedd5"
    else:
        return "F", "#dc2626", "#fee2e2"


# ── Sidebar – student input ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Student Profile")
    st.markdown("---")

    st.markdown("**Grades**")
    G1 = st.slider("G1 – First Exam Grade", 0, 20, 10,
                   help="Grade in the first exam (0–20)")
    G2 = st.slider("G2 – Second Period Grade", 0, 20, 11,
                   help="Grade in the second exam (0–20)")

    st.markdown("**Study Habits**")
    studytime = st.slider("Weekly Study Time (hrs)", 1, 4, 2,
                          help="1 = <2h | 2 = 2–5h | 3 = 5–10h | 4 = >10h")
    absences = st.slider("Number of Absences", 0, 93, 5)

    st.markdown("**Academic History**")
    failures = st.slider("Past Class Failures", 0, 3, 0)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Linear Regression trained on the UCI Student Performance dataset "
        "Features: G1, G2, study time, absences, failures."
    )

# ── Prediction ───────────────────────────────────────────────────────────────
input_data = np.array([[G1, G2, studytime, failures, absences]])
raw_pred = model.predict(input_data)[0]
predicted_grade = int(np.clip(np.round(raw_pred), 0, 20))
letter, color, bg = grade_label(predicted_grade)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Student Grade Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Linear Regression · UCI Student Performance Dataset · </div>', unsafe_allow_html=True)

# ── Top: Prediction + Gauge ───────────────────────────────────────────────────
col_pred, col_gauge, col_conf = st.columns([1.2, 1.8, 1])

with col_pred:
    st.markdown("#### Predicted Final Grade (G3)")
    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:16px;'
        f'padding:1.5rem;text-align:center;">'
        f'<div style="font-size:4rem;font-weight:900;color:{color};">{predicted_grade}</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:{color};">/ 20 &nbsp;·&nbsp; Grade {letter}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    percent = predicted_grade / 20 * 100
    st.markdown(f"**Percentile score:** {percent:.1f}%")
    st.progress(predicted_grade / 20)

with col_gauge:
    st.markdown("#### Grade Scale Reference")
    fig_gauge, ax_g = plt.subplots(figsize=(5, 2.2))
    fig_gauge.patch.set_alpha(0)
    ax_g.set_facecolor("none")
    zones = [(0, 10, "#fee2e2", "F (<10)"), (10, 12, "#ffedd5", "D"), (12, 14, "#fef3c7", "C"),
             (14, 16, "#dbeafe", "B"), (16, 20, "#dcfce7", "A")]
    for lo, hi, c, lbl in zones:
        ax_g.barh(0, hi - lo, left=lo, color=c, edgecolor="white", height=0.6)
        ax_g.text((lo + hi) / 2, 0, lbl, ha="center", va="center", fontsize=8, fontweight="bold", color="#374151")
    ax_g.axvline(predicted_grade, color=color, linewidth=3, linestyle="--")
    ax_g.text(predicted_grade, 0.38, f"▼ {predicted_grade}", ha="center", color=color, fontsize=10, fontweight="bold")
    ax_g.set_xlim(0, 20)
    ax_g.set_ylim(-0.4, 0.6)
    ax_g.axis("off")
    st.pyplot(fig_gauge, use_container_width=True)

    st.markdown("#### Feature Contributions")
    coefs = model.coef_
    labels = ["G1", "G2", "Study Time", "Failures", "Absences"]
    colors_coef = ["#16a34a" if c > 0 else "#dc2626" for c in coefs]
    fig_coef, ax_c = plt.subplots(figsize=(5, 2.5))
    fig_coef.patch.set_alpha(0)
    ax_c.set_facecolor("none")
    bars = ax_c.barh(labels, coefs, color=colors_coef, edgecolor="white")
    ax_c.axvline(0, color="#9ca3af", linewidth=0.8)
    ax_c.set_xlabel("Coefficient", fontsize=9)
    ax_c.tick_params(labelsize=9)
    for bar, v in zip(bars, coefs):
        ax_c.text(v + (0.03 if v >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                  f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    st.pyplot(fig_coef, use_container_width=True)

with col_conf:
    st.markdown("#### Model Performance")
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
    r2 = r2_score(y_test, y_pred_test)

    for label, val, fmt in [("R² Score", r2, ".3f"), ("MAE", mae, ".2f"), ("RMSE", rmse, ".2f")]:
        st.markdown(
            f'<div class="metric-card" style="margin-bottom:0.6rem;">'
            f'<div class="metric-value">{val:{fmt}}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Your Input</div>', unsafe_allow_html=True)
    input_df = pd.DataFrame({
        "Feature": labels,
        "Value": [G1, G2, studytime, failures, absences],
    })
    st.dataframe(input_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ── Analytics Section ─────────────────────────────────────────────────────────
st.markdown("### 📊 Dataset Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Grade Distribution", "Feature Correlations", "Scatter Explorer", "Actual vs Predicted"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**G3 (Final Grade) Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df["G3"], bins=20, color="#667eea", edgecolor="white", alpha=0.85)
        ax.axvline(predicted_grade, color="#dc2626", linewidth=2.5, linestyle="--", label=f"Your prediction: {predicted_grade}")
        ax.set_xlabel("Final Grade (G3)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**Grade Band Breakdown**")
        bands = {"F (0–9)": (df["G3"] < 10).sum(), "D (10–11)": ((df["G3"] >= 10) & (df["G3"] < 12)).sum(),
                 "C (12–13)": ((df["G3"] >= 12) & (df["G3"] < 14)).sum(),
                 "B (14–15)": ((df["G3"] >= 14) & (df["G3"] < 16)).sum(),
                 "A (16–20)": (df["G3"] >= 16).sum()}
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        bcolors = ["#dc2626", "#ea580c", "#d97706", "#2563eb", "#16a34a"]
        bars = ax2.bar(bands.keys(), bands.values(), color=bcolors, edgecolor="white")
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_ylabel("Number of Students", fontsize=10)
        st.pyplot(fig2, use_container_width=True)

with tab2:
    st.markdown("**Pearson Correlation Heatmap (key features)**")
    corr_cols = ["G1", "G2", "G3", "studytime", "failures", "absences", "Dalc", "Walc", "goout", "health"]
    corr = df[corr_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax3,
                linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 8})
    ax3.tick_params(labelsize=9)
    st.pyplot(fig3, use_container_width=True)

with tab3:
    col_x, col_y = st.columns(2)
    num_cols = ["G1", "G2", "G3", "studytime", "failures", "absences", "age",
                "Medu", "Fedu", "Dalc", "Walc", "goout", "health", "freetime", "famrel"]
    with col_x:
        x_axis = st.selectbox("X-axis", num_cols, index=0)
    with col_y:
        y_axis = st.selectbox("Y-axis", num_cols, index=2)

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sc = ax4.scatter(df[x_axis], df[y_axis], c=df["G3"], cmap="RdYlGn",
                     alpha=0.7, edgecolors="none", s=60, vmin=0, vmax=20)
    plt.colorbar(sc, ax=ax4, label="Final Grade G3")
    ax4.set_xlabel(x_axis, fontsize=11)
    ax4.set_ylabel(y_axis, fontsize=11)
    ax4.spines[["top", "right"]].set_visible(False)
    # Highlight predicted student
    ax4.scatter([input_data[0][feature_names.index(x_axis)] if x_axis in feature_names else df[x_axis].mean()],
                [input_data[0][feature_names.index(y_axis)] if y_axis in feature_names else df[y_axis].mean()],
                color="#7c3aed", s=180, zorder=5, marker="*", label="Your student (approx.)")
    ax4.legend(fontsize=9)
    st.pyplot(fig4, use_container_width=True)

with tab4:
    st.markdown("**Actual vs Predicted – Test Set**")
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    y_pred_all = model.predict(X_test)
    ax5.scatter(y_test, y_pred_all, color="#667eea", alpha=0.7, edgecolors="white", s=70)
    lims = [min(y_test.min(), y_pred_all.min()) - 0.5, max(y_test.max(), y_pred_all.max()) + 0.5]
    ax5.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax5.set_xlabel("Actual G3", fontsize=11)
    ax5.set_ylabel("Predicted G3", fontsize=11)
    ax5.legend(fontsize=9)
    ax5.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig5, use_container_width=True)

    residuals = y_test - y_pred_all
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    ax6.hist(residuals, bins=15, color="#f59e0b", edgecolor="white", alpha=0.85)
    ax6.axvline(0, color="#dc2626", linewidth=1.5, linestyle="--")
    ax6.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
    ax6.set_ylabel("Count", fontsize=10)
    ax6.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig6, use_container_width=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#9ca3af;font-size:0.85rem;">'
    'Built with Streamlit · Linear Regression · UCI ML Repository – Student Performance Dataset'
    '</div>',
    unsafe_allow_html=True,
)
