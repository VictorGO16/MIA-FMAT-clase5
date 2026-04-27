import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import time
import itertools
import re
from scipy import stats
from scipy.special import comb, gammaln, logsumexp
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==================================================================
#                       CONFIGURACIÓN GLOBAL
# ==================================================================
st.set_page_config(
    page_title="MIA - Probabilidad para IA",
    layout="wide",
)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#2563EB", "#D97706", "#059669", "#7C3AED", "#DC2626", "#475569"])
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.6,
    "grid.color": "#D9E2EC",
    "grid.linewidth": 0.8,
    "figure.dpi": 100,
    "figure.facecolor": "#FCFCFD",
    "axes.facecolor": "#FCFCFD",
    "savefig.facecolor": "#FCFCFD",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "axes.titleweight": "semibold",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0F172A",
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CBD5E1",
    "legend.framealpha": 0.96,
    "legend.loc": "upper right",
    "lines.linewidth": 2.2,
})

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 30%),
            linear-gradient(180deg, #F8FAFC 0%, #F4F6F8 100%);
    }
    .main .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.8rem;
        max-width: 1180px;
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.01em;
        color: #0F172A;
    }
    p, li {
        line-height: 1.62;
    }
    .mia-hero {
        padding: 1.1rem 1.25rem 1rem 1.25rem;
        margin: 0 0 1.1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(241,245,249,0.92));
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
    }
    .mia-hero-kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .mia-hero h1 {
        margin: 0;
        font-size: 2.15rem;
        font-weight: 700;
    }
    .mia-hero p {
        margin: 0.45rem 0 0 0;
        color: #475569;
        font-size: 1rem;
    }
    .mia-progress-track {
        margin-top: 0.85rem;
        height: 6px;
        width: 100%;
        background: rgba(148,163,184,0.22);
        border-radius: 999px;
        overflow: hidden;
    }
    .mia-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        border-radius: 999px;
        transition: width 0.35s ease;
    }
    .mia-prevnext-spacer {
        margin-top: 2.2rem;
        border-top: 1px dashed rgba(148,163,184,0.35);
        padding-top: 1.1rem;
    }
    .mia-prevnext-chip {
        text-align: center;
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        padding: 0.55rem 0;
        letter-spacing: 0.04em;
    }
    .mia-section-header {
        padding: 1rem 1.1rem 0.95rem 1.1rem;
        margin: 0.15rem 0 1.05rem 0;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 0.95rem;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }
    .mia-section-eyebrow {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.28rem;
    }
    .mia-section-header h2 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .mia-section-header p {
        margin: 0.42rem 0 0 0;
        color: #475569;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.16);
        padding: 0.75rem 0.9rem;
        border-radius: 0.8rem;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 0.8rem;
        background: rgba(255,255,255,0.74);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.035);
    }
    div[data-testid="stExpander"] details summary {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        border: 1px solid #D9E2EC;
        background: rgba(255,255,255,0.72);
        color: #334155;
        padding: 0.45rem 0.9rem;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background: #0F172A;
        color: white;
        border-color: #0F172A;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F7F9FC 0%, #EEF2F7 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .mia-sidebar-group {
        margin: 0.2rem 0 0.9rem 0;
    }
    .mia-sidebar-group-title {
        margin: 0 0 0.35rem 0.2rem;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 700;
    }
    a.mia-sidebar-link {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        padding: 0.52rem 0.6rem;
        margin: 0 0 0.14rem 0;
        border-radius: 0.82rem;
        color: #0F172A !important;
        text-decoration: none !important;
        border: 1px solid transparent;
        transition: all 0.18s ease;
    }
    a.mia-sidebar-link:hover {
        background: rgba(255,255,255,0.92);
        border-color: #D9E2EC;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    a.mia-sidebar-link.active {
        background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(255,255,255,0.95));
        border-color: rgba(37,99,235,0.24);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
    }
    .mia-sidebar-index {
        width: 1.75rem;
        min-width: 1.75rem;
        height: 1.75rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.77rem;
        font-weight: 700;
        background: rgba(148, 163, 184, 0.12);
        color: #334155;
    }
    a.mia-sidebar-link.active .mia-sidebar-index {
        background: #DBEAFE;
        color: #1D4ED8;
    }
    .mia-sidebar-text {
        line-height: 1.2;
        font-size: 0.93rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================================
#                    UTILIDADES PEDAGÓGICAS
# ==================================================================
def motivation(text):
    st.markdown("### Motivación")
    st.markdown(text)

def prerequisites_box(prereqs_md):
    with st.expander("Prerrequisitos y vocabulario base"):
        st.markdown(prereqs_md)

def how_to_read(text):
    with st.expander("Cómo leer el gráfico o simulación", expanded=True):
        st.markdown(text)
        st.markdown(
            "Conviene mirar qué variable cambia en el eje horizontal, cuál en el vertical, y cómo se modifica la forma del gráfico al mover los parámetros. "
            "La meta no es sólo obtener un número, sino relacionar el comportamiento visual con la fórmula y el concepto."
        )

def ai_bridge(text):
    st.markdown("### Conexión con IA")
    st.markdown(text)

def worked_example(title):
    st.markdown(f"### Ejemplo resuelto: {title}")

def interactive_header(title):
    st.markdown(f"### Laboratorio interactivo: {title}")

def self_check_header():
    st.markdown("### Autoevaluación")
    st.caption("Conviene responder antes de mirar la solución. El feedback explica el criterio, no sólo el resultado.")

def insight(text):
    st.markdown(f"> **Idea clave.** {text}")

def pitfall(text):
    st.markdown(f"> **Error frecuente.** {text}")

def _looks_like_latex(symbol):
    return bool(re.search(r"[\\_^{}]", symbol)) or any(token in symbol for token in ["P(", "E[", "Var", "Cov", "arg", "log", "sum", "prod"])

def interactive_guide(controls=None, procedure=None, observe=None, expanded=False):
    with st.expander("Qué hace este laboratorio y cómo usarlo", expanded=expanded):
        if controls:
            st.markdown("**Qué controla cada parámetro**")
            for name, desc in controls:
                st.markdown(f"- `{name}`: {desc}")
        if procedure:
            st.markdown("**Qué procedimiento se ejecuta**")
            st.markdown(procedure)
        if observe:
            st.markdown("**Qué conviene observar**")
            st.markdown(observe)

def formula_walkthrough(title, formula=None, terms=None, steps=None, expanded=False):
    with st.expander(title, expanded=expanded):
        if formula:
            st.latex(formula)
        if terms:
            st.markdown("**Qué significa cada símbolo o término**")
            for symbol, meaning in terms.items():
                c1, c2 = st.columns([1.2, 4.8])
                with c1:
                    if _looks_like_latex(symbol):
                        st.latex(symbol)
                    else:
                        st.markdown(f"**{symbol}**")
                with c2:
                    st.markdown(meaning)
        if steps:
            st.markdown("**Explicación en palabras del procedimiento o derivación**")
            for idx, step in enumerate(steps, start=1):
                st.markdown(f"{idx}. {step}")

def gaussian_logpdf_vector(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

def polish_axes(ax):
    axes = np.ravel(ax) if isinstance(ax, (list, tuple, np.ndarray)) else [ax]
    for a in axes:
        a.set_facecolor("#FCFCFD")
        a.grid(True, color="#D9E2EC", alpha=0.7, linewidth=0.8)
        a.tick_params(colors="#334155", labelsize=9.5)
        for spine in ["left", "bottom"]:
            if spine in a.spines:
                a.spines[spine].set_color("#CBD5E1")
                a.spines[spine].set_linewidth(0.9)
        legend = a.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_edgecolor("#CBD5E1")
            frame.set_linewidth(0.8)
            frame.set_facecolor("white")
            frame.set_alpha(0.96)

def polish_figure(fig):
    fig.patch.set_facecolor("#FCFCFD")

if not hasattr(st, "_mia_original_pyplot"):
    st._mia_original_pyplot = st.pyplot

def _styled_pyplot(fig=None, *args, **kwargs):
    if fig is not None and hasattr(fig, "axes"):
        try:
            polish_axes(fig.axes)
            polish_figure(fig)
        except Exception:
            pass
    result = st._mia_original_pyplot(fig, *args, **kwargs)
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass
    return result

st.pyplot = _styled_pyplot

def quiz(question, options, correct_idx, feedback_ok, feedback_wrong, key):
    st.markdown(f"**{question}**")
    ans = st.radio("Opciones", options, key=key, index=None, label_visibility="collapsed")
    if ans is None:
        st.caption("_Selecciona una opción para ver el feedback._")
        return
    if options.index(ans) == correct_idx:
        st.success(f"Correcto. {feedback_ok}")
    else:
        st.error(f"No es esa. {feedback_wrong}")

def section_title(title, subtitle=None):
    if ". " in title:
        sec_no, sec_name = title.split(". ", 1)
        eyebrow = f"Sección {sec_no}"
    else:
        sec_name = title
        eyebrow = "Sección"
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="mia-section-header">
            <div class="mia-section-eyebrow">{eyebrow}</div>
            <h2>{sec_name}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==================================================================
# SECCIÓN 1 — ESPACIOS Y AXIOMAS DE KOLMOGOROV
# ==================================================================
def sec_kolmogorov():
    section_title(
        "1. Espacios de Probabilidad y Axiomas de Kolmogorov",
        "El lenguaje formal para hablar de azar sin ambigüedad."
    )
    motivation(
        "Antes de calcular nada, necesitamos un marco que nos diga **qué es una probabilidad** "
        "y qué reglas cumple. Sin este marco, frases como «la probabilidad de X es 120%» no se pueden rechazar. "
        "Kolmogorov (1933) dio los 3 axiomas mínimos que hacen que todo encaje."
    )
    prerequisites_box(
        "- **Conjunto**: colección de elementos (ej. {1,2,3}).\n"
        "- **Unión** A∪B: elementos que están en A o B (o ambos).\n"
        "- **Intersección** A∩B: elementos que están en A y también en B.\n"
        "- **Complemento** Aᶜ: elementos del universo que NO están en A.\n"
        "- **Disjuntos**: dos conjuntos son disjuntos si A∩B = ∅ (no comparten elementos)."
    )

    st.markdown("### Construcción formal")
    st.markdown(
        "Un **espacio de probabilidad** es una terna $(\\Omega, \\mathcal{A}, P)$ donde:\n"
        "- $\\Omega$ (Omega) es el **espacio muestral**: todos los resultados posibles del experimento.\n"
        "- $\\mathcal{A}$ es una **σ-álgebra**: la colección de subconjuntos de $\\Omega$ a los que "
        "podemos asignar probabilidad (llamados **eventos**).\n"
        "- $P: \\mathcal{A} \\to [0,1]$ es la función de probabilidad."
    )
    insight(
        "El objeto central no es un número aislado sino una estructura completa: qué resultados son posibles, "
        "qué subconjuntos consideraremos eventos y qué regla asigna probabilidad a cada uno."
    )
    st.markdown("**Los 3 axiomas de Kolmogorov:**")
    st.latex(r"\textbf{A1 (No negatividad):}\quad P(A) \geq 0 \ \ \forall A \in \mathcal{A}")
    st.latex(r"\textbf{A2 (Normalización):}\quad P(\Omega) = 1")
    st.latex(r"\textbf{A3 (Aditividad numerable):}\quad P\Big(\bigcup_{i=1}^{\infty} A_i\Big) = \sum_{i=1}^{\infty} P(A_i) \quad \text{si los } A_i \text{ son disjuntos}")
    st.markdown("**Consecuencias inmediatas** (todo lo demás se deduce de estos 3):")
    st.latex(r"P(\emptyset) = 0, \quad P(A^c) = 1 - P(A), \quad A \subseteq B \Rightarrow P(A) \le P(B)")
    st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B) \quad \text{(inclusión-exclusión)}")
    formula_walkthrough(
        "Lectura precisa de la terna $(\\Omega, \\mathcal A, P)$ y de los axiomas",
        formula=r"(\Omega, \mathcal A, P)",
        terms={
            r"\Omega": "Espacio muestral: todos los resultados que el modelo admite.",
            r"\mathcal A": "Colección de eventos a los que sí les vamos a asignar probabilidad.",
            r"P": "Función que asigna a cada evento un número entre 0 y 1.",
        },
        steps=[
            "A1 impide probabilidades negativas: una probabilidad puede ser muy pequeña, pero no puede ser menor que cero.",
            "A2 fija la escala completa: la probabilidad total del universo posible es exactamente 1.",
            "A3 dice cómo se suma probabilidad cuando los eventos no se traslapan. Esa es la base de todas las reglas de suma.",
            "La fórmula de inclusión-exclusión resta la intersección porque al sumar $P(A)$ y $P(B)$ la parte común se contó dos veces.",
        ],
        expanded=True,
    )
    st.markdown(
        "En sustantivo: esta terna fija primero **qué puede ocurrir**, luego **qué subconjuntos de resultados vamos a considerar eventos**, "
        "y finalmente **qué peso probabilístico recibe cada evento**. Los axiomas no son un truco algebraico; son las reglas mínimas para que esa asignación sea coherente."
    )

    worked_example("dado justo de 6 caras")
    st.markdown(
        "- $\\Omega = \\{1,2,3,4,5,6\\}$, todos equiprobables.\n"
        "- Evento **Par** = $\\{2,4,6\\}$ → $P(\\text{Par}) = 3/6 = 0.5$.\n"
        "- Evento **Mayor que 4** = $\\{5,6\\}$ → $P = 2/6 = 1/3$.\n"
        "- $P(\\text{Par} \\cup \\text{Mayor4}) = P(\\text{Par}) + P(\\text{Mayor4}) - P(\\text{Par}\\cap\\text{Mayor4})$\n"
        "  $= 1/2 + 1/3 - 1/6 = 2/3$."
    )
    pitfall(
        "Confundir resultado elemental con evento. En un dado, el resultado '4' es un punto de $\\Omega$; "
        "el evento 'salir par' es un subconjunto de $\\Omega$."
    )

    interactive_header("Álgebra de eventos sobre un dado justo")
    interactive_guide(
        controls=[
            ("Evento A", "elige qué caras del dado pertenecen al evento A."),
            ("Evento B", "elige qué caras pertenecen al evento B."),
        ],
        procedure=(
            "La app trata cada cara del dado como un resultado elemental con probabilidad $1/6$. "
            "A partir de los conjuntos elegidos calcula intersección, unión, complemento y sus probabilidades."
        ),
        observe=(
            "Fíjate en qué caras cuentan una sola vez para la unión, cuáles están en ambos eventos y cómo cambia la fórmula "
            "$P(A\\cup B)=P(A)+P(B)-P(A\\cap B)$ cuando la intersección crece o desaparece."
        ),
    )
    col1, col2 = st.columns([1, 2])
    faces = [1, 2, 3, 4, 5, 6]
    with col1:
        event_a = st.multiselect("Evento A", faces, default=[2, 4, 6], key="kolm_event_a")
        event_b = st.multiselect("Evento B", faces, default=[4, 5, 6], key="kolm_event_b")
        set_a, set_b = set(event_a), set(event_b)
        inter = sorted(set_a & set_b)
        union = sorted(set_a | set_b)
        comp_a = [x for x in faces if x not in set_a]
        p_a = len(set_a) / 6
        p_b = len(set_b) / 6
        p_inter = len(inter) / 6
        p_union = len(union) / 6
        st.metric("P(A)", f"{p_a:.3f}")
        st.metric("P(B)", f"{p_b:.3f}")
        st.metric("P(A ∩ B)", f"{p_inter:.3f}")
        st.metric("P(A ∪ B)", f"{p_union:.3f}")
        st.markdown(f"$A^c = {comp_a}$")
        st.latex(
            rf"P(A\cup B) = P(A) + P(B) - P(A\cap B) = {p_a:.3f} + {p_b:.3f} - {p_inter:.3f} = {p_union:.3f}"
        )
    with col2:
        categories = []
        colors = []
        color_map = {
            "A ∩ B": "#1f77b4",
            "Sólo A": "#17becf",
            "Sólo B": "#ff7f0e",
            "Fuera de A y B": "#c7c7c7",
        }
        for face in faces:
            if face in set_a and face in set_b:
                categories.append("A ∩ B")
            elif face in set_a:
                categories.append("Sólo A")
            elif face in set_b:
                categories.append("Sólo B")
            else:
                categories.append("Fuera de A y B")
            colors.append(color_map[categories[-1]])
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.bar(faces, np.repeat(1/6, len(faces)), color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(faces)
        ax.set_ylim(0, 0.24)
        ax.set_xlabel("Cara del dado")
        ax.set_ylabel("Probabilidad")
        handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
        ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            pd.DataFrame(
                {
                    "cara": faces,
                    "en A": [face in set_a for face in faces],
                    "en B": [face in set_b for face in faces],
                    "categoría": categories,
                }
            ),
            hide_index=True,
        )
    how_to_read(
        "Cada barra representa un resultado elemental con masa 1/6. Los colores permiten ver qué resultados "
        "aportan a $A$, a $B$, a la intersección y a la unión."
    )

    interactive_header("Frecuencia relativa vs probabilidad teórica")
    st.caption("Lanza una moneda o dado muchas veces. La frecuencia empírica converge a la P teórica (LLN, que veremos en la sección 15).")
    interactive_guide(
        controls=[
            ("Experimento", "elige si quieres observar una moneda o un dado."),
            ("Número de lanzamientos", "fija cuántas repeticiones tendrá el experimento."),
            ("Simular", "genera una nueva realización aleatoria del experimento."),
        ],
        procedure=(
            "Se generan lanzamientos i.i.d. según el modelo teórico y luego se compara la frecuencia relativa observada de cada resultado "
            "con la probabilidad exacta del modelo."
        ),
        observe=(
            "Con pocas repeticiones la frecuencia empírica fluctúa bastante. A medida que aumentas el número de lanzamientos, "
            "las barras empíricas deberían acercarse a las teóricas."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        exp_type = st.radio("Experimento", ["Moneda (2 resultados)", "Dado (6 resultados)"], key="kolm_exp")
        n_trials = st.slider("Número de lanzamientos", 10, 5000, 500, step=10, key="kolm_n")
        if st.button("Simular", key="kolm_btn"):
            st.session_state["kolm_seed"] = np.random.randint(0, 1e6)
        seed = st.session_state.get("kolm_seed", 42)
    rng = np.random.default_rng(seed)
    if exp_type.startswith("Moneda"):
        outcomes = rng.integers(0, 2, size=n_trials)
        labels = ["Cara", "Cruz"]
        theoretical = [0.5, 0.5]
    else:
        outcomes = rng.integers(1, 7, size=n_trials)
        labels = ["1", "2", "3", "4", "5", "6"]
        theoretical = [1/6] * 6
    emp = [(outcomes == i).mean() if exp_type.startswith("Moneda") else (outcomes == i+1).mean()
           for i in range(len(labels))]
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, emp, width=0.4, label="Frecuencia empírica", color="#4C72B0")
        ax.bar(x + 0.2, theoretical, width=0.4, label="P teórica", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Probabilidad"); ax.set_ylim(0, max(max(emp), max(theoretical)) * 1.3)
        ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read("Las barras azules son lo observado en la simulación; las naranjas son lo que predice el modelo teórico. Si subes n, las azules se acercan a las naranjas.")

    self_check_header()
    quiz(
        "Si $P(A) = 0.6$ y $P(B) = 0.5$, ¿puede ser que $A$ y $B$ sean disjuntos?",
        ["Sí, siempre es posible", "No, imposible", "Sólo si son independientes"],
        1,
        "Si fueran disjuntos, $P(A \\cup B) = 0.6 + 0.5 = 1.1 > 1$, violando A2.",
        "Recuerda: $P(\\Omega)=1$ es el tope. Si la suma excede 1, no pueden ser disjuntos.",
        key="kolm_q1"
    )
    quiz(
        "$P(A) = 0.4$, $P(B) = 0.5$, $P(A\\cap B) = 0.2$. ¿Cuánto vale $P(A\\cup B)$?",
        ["0.9", "0.7", "0.2", "1.1"],
        1,
        "Inclusión-exclusión: $0.4 + 0.5 - 0.2 = 0.7$.",
        "Usa $P(A\\cup B) = P(A) + P(B) - P(A\\cap B)$.",
        key="kolm_q2"
    )

    ai_bridge(
        "En clasificación multi-clase, un modelo asigna probabilidades $P(y=k|x)$ a $K$ clases mutuamente "
        "excluyentes. El axioma A3 (aditividad) exige $\\sum_k P(y=k|x) = 1$ — por eso al final de una red "
        "neuronal ponemos **softmax**: garantiza que las salidas son una distribución de probabilidad válida."
    )

# ==================================================================
# SECCIÓN 2 — LAPLACE Y COMBINATORIA
# ==================================================================
def sec_laplace():
    section_title(
        "2. Regla de Laplace y Combinatoria",
        "Cuando todos los resultados son equiprobables, contar es calcular probabilidad."
    )
    motivation(
        "Si el espacio muestral es **finito y todos los resultados equiprobables**, la probabilidad de un "
        "evento se reduce a *contar*. Pero contar bien no es trivial: hay que saber cuándo importa el orden, "
        "cuándo hay reemplazo, cuándo los objetos son distinguibles."
    )
    prerequisites_box(
        "- **Factorial**: $n! = n\\cdot(n-1)\\cdot\\ldots\\cdot 1$. Cuenta ordenamientos de $n$ objetos.\n"
        "- **Permutación**: arreglo ordenado.\n"
        "- **Combinación**: subconjunto (orden no importa).\n"
        "- **Coeficiente binomial** $\\binom{n}{k} = \\dfrac{n!}{k!(n-k)!}$: formas de elegir $k$ de $n$ sin orden."
    )
    st.markdown("### Construcción")
    st.latex(r"P(A) = \frac{|A|}{|\Omega|} \quad \text{(Regla de Laplace, sólo si todos los resultados son equiprobables)}")
    st.markdown(
        "| Con orden / sin orden | Con reemplazo | Sin reemplazo |\n"
        "|---|---|---|\n"
        "| **Con orden** | $n^k$ | $\\dfrac{n!}{(n-k)!}$ |\n"
        "| **Sin orden** | $\\binom{n+k-1}{k}$ | $\\binom{n}{k}$ |\n"
    )
    formula_walkthrough(
        "Cuándo usar cada fórmula de conteo",
        terms={
            "Orden importa": "Intercambiar dos elementos genera un resultado distinto.",
            "Reemplazo": "Después de elegir un objeto, sigue disponible para volver a ser elegido.",
            r"\binom{n}{k}": "Número de subconjuntos de tamaño $k$ tomados desde un conjunto de $n$ elementos distintos.",
        },
        steps=[
            "Si en cada una de las $k$ elecciones tienes nuevamente $n$ opciones y el orden importa, el conteo es $n^k$.",
            "Si el orden importa pero no puedes repetir, la primera elección tiene $n$ opciones, la segunda $n-1$, y así sucesivamente: eso produce $n!/(n-k)!$.",
            "Si el orden no importa y tampoco hay reemplazo, cada subconjunto de tamaño $k$ se cuenta una vez: aparece $\\binom{n}{k}$.",
            "Si el orden no importa y sí hay reemplazo, el problema equivale a distribuir $k$ marcas entre $n$ categorías: la cuenta es $\\binom{n+k-1}{k}$.",
        ],
        expanded=True,
    )
    st.markdown(
        "La idea conceptual aquí es separar dos decisiones lógicas antes de contar: si dos secuencias con los mismos elementos pero en distinto orden "
        "deben considerarse distintas, y si un objeto puede volver a aparecer después de haber sido elegido."
    )
    pitfall(
        "Usar combinaciones cuando el orden sí importa, o permutaciones cuando el orden no importa. "
        "Antes de escribir una fórmula, conviene decidir explícitamente esas dos preguntas."
    )

    interactive_header("Selector de régimen de conteo")
    interactive_guide(
        controls=[
            ("n objetos distintos", "tamaño del conjunto base del que vas a elegir."),
            ("¿Hay reemplazo?", "decide si un mismo objeto puede aparecer más de una vez."),
            ("k elecciones", "cuántas posiciones o selecciones realizas."),
            ("¿Importa el orden?", "decide si dos elecciones con los mismos elementos pero distinto orden cuentan distinto."),
        ],
        procedure=(
            "Según esas dos decisiones lógicas, la app selecciona automáticamente el régimen correcto de conteo y, cuando el tamaño lo permite, "
            "enumera explícitamente algunos resultados posibles."
        ),
        observe=(
            "La enumeración sirve para auditar la fórmula. Si ves resultados que para tu problema deberían ser equivalentes, entonces elegiste mal el régimen."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        n_items = st.slider("n objetos distintos", 2, 10, 5, key="lap_n_items")
        replacement = st.radio("¿Hay reemplazo?", ["Sí", "No"], key="lap_replacement")
        k_max = 6 if replacement == "Sí" else n_items
        k_choices = st.slider("k elecciones", 1, k_max, min(3, k_max), key="lap_k_choices")
        order = st.radio("¿Importa el orden?", ["Sí", "No"], key="lap_order")
    if order == "Sí" and replacement == "Sí":
        count_value = n_items ** k_choices
        formula = rf"{n_items}^{k_choices} = {count_value}"
        regime = "variaciones con reemplazo"
        outcomes = list(itertools.product(range(1, n_items + 1), repeat=k_choices))
    elif order == "Sí" and replacement == "No":
        count_value = int(math.factorial(n_items) / math.factorial(n_items - k_choices))
        formula = rf"\frac{{{n_items}!}}{{({n_items}-{k_choices})!}} = {count_value}"
        regime = "variaciones sin reemplazo"
        outcomes = list(itertools.permutations(range(1, n_items + 1), k_choices))
    elif order == "No" and replacement == "Sí":
        count_value = int(comb(n_items + k_choices - 1, k_choices, exact=True))
        formula = rf"\binom{{{n_items}+{k_choices}-1}}{{{k_choices}}} = {count_value}"
        regime = "combinaciones con reemplazo"
        outcomes = list(itertools.combinations_with_replacement(range(1, n_items + 1), k_choices))
    else:
        count_value = int(comb(n_items, k_choices, exact=True))
        formula = rf"\binom{{{n_items}}}{{{k_choices}}} = {count_value}"
        regime = "combinaciones sin reemplazo"
        outcomes = list(itertools.combinations(range(1, n_items + 1), k_choices))
    with col2:
        st.metric("Conteo total", f"{count_value:,}")
        st.markdown(f"**Régimen identificado:** {regime}.")
        st.latex(formula)
        preview = pd.DataFrame({"Primeros resultados posibles": [str(outcome) for outcome in outcomes[: min(12, len(outcomes))]]})
        st.dataframe(preview, hide_index=True, use_container_width=True)
    how_to_read(
        "La tabla enumera explícitamente resultados cuando el tamaño es manejable. Si dos filas difieren sólo "
        "en el orden y el régimen elegido considera el orden irrelevante, entonces ese régimen no es el correcto."
    )

    worked_example("paradoja del cumpleaños")
    st.markdown(
        "Pregunta: ¿cuál es la probabilidad de que entre $n$ personas al menos dos compartan cumpleaños?\n\n"
        "Estrategia: es más fácil calcular el **complemento** (todos distintos) y restar a 1:"
    )
    st.latex(r"P(\text{coincidencia}) = 1 - \frac{365 \cdot 364 \cdot \ldots \cdot (365-n+1)}{365^n}")
    st.markdown("Con $n=23$ ya da ~50.7%. Contraintuitivo porque uno piensa «365 días, tendrían que ser muchas más personas».")

    interactive_header("Cumpleaños compartidos")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_people = st.slider("Número de personas", 2, 100, 23, key="bday_n")
        n_runs = st.slider("Experimentos Monte Carlo", 200, 5000, 1000, step=200, key="bday_runs")
        def birthday_prob(n):
            if n > 365: return 1.0
            return 1 - np.exp(sum(np.log(1 - i/365) for i in range(n)))
        p = birthday_prob(n_people)
        st.metric("P(coincidencia)", f"{p:.4f}", f"{p*100:.2f}%")
        rng = np.random.default_rng(123)
        sim = 0
        for _ in range(n_runs):
            birthdays = rng.integers(0, 365, size=n_people)
            sim += int(len(np.unique(birthdays)) < n_people)
        p_sim = sim / n_runs
        st.metric("Monte Carlo", f"{p_sim:.4f}", f"error abs. {abs(p-p_sim):.4f}")
    with col2:
        xs = np.arange(2, 101)
        ys = [birthday_prob(i) for i in xs]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(xs, ys, color="#4C72B0", lw=2)
        ax.axvline(n_people, color="#DD8452", ls="--", alpha=0.7)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.scatter([n_people], [p_sim], color="#55A868", s=70, zorder=5, label="Monte Carlo")
        ax.set_xlabel("n personas"); ax.set_ylabel("P(al menos 2 comparten cumpleaños)")
        ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read("Eje x: cantidad de personas; eje y: probabilidad de coincidencia. Nota cómo cruza 50% alrededor de n=23.")

    worked_example("póker: probabilidad de obtener un par")
    st.markdown(
        "5 cartas de una baraja de 52. $|\\Omega| = \\binom{52}{5} = 2{,}598{,}960$.\n\n"
        "**Par exactamente**: elegir el valor del par $\\binom{13}{1}$, elegir 2 palos $\\binom{4}{2}$, "
        "elegir 3 valores distintos restantes $\\binom{12}{3}$, elegir un palo para cada uno $4^3$."
    )
    num_par = comb(13, 1, exact=True) * comb(4, 2, exact=True) * comb(12, 3, exact=True) * (4**3)
    den_par = comb(52, 5, exact=True)
    st.latex(rf"P(\text{{par}}) = \frac{{{num_par}}}{{{den_par}}} \approx {num_par/den_par:.4f}")
    formula_walkthrough(
        "Despiece del conteo del par en póker",
        steps=[
            "Primero eliges qué valor forma el par: 13 posibilidades, una por rango.",
            "Luego eliges cuáles 2 de los 4 palos tendrán esas cartas: $\\binom{4}{2}$.",
            "Las otras 3 cartas deben tener valores distintos entre sí y distintos del par: eso da $\\binom{12}{3}$.",
            "Cada uno de esos 3 valores puede venir en cualquiera de sus 4 palos: por eso multiplicas por $4^3$.",
            "Finalmente divides por $\\binom{52}{5}$ porque el espacio muestral son todas las manos de 5 cartas, sin importar el orden."
        ],
    )

    self_check_header()
    quiz(
        "Tres amigos se sientan en una fila de 3 sillas. ¿Cuántos ordenamientos posibles hay?",
        ["3", "6", "9", "27"],
        1,
        "$3! = 6$. Permutaciones sin reemplazo.",
        "Son 3 posiciones distintas y 3 personas distintas: $3!$.",
        key="lap_q1"
    )
    quiz(
        "De 10 alumnos, ¿cuántos comités de 3 puedo formar (sin jerarquías)?",
        ["30", "720", "120", "1000"],
        2,
        "$\\binom{10}{3} = 120$. Sin orden ni reemplazo.",
        "Sin orden → combinación: $\\binom{10}{3} = 120$.",
        key="lap_q2"
    )
    ai_bridge(
        "La **combinatoria** es el esqueleto de muchos cálculos en IA: número de particiones en cross-validation, "
        "hipótesis posibles en búsqueda, muestras en bootstrap. El *birthday problem* reaparece en **hash collisions** "
        "(cuántas claves distintas hacen falta para esperar una colisión en un hash de $B$ buckets: $\\approx \\sqrt{B}$)."
    )

# ==================================================================
# SECCIÓN 3 — PROBABILIDAD CONDICIONAL
# ==================================================================
def sec_condicional():
    section_title(
        "3. Probabilidad Condicional e Independencia",
        "Cómo actualizar probabilidades cuando llega información nueva."
    )
    motivation(
        "En el mundo real rara vez calculamos $P(A)$ aislada. Siempre tenemos **información de contexto** "
        "(«sabiendo que llueve, ¿cuál es la probabilidad de que la calle esté mojada?»). La probabilidad "
        "condicional formaliza este razonamiento y es la base de Bayes y de todo modelo probabilístico en ML."
    )
    prerequisites_box(
        "- Probabilidad de un evento $P(A)$.\n"
        "- Intersección $A \\cap B$ = ambos ocurren.\n"
        "- $P(A)$ definida sobre un espacio $(\\Omega, \\mathcal{A}, P)$."
    )
    st.markdown("### Construcción")
    st.latex(r"P(A\mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0")
    st.markdown("Interpretación: *de todos los casos donde $B$ ocurrió, qué fracción también tiene $A$*.")
    st.markdown("**Regla del producto** (despejando):")
    st.latex(r"P(A \cap B) = P(A\mid B)\,P(B) = P(B\mid A)\,P(A)")
    st.markdown("**Independencia**: $A$ y $B$ son independientes si saber uno no cambia la probabilidad del otro:")
    st.latex(r"P(A\mid B) = P(A) \iff P(A \cap B) = P(A)\,P(B)")
    formula_walkthrough(
        "Lectura operativa de $P(A\\mid B)$",
        formula=r"P(A\mid B)=\frac{P(A\cap B)}{P(B)}",
        terms={
            r"P(A\cap B)": "Probabilidad de que ocurran simultáneamente $A$ y $B$.",
            r"P(B)": "Tamaño probabilístico del universo reducido en el que ahora estamos condicionando.",
            r"P(A\mid B)": "Proporción de casos con $B$ en los que además ocurre $A$.",
        },
        steps=[
            "Condicionar en $B$ significa que ya no estás mirando todo el espacio muestral original, sino sólo la región donde $B$ ocurrió.",
            "Dentro de esa región reducida, el evento relevante es $A\\cap B$: los casos compatibles con ambas afirmaciones.",
            "La independencia aparece cuando esa renormalización no cambia el valor de la probabilidad de $A$.",
        ],
        expanded=True,
    )
    st.markdown(
        "En lenguaje simple: primero restringes la atención al mundo donde $B$ ya ocurrió. "
        "Luego preguntas qué proporción de ese mundo restringido también satisface $A$."
    )
    insight(
        "La dirección importa: $P(A\\mid B)$ y $P(B\\mid A)$ casi nunca coinciden, porque normalizan sobre universos distintos."
    )
    pitfall(
        "Confundir condicionalidad con causalidad. Que $P(A\\mid B)$ sea grande no implica que $B$ cause $A$; "
        "sólo dice que, entre los casos donde $B$ ocurrió, $A$ es frecuente."
    )

    interactive_header("Tabla 2×2: leer condicionales e independencia")
    interactive_guide(
        controls=[
            ("Tamaño de la población sintética", "determina cuántos casos totales usarás en la tabla."),
            ("P(B)", "proporción de casos que pertenecen al grupo B."),
            ("P(A|B)", "fracción de casos con A dentro del grupo B."),
            ("P(A|¬B)", "fracción de casos con A fuera del grupo B."),
        ],
        procedure=(
            "La app construye una tabla 2×2 consistente con esos parámetros y calcula tanto probabilidades marginales como condicionales."
        ),
        observe=(
            "Compara $P(A|B)$ con $P(A)$. Si coinciden, saber que ocurrió $B$ no cambia la probabilidad de $A$; "
            "si difieren, hay dependencia."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        pop = st.slider("Tamaño de la población sintética", 1000, 50000, 10000, step=1000, key="cond_pop")
        p_b = st.slider("P(B)", 0.05, 0.95, 0.40, step=0.01, key="cond_pb")
        p_a_given_b = st.slider("P(A|B)", 0.0, 1.0, 0.75, step=0.01, key="cond_pagb")
        p_a_given_notb = st.slider("P(A|¬B)", 0.0, 1.0, 0.25, step=0.01, key="cond_pagnotb")
        n_b = int(round(pop * p_b))
        n_notb = pop - n_b
        n_ab = int(round(n_b * p_a_given_b))
        n_notab = n_b - n_ab
        n_a_notb = int(round(n_notb * p_a_given_notb))
        n_nota_notb = n_notb - n_a_notb
        p_a = (n_ab + n_a_notb) / pop
        p_b_given_a = n_ab / max(n_ab + n_a_notb, 1)
        independence_gap = abs(p_a_given_b - p_a)
        st.metric("P(A)", f"{p_a:.3f}")
        st.metric("P(A|B)", f"{p_a_given_b:.3f}")
        st.metric("P(B|A)", f"{p_b_given_a:.3f}")
        st.metric("|P(A|B) - P(A)|", f"{independence_gap:.3f}")
        if independence_gap < 0.02 and abs(p_a_given_notb - p_a) < 0.02:
            st.caption("Con estos parámetros, A y B están cerca de ser independientes.")
        else:
            st.caption("Aquí conocer B sí altera la probabilidad de A, así que no hay independencia.")
    with col2:
        table = pd.DataFrame(
            {
                "B": [n_ab, n_notab, n_b],
                "¬B": [n_a_notb, n_nota_notb, n_notb],
                "Total": [n_ab + n_a_notb, n_notab + n_nota_notb, pop],
            },
            index=["A", "¬A", "Total"],
        )
        st.dataframe(table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.bar(["B", "¬B"], [n_ab / max(n_b, 1), n_a_notb / max(n_notb, 1)], color=["#4C72B0", "#DD8452"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proporción de A dentro de cada grupo")
        ax.set_title("Comparación entre P(A|B) y P(A|¬B)")
        st.pyplot(fig)
        plt.close(fig)
        st.latex(
            rf"P(A)=\frac{{{n_ab}+{n_a_notb}}}{{{pop}}}={p_a:.3f}, \qquad P(B\mid A)=\frac{{{n_ab}}}{{{n_ab+n_a_notb}}}={p_b_given_a:.3f}"
        )
    how_to_read(
        "La tabla separa claramente qué universo se usa para cada condicional. La columna B sirve para $P(A|B)$; "
        "la fila A sirve para $P(B|A)$."
    )

    worked_example("Monty Hall")
    st.markdown(
        "Tres puertas, tras una hay un auto, tras las otras dos una cabra. Eliges una puerta; el presentador "
        "(que sabe dónde está el auto) abre otra con una cabra. ¿Conviene **cambiar** tu elección?\n\n"
        "Razonamiento condicional: al elegir primero, $P(\\text{auto}) = 1/3$. Esa probabilidad no cambia "
        "cuando abren otra puerta. La puerta restante concentra el $2/3$ complementario.\n\n"
        "**Conclusión**: cambiar gana con probabilidad **2/3**; quedarse con **1/3**."
    )
    interactive_header("Simulador Monty Hall")
    interactive_guide(
        controls=[
            ("Número de puertas", "tamaño total del problema de Monty Hall generalizado."),
            ("Puertas que abre el presentador", "cuántas puertas con cabra abre el presentador después de tu elección inicial."),
            ("Simulaciones", "cuántas repeticiones Monte Carlo usar para estimar probabilidades."),
        ],
        procedure=(
            "Primero eliges una puerta al azar. Luego el presentador, que sabe dónde está el premio, abre sólo puertas con cabra. "
            "Finalmente, la estrategia 'cambiar' en esta app significa elegir **una sola** puerta al azar entre las que siguen cerradas y no fueron tu elección inicial."
        ),
        observe=(
            "Si quedan muchas puertas posibles después de que el presentador abre algunas, cambiar a una puerta elegida al azar ya no equivale a capturar toda la masa $(n-1)/n$. "
            "Sólo capturas una fracción de esa masa."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        n_doors = st.slider("Número de puertas", 3, 20, 3, key="mh_doors")
        max_open = n_doors - 2
        if max_open == 1:
            n_opened = 1
            st.markdown("**Puertas que abre el presentador:** 1")
        else:
            n_opened = st.slider("Puertas que abre el presentador", 1, max_open, 1, key="mh_open")
        n_sim = st.slider("Simulaciones", 500, 20000, 5000, step=500, key="mh_sim")
        st.markdown(
            "Aquí el presentador **no abre una puerta al azar**: abre sólo puertas con cabra y nunca la que elegiste. "
            "Esa información adicional es exactamente lo que hace que cambiar sea ventajoso."
        )
        rng = np.random.default_rng(0)
        wins_stay = 0; wins_switch = 0
        for _ in range(n_sim):
            car = rng.integers(n_doors); pick = rng.integers(n_doors)
            candidates = [d for d in range(n_doors) if d != pick and d != car]
            if len(candidates) < n_opened:
                opened = candidates
            else:
                opened = list(rng.choice(candidates, size=n_opened, replace=False))
            remaining = [d for d in range(n_doors) if d != pick and d not in opened]
            switch = rng.choice(remaining) if remaining else pick
            wins_stay += int(pick == car); wins_switch += int(switch == car)
        p_stay = wins_stay / n_sim; p_switch = wins_switch / n_sim
        remaining_choices = n_doors - 1 - n_opened
        theoretical_stay = 1 / n_doors
        theoretical_switch = (n_doors - 1) / (n_doors * remaining_choices)
        st.metric("P(ganar quedándose)", f"{p_stay:.3f}")
        st.metric("P(ganar cambiando)", f"{p_switch:.3f}")
        st.markdown(
            "Para esta implementación, la estrategia de cambio consiste en escoger **una** puerta al azar entre las que siguen cerradas."
        )
        st.latex(rf"P(\mathrm{{stay}})=\frac{{1}}{{{n_doors}}}={theoretical_stay:.3f}")
        st.latex(
            rf"P(\mathrm{{switch\ al\ azar}})=\frac{{{n_doors}-1}}{{{n_doors}}}\cdot\frac{{1}}{{{remaining_choices}}}={theoretical_switch:.3f}"
        )
        st.caption(
            f"Aquí n={n_doors}, k={n_opened}, y después de abrir k puertas quedan {remaining_choices} puertas candidatas para cambiar."
        )
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Quedarse", "Cambiar"], [p_stay, p_switch], color=["#DD8452", "#4C72B0"])
        ax.set_ylim(0, 1); ax.set_ylabel("P(ganar)")
        for i, v in enumerate([p_stay, p_switch]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        polish_axes(ax)
        polish_figure(fig)
        st.pyplot(fig); plt.close(fig)
    how_to_read(
        "La barra azul y la naranja muestran probabilidades de ganar estimadas por simulación. "
        "Cuando sólo queda una puerta disponible para cambiar, la estrategia de cambio recoge casi toda la probabilidad de que tu primera elección haya sido errónea. "
        "Si quedan varias puertas cerradas y cambias a una sola elegida al azar, esa ventaja se reparte entre esas puertas restantes."
    )

    self_check_header()
    quiz(
        "Si $P(A|B) = P(A)$, entonces...",
        ["$A$ y $B$ son disjuntos", "$A$ y $B$ son independientes", "$P(A)=P(B)$"],
        1,
        "Esa es la definición de independencia: saber $B$ no cambia la probabilidad de $A$.",
        "Disjunto y independiente son cosas distintas: disjuntos con $P>0$ NUNCA son independientes.",
        key="cond_q1"
    )
    ai_bridge(
        "**Independencia condicional** ($X \\perp Y \\mid Z$) es la piedra angular de **redes bayesianas** "
        "y **modelos gráficos probabilísticos**: permite factorizar distribuciones conjuntas enormes en "
        "productos de piezas pequeñas. Sin ella, no podríamos entrenar Naïve Bayes ni hacer inferencia eficiente."
    )

# ==================================================================
# SECCIÓN 4 — TEOREMA DE BAYES
# ==================================================================
def sec_bayes():
    section_title(
        "4. Teorema de Bayes",
        "La fórmula que invierte condicionales y actualiza creencias con evidencia."
    )
    motivation(
        "Muchas veces conocemos $P(\\text{evidencia} \\mid \\text{causa})$ (ej: sensibilidad de un test) pero "
        "queremos $P(\\text{causa} \\mid \\text{evidencia})$ (¿realmente estoy enfermo dado que el test dio positivo?). "
        "Bayes invierte eso y es la herramienta conceptual más importante para razonar con incertidumbre."
    )
    prerequisites_box(
        "- Probabilidad condicional $P(A\\mid B) = P(A\\cap B)/P(B)$.\n"
        "- Regla del producto.\n"
        "- **Ley de probabilidad total**: si $B_1,\\ldots,B_n$ partición de $\\Omega$, $P(A) = \\sum_i P(A\\mid B_i)P(B_i)$."
    )
    st.markdown("### Construcción")
    st.latex(r"P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)} = \frac{P(E\mid H)\,P(H)}{\sum_i P(E\mid H_i)\,P(H_i)}")
    st.markdown(
        "- $P(H)$: **prior** (creencia antes de ver la evidencia).\n"
        "- $P(E\\mid H)$: **verosimilitud** (qué tan bien explica $H$ lo observado).\n"
        "- $P(E)$: **evidencia** (constante de normalización).\n"
        "- $P(H\\mid E)$: **posterior** (creencia actualizada)."
    )
    formula_walkthrough(
        "Derivación de Bayes desde probabilidad condicional y probabilidad total",
        terms={
            r"H": "Hipótesis o causa que quieres evaluar.",
            r"E": "Evidencia observada.",
            r"P(H)": "Creencia inicial en la hipótesis antes de ver la evidencia.",
            r"P(E\mid H)": "Probabilidad de observar la evidencia si la hipótesis fuera cierta.",
            r"P(E)": "Probabilidad total de observar la evidencia bajo todas las hipótesis posibles.",
            r"P(H\mid E)": "Creencia actualizada en la hipótesis después de observar la evidencia.",
        },
        steps=[
            "Empieza con la definición $P(H\\mid E)=P(H\\cap E)/P(E)$.",
            "La regla del producto dice también que $P(H\\cap E)=P(E\\mid H)P(H)$.",
            "Sustituyendo, aparece $P(H\\mid E)=P(E\\mid H)P(H)/P(E)$.",
            "Si las hipótesis $H_1,\\dots,H_m$ forman una partición, entonces la evidencia total se obtiene sumando todos los caminos posibles hacia $E$: $P(E)=\\sum_i P(E\\mid H_i)P(H_i)$.",
            "Ese denominador es la pieza que suele olvidarse y es justamente la que corrige intuiciones erróneas de 'el test es 99% exacto, luego la enfermedad es 99% probable'."
        ],
        expanded=True,
    )
    insight(
        "Bayes no sólo actualiza un número: separa con precisión tres objetos distintos que solemos mezclar al razonar informalmente: prevalencia, calidad del test y probabilidad posterior."
    )

    worked_example("test médico raro (paradoja de baja prevalencia)")
    st.markdown(
        "Enfermedad con prevalencia $P(D)=0.001$. Test con sensibilidad $P(+\\mid D)=0.99$ y "
        "especificidad $P(-\\mid \\lnot D)=0.99$. Tests positivo; ¿cuál es $P(D\\mid +)$?"
    )
    st.latex(
        r"P(D\mid +) = \frac{0.99 \cdot 0.001}{0.99 \cdot 0.001 + 0.01 \cdot 0.999} \approx 0.090"
    )
    st.markdown(
        "**Sólo 9% de probabilidad real**, pese al test «99% confiable». Los falsos positivos entre "
        "la enorme población sana dominan."
    )

    worked_example("test de drogas (Clase 6)")
    st.markdown(
        "Prevalencia $P(D)=0.10$. Test con sensibilidad $P(+\\mid D)=0.9$.\n\n"
        "**Caso A** — especificidad $P(-\\mid \\lnot D)=0.9$:"
    )
    st.latex(r"P(D\mid +) = \frac{0.9\cdot 0.1}{0.9\cdot 0.1 + 0.1\cdot 0.9} = \frac{0.09}{0.18} = 0.5")
    st.markdown("**Caso B** — especificidad baja $P(-\\mid\\lnot D)=0.8$:")
    st.latex(r"P(D\mid +) = \frac{0.9\cdot 0.1}{0.9\cdot 0.1 + 0.2\cdot 0.9} = \frac{0.09}{0.27} = \tfrac{1}{3}")
    st.markdown(
        "Lección: pequeños cambios en especificidad tiran abajo la confianza del test cuando la "
        "prevalencia es baja. Es la razón por la que tests de tamizaje se confirman con segundos tests."
    )
    pitfall(
        "Confundir sensibilidad con valor predictivo positivo. Sensibilidad responde 'si realmente hay enfermedad, ¿el test sale positivo?'; "
        "el posterior responde 'si salió positivo, ¿qué tan probable es que haya enfermedad?'. Son preguntas distintas."
    )

    interactive_header("Bayes con frecuencias naturales y análisis paramétrico")
    interactive_guide(
        controls=[
            ("Prevalencia P(D)", "qué fracción de la población realmente tiene la condición."),
            ("Sensibilidad P(+|D)", "qué tan a menudo el test detecta correctamente un caso enfermo."),
            ("Especificidad P(-|¬D)", "qué tan a menudo el test descarta correctamente un caso sano."),
            ("Población de referencia", "cuántos casos concretos usarás para traducir probabilidades a conteos."),
        ],
        procedure=(
            "Primero se calcula el posterior exacto con Bayes. Luego se traduce ese cálculo a una tabla de frecuencias naturales: "
            "verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos."
        ),
        observe=(
            "La intuición mejora mucho cuando miras conteos. Si hay muy pocos enfermos pero muchísimos sanos, incluso un pequeño porcentaje de falsos positivos "
            "puede superar en número a los verdaderos positivos."
        ),
    )
    prior = st.slider("Prevalencia P(D)", 0.001, 0.5, 0.01, step=0.001, format="%.3f", key="bay_prior")
    sens = st.slider("Sensibilidad P(+|D)", 0.5, 1.0, 0.99, step=0.01, key="bay_sens")
    spec = st.slider("Especificidad P(-|¬D)", 0.5, 1.0, 0.99, step=0.01, key="bay_spec")
    pop = st.slider("Población de referencia para frecuencias naturales", 1000, 100000, 10000, step=1000, key="bay_pop")
    fpr = 1 - spec
    num = sens * prior
    den = num + fpr * (1 - prior)
    post = num / den if den > 0 else 0
    post_neg = ((1 - sens) * prior) / (((1 - sens) * prior) + spec * (1 - prior))
    tabs = st.tabs(["Frecuencias naturales", "Posterior vs prevalencia"])
    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            n_d = int(round(pop * prior))
            n_notd = pop - n_d
            tp = int(round(n_d * sens))
            fn = n_d - tp
            tn = int(round(n_notd * spec))
            fp = n_notd - tn
            st.metric("P(D | +)", f"{post:.4f}")
            st.metric("P(D | −)", f"{post_neg:.4f}")
            st.metric("Verdaderos positivos", f"{tp:,}")
            st.metric("Falsos positivos", f"{fp:,}")
            st.latex(rf"P(D\mid +)=\frac{{TP}}{{TP+FP}}=\frac{{{tp}}}{{{tp}+{fp}}}={post:.4f}")
            st.latex(rf"P(+)=P(+\mid D)P(D)+P(+\mid \neg D)P(\neg D)={den:.4f}")
        with col2:
            freq_table = pd.DataFrame(
                {
                    "D": [tp, fn, n_d],
                    "¬D": [fp, tn, n_notd],
                    "Total": [tp + fp, fn + tn, pop],
                },
                index=["+", "-", "Total"],
            )
            st.dataframe(freq_table, use_container_width=True)
            fig, ax = plt.subplots(figsize=(7, 3.1))
            ax.bar(["Positivos"], [tp], color="#4C72B0", label="Verdaderos positivos")
            ax.bar(["Positivos"], [fp], bottom=[tp], color="#DD8452", label="Falsos positivos")
            ax.set_ylabel("Número de casos")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        how_to_read(
            "El numerador de Bayes es la fracción de positivos que realmente vienen de enfermos. "
            "El denominador incluye todos los positivos, tanto verdaderos como falsos."
        )
    with tabs[1]:
        priors = np.linspace(0.001, 0.5, 200)
        posts = sens * priors / (sens * priors + fpr * (1 - priors))
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.plot(priors, posts, color="#4C72B0", lw=2)
        ax.axvline(prior, color="#DD8452", ls="--", label=f"prior={prior:.3f}")
        ax.axhline(post, color="#55A868", ls=":", label=f"posterior={post:.3f}")
        ax.set_xlabel("Prevalencia P(D)")
        ax.set_ylabel("P(D | +)")
        ax.set_xscale("log")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "La curva muestra por qué el valor predictivo positivo es extremadamente sensible a la prevalencia. "
            "Con baja prevalencia, incluso un test muy bueno puede producir muchos falsos positivos."
        )

    self_check_header()
    quiz(
        "Un test tiene sens=spec=0.95 y prevalencia=0.001. Aproximadamente, $P(D|+)$ vale...",
        ["~0.95", "~0.50", "~0.019", "~0.001"],
        2,
        "$P(D|+) = 0.95\\cdot 0.001 / (0.95\\cdot 0.001 + 0.05\\cdot 0.999) \\approx 0.019$.",
        "Con baja prevalencia los falsos positivos dominan; no es ~0.95.",
        key="bay_q1"
    )
    ai_bridge(
        "Bayes es el esqueleto de casi todo ML probabilístico: **filtros de spam** ($P(\\text{spam}\\mid \\text{palabras})$), "
        "**modelos generativos** ($P(y\\mid x) \\propto P(x\\mid y)P(y)$), **Naïve Bayes** (sección siguiente), "
        "**aprendizaje bayesiano** (posterior sobre parámetros), **diffusion models** (score $\\propto \\nabla \\log p$)."
    )

# ==================================================================
# SECCIÓN 5 — NAÏVE BAYES
# ==================================================================
def sec_naive_bayes():
    section_title(
        "5. Clasificador Naïve Bayes",
        "Bayes + asunción de independencia condicional = un clasificador simple y sorprendentemente efectivo."
    )
    motivation(
        "Calcular $P(x_1,\\ldots,x_d\\mid y)$ para datos de alta dimensión es intratable (demasiados parámetros). "
        "**Naïve Bayes** asume que los features son independientes **dada** la clase. La asunción es casi siempre "
        "falsa, pero el clasificador resultante es robusto, rápido y suele ser un baseline difícil de batir."
    )
    prerequisites_box(
        "- Teorema de Bayes.\n"
        "- Independencia condicional.\n"
        "- Distribución Gaussiana (la veremos en sección 7, aquí la usamos por adelantado).\n"
        "- **MAP**: maximizar $P(y\\mid x)$ sobre las clases."
    )
    st.markdown("### Construcción")
    st.latex(r"P(y\mid x) = \frac{P(y)\,P(x\mid y)}{P(x)} \propto P(y)\prod_{j=1}^d P(x_j\mid y)")
    st.markdown("**Decisión MAP**:")
    st.latex(r"\hat y = \arg\max_y\ P(y)\prod_{j=1}^d P(x_j\mid y)")
    st.markdown("**Truco del logaritmo** (evita underflow, convierte producto en suma):")
    st.latex(r"\hat y = \arg\max_y\ \log P(y) + \sum_{j=1}^d \log P(x_j\mid y)")
    st.markdown(
        "En **Gaussian NB**, cada $P(x_j\\mid y)$ es una normal con parámetros $\\mu_{y,j}, \\sigma_{y,j}^2$ "
        "estimados por MLE desde los datos de la clase $y$."
    )
    formula_walkthrough(
        "Lectura rigurosa de la regla de decisión",
        terms={
            r"y": "clase que queremos predecir.",
            r"x_j": "valor observado del atributo $j$ de la observación.",
            r"P(y)": "probabilidad a priori de cada clase.",
            r"P(x_j\mid y)": "compatibilidad del atributo $j$ con la clase $y$.",
            r"\hat y": "clase finalmente elegida por el clasificador.",
        },
        steps=[
            "Para cada clase candidata $c$, el clasificador construye un puntaje proporcional a $P(y=c)\\prod_j P(x_j\\mid y=c)$.",
            "El término a priori $P(y=c)$ favorece clases frecuentes; los términos $P(x_j\\mid y=c)$ miden compatibilidad feature por feature.",
            "La hipótesis naïve no dice que los features sean independientes en general, sino sólo después de fijar la clase $y$.",
            "El logaritmo no cambia qué clase maximiza el puntaje, pero transforma productos de muchos términos pequeños en una suma numéricamente estable."
        ],
        expanded=True,
    )
    insight(
        "La frontera de decisión puede ser útil incluso si las probabilidades absolutas no están perfectamente calibradas. "
        "Esa es una de las razones por las que Naïve Bayes sobrevive como baseline serio."
    )
    pitfall(
        "Pensar que el denominador $P(x)$ se 'elimina' porque sea irrelevante conceptualmente. No lo es: sólo se cancela al comparar clases para una misma observación."
    )

    worked_example("Gaussian NB desde cero con numpy")
    st.markdown("Entrenamiento: para cada clase $c$ y cada feature $j$, estima $\\mu_{c,j}, \\sigma_{c,j}^2$ con los datos de la clase.")
    st.code(
        "import numpy as np\n"
        "class GaussianNBFromScratch:\n"
        "    def fit(self, X, y):\n"
        "        self.classes_ = np.unique(y)\n"
        "        self.priors_ = np.array([(y==c).mean() for c in self.classes_])\n"
        "        self.mu_ = np.array([X[y==c].mean(axis=0) for c in self.classes_])\n"
        "        self.var_ = np.array([X[y==c].var(axis=0) + 1e-9 for c in self.classes_])\n"
        "        return self\n"
        "    def predict_log_proba(self, X):\n"
        "        lp = np.log(self.priors_)[None, :]\n"
        "        for c_idx in range(len(self.classes_)):\n"
        "            diff2 = (X[:, None, :] - self.mu_[None, c_idx, :])**2\n"
        "            ll = -0.5*np.sum(np.log(2*np.pi*self.var_[c_idx]) + diff2/self.var_[c_idx], axis=-1)\n"
        "            lp = lp + 0  # see full impl; shown schematic\n"
        "        return lp\n",
        language="python"
    )
    st.markdown("Comparación con sklearn sobre el dataset Wine:")

    @st.cache_data
    def run_wine_nb_bundle(test_size=0.3, seed=42):
        data = load_wine()
        X, y = data.data, data.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        skn = GaussianNB().fit(X_tr, y_tr)
        acc_sk = accuracy_score(y_te, skn.predict(X_te))
        classes = np.unique(y_tr)
        mu = np.stack([X_tr[y_tr==c].mean(axis=0) for c in classes])
        var = np.stack([X_tr[y_tr==c].var(axis=0) + 1e-9 for c in classes])
        priors = np.array([(y_tr==c).mean() for c in classes])
        log_prior = np.log(priors)
        def predict_scratch(X):
            logprobs = np.zeros((len(X), len(classes)))
            for ci in range(len(classes)):
                ll = gaussian_logpdf_vector(X, mu[ci], var[ci]).sum(axis=1)
                logprobs[:, ci] = log_prior[ci] + ll
            return classes[np.argmax(logprobs, axis=1)]
        acc_np = accuracy_score(y_te, predict_scratch(X_te))
        return {
            "acc_sk": acc_sk,
            "acc_np": acc_np,
            "feature_names": list(data.feature_names),
            "target_names": list(data.target_names),
            "X_train": X_tr,
            "X_test": X_te,
            "y_train": y_tr,
            "y_test": y_te,
            "classes": classes,
            "mu": mu,
            "var": var,
            "priors": priors,
            "log_prior": log_prior,
            "X_all": X,
            "y_all": y,
        }

    bundle = run_wine_nb_bundle()
    acc_sk, acc_np = bundle["acc_sk"], bundle["acc_np"]
    fnames, tnames = bundle["feature_names"], bundle["target_names"]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy sklearn", f"{acc_sk:.3f}")
    c2.metric("Accuracy numpy desde cero", f"{acc_np:.3f}")
    st.caption(f"Dataset: Wine ({len(fnames)} features, {len(tnames)} clases: {', '.join(tnames)}).")

    interactive_header("Descomponer una predicción en evidencia atributo por atributo")
    interactive_guide(
        controls=[
            ("Observación del conjunto de test", "elige qué ejemplo real del dataset Wine quieres analizar."),
            ("Atributos más influyentes a mostrar", "cuántos atributos con mayor impacto en la decisión quieres inspeccionar."),
        ],
        procedure=(
            "Para la observación elegida, la app calcula el puntaje logarítmico de cada clase y luego compara atributo por atributo "
            "qué términos favorecen a la clase ganadora y cuáles a la segunda mejor."
        ),
        observe=(
            "Esto convierte la decisión del modelo en algo auditable: puedes ver qué atributos empujan fuertemente hacia una clase y cuáles generan duda."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        sample_idx = st.slider("Observación del conjunto de test", 0, len(bundle["X_test"]) - 1, 0, key="nb_sample")
        top_k = st.slider("Atributos más influyentes a mostrar", 3, min(10, len(fnames)), 6, key="nb_topk")
    x0 = bundle["X_test"][sample_idx]
    y_true = bundle["y_test"][sample_idx]
    log_lik = np.vstack([
        gaussian_logpdf_vector(x0, bundle["mu"][ci], bundle["var"][ci])
        for ci in range(len(bundle["classes"]))
    ])
    scores = bundle["log_prior"] + log_lik.sum(axis=1)
    probs = np.exp(scores - logsumexp(scores))
    ordering = np.argsort(scores)
    pred_idx = ordering[-1]
    runner_idx = ordering[-2]
    pred_class = bundle["classes"][pred_idx]
    runner_class = bundle["classes"][runner_idx]
    feature_margin = log_lik[pred_idx] - log_lik[runner_idx]
    top_idx = np.argsort(np.abs(feature_margin))[-top_k:][::-1]
    with col1:
        st.metric("Clase predicha", tnames[pred_class])
        st.metric("Clase real", tnames[y_true])
        st.metric("Posterior normalizado", f"{probs[pred_idx]:.4f}")
        st.metric("Margen sobre segunda clase", f"{scores[pred_idx] - scores[runner_idx]:.3f}")
        st.latex(r"s_c(x)=\log P(y=c)+\sum_j \log P(x_j\mid y=c)")
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
        axes[0].bar(
            tnames,
            scores,
            color=["#4C72B0" if i == pred_idx else "#BFC8D6" for i in range(len(scores))],
        )
        axes[0].set_ylabel("score logarítmico")
        axes[0].set_title("Puntaje por clase")
        axes[1].barh([fnames[i] for i in top_idx][::-1], feature_margin[top_idx][::-1], color="#DD8452")
        axes[1].axvline(0, color="black", lw=1)
        axes[1].set_xlabel(f"Ventaja de {tnames[pred_class]} sobre {tnames[runner_class]}")
        axes[1].set_title("Contribuciones más decisivas")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(
        pd.DataFrame(
            {
                "feature": [fnames[i] for i in top_idx],
                "valor observado": [round(float(x0[i]), 4) for i in top_idx],
                f"log p(x_j|{tnames[pred_class]})": [round(float(log_lik[pred_idx, i]), 4) for i in top_idx],
                f"log p(x_j|{tnames[runner_class]})": [round(float(log_lik[runner_idx, i]), 4) for i in top_idx],
                "diferencia": [round(float(feature_margin[i]), 4) for i in top_idx],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    how_to_read(
        "Una diferencia positiva favorece la clase predicha; una negativa favorece a la segunda mejor clase. "
        "La decisión final es la suma de todas estas contribuciones más el prior."
    )

    interactive_header("Visualizar frontera con 2 features (Wine)")
    col1, col2 = st.columns([1, 2])
    with col1:
        feat_x = st.selectbox("Feature X", options=list(range(len(fnames))),
                              format_func=lambda i: fnames[i], index=0, key="nb_fx")
        feat_y = st.selectbox("Feature Y", options=list(range(len(fnames))),
                              format_func=lambda i: fnames[i], index=6, key="nb_fy")

    @st.cache_data(show_spinner=False)
    def _nb_decision_boundary(fx: int, fy: int):
        X2 = bundle["X_all"][:, [fx, fy]]
        y2 = bundle["y_all"]
        model = GaussianNB().fit(X2, y2)
        x_min, x_max = X2[:, 0].min()-0.5, X2[:, 0].max()+0.5
        y_min, y_max = X2[:, 1].min()-0.5, X2[:, 1].max()+0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        return xx, yy, Z, X2, y2

    xx, yy, Z, X2, y2 = _nb_decision_boundary(feat_x, feat_y)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
        for c in np.unique(y2):
            ax.scatter(X2[y2==c, 0], X2[y2==c, 1], label=tnames[c], s=22, edgecolor="k", alpha=0.8)
        ax.set_xlabel(fnames[feat_x]); ax.set_ylabel(fnames[feat_y])
        ax.legend()
        st.pyplot(fig)
    how_to_read("Zonas coloreadas: región donde el clasificador predice cada clase. Puntos: datos reales coloreados por su etiqueta verdadera.")

    self_check_header()
    quiz(
        "¿Por qué se llama «naïve»?",
        ["Porque sólo funciona con pocos datos",
         "Porque asume que los features son independientes dada la clase",
         "Porque ignora las probabilidades a priori"],
        1,
        "La asunción de independencia condicional es típicamente falsa pero simplifica mucho el cálculo.",
        "Revisa la fórmula: el producto $\\prod P(x_j\\mid y)$ sólo se justifica bajo independencia condicional.",
        key="nb_q1"
    )
    quiz(
        "¿Por qué usar log en la decisión MAP?",
        ["Por velocidad", "Porque evita underflow numérico al multiplicar muchas probabilidades pequeñas", "Para ganar precisión simbólica"],
        1,
        "Productos de 0.001 colapsan a 0 en float32; sumar logs los mantiene estables.",
        "El log convierte producto en suma y evita que los productos se vayan a 0.",
        key="nb_q2"
    )
    ai_bridge(
        "Naïve Bayes es el **baseline** clásico en NLP (clasificación de texto, detección de spam). Aunque "
        "las palabras claramente no son independientes, el clasificador es robusto porque la **frontera de decisión** "
        "suele colocarse bien aunque las probabilidades absolutas estén mal calibradas."
    )

# ==================================================================
# SECCIÓN 6 — VARIABLES ALEATORIAS: PMF, PDF, CDF
# ==================================================================
def sec_va_cdf():
    section_title(
        "6. Variables Aleatorias: PMF, PDF y CDF",
        "El objeto matemático que convierte eventos en números manejables."
    )
    motivation(
        "Una **variable aleatoria** $X$ no es «aleatoria», es una *función* que asigna un número a cada "
        "resultado del espacio muestral. Esto nos permite sumarlas, promediarlas, graficarlas y, sobre todo, "
        "caracterizarlas por su **distribución**: PMF (discreta), PDF (continua) o CDF (ambas)."
    )
    prerequisites_box(
        "- Espacio muestral $\\Omega$, eventos, probabilidad.\n"
        "- Función $X: \\Omega \\to \\mathbb{R}$.\n"
        "- Integral para continuas (a nivel intuitivo)."
    )
    st.markdown("### Construcción")
    st.markdown(
        "**VA discreta** toma valores numerables ($\\{0,1,2,...\\}$). La describe su **PMF**:\n"
    )
    st.latex(r"p_X(k) = P(X = k), \quad \sum_k p_X(k) = 1")
    st.markdown("**VA continua** toma valores en un intervalo. La describe su **PDF**:")
    st.latex(r"P(a < X \leq b) = \int_a^b f_X(x)\,dx, \quad \int_{-\infty}^{\infty} f_X(x)\,dx = 1")
    st.warning("Para VA continuas, $P(X=x) = 0$ para todo $x$. Sólo tienen probabilidad los *intervalos*.")
    st.markdown("**CDF — definición formal** (unifica discretas y continuas):")
    st.latex(r"F_X(x) = P(X \leq x)")
    st.markdown("Propiedades de la CDF (que la caracterizan):")
    st.markdown(
        "1. **Monótona no decreciente**: $x_1 < x_2 \\Rightarrow F(x_1) \\leq F(x_2)$.\n"
        "2. **Límites**: $\\lim_{x\\to-\\infty}F(x)=0$, $\\lim_{x\\to\\infty}F(x)=1$.\n"
        "3. **Continua por la derecha**: $\\lim_{h\\to 0^+} F(x+h) = F(x)$.\n"
        "4. Relación con PMF/PDF:"
    )
    st.latex(r"\text{Discreta: } F(x) = \sum_{k\leq x} p(k); \quad \text{Continua: } F(x) = \int_{-\infty}^x f(u)\,du, \quad f(x) = F'(x)")
    formula_walkthrough(
        "Qué cambia entre PMF, PDF y CDF",
        terms={
            "PMF": "Asigna masa puntual a cada valor posible de una variable discreta.",
            "PDF": "No da probabilidad puntual; da densidad. La probabilidad real se obtiene integrando sobre intervalos.",
            "CDF": "Acumula probabilidad desde $-\\infty$ hasta el punto consultado y existe tanto en el caso discreto como continuo.",
        },
        steps=[
            "En variables discretas, puedes sumar probabilidades de puntos individuales porque esos puntos tienen masa positiva.",
            "En variables continuas, un punto aislado no tiene ancho y por eso su probabilidad es cero; la densidad sólo adquiere significado al integrarse.",
            "La CDF unifica ambos mundos: siempre responde 'qué probabilidad se ha acumulado hasta aquí'.",
            "En continuas diferenciables, la PDF es la derivada de la CDF; en discretas, la CDF avanza por saltos."
        ],
        expanded=True,
    )
    st.markdown(
        "La diferencia sustantiva es esta: en el mundo discreto puedes repartir probabilidad como fichas sobre puntos aislados; "
        "en el continuo sólo puedes repartir densidad y recuperar probabilidad acumulando área sobre intervalos."
    )
    pitfall(
        "Leer $f(x)$ como si fuera una probabilidad. En una continua, una densidad puede ser mayor que 1 sin violar nada; "
        "la probabilidad siempre vive en áreas bajo la curva."
    )

    worked_example("dos monedas justas — $X$ = número de caras")
    st.markdown("$\\Omega = \\{CC, CX, XC, XX\\}$, cada resultado con probabilidad $1/4$. $X(CC)=2, X(CX)=X(XC)=1, X(XX)=0$.")
    df_moneda = pd.DataFrame({
        "k": [0, 1, 2],
        "P(X=k)": [0.25, 0.5, 0.25],
        "F(k)=P(X≤k)": [0.25, 0.75, 1.0]
    })
    st.dataframe(df_moneda, hide_index=True)

    worked_example("urna con 3 rojas y 2 azules, extraer 2 sin reemplazo, $X$ = # rojas")
    st.markdown("PMF usando combinatoria (hipergeométrica):")
    st.latex(r"P(X=k) = \frac{\binom{3}{k}\binom{2}{2-k}}{\binom{5}{2}}, \quad k \in \{0,1,2\}")
    vals = [0,1,2]
    pmf = [comb(3, k, exact=True)*comb(2, 2-k, exact=True) / comb(5, 2, exact=True) for k in vals]
    st.dataframe(pd.DataFrame({"k": vals, "P(X=k)": pmf, "F(k)": np.cumsum(pmf)}), hide_index=True)

    interactive_header("Probabilidad de intervalos: masa/densidad y CDF en paralelo")
    interactive_guide(
        controls=[
            ("Distribución", "elige si quieres estudiar una variable discreta o continua."),
            ("Parámetros", "determinan la forma concreta de la distribución elegida."),
            ("Intervalo [a,b]", "es el rango de valores cuya probabilidad quieres calcular."),
        ],
        procedure=(
            "La app calcula la probabilidad del intervalo de dos maneras equivalentes: como suma o área en la gráfica de masa/densidad, "
            "y como diferencia de valores de la CDF."
        ),
        observe=(
            "En discretas verás barras individuales que se suman; en continuas verás un área sombreada. "
            "En ambos casos la CDF resume exactamente esa misma probabilidad acumulada."
        ),
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        dist_type = st.radio("Distribución", ["Binomial (discreta)", "Normal (continua)", "Exponencial (continua)"], key="cdf_kind")
        if dist_type.startswith("Binomial"):
            n = st.slider("n", 5, 40, 20, key="cdf_bin_n")
            p = st.slider("p", 0.05, 0.95, 0.40, step=0.05, key="cdf_bin_p")
            a_bin, b_bin = st.slider("Intervalo [a,b]", 0, n, (6, 10), key="cdf_bin_int")
        elif dist_type.startswith("Normal"):
            mu = st.slider("μ", -3.0, 3.0, 0.0, step=0.1, key="cdf_norm_mu")
            sig = st.slider("σ", 0.2, 3.0, 1.0, step=0.1, key="cdf_norm_sig")
            a_cont, b_cont = st.slider("Intervalo [a,b]", mu - 4 * sig, mu + 4 * sig, (mu - sig, mu + sig), key="cdf_norm_int")
        else:
            lam = st.slider("λ", 0.2, 3.0, 1.0, step=0.1, key="cdf_exp_lam")
            a_cont, b_cont = st.slider("Intervalo [a,b]", 0.0, 8.0 / lam, (0.5 / lam, 2.0 / lam), key="cdf_exp_int")
    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
        if dist_type.startswith("Binomial"):
            ks = np.arange(0, n + 1)
            pmf_vals = stats.binom.pmf(ks, n, p)
            cdf_vals = stats.binom.cdf(ks, n, p)
            mask = (ks >= a_bin) & (ks <= b_bin)
            interval_prob = pmf_vals[mask].sum()
            ax1.bar(ks, pmf_vals, color=["#4C72B0" if m else "#CFD7E3" for m in mask])
            ax1.set_title(f"PMF Binomial({n}, {p:.2f})")
            ax1.set_xlabel("k")
            ax2.step(ks, cdf_vals, where="post", color="#DD8452")
            ax2.axvline(a_bin - 1e-6, color="gray", ls=":")
            ax2.axvline(b_bin, color="gray", ls=":")
            ax2.set_title("CDF")
            ax2.set_xlabel("k")
            st.latex(
                rf"P({a_bin}\le X \le {b_bin})=\sum_{{k={a_bin}}}^{{{b_bin}}} p_X(k)=F({b_bin})-F({a_bin-1})={interval_prob:.4f}"
            )
        elif dist_type.startswith("Normal"):
            xs = np.linspace(mu - 4 * sig, mu + 4 * sig, 600)
            pdf_vals = stats.norm.pdf(xs, mu, sig)
            cdf_vals = stats.norm.cdf(xs, mu, sig)
            interval_prob = stats.norm.cdf(b_cont, mu, sig) - stats.norm.cdf(a_cont, mu, sig)
            mask = (xs >= a_cont) & (xs <= b_cont)
            ax1.plot(xs, pdf_vals, color="#4C72B0")
            ax1.fill_between(xs[mask], pdf_vals[mask], color="#4C72B0", alpha=0.35)
            ax1.set_title(f"PDF N({mu:.2f}, {sig**2:.2f})")
            ax2.plot(xs, cdf_vals, color="#DD8452")
            ax2.axvline(a_cont, color="gray", ls=":")
            ax2.axvline(b_cont, color="gray", ls=":")
            ax2.set_title("CDF")
            st.latex(
                rf"P({a_cont:.2f}\le X \le {b_cont:.2f})=F({b_cont:.2f})-F({a_cont:.2f})={interval_prob:.4f}"
            )
        else:
            xs = np.linspace(0, 8.0 / lam, 600)
            pdf_vals = stats.expon.pdf(xs, scale=1 / lam)
            cdf_vals = stats.expon.cdf(xs, scale=1 / lam)
            interval_prob = stats.expon.cdf(b_cont, scale=1 / lam) - stats.expon.cdf(a_cont, scale=1 / lam)
            mask = (xs >= a_cont) & (xs <= b_cont)
            ax1.plot(xs, pdf_vals, color="#4C72B0")
            ax1.fill_between(xs[mask], pdf_vals[mask], color="#4C72B0", alpha=0.35)
            ax1.set_title(f"PDF Exp({lam:.2f})")
            ax2.plot(xs, cdf_vals, color="#DD8452")
            ax2.axvline(a_cont, color="gray", ls=":")
            ax2.axvline(b_cont, color="gray", ls=":")
            ax2.set_title("CDF")
            st.latex(
                rf"P({a_cont:.2f}\le X \le {b_cont:.2f})=F({b_cont:.2f})-F({a_cont:.2f})={interval_prob:.4f}"
            )
        for ax in (ax1, ax2):
            ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    how_to_read(
        "En la figura izquierda se marca la masa o el área del intervalo consultado. La figura derecha muestra "
        "la misma probabilidad leída como diferencia de valores de la CDF."
    )

    self_check_header()
    quiz(
        "¿Puede $F(x)$ tomar valor $1.2$ en algún punto?",
        ["Sí, si la PDF es muy alta", "No, siempre está en [0,1]", "Sólo para VA discretas"],
        1,
        "$F(x) = P(X\\leq x) \\in [0,1]$ por definición.",
        "Es una probabilidad: $[0,1]$.",
        key="cdf_q1"
    )
    quiz(
        "Para una VA continua, $P(X = 3)$ vale...",
        ["$f(3)$", "0", "$F(3)$"],
        1,
        "Las continuas no ponen masa en puntos; sólo en intervalos.",
        "Recuerda: en continuas $P(X=c)=0$. La densidad $f(3)$ no es una probabilidad.",
        key="cdf_q2"
    )
    ai_bridge(
        "La CDF es clave en **muestreo inverso** (inverse-CDF sampling): para generar $X$ con distribución $F$, "
        "basta muestrear $U \\sim \\text{Unif}(0,1)$ y devolver $F^{-1}(U)$. Es como se generan variables aleatorias "
        "exóticas en simulaciones, y aparece en **diffusion models** y **flow-based generative models**."
    )

# ==================================================================
# SECCIÓN 7 — CATÁLOGO DE DISTRIBUCIONES
# ==================================================================
def sec_distribuciones():
    section_title(
        "7. Catálogo de Distribuciones",
        "Los patrones reutilizables que describen la mayoría de fenómenos aleatorios."
    )
    motivation(
        "No inventas una distribución nueva cada vez: hay una docena de **familias paramétricas** que modelan "
        "casi todo (éxito/fracaso, conteos, tiempos de espera, errores, proporciones...). Conocer su forma, "
        "sus parámetros y cuándo aplicarlas es un superpoder."
    )
    prerequisites_box(
        "- PMF, PDF, CDF.\n"
        "- Factorial y $\\binom{n}{k}$.\n"
        "- Integral (usaremos integrales definidas sólo como cajas negras)."
    )
    st.markdown(
        "Un punto importante: una distribución no es sólo una fórmula. Es una **historia probabilística** sobre qué variable estás modelando, "
        "qué significan sus parámetros y qué comportamientos quedan descartados por esa elección."
    )

    tabs = st.tabs([
        "Discretas", "Continuas", "Aproximaciones", "Memoryless (Exp)", "Explorador interactivo"
    ])

    with tabs[0]:
        st.markdown("#### Distribuciones discretas")
        st.markdown(
            "| Distribución | PMF | $E[X]$ | $\\text{Var}[X]$ | Cuándo usar |\n"
            "|---|---|---|---|---|\n"
            "| **Bernoulli($p$)** | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | Un ensayo con éxito/fracaso |\n"
            "| **Binomial($n,p$)** | $\\binom{n}{x}p^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | # éxitos en $n$ ensayos |\n"
            "| **Geométrica($p$)** | $(1-p)^{x-1}p$ | $1/p$ | $(1-p)/p^2$ | # ensayos hasta 1er éxito |\n"
            "| **Hipergeom.** | $\\binom{K}{x}\\binom{N-K}{n-x}/\\binom{N}{n}$ | $nK/N$ | ... | muestreo sin reemplazo |\n"
            "| **Poisson($\\lambda$)** | $\\lambda^x e^{-\\lambda}/x!$ | $\\lambda$ | $\\lambda$ | # eventos raros por unidad de tiempo |\n"
            "| **Multinomial($n,\\mathbf p$)** | $\\frac{n!}{x_1!\\cdots x_k!}p_1^{x_1}\\cdots p_k^{x_k}$ | $np_i$ | $np_i(1-p_i)$ | # éxitos en $k$ categorías |\n"
        )
        st.info("**Varianza de Bernoulli** = $p(1-p)$ → máxima en $p=0.5$ (máxima incertidumbre).")

    with tabs[1]:
        st.markdown("#### Distribuciones continuas")
        st.markdown(
            "| Distribución | PDF | $E[X]$ | $\\text{Var}[X]$ | Cuándo usar |\n"
            "|---|---|---|---|---|\n"
            "| **Uniforme($a,b$)** | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | Equiprobable en intervalo |\n"
            "| **Normal($\\mu,\\sigma^2$)** | $\\frac{1}{\\sqrt{2\\pi}\\sigma}e^{-(x-\\mu)^2/2\\sigma^2}$ | $\\mu$ | $\\sigma^2$ | Suma de muchos efectos (CLT) |\n"
            "| **Exponencial($\\lambda$)** | $\\lambda e^{-\\lambda x}$ | $1/\\lambda$ | $1/\\lambda^2$ | Tiempo entre eventos raros |\n"
            "| **Gamma($\\alpha,\\lambda$)** | $\\frac{\\lambda^\\alpha}{\\Gamma(\\alpha)}x^{\\alpha-1}e^{-\\lambda x}$ | $\\alpha/\\lambda$ | $\\alpha/\\lambda^2$ | Suma de Exp; tiempos de espera |\n"
            "| **Beta($\\alpha,\\beta$)** | $\\propto x^{\\alpha-1}(1-x)^{\\beta-1}$ | $\\alpha/(\\alpha+\\beta)$ | ... | Prior de probabilidades |\n"
            "| **Student-$t_\\nu$** | $\\propto(1+x^2/\\nu)^{-(\\nu+1)/2}$ | $0$ ($\\nu>1$) | $\\nu/(\\nu-2)$ ($\\nu>2$) | Muestras pequeñas, colas pesadas |\n"
        )
        st.latex(r"\text{Integral gaussiana (útil recordar):}\quad \int_{-\sigma}^{\sigma} f_{N(0,\sigma^2)}(x)\,dx \approx \tfrac{2}{3} \ \Rightarrow\ P(\mu-\sigma\le X\le \mu+\sigma)\approx 68\%")

    with tabs[2]:
        st.markdown("#### Aproximaciones útiles a Binomial")
        st.markdown(
            "- **Binomial → Poisson** cuando $n$ grande, $p$ pequeño, $\\lambda=np$ moderado "
            "(regla: $n>20$ y $np<10$).\n"
            "- **Binomial → Normal** cuando $np(1-p)\\gtrsim 10$:\n"
        )
        st.latex(r"\text{Bin}(n,p) \approx \mathcal N\!\left(np,\,np(1-p)\right)")
        n = st.slider("n (para Binomial)", 10, 300, 50, key="apx_n")
        p = st.slider("p", 0.01, 0.5, 0.1, key="apx_p")
        ks = np.arange(0, n+1)
        bin_pmf = stats.binom.pmf(ks, n, p)
        poi_pmf = stats.poisson.pmf(ks, n*p)
        xs = np.linspace(0, n, 400)
        nor_pdf = stats.norm.pdf(xs, loc=n*p, scale=np.sqrt(n*p*(1-p)))
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(ks, bin_pmf, color="#4C72B0", alpha=0.6, label="Binomial")
        ax.plot(ks, poi_pmf, "o-", color="#DD8452", label="Poisson (λ=np)")
        ax.plot(xs, nor_pdf, color="#55A868", lw=2, label="Normal (aprox)")
        ax.set_xlim(0, max(2*n*p + 3, 15)); ax.legend()
        st.pyplot(fig); plt.close(fig)
        how_to_read("Cuando $n$ crece y $p$ es chica, Poisson pega el punto discreto. Cuando $np(1-p)$ es suficientemente grande, la curva Normal alinea su perfil con la Binomial.")

    with tabs[3]:
        worked_example("propiedad de falta de memoria (GPU)")
        st.markdown(
            "Una GPU tiene un tiempo de vida $T \\sim \\text{Exp}(\\lambda)$ con media $100$ hrs ($\\lambda=1/100$). "
            "Llevo 50 hrs usándola y sigue funcionando. ¿Probabilidad de que dure otras 100?\n\n"
            "La propiedad de **falta de memoria** de la exponencial:"
        )
        st.latex(r"P(T > s + t \mid T > s) = P(T > t)")
        st.latex(r"\Rightarrow P(T > 150 \mid T > 50) = P(T > 100) = e^{-100/100} = e^{-1} \approx 0.368")
        st.info(
            "La Exponencial **no recuerda cuánto llevaba funcionando**. La probabilidad de durar 100 hrs "
            "más es la misma estando nuevo o con 50 hrs. Es la *única* distribución continua con esta propiedad."
        )

    with tabs[4]:
        interactive_header("Explorador de distribuciones")
        interactive_guide(
            controls=[
                ("Distribución", "elige la familia probabilística que quieres estudiar."),
                ("Parámetros", "controlan ubicación, dispersión, asimetría o escala según la familia."),
            ],
            procedure=(
                "La app dibuja la PMF o la PDF de la familia elegida con los parámetros seleccionados."
            ),
            observe=(
                "Pregunta siempre qué cambia al mover cada parámetro: si desplaza la distribución, si la vuelve más dispersa, "
                "si concentra masa cerca de cero o si hace las colas más pesadas."
            ),
        )
        dist = st.selectbox(
            "Distribución",
            ["Bernoulli","Binomial","Poisson","Geométrica","Normal","Exponencial","Gamma","Beta","Student-t","Uniforme"],
            key="exp_dist"
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            if dist == "Bernoulli":
                p = st.slider("p", 0.0, 1.0, 0.5, key="exp_p")
                xs = np.array([0, 1]); pmf = np.array([1-p, p]); is_disc = True
            elif dist == "Binomial":
                n = st.slider("n", 1, 100, 20, key="exp_n")
                p = st.slider("p", 0.0, 1.0, 0.4, key="exp_p2")
                xs = np.arange(0, n+1); pmf = stats.binom.pmf(xs, n, p); is_disc = True
            elif dist == "Poisson":
                lam = st.slider("λ", 0.1, 20.0, 3.0, key="exp_lam")
                xs = np.arange(0, int(3*lam)+5); pmf = stats.poisson.pmf(xs, lam); is_disc = True
            elif dist == "Geométrica":
                p = st.slider("p", 0.01, 1.0, 0.3, key="exp_gp")
                xs = np.arange(1, 30); pmf = stats.geom.pmf(xs, p); is_disc = True
            elif dist == "Normal":
                mu = st.slider("μ", -5.0, 5.0, 0.0, key="exp_mu"); sig = st.slider("σ", 0.1, 4.0, 1.0, key="exp_sig")
                xs = np.linspace(mu-4*sig, mu+4*sig, 400); pmf = stats.norm.pdf(xs, mu, sig); is_disc = False
            elif dist == "Exponencial":
                lam = st.slider("λ", 0.1, 3.0, 1.0, key="exp_elam")
                xs = np.linspace(0, 5/lam, 400); pmf = stats.expon.pdf(xs, scale=1/lam); is_disc = False
            elif dist == "Gamma":
                a = st.slider("α", 0.5, 10.0, 2.0, key="exp_ga"); lam = st.slider("λ", 0.1, 3.0, 1.0, key="exp_glam")
                xs = np.linspace(0, (a+3)/lam, 400); pmf = stats.gamma.pdf(xs, a, scale=1/lam); is_disc = False
            elif dist == "Beta":
                a = st.slider("α", 0.1, 10.0, 2.0, key="exp_ba"); b = st.slider("β", 0.1, 10.0, 2.0, key="exp_bb")
                xs = np.linspace(0.001, 0.999, 400); pmf = stats.beta.pdf(xs, a, b); is_disc = False
            elif dist == "Student-t":
                nu = st.slider("ν", 1, 50, 5, key="exp_nu")
                xs = np.linspace(-6, 6, 400); pmf = stats.t.pdf(xs, nu); is_disc = False
            else:
                a = st.slider("a", -5.0, 5.0, 0.0, key="exp_ua"); b = st.slider("b", a+0.1, a+10.0, a+1.0, key="exp_ub")
                xs = np.linspace(a-0.5, b+0.5, 400); pmf = stats.uniform.pdf(xs, a, b-a); is_disc = False
        with col2:
            fig, ax = plt.subplots(figsize=(7.5, 3.3))
            if is_disc:
                ax.bar(xs, pmf, color="#4C72B0")
                ax.set_ylabel("P(X=k)")
            else:
                ax.fill_between(xs, pmf, color="#4C72B0", alpha=0.55)
                ax.plot(xs, pmf, color="#4C72B0")
                ax.set_ylabel("f(x)")
            ax.set_title(dist)
            st.pyplot(fig); plt.close(fig)

    self_check_header()
    quiz(
        "Para $\\text{Bernoulli}(p)$ la varianza es máxima cuando...",
        ["$p=0$", "$p=0.5$", "$p=1$"],
        1,
        "$p(1-p)$ es máxima en $p=0.5$ (máxima incertidumbre).",
        "Deriva $p(1-p)$ e iguala a cero, o grafica la parábola.",
        key="dist_q1"
    )
    quiz(
        "Sigo lanzando una moneda justa hasta la primera cara. # de lanzamientos ~",
        ["Binomial", "Poisson", "Geométrica", "Normal"],
        2,
        "Número de ensayos hasta el primer éxito = Geométrica.",
        "Binomial cuenta éxitos en $n$ fijos; acá el contador es cuántos ensayos hasta 1er éxito.",
        key="dist_q2"
    )
    ai_bridge(
        "- **Bernoulli / Binomial / Categorical**: salidas de clasificación.\n"
        "- **Gaussiana**: ruido en regresión, priors de pesos (L2 regularization = prior $\\mathcal N(0,\\sigma^2)$).\n"
        "- **Poisson / Exponencial**: conteos y latencias (RL, colas, modelos de eventos).\n"
        "- **Beta / Dirichlet**: priors conjugados de probabilidades (topic models, bandits, Thompson sampling)."
    )

# ==================================================================
# SECCIÓN 8 — MLE Y ENTROPÍA CRUZADA
# ==================================================================
def sec_mle():
    section_title(
        "8. Máxima Verosimilitud (MLE) y Entropía Cruzada",
        "Cómo aprendemos parámetros a partir de datos observados."
    )
    motivation(
        "Dado un modelo probabilístico con parámetros desconocidos y un dataset, ¿qué parámetros explican "
        "mejor los datos? La respuesta clásica: los que hacen los datos **más probables**. Esto es MLE, "
        "y la *loss* de casi toda red neuronal supervisada es MLE disfrazada."
    )
    prerequisites_box(
        "- Distribuciones paramétricas (Bernoulli, Normal).\n"
        "- Derivadas (igualar a cero para encontrar extremos).\n"
        "- Independencia: $P(x_1,\\ldots,x_n \\mid \\theta) = \\prod_i P(x_i\\mid\\theta)$."
    )
    st.markdown("### Construcción")
    st.latex(r"\mathcal{L}(\theta) = \prod_{i=1}^n p(x_i \mid \theta) \quad \text{(verosimilitud)}")
    st.latex(r"\ell(\theta) = \log \mathcal{L}(\theta) = \sum_i \log p(x_i\mid\theta) \quad \text{(log-verosimilitud)}")
    st.latex(r"\hat\theta_{MLE} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \big[-\ell(\theta)\big] \quad \text{(NLL)}")
    formula_walkthrough(
        "Qué está optimizando realmente MLE",
        terms={
            r"\theta": "parámetro o conjunto de parámetros del modelo.",
            r"x_i": "observación número $i$ del dataset.",
            r"\mathcal{L}(\theta)": "verosimilitud: qué tan bien explica el parámetro a todos los datos observados.",
            r"\ell(\theta)": "log-verosimilitud: la misma información, pero en escala logarítmica.",
            r"\hat\theta_{MLE}": "valor del parámetro que maximiza la log-verosimilitud.",
        },
        steps=[
            "La verosimilitud $\\mathcal L(\\theta)$ se lee con los datos fijos y el parámetro variable: pregunta qué valor de $\\theta$ hace más plausibles las observaciones ya vistas.",
            "La independencia i.i.d. convierte una probabilidad conjunta difícil en un producto de términos sencillos.",
            "Tomar logaritmo no cambia el máximo porque el log es monótono creciente, pero convierte productos en sumas y vuelve la optimización mucho más tratable.",
            "Minimizar la negativa de la log-verosimilitud es sólo una convención conveniente: en aprendizaje automático solemos minimizar pérdidas."
        ],
        expanded=True,
    )
    insight(
        "MLE no 'adivina' parámetros verdaderos observando el futuro: selecciona el parámetro que explica mejor el dataset ya observado bajo el modelo elegido."
    )

    worked_example("MLE de Bernoulli = media muestral")
    st.markdown("Dados $n$ resultados $x_i\\in\\{0,1\\}$ con $k=\\sum x_i$ éxitos:")
    st.latex(r"\ell(p) = k\log p + (n-k)\log(1-p)")
    st.latex(r"\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \Rightarrow \hat p_{MLE} = \frac{k}{n}")
    st.markdown("Interpretación: el mejor estimador de $p$ es *simplemente la fracción observada de éxitos*.")

    worked_example("MLE de la Gaussiana")
    st.markdown("Dados $x_1,\\ldots,x_n \\sim \\mathcal{N}(\\mu, \\sigma^2)$:")
    st.latex(r"\hat\mu = \bar x = \frac{1}{n}\sum_i x_i, \quad \hat\sigma^2 = \frac{1}{n}\sum_i (x_i-\bar x)^2")

    worked_example("de NLL de Bernoulli → entropía cruzada binaria")
    st.markdown(
        "En clasificación, el modelo produce $\\hat y_i = P(y=1\\mid x_i; \\theta)$. La NLL es:"
    )
    st.latex(r"-\ell(\theta) = -\sum_i [y_i \log \hat y_i + (1-y_i)\log(1-\hat y_i)]")
    st.info("Eso es *exactamente* la **binary cross-entropy loss**. No son dos funciones distintas: minimizar BCE = MLE de Bernoulli.")

    interactive_header("Superficie de log-verosimilitud (Bernoulli)")
    interactive_guide(
        controls=[
            ("n lanzamientos", "cantidad total de observaciones Bernoulli disponibles."),
            ("k/n (éxitos observados)", "fracción de éxitos en la muestra."),
        ],
        procedure=(
            "Se construye la función de log-verosimilitud de un parámetro Bernoulli $p$ dado el número observado de éxitos y fracasos."
        ),
        observe=(
            "El máximo de la curva marca el valor de $p$ que mejor explica la muestra. "
            "Cuando $n$ crece, la curva se vuelve más aguda: pequeñas desviaciones del estimador óptimo se penalizan más."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.slider("n lanzamientos", 5, 200, 50, key="mle_n")
        k_ratio = st.slider("k/n (éxitos observados)", 0.0, 1.0, 0.6, key="mle_k")
        k = int(round(k_ratio * n))
        st.markdown(f"**Observado**: $k={k}, n-k={n-k}$")
        st.markdown(f"**MLE**: $\\hat p = {k/n:.3f}$")
    with col2:
        ps = np.linspace(0.001, 0.999, 400)
        ll = k*np.log(ps) + (n-k)*np.log(1-ps)
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.plot(ps, ll, color="#4C72B0", lw=2)
        ax.axvline(k/n, color="#DD8452", ls="--", label=f"$\\hat p={k/n:.2f}$")
        ax.set_xlabel("p"); ax.set_ylabel("log-verosimilitud")
        ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read("La curva es cóncava con un único máximo — ese máximo es $\\hat p$. Más datos → pico más estrecho → mayor certeza.")

    interactive_header("Cómo castiga la entropía cruzada una predicción")
    col1, col2 = st.columns([1, 2])
    with col1:
        y_obs = st.radio("Etiqueta observada y", [0, 1], horizontal=True, key="bce_y")
        q_hat = st.slider("Predicción q = P(y=1|x)", 0.001, 0.999, 0.7, step=0.001, key="bce_q")
        loss = -(y_obs * np.log(q_hat) + (1 - y_obs) * np.log(1 - q_hat))
        st.metric("Pérdida BCE del ejemplo", f"{loss:.4f}")
        if y_obs == 1:
            st.latex(rf"\text{{BCE}} = -\log({q_hat:.3f}) = {loss:.4f}")
        else:
            st.latex(rf"\text{{BCE}} = -\log(1-{q_hat:.3f}) = {loss:.4f}")
        st.markdown(
            "Aquí $q$ es la probabilidad que el modelo asigna a la clase 1. Si el dato observado es $y=1$, "
            "la pérdida castiga probabilidades pequeñas; si $y=0$, castiga probabilidades grandes."
        )
    with col2:
        qs = np.linspace(0.001, 0.999, 400)
        loss_y1 = -np.log(qs)
        loss_y0 = -np.log(1 - qs)
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.plot(qs, loss_y1, label="y=1", color="#4C72B0", lw=2)
        ax.plot(qs, loss_y0, label="y=0", color="#DD8452", lw=2)
        ax.scatter([q_hat], [loss], color="#222222", s=60, zorder=5)
        ax.set_ylim(0, 7)
        ax.set_xlabel("q = P(y=1|x)")
        ax.set_ylabel("BCE")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    how_to_read(
        "Cuando el modelo está seguro y se equivoca, la pérdida crece abruptamente. En cambio, una predicción segura y correcta tiene pérdida cercana a 0."
    )

    self_check_header()
    quiz(
        "¿Por qué trabajamos con log-verosimilitud en vez de la verosimilitud?",
        ["Porque es más grande", "Por estabilidad numérica y por convertir producto en suma (más fácil derivar)", "Porque es siempre positiva"],
        1,
        "Productos de muchas probabilidades pequeñas dan underflow; el log los convierte en sumas manejables.",
        "Piensa: ¿qué pasa si multiplicas mil numeritos como $10^{-5}$?",
        key="mle_q1"
    )
    ai_bridge(
        "Casi toda **loss** en deep learning es un caso particular de MLE:\n"
        "- MSE = MLE de $y = f(x) + \\mathcal N(0,\\sigma^2)$.\n"
        "- Binary cross-entropy = MLE de $y \\sim \\text{Bernoulli}(\\sigma(f(x)))$.\n"
        "- Categorical cross-entropy = MLE de $y \\sim \\text{Categorical}(\\text{softmax}(f(x)))$.\n"
        "Cuando optimizas una red con SGD, estás haciendo MLE estocástica."
    )

# ==================================================================
# SECCIÓN 9 — ESPERANZA, VARIANZA, JENSEN
# ==================================================================
def sec_esperanza_jensen():
    section_title(
        "9. Valor Esperado, Varianza y Desigualdad de Jensen",
        "Dos números que resumen una distribución + la desigualdad que relaciona esperanza y funciones no lineales."
    )
    motivation(
        "Aun sin conocer la distribución completa, dos números bastan para muchos propósitos: **dónde se "
        "concentra** (esperanza) y **cuán dispersa es** (varianza). Jensen nos da la regla fundamental para "
        "meter/sacar funciones del operador $E[\\cdot]$, y aparece en ELBO, bound de entropía, y muchos más."
    )
    prerequisites_box(
        "- PMF / PDF.\n"
        "- Sumas e integrales.\n"
        "- Función convexa: $f''\\ge 0$ (ej: $x^2$, $e^x$, $-\\log x$). Cóncava: $f''\\le 0$ (ej: $\\log x$, $\\sqrt x$)."
    )
    st.markdown("### Construcción")
    st.latex(r"E[X] = \sum_k k\,p_X(k) \quad \text{(discreta)}, \quad E[X] = \int x\,f_X(x)\,dx \quad \text{(continua)}")
    st.markdown("**Linealidad** (la propiedad más útil de $E$):")
    st.latex(r"E[aX+bY+c] = aE[X]+bE[Y]+c \quad \textbf{aunque } X,Y \text{ no sean independientes}")
    st.latex(r"\text{Var}(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2 \ge 0")
    st.latex(r"\text{Var}(aX+b) = a^2\text{Var}(X), \quad \text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y) \text{ si } X\perp Y")
    formula_walkthrough(
        "Cómo interpretar esperanza y varianza en lenguaje simple",
        terms={
            r"E[X]": "promedio de largo plazo o centro de masa de la distribución.",
            r"\text{Var}(X)": "medida de dispersión: cuánto se alejan típicamente los valores respecto de la media.",
            r"E[X^2] - E[X]^2": "forma computacional práctica de la varianza.",
        },
        steps=[
            "La esperanza no necesariamente coincide con un valor frecuente; resume dónde se equilibra probabilísticamente la distribución.",
            "La varianza compara cada valor con la media, eleva al cuadrado esas diferencias y luego las promedia.",
            "La linealidad de la esperanza es especialmente poderosa porque no requiere independencia."
        ],
    )

    worked_example("suma de dos dados = 7 en promedio")
    st.markdown("$X, Y \\sim \\text{Uniforme}\\{1,\\ldots,6\\}$, $E[X]=3.5$. Por linealidad: $E[X+Y]=7$. No hace falta sumar 36 casos.")

    worked_example("problema del guardarropa vía indicadores")
    st.markdown(
        "$n$ personas dejan sus abrigos y al salir cada uno recibe uno al azar. $X$ = # personas que reciben su propio abrigo.\n\n"
        "Truco: $X = \\sum_{i=1}^n X_i$ donde $X_i=1$ si la persona $i$ recibe su abrigo. $P(X_i=1)=1/n$."
    )
    st.latex(r"E[X] = \sum_i E[X_i] = \sum_i \tfrac{1}{n} = 1")
    st.info("**Sorprendente**: da 1 sin importar cuántas personas. Los $X_i$ NO son independientes, pero la linealidad no lo requiere.")

    worked_example("Varianza de Bernoulli")
    st.markdown("$X\\sim \\text{Bernoulli}(p)$: $E[X]=p$, $E[X^2]=p$.")
    st.latex(r"\text{Var}(X) = E[X^2] - E[X]^2 = p - p^2 = p(1-p)")

    st.markdown("### Desigualdad de Jensen")
    st.markdown("Si $f$ es **convexa**:")
    st.latex(r"E[f(X)] \geq f(E[X])")
    st.markdown("Si $f$ es **cóncava**:")
    st.latex(r"E[f(X)] \leq f(E[X])")
    st.markdown("Igualdad si y sólo si $f$ es lineal en el rango de $X$, o $X$ es constante.")

    interactive_header("Visualización de Jensen")
    interactive_guide(
        controls=[
            ("Función", "elige si quieres una función convexa o cóncava."),
            ("E[X]", "fija aproximadamente la ubicación central de la variable."),
            ("σ", "controla cuánta dispersión tiene la variable alrededor de su media."),
        ],
        procedure=(
            "Se simula una variable aleatoria con la media y dispersión elegidas, se evalúa $f$ sobre esa variable y luego se compara "
            "$E[f(X)]$ con $f(E[X])$."
        ),
        observe=(
            "Si la función es convexa, la dispersión empuja el promedio de $f(X)$ hacia arriba; si es cóncava, lo empuja hacia abajo."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        f_type = st.radio("Función", ["x² (convexa)", "log(x) (cóncava)", "e^x (convexa)"], key="jen_f")
        mu = st.slider("E[X]", 0.5, 4.0, 2.0, key="jen_mu")
        sig = st.slider("σ (dispersión de X)", 0.1, 2.0, 0.8, key="jen_sig")
    rng = np.random.default_rng(0)
    xs = rng.normal(mu, sig, 5000)
    if "x²" in f_type:
        f = lambda x: x**2; xp = np.linspace(max(0.01, mu-3*sig), mu+3*sig, 200)
    elif "log" in f_type:
        xs = np.clip(xs, 0.05, None)
        f = lambda x: np.log(x); xp = np.linspace(0.05, mu+3*sig, 200)
    else:
        f = lambda x: np.exp(x); xp = np.linspace(mu-3*sig, mu+3*sig, 200)
    EfX = np.mean(f(xs)); fEX = f(mu)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(xp, f(xp), color="#4C72B0", lw=2, label="f(x)")
        ax.scatter([mu], [fEX], color="#DD8452", s=90, zorder=5, label=f"f(E[X])={fEX:.2f}")
        ax.axhline(EfX, color="#55A868", ls="--", label=f"E[f(X)]={EfX:.2f}")
        ax.axvline(mu, color="gray", ls=":", alpha=0.5)
        ax.legend(); ax.set_xlabel("x")
        st.pyplot(fig); plt.close(fig)
    delta = EfX - fEX
    st.metric("E[f(X)] − f(E[X])", f"{delta:.4f}", "convexa → ≥0" if "convexa" in f_type else "cóncava → ≤0")

    self_check_header()
    quiz(
        "Si $X$ es constante, ¿qué pasa con la desigualdad de Jensen?",
        ["No aplica", "Se cumple con igualdad", "Se invierte"],
        1,
        "Si $X=c$ entonces $E[f(X)] = f(c) = f(E[X])$.",
        "Piensa: sin variabilidad, no hay nada que disperse.",
        key="jen_q1"
    )
    quiz(
        "10 personas entran al guardarropa. $E[\\#\\text{aciertos}]$ vale...",
        ["0.1", "1", "10", "depende"],
        1,
        "Por indicadores: $E=\\sum_i 1/n = 1$, independiente de $n$.",
        "No hace falta calcular permutaciones; usa linealidad + indicadores.",
        key="jen_q2"
    )
    ai_bridge(
        "**ELBO** (Evidence Lower Bound), core de **VAEs** y **variational inference**, sale de aplicar "
        "Jensen a $\\log$ (cóncava):"
    )
    st.latex(r"\log p(x) = \log \int p(x,z)dz \geq \int q(z)\log\frac{p(x,z)}{q(z)}dz = \text{ELBO}")
    st.markdown(
        "Sin Jensen, no hay VAE. Jensen también aparece en **EM**, **contrastive learning**, y bounds de entropía."
    )

# ==================================================================
# SECCIÓN 10 — FGM, COVARIANZA, CORRELACIÓN
# ==================================================================
def sec_fgm_cov():
    section_title(
        "10. FGM, Covarianza y Correlación",
        "Herramientas para comparar distribuciones (FGM) y cuantificar relaciones lineales entre variables."
    )
    motivation(
        "La **FGM** empaqueta *todos* los momentos ($E[X], E[X^2], ...$) en una sola función: si dos VAs "
        "tienen la misma FGM, son iguales en distribución. Útil para probar propiedades de sumas. "
        "La **covarianza** y **correlación** miden qué tan juntas se mueven dos VAs."
    )
    prerequisites_box(
        "- $E[X]$, $\\text{Var}(X)$.\n"
        "- Serie de Taylor de $e^{tX}$.\n"
        "- Derivadas parciales."
    )
    st.markdown("### Construcción")
    st.latex(r"M_X(t) = E[e^{tX}] = 1 + tE[X] + \tfrac{t^2}{2!}E[X^2] + \tfrac{t^3}{3!}E[X^3] + \ldots")
    st.latex(r"M_X^{(k)}(0) = E[X^k]")
    st.markdown("**Propiedades clave**:")
    st.markdown(
        "- Unicidad: misma FGM ⇒ misma distribución.\n"
        "- Suma de independientes: $M_{X+Y}(t) = M_X(t)\\,M_Y(t)$.\n"
        "- Ejemplo Gamma($\\alpha, \\lambda$): $M(t) = (\\lambda/(\\lambda-t))^\\alpha$ para $t<\\lambda$."
    )
    worked_example("E[X] y Var[X] de Gamma usando FGM")
    st.latex(r"M'(0) = \alpha/\lambda = E[X], \quad M''(0) = \alpha(\alpha+1)/\lambda^2 \Rightarrow \text{Var}(X)=\alpha/\lambda^2")

    st.markdown("### Covarianza y correlación")
    st.latex(r"\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]")
    st.latex(r"\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1, 1]")
    st.markdown("- $\\rho=0$: sin relación lineal (NO implica independencia salvo en gaussianas).")
    st.markdown("- $\\text{Var}(X+Y) = \\text{Var}(X) + \\text{Var}(Y) + 2\\text{Cov}(X,Y)$.")
    formula_walkthrough(
        "Por qué covarianza y correlación no capturan toda la dependencia",
        terms={
            r"\text{Cov}(X,Y)": "mide si $X$ y $Y$ tienden a estar simultáneamente sobre o bajo sus medias.",
            r"\rho(X,Y)": "covarianza normalizada por las desviaciones estándar, para obtener una escala entre -1 y 1.",
            r"E[XY]": "promedio del producto conjunto; recoge si valores grandes de una variable tienden a convivir con valores grandes de la otra.",
        },
        steps=[
            "La covarianza compara variaciones conjuntas alrededor de las medias. Si cuando $X$ está sobre su media, $Y$ también tiende a estarlo, la covarianza es positiva.",
            "La correlación normaliza esa covarianza por las escalas de ambas variables para producir un número entre -1 y 1.",
            "Ambas medidas son lineales: detectan alineación tipo recta. Si la relación real es curvada o simétrica, pueden dar 0 aun cuando exista dependencia fuerte.",
            "En gaussianas, linealidad y dependencia coinciden de forma especial; fuera de ese caso, no."
        ],
    )

    worked_example("Cov(X, X+Y) con X, Y independientes de varianza 1")
    st.latex(r"\text{Cov}(X, X+Y) = \text{Cov}(X,X) + \text{Cov}(X,Y) = \text{Var}(X) + 0 = 1")

    interactive_header("Dependencia lineal y no lineal")
    tabs = st.tabs(["Correlación ajustable", "Dependencia con ρ≈0"])
    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            rho = st.slider("ρ objetivo", -1.0, 1.0, 0.7, step=0.05, key="cov_rho")
            n = st.slider("n muestras", 50, 2000, 400, key="cov_n")
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((n, 2))
        L = np.array([[1.0, 0.0], [rho, np.sqrt(max(1 - rho**2, 1e-9))]])
        X = Z @ L.T
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            emp_rho = np.corrcoef(X.T)[0, 1]
            ax.set_title(f"ρ teórica={rho:.2f}, ρ empírica={emp_rho:.2f}")
            st.pyplot(fig)
            plt.close(fig)
        how_to_read("Cuando la nube se alinea sobre una recta ascendente o descendente, la correlación captura bien la estructura.")
    with tabs[1]:
        col1, col2 = st.columns([1, 2])
        with col1:
            relation = st.selectbox(
                "Patrón de dependencia",
                ["Parábola: Y = X² + ruido", "Círculo con radio casi fijo", "Seno: Y = sin(X) + ruido"],
                key="cov_nonlin",
            )
            n2 = st.slider("n muestras", 100, 3000, 700, key="cov_n2")
            noise = st.slider("Ruido", 0.0, 1.0, 0.15, step=0.05, key="cov_noise")
        rng = np.random.default_rng(22)
        if relation.startswith("Parábola"):
            x = rng.normal(0, 1, n2)
            y = x**2 + noise * rng.normal(size=n2)
        elif relation.startswith("Círculo"):
            ang = rng.uniform(0, 2 * np.pi, n2)
            rad = 1 + noise * rng.normal(size=n2)
            x = rad * np.cos(ang)
            y = rad * np.sin(ang)
        else:
            x = rng.uniform(-np.pi, np.pi, n2)
            y = np.sin(x) + noise * rng.normal(size=n2)
        emp_cov = np.cov(x, y, ddof=1)[0, 1]
        emp_rho = np.corrcoef(x, y)[0, 1]
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(x, y, alpha=0.35, s=12, color="#4C72B0")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"cov={emp_cov:.3f}, ρ={emp_rho:.3f}")
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Aquí hay dependencia visible aunque la correlación pueda quedar cerca de cero. "
                "La estructura no es lineal."
            )

    self_check_header()
    quiz(
        "Si $\\rho(X,Y)=0$, entonces $X\\perp Y$.",
        ["Verdadero siempre", "Verdadero sólo para gaussianas", "Falso siempre"],
        1,
        "Ejemplo: $Y=X^2$ con $X\\sim N(0,1)$: $\\rho=0$ pero $Y$ depende totalmente de $X$.",
        "$\\rho=0$ descarta relación *lineal*, no cualquier dependencia.",
        key="cov_q1"
    )
    ai_bridge(
        "La **matriz de covarianza** $\\Sigma$ es la base de PCA (siguiente sección), whitening y la "
        "Gaussiana multivariada. En deep learning, **batch normalization** puede verse como un whitening "
        "aproximado: restar media y dividir por desv. std."
    )

# ==================================================================
# SECCIÓN 11 — GAUSSIANA MULTIVARIADA Y PCA
# ==================================================================
def sec_pca():
    section_title(
        "11. Gaussiana Multivariada y PCA vía SVD",
        "De una variable a muchas correlacionadas: la geometría de elipses y cómo encontrar sus ejes principales."
    )
    motivation(
        "Cuando trabajas con datos vectoriales, las features suelen estar correlacionadas. La Gaussiana "
        "multivariada modela esa estructura con una **matriz de covarianza** $\\Sigma$. **PCA** encuentra "
        "las direcciones de máxima varianza — sus ejes principales — y es la herramienta estándar de "
        "reducción de dimensionalidad."
    )
    prerequisites_box(
        "- Covarianza, matriz $\\Sigma$.\n"
        "- Autovalores / autovectores (a nivel intuitivo: direcciones preservadas bajo la transformación).\n"
        "- SVD: $X = U\\,S\\,V^T$ (descomposición en valores singulares)."
    )
    st.markdown("### Gaussiana multivariada")
    st.latex(r"f_{\mathbf X}(\mathbf x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf x-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\right)")
    st.markdown("Las curvas de nivel (*isoprobabilidad*) son **elipses** con ejes = autovectores de $\\Sigma$ y longitudes proporcionales a $\\sqrt{\\lambda_i}$.")
    formula_walkthrough(
        "Qué dice en sustantivo la fórmula de la gaussiana multivariada",
        terms={
            r"\boldsymbol\mu": "vector de medias: el centro de la nube de datos.",
            r"\Sigma": "matriz de covarianza: describe escalas y correlaciones entre coordenadas.",
            r"\Sigma^{-1}": "penaliza desviaciones según la geometría de la covarianza; moverse en una dirección muy variable cuesta menos que moverse en una muy rígida.",
            r"|\Sigma|": "ajuste de normalización ligado al volumen característico de la distribución.",
        },
        steps=[
            "La exponencial decrece cuando te alejas del centro $\\mu$.",
            "Pero no mide distancia euclidiana común: usa una distancia deformada por $\\Sigma$.",
            "Por eso las curvas de igual densidad son elipses orientadas según los autovectores de la covarianza."
        ],
    )

    interactive_header("Gaussiana bivariada — heatmap con covarianza ajustable")
    interactive_guide(
        controls=[
            ("σ₁, σ₂", "controlan la dispersión en cada eje principal antes de considerar la correlación."),
            ("ρ", "controla la inclinación y la correlación lineal entre ambas variables."),
        ],
        procedure=(
            "Con esos parámetros se construye la matriz de covarianza $\\Sigma$, se evalúa la densidad gaussiana bivariada en una malla y se dibujan sus curvas de nivel."
        ),
        observe=(
            "Cuando $\\rho=0$ la elipse no se inclina. Cuando $|\\rho|$ crece, la elipse gira y se alarga siguiendo la dirección de dependencia."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        s1 = st.slider("σ₁", 0.3, 3.0, 1.0, key="pca_s1")
        s2 = st.slider("σ₂", 0.3, 3.0, 1.5, key="pca_s2")
        rho = st.slider("ρ", -0.95, 0.95, 0.6, key="pca_rho")
    Sigma = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])
    xs = np.linspace(-4, 4, 200); ys = np.linspace(-4, 4, 200)
    XX, YY = np.meshgrid(xs, ys)
    pos = np.dstack((XX, YY))
    rv = stats.multivariate_normal(mean=[0,0], cov=Sigma)
    Z = rv.pdf(pos)
    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cf = ax.contourf(XX, YY, Z, levels=20, cmap="viridis")
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        for i in range(2):
            v = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2
            ax.plot([0, v[0]], [0, v[1]], color="red", lw=2)
        ax.set_aspect("equal")
        ax.set_title(f"Σ={np.round(Sigma,2).tolist()}, autovalores={np.round(eigvals,2).tolist()}")
        st.pyplot(fig); plt.close(fig)
    how_to_read("Las flechas rojas son los **autovectores** de Σ (ejes de la elipse), con longitudes proporcionales a $\\sqrt{\\lambda_i}$.")

    st.markdown("### PCA vía SVD")
    st.markdown(
        "Dado un dataset $X \\in \\mathbb{R}^{N\\times d}$ **centrado**, computamos SVD:"
    )
    st.latex(r"X = U\,S\,V^T")
    st.markdown("Las columnas de $V$ son las **direcciones principales**, y la covarianza muestral se factoriza así:")
    st.latex(r"\hat\Sigma = \frac{X^T X}{N-1} = V\,\frac{S^2}{N-1}\,V^T")
    st.markdown("→ autovalores de $\\hat\\Sigma$ = $S_i^2/(N-1)$. Los primeros $k$ autovectores forman el mejor subespacio $k$-dim en sentido de mínimo error cuadrático.")
    st.markdown(
        "En palabras: PCA busca un nuevo sistema de coordenadas donde la primera dirección capture la mayor variabilidad posible, "
        "la segunda capture la mayor variabilidad restante y así sucesivamente."
    )

    interactive_header("PCA sobre una nube 2D rotada")
    interactive_guide(
        controls=[
            ("ángulo de rotación", "gira la nube de puntos original."),
            ("razón de ejes", "controla cuán alargada es la nube en una dirección respecto de la otra."),
            ("n puntos", "tamaño de la muestra simulada."),
        ],
        procedure=(
            "Se genera una nube elíptica, se rota, se centra y luego se aplica SVD para encontrar las direcciones principales."
        ),
        observe=(
            "PC1 debe alinearse con la dirección de mayor alargamiento de la nube. Si la nube se vuelve casi circular, la distinción entre componentes principales se debilita."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        theta = st.slider("ángulo de rotación (°)", 0, 180, 30, key="pca_theta") * np.pi / 180
        stretch = st.slider("razón de ejes (aspect)", 1.0, 6.0, 3.0, key="pca_stretch")
        n2 = st.slider("n puntos", 100, 3000, 500, key="pca_n")
    rng2 = np.random.default_rng(2)
    cloud = rng2.standard_normal((n2, 2)) * np.array([stretch, 1.0])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cloud = cloud @ R.T
    cloud -= cloud.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(cloud, full_matrices=False)
    var_explained = S**2 / (n2-1)
    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(cloud[:, 0], cloud[:, 1], alpha=0.3, s=10)
        for i in range(2):
            v = Vt[i] * np.sqrt(var_explained[i]) * 2
            ax.plot([0, v[0]], [0, v[1]], lw=3, label=f"PC{i+1} var={var_explained[i]:.2f}")
        ax.set_aspect("equal"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read("PC1 apunta en la dirección de máxima varianza. Si los datos están muy estirados en una dirección, PC1 la recupera.")

    self_check_header()
    quiz(
        "Los ejes principales que devuelve PCA son los autovectores de...",
        ["X", "X^T X / (N-1)", "la matriz de correlación de los targets"],
        1,
        "PCA busca las direcciones de máxima varianza = autovectores de la covarianza muestral.",
        "Piensa: SVD de $X$ ⇒ autovectores de $X^TX$ en $V$.",
        key="pca_q1"
    )
    ai_bridge(
        "PCA aparece en: **compresión de imágenes** (eigenfaces), **visualización** (2D/3D scatter plots), "
        "**preprocesamiento** (whitening antes de k-means), **análisis de activaciones** en redes neuronales. "
        "Los **autoencoders** lineales con loss cuadrática aprenden exactamente PCA."
    )

# ==================================================================
# SECCIÓN 12 — MALDICIÓN DE LA DIMENSIONALIDAD
# ==================================================================
def sec_curse():
    section_title(
        "12. Maldición de la Dimensionalidad",
        "Por qué la alta dimensión rompe nuestras intuiciones geométricas — y qué implica para ML."
    )
    motivation(
        "En 2D y 3D nuestra intuición funciona. En 100 dimensiones no. Las distancias se uniformizan, "
        "los volúmenes se concentran en los bordes, y algoritmos como kNN se degradan. Entender esto es "
        "fundamental para diseñar modelos que funcionen en dimensiones típicas de ML (cientos o miles)."
    )
    prerequisites_box(
        "- Volumen de una esfera $d$-dim: $V_d(r) = \\dfrac{\\pi^{d/2}}{\\Gamma(d/2+1)} r^d$.\n"
        "- Volumen de un hipercubo $d$-dim de lado $2r$: $(2r)^d$.\n"
        "- Distancias euclidianas en $\\mathbb{R}^d$."
    )
    st.markdown("### Construcción")
    st.markdown("**Razón volumen esfera / volumen cubo**:")
    st.latex(r"\frac{V_d(r)}{(2r)^d} = \frac{\pi^{d/2}}{2^d\,\Gamma(d/2+1)}")
    formula_walkthrough(
        "Qué está diciendo realmente esta razón de volúmenes",
        terms={
            r"V_d(r)": "volumen de la esfera de radio $r$ en dimensión $d$.",
            r"(2r)^d": "volumen del hipercubo de lado $2r$ que contiene a esa esfera.",
            r"\Gamma(d/2+1)": "generalización del factorial que aparece al calcular volúmenes en dimensión arbitraria.",
        },
        steps=[
            "La razón compara cuánto del cubo está realmente ocupado por la esfera inscrita.",
            "En baja dimensión la esfera ocupa una fracción razonable del cubo.",
            "Cuando la dimensión crece, esa fracción colapsa rápidamente: la mayor parte del volumen del cubo queda lejos del centro y cerca de las esquinas."
        ],
    )
    st.markdown("Tabla (esfera inscrita en cubo de lado 2):")
    dims = [1,2,3,4,5,10,20,50]
    ratios = []
    for d in dims:
        log_vs = (d/2)*np.log(np.pi) - gammaln(d/2+1)
        ratio = np.exp(log_vs) / (2.0**d)
        ratios.append(ratio)
    df_curse = pd.DataFrame({"d": dims, "V_esfera / V_cubo": [f"{r:.3e}" if r < 1e-3 else f"{r:.4f}" for r in ratios]})
    st.dataframe(df_curse, hide_index=True)
    st.info(
        "En **2D** la esfera inscrita ocupa 78.5% del cubo; en **3D**, 52.4%; en **4D**, 30.8%; "
        "en **10D**, 0.25%; en **20D**, $2\\times 10^{-8}$. *Casi todo el volumen del hipercubo está en las esquinas.*"
    )

    interactive_header("Simulación: ¿cuántos puntos aleatorios caen en la esfera?")
    interactive_guide(
        controls=[
            ("dimensión", "elige en cuántas dimensiones vives dentro del hipercubo y la esfera."),
            ("# puntos", "cantidad de puntos uniformes generados en el cubo."),
        ],
        procedure=(
            "La app genera puntos uniformes en el cubo $[-1,1]^d$ y cuenta qué fracción cae dentro de la esfera unitaria, "
            "es decir, satisface $x_1^2+\\cdots+x_d^2\\le 1$."
        ),
        observe=(
            "Incluso con muchos puntos, la proporción de aciertos cae muy rápido al aumentar la dimensión. "
            "Eso visualiza por qué la intuición geométrica de 2D y 3D deja de funcionar en alta dimensión."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        d_sim = st.slider("dimensión", 1, 20, 5, key="curse_d")
        n_sim = st.slider("# puntos", 1000, 50000, 10000, step=1000, key="curse_n")
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1, 1, size=(n_sim, d_sim))
    inside = (np.sum(pts**2, axis=1) <= 1).mean()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ds = np.arange(1, 21)
        rr = [np.exp((d/2)*np.log(np.pi) - gammaln(d/2+1)) / (2.0**d) for d in ds]
        ax.plot(ds, rr, "o-", color="#4C72B0", label="teórico")
        ax.scatter([d_sim], [inside], color="#DD8452", s=80, zorder=5, label=f"sim d={d_sim}: {inside:.4f}")
        ax.set_yscale("log"); ax.set_xlabel("dimensión d"); ax.set_ylabel("V esfera / V cubo (log)")
        ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read(
        "La escala vertical es logarítmica porque la razón cae extremadamente rápido. "
        "El punto destacado muestra la estimación por simulación para la dimensión elegida y debe alinearse con la curva teórica."
    )

    st.markdown("### Consecuencias para ML")
    st.markdown(
        "1. **Distancias se uniformizan**: en alta $d$, $\\max\\|x_i-x_j\\| \\approx \\min\\|x_i-x_j\\|$, "
        "kNN pierde discriminación.\n"
        "2. **Densidad cae exponencialmente**: para cubrir un cubo con resolución $\\epsilon$ hacen falta "
        "$(1/\\epsilon)^d$ puntos.\n"
        "3. **Los datos viven en subvariedades**: por eso funcionan PCA, autoencoders y embeddings."
    )
    self_check_header()
    quiz(
        "En 20D, la esfera inscrita ocupa del cubo aproximadamente...",
        ["78%", "50%", "0.25%", "$2\\times 10^{-8}$"],
        3,
        "Cae exponencialmente con la dimensión.",
        "Revisa la tabla — en 20D ya estamos en $10^{-8}$.",
        key="curse_q1"
    )
    ai_bridge(
        "La maldición es la razón por la que **los embeddings funcionan**: aunque una imagen de 256×256×3 "
        "vive en $\\mathbb{R}^{196608}$, los datos reales ocupan una subvariedad de dimensión mucho menor. "
        "Autoencoders, contrastive learning y self-supervised representation learning explotan esto."
    )

# ==================================================================
# SECCIÓN 13 — DESIGUALDADES DE CONCENTRACIÓN
# ==================================================================
def sec_concentration():
    section_title(
        "13. Desigualdades de Concentración",
        "Cotas sobre cuánto se aleja una VA de su media, sin conocer la distribución exacta."
    )
    motivation(
        "No siempre podemos calcular $P(|X-\\mu|>\\epsilon)$ directamente — la distribución puede ser compleja. "
        "**Markov, Chebyshev y Hoeffding** dan cotas universales crecientemente más finas. Son la base de "
        "la teoría de generalización en ML y de muchísimos resultados de muestreo."
    )
    prerequisites_box(
        "- $E[X]$, $\\text{Var}(X)$.\n"
        "- VAs acotadas (para Hoeffding)."
    )
    st.markdown("### Construcción")
    st.markdown("**Markov** (para $X\\ge 0$):")
    st.latex(r"P(X \geq a) \leq \frac{E[X]}{a}")
    st.markdown("**Chebyshev** (a partir de Markov aplicado a $(X-\\mu)^2$):")
    st.latex(r"P(|X-\mu| \geq k\sigma) \leq \frac{1}{k^2}")
    st.markdown("**Hoeffding** (si cada $X_i \\in [a,b]$, i.i.d.):")
    st.latex(r"P\!\left(\big|\bar X_n - \mu\big| \geq t\right) \leq 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)")
    formula_walkthrough(
        "Qué garantizan realmente estas desigualdades",
        terms={
            r"X": "variable aleatoria individual.",
            r"\mu": "valor esperado o media.",
            r"\sigma^2": "varianza.",
            r"\bar X_n": "promedio de $n$ observaciones i.i.d.",
            r"\epsilon": "tolerancia o desviación máxima aceptable.",
        },
        steps=[
            "No entregan la probabilidad exacta del evento raro, sino una cota superior válida bajo hipótesis muy generales.",
            "Markov sólo necesita no negatividad; por eso es muy universal, pero también muy laxa.",
            "Chebyshev incorpora varianza y ya controla desviaciones alrededor de la media.",
            "Hoeffding aprovecha que las variables están acotadas en un intervalo fijo y por eso logra un decaimiento exponencial en $n$."
        ],
    )

    worked_example("¿Cuántas muestras necesito para estar seguro?")
    st.markdown(
        "Queremos $P(|\\bar X_n - \\mu| \\geq 0.05) \\leq 0.05$ con $X_i \\in [0,1]$.\n\n"
        "Hoeffding: $2\\exp(-2n(0.05)^2) \\leq 0.05 \\Rightarrow n \\geq \\dfrac{\\ln(40)}{0.005} \\approx 738$."
    )
    st.latex(r"n \geq \frac{\ln(2/\delta)}{2\epsilon^2} = \frac{\ln(40)}{2\cdot 0.0025} \approx 738")

    interactive_header("Comparador de cotas")
    col1, col2 = st.columns([1, 2])
    with col1:
        eps = st.slider("ε", 0.01, 0.5, 0.1, step=0.01, key="conc_eps")
        delta = st.slider("δ (probabilidad de fallar)", 0.001, 0.5, 0.05, step=0.001, key="conc_delta")
        n_h = np.ceil(np.log(2/delta) / (2*eps**2))
        var_unit = 0.25
        n_c = np.ceil(var_unit / (eps**2 * delta))
        st.metric("n (Hoeffding)", int(n_h))
        st.metric("n (Chebyshev, σ²≤0.25)", int(n_c))
    with col2:
        ns = np.arange(10, 2000)
        p_hoef = 2*np.exp(-2*ns*eps**2)
        p_cheb = np.minimum(1.0, var_unit/(ns*eps**2))
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.plot(ns, p_hoef, label="Hoeffding", color="#4C72B0", lw=2)
        ax.plot(ns, p_cheb, label="Chebyshev", color="#DD8452", lw=2)
        ax.axhline(delta, ls=":", color="gray", label=f"δ={delta}")
        ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("cota superior de P(|X̄−μ|≥ε)")
        ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read("Hoeffding (azul) decae exponencialmente, Chebyshev (naranja) sólo polinomialmente. Misma ε, Hoeffding necesita muchas menos muestras.")

    interactive_header("Cotas teóricas versus comportamiento empírico")
    interactive_guide(
        controls=[
            ("Fuente", "elige la distribución real desde la que se simulan los datos."),
            ("n", "tamaño de cada muestra usada para formar el promedio."),
            ("ε empírico", "umbral de desviación respecto de la media."),
            ("Repeticiones Monte Carlo", "número de muestras independientes usadas para estimar la probabilidad real."),
        ],
        procedure=(
            "Se simulan muchas medias muestrales, se estima empíricamente la probabilidad del evento $|\\bar X_n-\\mu|\\ge \\epsilon$, "
            "y luego se compara con las cotas de Chebyshev y Hoeffding."
        ),
        observe=(
            "La probabilidad empírica suele ser bastante menor que las cotas. Eso no es un problema: una cota está diseñada para garantizar por arriba, no para ser exacta."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        source = st.radio("Fuente", ["Bernoulli(0.3)", "Uniforme(0,1)"], key="conc_src")
        n_emp = st.slider("n", 5, 500, 60, key="conc_nemp")
        eps_emp = st.slider("ε empírico", 0.01, 0.4, 0.10, step=0.01, key="conc_epsemp")
        reps = st.slider("Repeticiones Monte Carlo", 500, 20000, 5000, step=500, key="conc_reps")
    rng = np.random.default_rng(202)
    if source.startswith("Bernoulli"):
        samples = rng.binomial(1, 0.3, size=(reps, n_emp))
        mu_emp = 0.3
        var_emp = 0.3 * 0.7
    else:
        samples = rng.uniform(0, 1, size=(reps, n_emp))
        mu_emp = 0.5
        var_emp = 1 / 12
    means = samples.mean(axis=1)
    empirical_tail = np.mean(np.abs(means - mu_emp) >= eps_emp)
    cheb_tail = min(1.0, var_emp / (n_emp * eps_emp**2))
    hoef_tail = min(1.0, 2 * np.exp(-2 * n_emp * eps_emp**2))
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
        axes[0].hist(means, bins=40, color="#4C72B0", alpha=0.75, density=True)
        axes[0].axvline(mu_emp - eps_emp, color="red", ls="--")
        axes[0].axvline(mu_emp + eps_emp, color="red", ls="--")
        axes[0].axvline(mu_emp, color="black", ls=":")
        axes[0].set_title("Distribución empírica de $\\bar X_n$")
        axes[1].bar(
            ["Empírica", "Chebyshev", "Hoeffding"],
            [empirical_tail, cheb_tail, hoef_tail],
            color=["#2F5D50", "#DD8452", "#4C72B0"],
        )
        axes[1].set_ylim(0, max(0.05, empirical_tail, cheb_tail, hoef_tail) * 1.25)
        axes[1].set_ylabel("P(|X̄−μ| ≥ ε)")
        axes[1].set_title("Comparación de probabilidades")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.metric("Probabilidad empírica", f"{empirical_tail:.4f}")
    st.metric("Cota de Chebyshev", f"{cheb_tail:.4f}")
    st.metric("Cota de Hoeffding", f"{hoef_tail:.4f}")
    how_to_read(
        "La barra empírica estima la probabilidad real en la simulación. Las otras dos barras no intentan acertar exactamente, "
        "sino garantizar una cota superior válida."
    )

    self_check_header()
    quiz(
        "¿Cuál desigualdad decae exponencialmente con n?",
        ["Markov", "Chebyshev", "Hoeffding"],
        2,
        "Hoeffding: $2e^{-2nt^2/(b-a)^2}$.",
        "Exponencial en $n$ es la estrella aquí.",
        key="conc_q1"
    )
    ai_bridge(
        "Hoeffding es la herramienta fundamental en **PAC-learning** (Probably Approximately Correct): "
        "permite acotar el riesgo de generalización de un clasificador con probabilidad al menos $1-\\delta$, "
        "conocido $n$. Aparece en **bandits**, **bounds de generalización**, **bootstrap** y **concentración de medidas** en alta dimensión."
    )

# ==================================================================
# SECCIÓN 14 — MUESTRAS Y TESTEO AGRUPADO
# ==================================================================
def sec_samples_pooled():
    section_title(
        "14. Muestras y Testeo Agrupado (pooled testing)",
        "Estimadores a partir de n observaciones + un caso clásico donde azar + agrupamiento baja costos."
    )
    motivation(
        "En estadística casi siempre trabajamos con **muestras** (subset de una población). Media muestral "
        "y varianza muestral son los estimadores estándar de $\\mu$ y $\\sigma^2$. Como aplicación, el "
        "**testeo agrupado** reduce drásticamente el número de tests cuando la prevalencia es baja."
    )
    prerequisites_box(
        "- $E[X]$, $\\text{Var}(X)$.\n"
        "- Independencia e idéntica distribución (i.i.d.)."
    )
    st.markdown("### Estimadores muestrales")
    st.latex(r"\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i, \quad E[\bar X_n]=\mu,\quad \text{Var}[\bar X_n] = \frac{\sigma^2}{n}")
    st.latex(r"S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X_n)^2 \quad \text{(insesgado de } \sigma^2\text{)}")
    st.caption("El $n-1$ (corrección de Bessel) viene de que $\\bar X$ ya 'consumió' un grado de libertad.")
    formula_walkthrough(
        "Qué significan los estimadores muestrales",
        terms={
            r"\bar X_n": "media muestral: el promedio de los datos observados.",
            r"\mu": "media verdadera de la población, que normalmente desconocemos.",
            r"S^2": "varianza muestral corregida para no subestimar sistemáticamente la varianza poblacional.",
            r"n-1": "corrección de Bessel: compensa el hecho de haber centrado los datos usando la propia muestra.",
        },
        steps=[
            "La media muestral resume el nivel promedio observado en el conjunto de datos.",
            "Su varianza cae como $1/n$: promediar muchas observaciones estabiliza el estimador.",
            "Para la varianza, usar $n$ en lugar de $n-1$ tiende a subestimar la dispersión real porque la media ya fue ajustada con esos mismos datos."
        ],
    )

    worked_example("testeo agrupado (pooled testing)")
    st.markdown(
        "Tenemos $N=1000$ muestras de sangre; prevalencia $p=0.01$. Dos esquemas:\n\n"
        "**A. Test individual**: 1000 tests.\n\n"
        "**B. Agrupar en bloques de $k$**: mezclar $k$ muestras y testar la mezcla. Si negativo → 1 test "
        "para $k$ personas. Si positivo → $1+k$ tests para ese grupo. Total esperado:"
    )
    st.latex(r"E[Z] = \frac{N}{k}\Big[1 + k\cdot (1 - (1-p)^k)\Big]")
    st.markdown("Para $N=1000, p=0.01, k=10$:")
    k0 = 10; p0 = 0.01; N0 = 1000
    ez = (N0/k0) * (1 + k0*(1 - (1-p0)**k0))
    st.latex(rf"E[Z] = \tfrac{{1000}}{{10}}[1 + 10(1 - 0.99^{{10}})] \approx {ez:.0f}")
    st.info(f"Pasamos de **1000 tests → ~{ez:.0f}**. Con el mejor $k$ se puede bajar aún más.")

    interactive_header("Pooled testing: óptimo en k")
    interactive_guide(
        controls=[
            ("Prevalencia p", "fracción esperada de individuos positivos en la población."),
            ("N total", "cantidad total de muestras que deseas testear."),
        ],
        procedure=(
            "Para cada tamaño de grupo $k$, la app calcula el número esperado total de tests: un test inicial por grupo "
            "más los tests individuales adicionales cuando un grupo sale positivo."
        ),
        observe=(
            "Si $k$ es muy pequeño, casi no ahorras tests. Si $k$ es muy grande, muchos grupos salen positivos y obligan a hacer demasiados tests de confirmación. "
            "Por eso aparece una curva en forma de U."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        p_slider = st.slider("Prevalencia p", 0.001, 0.2, 0.01, step=0.001, format="%.3f", key="pool_p")
        N_slider = st.slider("N total", 100, 10000, 1000, step=100, key="pool_N")
    ks = np.arange(1, 51)
    EZ = (N_slider/ks) * (1 + ks*(1 - (1-p_slider)**ks))
    k_opt = ks[np.argmin(EZ)]
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(ks, EZ, color="#4C72B0", lw=2)
        ax.axhline(N_slider, color="gray", ls="--", label=f"Sin pooling = {N_slider}")
        ax.axvline(k_opt, color="#DD8452", ls=":", label=f"k óptimo = {k_opt}")
        ax.set_xlabel("tamaño de grupo k"); ax.set_ylabel("E[# tests]")
        ax.legend()
        st.pyplot(fig); plt.close(fig)
        st.metric("E[# tests] óptimo", f"{EZ.min():.1f}", f"vs {N_slider} sin pooling")
    how_to_read("Curva U: k=1 es test individual (costo N); k grande da muchos reruns. El fondo es el óptimo.")

    self_check_header()
    quiz(
        "El estimador $S^2$ usa divisor $n-1$ porque...",
        ["Es más chico", "Compensa que $\\bar X$ usó un grado de libertad (insesgado)", "Es arbitrario"],
        1,
        "$E[S^2]=\\sigma^2$ con el divisor $n-1$; con $n$ sería sesgado hacia abajo.",
        "Corrección de Bessel: compensa el uso de $\\bar X$ como centro.",
        key="smp_q1"
    )
    ai_bridge(
        "**Minibatch SGD** es precisamente muestreo: estimamos $\\nabla \\mathcal{L}$ con un *batch* y "
        "aceptamos varianza $\\sigma^2/|batch|$. **Bootstrap**: muestreamos con reemplazo para estimar "
        "distribuciones de estadísticos. **Negative sampling** en word2vec: agrupa pares para reducir costo."
    )

# ==================================================================
# SECCIÓN 15 — LEYES LÍMITE: LLN Y CLT
# ==================================================================
def sec_limits():
    section_title(
        "15. Leyes Límite: LLN, Kolmogorov, CLT, CLT Multivariado",
        "Los dos teoremas que explican por qué funcionan los promedios — y por qué la normal aparece por todos lados."
    )
    motivation(
        "**LLN**: el promedio muestral converge a la media poblacional. **CLT**: las fluctuaciones de ese "
        "promedio alrededor de $\\mu$ son *aproximadamente Gaussianas* con varianza $\\sigma^2/n$, "
        "independientemente de la distribución original. Estos dos resultados sostienen casi toda la estadística inferencial."
    )
    prerequisites_box(
        "- $\\bar X_n = \\tfrac{1}{n}\\sum X_i$, i.i.d.\n"
        "- Convergencia en probabilidad vs convergencia casi segura (la distinción es fina; con saber la idea basta).\n"
        "- Estandarización: $Z = (X-\\mu)/\\sigma$."
    )
    st.markdown("### Ley de los Grandes Números")
    st.markdown("**Débil (LLN débil)** — convergencia en probabilidad:")
    st.latex(r"\forall \epsilon>0: \lim_{n\to\infty} P(|\bar X_n - \mu|>\epsilon) = 0")
    st.markdown("**Fuerte (LLN fuerte)** — convergencia casi segura:")
    st.latex(r"P\!\left(\lim_{n\to\infty} \bar X_n = \mu\right) = 1")
    st.caption(
        "Diferencia intuitiva: la débil dice «es improbable estar lejos»; la fuerte dice «la trayectoria "
        "completa converge». La fuerte implica la débil, no al revés."
    )
    st.markdown("**Desigualdad de Kolmogorov** (máxima de sumas parciales):")
    st.latex(r"P\!\left(\max_{1\leq k\leq n} |S_k - k\mu| \geq \epsilon\right) \leq \frac{\text{Var}(S_n)}{\epsilon^2}")

    st.markdown("### Teorema Central del Límite (CLT)")
    st.latex(r"\frac{\bar X_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal N(0,1)")
    st.markdown(
        "Para $n$ grande, **sin importar la distribución de $X_i$** (con varianza finita), $\\bar X_n$ es "
        "aproximadamente $\\mathcal N(\\mu, \\sigma^2/n)$."
    )
    st.markdown("**CLT multivariado**:")
    st.latex(r"\sqrt{n}(\bar{\mathbf X}_n - \boldsymbol\mu) \xrightarrow{d} \mathcal N_d(\mathbf 0, \Sigma)")
    formula_walkthrough(
        "Qué prometen LLN y CLT, y qué no",
        terms={
            r"\bar X_n": "promedio de la muestra.",
            r"\mu": "media poblacional verdadera.",
            r"\sigma": "desviación estándar de la variable original.",
            r"\sqrt{n}": "escala natural de las fluctuaciones del promedio.",
            r"\mathcal N(0,1)": "normal estándar, la forma límite de la variable estandarizada.",
        },
        steps=[
            "LLN habla de consistencia del promedio: con muchas muestras, $\\bar X_n$ se acerca a $\\mu$.",
            "CLT habla de la forma de las fluctuaciones alrededor de $\\mu$: tras escalar por $\\sqrt n$, esas fluctuaciones se vuelven aproximadamente gaussianas.",
            "LLN no dice a qué velocidad ocurre la convergencia; CLT sí cuantifica la escala típica del error mediante $\\sigma/\\sqrt n$.",
            "CLT no exige normalidad de la fuente, pero sí independencia (o condiciones similares) y varianza finita."
        ],
    )

    worked_example("Chebyshev vs CLT para dimensionar muestras")
    st.markdown(
        "$X_i\\in[0,1]$, queremos $P(|\\bar X_n - \\mu|>0.05) \\leq 0.05$.\n\n"
        "**Chebyshev** con la cota genérica $\\sigma^2\\le 0.25$: "
        "$\\sigma^2/(n\\epsilon^2) \\leq 0.05 \\Rightarrow n \\geq 0.25/(0.0025\\cdot 0.05) = 2000$.\n\n"
        "**CLT** (aproximación): si tomamos el peor caso $\\sigma=0.5$, entonces para un intervalo bilateral del 95% necesitamos "
        "$1.96\\,\\sigma/\\sqrt n \\le 0.05$. Eso da $\\sqrt n \\ge 1.96\\cdot 0.5/0.05 = 19.6$ y por tanto "
        "$n \\ge 19.6^2 \\approx 384.2$.\n\n"
        "→ CLT sigue siendo bastante más eficiente que Chebyshev cuando aplica, pero no por un factor tan extremo como decía el texto anterior."
    )

    interactive_header("LLN y CLT en acción")
    tabs = st.tabs(["LLN (trayectorias)", "CLT (histograma de promedios)"])
    rng = np.random.default_rng(7)
    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            dist_name = st.radio("Distribución fuente", ["Bernoulli(0.3)", "Exponencial(1)", "Uniforme(0,1)"], key="lln_dist")
            n_paths = st.slider("# trayectorias", 3, 30, 8, key="lln_paths")
            nmax = st.slider("n máximo", 100, 5000, 2000, key="lln_nmax")
        if dist_name.startswith("Bernoulli"):
            data = rng.binomial(1, 0.3, size=(n_paths, nmax)); mu_true = 0.3
        elif dist_name.startswith("Exp"):
            data = rng.exponential(1.0, size=(n_paths, nmax)); mu_true = 1.0
        else:
            data = rng.uniform(0, 1, size=(n_paths, nmax)); mu_true = 0.5
        means = np.cumsum(data, axis=1) / np.arange(1, nmax+1)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 3.3))
            for i in range(n_paths):
                ax.plot(means[i], alpha=0.5)
            ax.axhline(mu_true, color="red", ls="--", label=f"μ={mu_true}")
            ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("X̄_n")
            ax.legend()
            st.pyplot(fig); plt.close(fig)
        how_to_read("Cada línea es una trayectoria distinta. Todas convergen al valor rojo μ conforme n crece. La dispersión baja como $1/\\sqrt n$.")
    with tabs[1]:
        col1, col2 = st.columns([1, 2])
        with col1:
            dist2 = st.radio("Fuente", ["Exponencial(1)", "Uniforme(0,1)", "Bernoulli(0.5)"], key="clt_dist")
            n_clt = st.slider("n (por promedio)", 2, 500, 30, key="clt_n")
            n_reps = st.slider("# repeticiones", 500, 20000, 5000, step=500, key="clt_reps")
        if dist2.startswith("Exp"):
            sample = rng.exponential(1.0, size=(n_reps, n_clt)); mu0, s0 = 1.0, 1.0
        elif dist2.startswith("Uni"):
            sample = rng.uniform(0, 1, size=(n_reps, n_clt)); mu0, s0 = 0.5, np.sqrt(1/12)
        else:
            sample = rng.binomial(1, 0.5, size=(n_reps, n_clt)); mu0, s0 = 0.5, 0.5
        means = sample.mean(axis=1)
        z = (means - mu0) / (s0 / np.sqrt(n_clt))
        with col2:
            fig, ax = plt.subplots(figsize=(7, 3.3))
            ax.hist(z, bins=50, density=True, color="#4C72B0", alpha=0.65, label="Promedios estandarizados")
            xs = np.linspace(-4, 4, 200)
            ax.plot(xs, stats.norm.pdf(xs), color="red", lw=2, label="N(0,1)")
            ax.legend(); ax.set_xlim(-4, 4)
            st.pyplot(fig); plt.close(fig)
        how_to_read("Aunque la fuente sea sesgada (Exponencial) o discreta (Bernoulli), el histograma del promedio estandarizado se parece a la campana estándar cuando n crece.")

    interactive_header("Cobertura de intervalos construidos con CLT")
    interactive_guide(
        controls=[
            ("Fuente para intervalos", "elige la distribución original de donde salen las muestras."),
            ("n por muestra", "tamaño de cada muestra usada para construir un intervalo."),
            ("Número de intervalos simulados", "cuántos intervalos independientes quieres generar."),
        ],
        procedure=(
            "Para cada muestra, la app construye un intervalo del 95% usando la aproximación del CLT y verifica si contiene o no la media verdadera."
        ),
        observe=(
            "Cada línea representa un intervalo distinto. Si la aproximación es buena, la fracción de intervalos que contienen la media verdadera debería acercarse a 0.95."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        source_ci = st.radio("Fuente para intervalos", ["Exponencial(1)", "Uniforme(0,1)", "Bernoulli(0.5)"], key="clt_ci_src")
        n_ci = st.slider("n por muestra", 5, 400, 40, key="clt_ci_n")
        n_intervals = st.slider("Número de intervalos simulados", 50, 1000, 200, step=50, key="clt_ci_rep")
    rng = np.random.default_rng(77)
    if source_ci.startswith("Exp"):
        draws = rng.exponential(1.0, size=(n_intervals, n_ci)); mu_ci, sigma_ci = 1.0, 1.0
    elif source_ci.startswith("Uni"):
        draws = rng.uniform(0, 1, size=(n_intervals, n_ci)); mu_ci, sigma_ci = 0.5, np.sqrt(1/12)
    else:
        draws = rng.binomial(1, 0.5, size=(n_intervals, n_ci)); mu_ci, sigma_ci = 0.5, 0.5
    means_ci = draws.mean(axis=1)
    half_width = 1.96 * sigma_ci / np.sqrt(n_ci)
    lowers = means_ci - half_width
    uppers = means_ci + half_width
    covered = (lowers <= mu_ci) & (mu_ci <= uppers)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        shown = min(80, n_intervals)
        for i in range(shown):
            color = "#4C72B0" if covered[i] else "#DD8452"
            ax.plot([lowers[i], uppers[i]], [i, i], color=color, lw=1.6)
            ax.plot(means_ci[i], i, "o", color=color, ms=3)
        ax.axvline(mu_ci, color="black", ls="--", label=f"μ={mu_ci}")
        ax.set_xlabel("Intervalo al 95% vía CLT")
        ax.set_ylabel("Índice de muestra")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    st.metric("Cobertura empírica", f"{covered.mean():.3f}")
    st.caption(
        "Las líneas azules contienen a la media verdadera y las naranjas no. Con $n$ suficientemente grande, "
        "la cobertura debería acercarse al 95%."
    )

    self_check_header()
    quiz(
        "CLT requiere que la distribución original sea...",
        ["Normal", "Simétrica", "Tener varianza finita y ser i.i.d."],
        2,
        "Esa es la hipótesis clave; no hace falta normalidad de los $X_i$.",
        "Justamente lo milagroso del CLT: no pide normalidad de la fuente.",
        key="lim_q1"
    )
    quiz(
        "$\\bar X_n$ tiene varianza...",
        ["$\\sigma^2$", "$\\sigma^2/n$", "$n\\sigma^2$"],
        1,
        "$\\text{Var}(\\bar X_n)=\\sigma^2/n$: más muestras → menos ruido en el promedio.",
        "Linealidad e independencia: $\\text{Var}(\\frac{1}{n}\\sum X_i) = \\frac{1}{n^2}\\sum \\sigma^2 = \\sigma^2/n$.",
        key="lim_q2"
    )
    ai_bridge(
        "**Minibatch SGD** usa CLT: el gradiente de un batch es $\\mathcal N(\\nabla\\mathcal L, \\Sigma/|B|)$ "
        "aproximadamente. Por eso batches más grandes → updates más estables. **Bootstrap** usa LLN para "
        "estimar distribuciones de estadísticos. **Intervalos de confianza** al $95\\%$ → directamente del CLT ($\\pm 1.96 \\sigma/\\sqrt n$)."
    )

# ==================================================================
# SECCIÓN 16 — ALGORITMOS ALEATORIZADOS
# ==================================================================
def sec_randomized():
    section_title(
        "16. Algoritmos Aleatorizados",
        "Usar azar no como obstáculo sino como herramienta algorítmica: Quicksort y Quickselect."
    )
    motivation(
        "Quicksort con pivote fijo tiene peor caso $O(n^2)$ (entrada ya ordenada). Con pivote **aleatorio** "
        "esperamos $O(n\\log n)$ *para toda entrada*. El azar no cambia la entrada — protege contra entradas adversariales. "
        "Es un patrón que reaparece en toda computación eficiente."
    )
    prerequisites_box(
        "- Notación $O(\\cdot)$.\n"
        "- Recursión.\n"
        "- Linealidad de esperanza e indicadores (sección 9)."
    )
    st.markdown("### Quicksort aleatorizado")
    st.markdown(
        "Pseudocódigo: elige un **pivote al azar**, particiona, recursiona izquierda y derecha. "
        "Sea $X_{ij}=1$ si los elementos $i$-ésimo y $j$-ésimo del array ordenado fueron comparados durante el algoritmo. "
        "Se compara sólo si uno es pivote *antes* de separarlos."
    )
    st.latex(r"P(X_{ij}=1) = \frac{2}{j-i+1}")
    st.markdown("Por linealidad de esperanza:")
    st.latex(r"E[\text{comp}] = \sum_{i<j}\frac{2}{j-i+1} = O(n\log n)")
    formula_walkthrough(
        "La intuición detrás de $P(X_{ij}=1)=2/(j-i+1)$",
        terms={
            r"X_{ij}": "indicador que vale 1 si los elementos de rango $i$ y $j$ se comparan durante Quicksort, y 0 en caso contrario.",
            r"j-i+1": "tamaño del bloque de elementos comprendido entre esos dos rangos en el arreglo ya ordenado.",
            r"E[\text{comp}]": "número esperado total de comparaciones del algoritmo.",
        },
        steps=[
            "Fija dos elementos del arreglo ya ordenado: el de rango $i$ y el de rango $j$.",
            "Esos dos elementos sólo pueden compararse mientras todos los elementos entre ellos sigan en el mismo subproblema.",
            "La primera vez que alguno de los elementos del bloque $\\{i,\\dots,j\\}$ es elegido como pivote, o bien sale el de rango $i$, o el de rango $j$, o uno interno.",
            "Sólo en los dos primeros casos $i$ y $j$ se comparan entre sí. Como todos los pivotes del bloque son igualmente probables, la probabilidad es $2/(j-i+1)$."
        ],
    )

    worked_example("Quickselect")
    st.markdown(
        "Pregunta: encontrar el $k$-ésimo más pequeño sin ordenar todo. Quicksort recursa en ambos lados; "
        "**Quickselect** sólo recursa en el lado que contiene el $k$. El análisis análogo da $E[\\#\\text{comp}]=O(n)$."
    )

    interactive_header("Benchmark: Quicksort determinista vs aleatorizado")
    interactive_guide(
        controls=[
            ("tamaño del array", "longitud del arreglo a ordenar."),
            ("Entrada", "elige si el arreglo inicial será aleatorio o ya ordenado."),
        ],
        procedure=(
            "La app ejecuta una versión determinista de Quicksort y otra con pivote aleatorio sobre la misma entrada, y cuenta comparaciones."
        ),
        observe=(
            "La comparación importante no es sólo quién gana en un caso puntual, sino cómo cambia el costo cuando la entrada es adversarial para el pivote fijo."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        n_bench = st.slider("tamaño del array", 100, 3000, 1000, step=100, key="qs_n")
        adv = st.radio("Entrada", ["Aleatoria", "Ya ordenada (worst-case deterministic)"], key="qs_adv")

    def _qs_det_count(a):
        comps = 0
        stack = [a]
        while stack:
            cur = stack.pop()
            if len(cur) <= 1:
                continue
            pivot = cur[0]
            comps += len(cur) - 1
            left = cur[1:][cur[1:] < pivot]
            right = cur[1:][cur[1:] >= pivot]
            stack.append(left); stack.append(right)
        return comps

    def _qs_rand_count(a, rng):
        comps = 0
        stack = [a]
        while stack:
            cur = stack.pop()
            if len(cur) <= 1:
                continue
            idx = int(rng.integers(len(cur)))
            pivot = cur[idx]
            rest = np.delete(cur, idx)
            comps += len(rest)
            left = rest[rest < pivot]; right = rest[rest >= pivot]
            stack.append(left); stack.append(right)
        return comps

    @st.cache_data(show_spinner=False)
    def _bench_qs_single(n_bench: int, scenario_key: str):
        rng = np.random.default_rng(5)
        arr = rng.permutation(n_bench) if scenario_key == "random" else np.arange(n_bench)
        c_det = _qs_det_count(arr.copy())
        c_rand = _qs_rand_count(arr.copy(), rng)
        return c_det, c_rand

    c_det, c_rand = _bench_qs_single(n_bench, "random" if adv.startswith("Aleatoria") else "sorted")
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(["Determinista", "Aleatorizado"], [c_det, c_rand], color=["#DD8452", "#4C72B0"])
        ax.set_ylabel("# comparaciones")
        for i, v in enumerate([c_det, c_rand]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        ax.set_title(f"n={n_bench}, n·log₂n={n_bench*math.log2(n_bench):.0f}, n²={n_bench**2:,}")
        st.pyplot(fig)
    how_to_read("Con entrada ordenada, determinista explota a $O(n^2)$. Aleatorizado se mantiene cerca de $n\\log n$ (entre las líneas de referencia en el título).")

    interactive_header("Variabilidad del costo de Quicksort aleatorizado")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_runs = st.slider("Número de corridas", 20, 400, 120, step=20, key="qs_runs")
        n_hist = st.slider("n para el histograma", 50, 1500, 400, step=50, key="qs_hist_n")
        scenario = st.radio("Entrada base", ["Aleatoria fija", "Ordenada"], key="qs_hist_input")

    @st.cache_data(show_spinner=False)
    def _bench_qs_histogram(n_hist: int, n_runs: int, scenario_key: str):
        rng_hist = np.random.default_rng(99)
        base_arr = rng_hist.permutation(n_hist) if scenario_key == "random" else np.arange(n_hist)
        counts = np.empty(n_runs, dtype=np.int64)
        for i in range(n_runs):
            seed = int(rng_hist.integers(1_000_000))
            counts[i] = _qs_rand_count(base_arr.copy(), np.random.default_rng(seed))
        return counts

    rand_counts = _bench_qs_histogram(
        n_hist, n_runs, "random" if scenario.startswith("Aleatoria") else "sorted"
    )
    with col2:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.hist(rand_counts, bins=20, color="#4C72B0", alpha=0.75)
        ax.axvline(np.mean(rand_counts), color="black", ls="--", label=f"media={np.mean(rand_counts):.0f}")
        ax.axvline(n_hist * math.log2(n_hist), color="#55A868", ls=":", label=r"$n\log_2 n$")
        ax.set_xlabel("# comparaciones")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    st.caption(
        "El costo aleatorizado es una variable aleatoria: no toma siempre el mismo valor, pero su escala típica se mantiene cerca de $n\\log n$."
    )

    self_check_header()
    quiz(
        "Con entrada ya ordenada, Quicksort con pivote = primer elemento tiene coste...",
        ["$O(n)$", "$O(n\\log n)$", "$O(n^2)$"],
        2,
        "Cada partición deja un lado vacío y otro de tamaño $n-1$: $\\sum k = O(n^2)$.",
        "Ese es exactamente el *worst case* que Quicksort aleatorizado evita en esperanza.",
        key="rand_q1"
    )
    ai_bridge(
        "El azar como recurso computacional aparece en toda IA: **dropout** (regulariza apagando neuronas al azar), "
        "**data augmentation**, **SGD** (submuestreo aleatorio de batches), **random forests** (features y filas aleatorias), "
        "**Monte Carlo tree search** (AlphaZero). Controlar la aleatoriedad — y sus garantías en esperanza — es "
        "parte del toolkit esencial."
    )

# ==================================================================
#                   NAVEGACIÓN / SIDEBAR
# ==================================================================
st.sidebar.title("MIA — Probabilidad para IA")
st.sidebar.caption("Derivaciones, ejemplos resueltos, laboratorios interactivos y conexiones con IA.")

SECTIONS = {
    "1. Espacios y Axiomas de Kolmogorov": sec_kolmogorov,
    "2. Regla de Laplace y Combinatoria": sec_laplace,
    "3. Probabilidad Condicional": sec_condicional,
    "4. Teorema de Bayes": sec_bayes,
    "5. Clasificador Naïve Bayes": sec_naive_bayes,
    "6. VA: PMF, PDF y CDF": sec_va_cdf,
    "7. Catálogo de Distribuciones": sec_distribuciones,
    "8. MLE y Entropía Cruzada": sec_mle,
    "9. Esperanza, Varianza y Jensen": sec_esperanza_jensen,
    "10. FGM, Covarianza y Correlación": sec_fgm_cov,
    "11. Gaussiana Multivariada y PCA": sec_pca,
    "12. Maldición de la Dimensionalidad": sec_curse,
    "13. Desigualdades de Concentración": sec_concentration,
    "14. Muestras y Testeo Agrupado": sec_samples_pooled,
    "15. Leyes Límite: LLN y CLT": sec_limits,
    "16. Algoritmos Aleatorizados": sec_randomized,
}

NAV_GROUPS = {
    "Fundamentos": [
        "1. Espacios y Axiomas de Kolmogorov",
        "2. Regla de Laplace y Combinatoria",
    ],
    "Condicional y Bayes": [
        "3. Probabilidad Condicional",
        "4. Teorema de Bayes",
        "5. Clasificador Naïve Bayes",
    ],
    "VA y distribuciones": [
        "6. VA: PMF, PDF y CDF",
        "7. Catálogo de Distribuciones",
        "8. MLE y Entropía Cruzada",
    ],
    "Momentos y estructura": [
        "9. Esperanza, Varianza y Jensen",
        "10. FGM, Covarianza y Correlación",
        "11. Gaussiana Multivariada y PCA",
    ],
    "Alta dimensión y concentración": [
        "12. Maldición de la Dimensionalidad",
        "13. Desigualdades de Concentración",
    ],
    "Muestras y leyes límite": [
        "14. Muestras y Testeo Agrupado",
        "15. Leyes Límite: LLN y CLT",
    ],
    "Aplicaciones algorítmicas": [
        "16. Algoritmos Aleatorizados",
    ],
}

def _section_slug(label):
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")

SECTION_TO_SLUG = {label: _section_slug(label) for label in SECTIONS}
SLUG_TO_SECTION = {slug: label for label, slug in SECTION_TO_SLUG.items()}

sidebar_css = """
<style>
    section[data-testid="stSidebar"] .mia-sidebar-group-title {
        margin: 0.6rem 0 0.35rem 0.2rem;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        align-items: flex-start;
        padding: 0.52rem 0.7rem;
        margin: 0 0 0.18rem 0;
        border-radius: 0.82rem;
        color: #0F172A;
        background: transparent;
        border: 1px solid transparent;
        font-weight: 500;
        line-height: 1.2;
        font-size: 0.93rem;
        transition: all 0.18s ease;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button div,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        align-items: flex-start;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {
        margin: 0;
        white-space: normal;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: rgba(255,255,255,0.92);
        border-color: #D9E2EC;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        color: #0F172A;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(255,255,255,0.95));
        border-color: rgba(37,99,235,0.24);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
        color: #1D4ED8;
    }
</style>
"""

if 'choice' not in st.session_state:
    raw_section = st.query_params.get("section", SECTION_TO_SLUG[list(SECTIONS.keys())[0]])
    if isinstance(raw_section, list):
        raw_section = raw_section[0]
    st.session_state.choice = SLUG_TO_SECTION.get(raw_section, list(SECTIONS.keys())[0])

st.sidebar.markdown(sidebar_css, unsafe_allow_html=True)

with st.sidebar:
    for group_name, labels in NAV_GROUPS.items():
        st.markdown(
            f'<div class="mia-sidebar-group-title">{group_name}</div>',
            unsafe_allow_html=True,
        )
        for label in labels:
            is_active = (label == st.session_state.choice)
            if st.button(
                label,
                key=f"navbtn_{SECTION_TO_SLUG[label]}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if label != st.session_state.choice:
                    st.session_state.choice = label
                    st.query_params["section"] = SECTION_TO_SLUG[label]
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Cada sección integra marco formal, ejemplos, exploración computacional y conexiones con aprendizaje automático.")

SECTION_LABELS = list(SECTIONS.keys())
SECTION_TO_GROUP = {label: group for group, labels in NAV_GROUPS.items() for label in labels}

def _goto_section(new_label):
    if new_label in SECTIONS and new_label != st.session_state.choice:
        st.session_state.choice = new_label
        st.query_params["section"] = SECTION_TO_SLUG[new_label]
        st.rerun()

_current_idx = SECTION_LABELS.index(st.session_state.choice)
_current_group = SECTION_TO_GROUP.get(st.session_state.choice, "")
_progress_pct = int(100 * (_current_idx + 1) / len(SECTION_LABELS))

st.markdown(
    f"""
    <div class="mia-hero">
        <div class="mia-hero-kicker">MIA · IMT3850 · {_current_group} · Sección {_current_idx + 1} de {len(SECTION_LABELS)}</div>
        <h1>Probabilidad para Inteligencia Artificial</h1>
        <p>Desarrollo conceptual, derivaciones paso a paso, laboratorios interactivos y lectura rigurosa de los modelos probabilísticos usados en IA.</p>
        <div class="mia-progress-track"><div class="mia-progress-fill" style="width:{_progress_pct}%;"></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

SECTIONS[st.session_state.choice]()

st.markdown("<div class='mia-prevnext-spacer'></div>", unsafe_allow_html=True)
c_prev, c_info, c_next = st.columns([5, 2, 5])
with c_prev:
    if _current_idx > 0:
        prev_label = SECTION_LABELS[_current_idx - 1]
        if st.button(f"← {prev_label}", use_container_width=True, key=f"prevbtn_{_current_idx}"):
            _goto_section(prev_label)
    else:
        st.markdown("&nbsp;", unsafe_allow_html=True)
with c_info:
    st.markdown(
        f"<div class='mia-prevnext-chip'>{_current_idx + 1} / {len(SECTION_LABELS)}</div>",
        unsafe_allow_html=True,
    )
with c_next:
    if _current_idx < len(SECTION_LABELS) - 1:
        next_label = SECTION_LABELS[_current_idx + 1]
        if st.button(f"{next_label} →", use_container_width=True, key=f"nextbtn_{_current_idx}"):
            _goto_section(next_label)
    else:
        st.markdown("&nbsp;", unsafe_allow_html=True)
