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
    .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
        padding: 0.18rem 0.1rem 0.28rem 0.1rem;
        margin: 0.35rem 0;
    }
    div[data-testid="stMetric"] label {
        white-space: normal;
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
    with st.expander("Punto de partida: prerrequisitos y vocabulario", expanded=True):
        st.markdown(prereqs_md)

def how_to_read(text, expanded=False):
    with st.expander("Cómo leer el gráfico o simulación", expanded=expanded):
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

def lab_columns(left=1.15, right=1.85):
    return st.columns([left, right], vertical_alignment="top")

def metric_grid(items, columns=2):
    cols = st.columns(columns)
    for idx, item in enumerate(items):
        label, value, *rest = item
        delta = rest[0] if rest else None
        cols[idx % columns].metric(label, value, delta)

def compact_dataframe(df, height=None):
    kwargs = {"hide_index": True, "width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)

def advanced_expander(title, expanded=False):
    return st.expander(f"Profundización: {title}", expanded=expanded)

def learning_goal(text):
    st.info(f"Objetivo de lectura: {text}")

def lab_note(text):
    st.caption(f"Lectura rápida: {text}")

def beginner_bridge(title, bullets):
    with st.expander(f"Si no sabes nada: {title}", expanded=True):
        for bullet in bullets:
            st.markdown(f"- {bullet}")

def lab_task(predict=None, manipulate=None, verify=None):
    pieces = []
    if predict:
        pieces.append(f"**Predice:** {predict}")
    if manipulate:
        pieces.append(f"**Manipula:** {manipulate}")
    if verify:
        pieces.append(f"**Verifica:** {verify}")
    if pieces:
        st.info("\n\n".join(pieces))

def minimum_takeaway(learned, use_when, do_not_conclude):
    with st.expander("Conclusión mínima", expanded=True):
        st.markdown(
            f"- **Qué aprendí:** {learned}\n"
            f"- **Cuándo lo uso:** {use_when}\n"
            f"- **Qué no debo concluir:** {do_not_conclude}"
        )

def latex_aligned(lines):
    body = r"\\".join(lines)
    st.latex(rf"\begin{{aligned}}{body}\end{{aligned}}")

def plain_language(title, text):
    with st.expander(title, expanded=True):
        st.markdown(text)

def notation_box(items, expanded=True):
    with st.expander("Notación en palabras", expanded=expanded):
        for symbol, meaning in items:
            c1, c2 = st.columns([1.1, 5.0])
            with c1:
                if _looks_like_latex(symbol):
                    st.latex(symbol)
                else:
                    st.markdown(f"**{symbol}**")
            with c2:
                st.markdown(meaning)

def real_world_case(title, situation, controls=None, takeaway=None, expanded=True):
    with st.expander(f"Caso práctico: {title}", expanded=expanded):
        st.markdown(situation)
        if controls:
            st.markdown("**Cómo leer los controles en este caso**")
            for name, desc in controls:
                st.markdown(f"- **{name}**: {desc}")
        if takeaway:
            st.markdown("**Qué decisión o lectura permite hacer**")
            st.markdown(takeaway)

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
                st.markdown(f"- **{name}**: {desc}")
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

def mia_pyplot(fig=None, *args, **kwargs):
    if fig is not None and hasattr(fig, "axes"):
        try:
            polish_axes(fig.axes)
            polish_figure(fig)
        except Exception:
            pass
    result = st.pyplot(fig, *args, **kwargs)
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass
    return result

def quiz(question, options, correct_idx, feedback_ok, feedback_wrong, key):
    st.markdown(f"**{question}**")
    has_math_options = any("$" in str(opt) or "\\" in str(opt) for opt in options)
    if has_math_options:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        for letter, option in zip(letters, options):
            st.markdown(f"**{letter}.** {option}")
        ans = st.radio("Opciones", letters, key=key, index=None, horizontal=True, label_visibility="collapsed")
    else:
        ans = st.radio("Opciones", options, key=key, index=None, label_visibility="collapsed")
    if ans is None:
        st.caption("_Selecciona una opción para ver el feedback._")
        return
    selected_idx = letters.index(ans) if has_math_options else options.index(ans)
    if selected_idx == correct_idx:
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

    plain_language(
        "Antes de la fórmula: qué problema resuelve esta sección",
        "Imagina que quieres construir un sistema que responde preguntas con incertidumbre: "
        "'¿saldrá cara?', '¿este correo es spam?', '¿este paciente tiene riesgo alto?'. "
        "Antes de calcular porcentajes, hay que dejar claro **qué cosas pueden ocurrir**, "
        "**qué preguntas vamos a permitir hacer** y **qué número asignaremos a cada pregunta**. "
        "Eso es exactamente lo que formaliza un espacio de probabilidad."
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
    st.markdown("**Cota de la unión (Boole)**: aunque los eventos se sobrepongan, la probabilidad de la unión nunca supera la suma de probabilidades individuales.")
    st.latex(r"P\Big(\bigcup_i E_i\Big)\le \sum_i P(E_i)")
    notation_box([
        (r"\Omega", "El universo de resultados posibles. En un dado, son las seis caras. En un clasificador, podrían ser las clases posibles."),
        (r"\mathcal A", "La colección de preguntas que vamos a considerar eventos. Por ejemplo: 'salió par', 'salió mayor que 4', 'el correo es spam'."),
        (r"P(A)", "La probabilidad del evento A. Es un número entre 0 y 1: 0 significa imposible, 1 significa seguro."),
        (r"A^c", "El complemento de A: todo lo que puede ocurrir y no pertenece a A."),
        (r"A\cap B", "La parte común: casos donde ocurren A y B a la vez."),
        (r"A\cup B", "La unión: casos donde ocurre A, ocurre B, o ocurren ambos."),
    ], expanded=True)
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
            "El laboratorio trata cada cara del dado como un resultado elemental con probabilidad $1/6$. "
            "A partir de los conjuntos elegidos calcula intersección, unión, complemento y sus probabilidades."
        ),
        observe=(
            "Fíjate en qué caras cuentan una sola vez para la unión, cuáles están en ambos eventos y cómo cambia la fórmula "
            "$P(A\\cup B)=P(A)+P(B)-P(A\\cap B)$ cuando la intersección crece o desaparece."
        ),
    )
    col1, col2 = lab_columns()
    faces = [1, 2, 3, 4, 5, 6]
    with col1:
        preset = st.selectbox(
            "Situación guiada",
            ["Par vs mayor que 4", "Número bajo vs número par", "Personalizado"],
            key="kolm_preset",
        )
        if preset == "Par vs mayor que 4":
            default_a, default_b = [2, 4, 6], [5, 6]
            st.caption("A = sale número par. B = sale un número mayor que 4.")
        elif preset == "Número bajo vs número par":
            default_a, default_b = [1, 2, 3], [2, 4, 6]
            st.caption("A = sale un número bajo (1, 2 o 3). B = sale número par.")
        else:
            default_a, default_b = [2, 4, 6], [4, 5, 6]
            st.caption("Define A y B como las preguntas que quieras hacer sobre el dado.")
        preset_key = _section_slug(preset)
        event_a = st.multiselect("Caras que cumplen A", faces, default=default_a, key=f"kolm_event_a_{preset_key}")
        event_b = st.multiselect("Caras que cumplen B", faces, default=default_b, key=f"kolm_event_b_{preset_key}")
        set_a, set_b = set(event_a), set(event_b)
        inter = sorted(set_a & set_b)
        union = sorted(set_a | set_b)
        comp_a = [x for x in faces if x not in set_a]
        p_a = len(set_a) / 6
        p_b = len(set_b) / 6
        p_inter = len(inter) / 6
        p_union = len(union) / 6
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        metric_grid([
            ("P(A): ocurre A", f"{p_a:.3f}"),
            ("P(B): ocurre B", f"{p_b:.3f}"),
            ("P(A y B)", f"{p_inter:.3f}"),
            ("P(A o B)", f"{p_union:.3f}"),
        ], columns=2)
        st.caption(
            f"Sólo A: {only_a or 'ninguna'} · Sólo B: {only_b or 'ninguna'} · A∩B: {inter or 'ninguna'}."
        )
        comp_text = "{" + ", ".join(map(str, comp_a)) + "}" if comp_a else "∅"
        st.markdown(f"**Caras fuera de A ($A^c$):** {comp_text}")
        latex_aligned([
            r"P(A\cup B) = P(A) + P(B) - P(A\cap B)",
            rf"P(A\cup B) = {p_a:.3f} + {p_b:.3f} - {p_inter:.3f} = {p_union:.3f}",
        ])
        st.caption(f"Lectura en conteos: la unión A∪B cubre {len(union)} de las 6 caras del dado.")
    with col2:
        st.info(
            "Todas las barras tienen altura 1/6 porque el dado es justo. "
            "Este gráfico es un mapa de pertenencia: el color muestra si cada cara está sólo en A, sólo en B, en ambos o fuera."
        )
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
        ax.set_title("Cada cara pesa 1/6; el color indica pertenencia a eventos")
        handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
        ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        fig.subplots_adjust(bottom=0.32)
        mia_pyplot(fig)
        plt.close(fig)
        st.dataframe(
            pd.DataFrame(
                {
                    "cara": faces,
                    "en A": ["sí" if face in set_a else "no" for face in faces],
                    "en B": ["sí" if face in set_b else "no" for face in faces],
                    "categoría": categories,
                }
            ),
            hide_index=True,
            width="stretch",
        )
    how_to_read(
        "Cada barra representa un resultado elemental con masa 1/6. Los colores permiten ver qué resultados "
        "aportan a $A$, a $B$, a la intersección y a la unión."
    )
    lab_note("en este gráfico la altura no cambia porque todas las caras pesan 1/6; la información está en el color de cada cara.")

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
    col1, col2 = lab_columns()
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
        mia_pyplot(fig); plt.close(fig)
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
    lab_task(
        predict="decide primero si importan order y replacement antes de mirar la fórmula.",
        manipulate="cambia el escenario cotidiano y luego ajusta n, k, replacement y order.",
        verify="revisa si los resultados listados calzan con la interpretación de tu problema.",
    )
    interactive_guide(
        controls=[
            ("n objetos distintos", "tamaño del conjunto base del que vas a elegir."),
            ("¿Hay reemplazo?", "decide si un mismo objeto puede aparecer más de una vez."),
            ("k elecciones", "cuántas posiciones o selecciones realizas."),
            ("¿Importa el orden?", "decide si dos elecciones con los mismos elementos pero distinto orden cuentan distinto."),
        ],
        procedure=(
            "Según esas dos decisiones lógicas, el laboratorio selecciona automáticamente el régimen correcto de conteo y, cuando el tamaño lo permite, "
            "enumera explícitamente algunos resultados posibles."
        ),
        observe=(
            "La enumeración sirve para auditar la fórmula. Si ves resultados que para tu problema deberían ser equivalentes, entonces elegiste mal el régimen."
        ),
    )
    col1, col2 = lab_columns()
    with col1:
        problem_type = st.selectbox(
            "Tipo de problema cotidiano",
            ["Clave PIN", "Podio de ganadores", "Comité sin cargos", "Sabores de helado con repetición", "Personalizado"],
            key="lap_problem_type",
        )
        if problem_type == "Clave PIN":
            st.caption("Ejemplo: una clave de 4 dígitos. El orden importa y se puede repetir un dígito.")
            suggested_n, suggested_k, suggested_replacement, suggested_order = 10, 4, "Sí", "Sí"
        elif problem_type == "Podio de ganadores":
            st.caption("Ejemplo: oro, plata y bronce entre varias personas. El orden importa y nadie puede ocupar dos puestos.")
            suggested_n, suggested_k, suggested_replacement, suggested_order = 8, 3, "No", "Sí"
        elif problem_type == "Comité sin cargos":
            st.caption("Ejemplo: elegir 3 personas para un comité. El orden no importa y no se repiten personas.")
            suggested_n, suggested_k, suggested_replacement, suggested_order = 8, 3, "No", "No"
        elif problem_type == "Sabores de helado con repetición":
            st.caption("Ejemplo: elegir 3 bolas entre varios sabores; puedes repetir sabor y el orden no importa.")
            suggested_n, suggested_k, suggested_replacement, suggested_order = 5, 3, "Sí", "No"
        else:
            st.caption("Ajusta manualmente las dos preguntas clave: ¿se repite? ¿importa el orden?")
            suggested_n, suggested_k, suggested_replacement, suggested_order = 5, 3, "Sí", "Sí"
        n_items = st.slider("cantidad de opciones disponibles (n)", 2, 10, suggested_n, key=f"lap_n_items_{problem_type}")
        replacement = st.radio("¿Hay reemplazo?", ["Sí", "No"], index=0 if suggested_replacement == "Sí" else 1, horizontal=True, key=f"lap_replacement_{problem_type}")
        k_max = 6 if replacement == "Sí" else n_items
        k_choices = st.slider("cantidad que eliges (k)", 1, k_max, min(suggested_k, k_max), key=f"lap_k_choices_{problem_type}")
        order = st.radio("¿Importa el orden?", ["Sí", "No"], index=0 if suggested_order == "Sí" else 1, horizontal=True, key=f"lap_order_{problem_type}")
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
        st.metric("Resultados posibles", f"{count_value:,}")
        st.markdown(f"**Régimen identificado:** {regime}.")
        st.latex(formula)
        preview = pd.DataFrame({"Primeros resultados posibles": [str(outcome) for outcome in outcomes[: min(12, len(outcomes))]]})
        compact_dataframe(preview, height=220)
        lab_note("este número es un conteo de casos posibles, no una probabilidad. La tabla permite ver si el orden está contando como distinto.")
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
    lab_task(
        predict="estima si 23 personas deberían quedar por debajo o por encima de 50% antes de mirar la curva.",
        manipulate="cambia el tamaño del grupo y la cantidad de corridas Monte Carlo.",
        verify="compara la curva exacta con el punto simulado.",
    )
    col1, col2 = lab_columns()
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
        st.metric("Estimación por simulación", f"{p_sim:.4f}", f"error abs. {abs(p-p_sim):.4f}")
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
        mia_pyplot(fig); plt.close(fig)
    how_to_read("Eje x: cantidad de personas; eje y: probabilidad de coincidencia. Nota cómo cruza 50% alrededor de n=23.")

    with advanced_expander("póker y conteos de cartas"):
        worked_example("póker: probabilidad de obtener un par")
        st.markdown(
            "5 cartas de una baraja de 52. $|\\Omega| = \\binom{52}{5} = 2{,}598{,}960$.\n\n"
            "**Par exactamente**: elegir el valor del par $\\binom{13}{1}$, elegir 2 palos $\\binom{4}{2}$, "
            "elegir 3 valores distintos restantes $\\binom{12}{3}$, elegir un palo para cada uno $4^3$."
        )
        num_par = comb(13, 1, exact=True) * comb(4, 2, exact=True) * comb(12, 3, exact=True) * (4**3)
        den_par = comb(52, 5, exact=True)
        latex_aligned([
            r"P(\text{par}) = \frac{\binom{13}{1}\binom{4}{2}\binom{12}{3}4^3}{\binom{52}{5}}",
            rf"P(\text{{par}})=\frac{{{num_par}}}{{{den_par}}}\approx {num_par/den_par:.4f}",
        ])
        num_poker = comb(13, 1, exact=True) * comb(4, 4, exact=True) * comb(48, 1, exact=True)
        num_full = comb(13, 1, exact=True) * comb(4, 3, exact=True) * comb(12, 1, exact=True) * comb(4, 2, exact=True)
        num_flush = 4 * comb(13, 5, exact=True)
        st.markdown("Otros conteos clásicos del mismo mazo:")
        latex_aligned([
            rf"P(\text{{póker}})=\frac{{13\binom{{4}}{{4}}\cdot48}}{{\binom{{52}}{{5}}}}\approx {num_poker/den_par:.5f}",
            rf"P(\text{{full house}})=\frac{{13\binom{{4}}{{3}}\cdot12\binom{{4}}{{2}}}}{{\binom{{52}}{{5}}}}\approx {num_full/den_par:.5f}",
            rf"P(\text{{color simple}})=\frac{{4\binom{{13}}{{5}}}}{{\binom{{52}}{{5}}}}\approx {num_flush/den_par:.5f}",
        ])
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

    worked_example("3 monedas honestas y monedas cargadas")
    st.markdown(
        "Con 3 monedas honestas hay $2^3=8$ resultados equiprobables. Exactamente 2 caras tiene 3 casos; "
        "al menos 2 caras tiene 4 casos; ninguna cara tiene 1 caso."
    )
    st.latex(r"P(X=2)=\frac{3}{8},\qquad P(X\ge 2)=\frac{4}{8},\qquad P(X=0)=\frac{1}{8}")
    st.markdown("Si cada moneda tiene $P(C)=p$ y $q=1-p$, los resultados ya no son equiprobables:")
    st.latex(r"P(X=2)=3p^2(1-p),\qquad P(X\ge 2)=p^3+3p^2(1-p)")

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
            ("P(A | B)", "fracción de casos con A dentro del grupo B."),
            ("P(A | no B)", "fracción de casos con A fuera del grupo B."),
        ],
        procedure=(
            "El laboratorio construye una tabla 2×2 consistente con esos parámetros y calcula tanto probabilidades marginales como condicionales."
        ),
        observe=(
            "Compara $P(A\\mid B)$ con $P(A)$. Si coinciden, saber que ocurrió $B$ no cambia la probabilidad de $A$; "
            "si difieren, hay dependencia."
        ),
    )
    col1, col2 = lab_columns()
    with col1:
        context = st.selectbox(
            "Contexto real",
            ["Recomendación y compra", "Spam y palabra oferta", "Paciente y test positivo"],
            key="cond_context",
        )
        if context == "Recomendación y compra":
            a_label, b_label = "compró", "vio recomendación"
        elif context == "Spam y palabra oferta":
            a_label, b_label = "es spam", "contiene 'oferta'"
        else:
            a_label, b_label = "está enfermo", "test salió positivo"
        st.caption(f"A = {a_label}. B = {b_label}. Los símbolos son abstractos; el contexto les da significado.")
        pop = st.slider("Tamaño de la población sintética", 1000, 50000, 10000, step=1000, key="cond_pop")
        p_b = st.slider(f"proporción que {b_label}: P(B)", 0.05, 0.95, 0.40, step=0.01, key="cond_pb")
        p_a_given_b = st.slider(f"entre quienes {b_label}, proporción que {a_label}: P(A | B)", 0.0, 1.0, 0.75, step=0.01, key="cond_pagb")
        p_a_given_notb = st.slider(f"entre quienes NO {b_label}, proporción que {a_label}: P(A | no B)", 0.0, 1.0, 0.25, step=0.01, key="cond_pagnotb")
        n_b = int(round(pop * p_b))
        n_notb = pop - n_b
        n_ab = int(round(n_b * p_a_given_b))
        n_notab = n_b - n_ab
        n_a_notb = int(round(n_notb * p_a_given_notb))
        n_nota_notb = n_notb - n_a_notb
        p_a = (n_ab + n_a_notb) / pop
        p_b_given_a = n_ab / max(n_ab + n_a_notb, 1)
        independence_gap = abs(p_a_given_b - p_a)
        metric_grid([
            ("P(A)", f"{p_a:.3f}"),
            ("P(A | B)", f"{p_a_given_b:.3f}"),
            ("P(B|A)", f"{p_b_given_a:.3f}"),
            ("Cambio al saber B", f"{independence_gap:.3f}"),
        ], columns=2)
        if independence_gap < 0.02 and abs(p_a_given_notb - p_a) < 0.02:
            st.caption("Con estos parámetros, A y B están cerca de ser independientes.")
        else:
            st.caption("Aquí conocer B sí altera la probabilidad de A, así que no hay independencia.")
        st.caption("Cambio cercano a 0 significa independencia aproximada; valores grandes indican que saber B cambia bastante la probabilidad de A.")
    with col2:
        table = pd.DataFrame(
            {
                f"B: {b_label}": [n_ab, n_notab, n_b],
                f"¬B: no {b_label}": [n_a_notb, n_nota_notb, n_notb],
                "Total": [n_ab + n_a_notb, n_notab + n_nota_notb, pop],
            },
            index=[f"A: {a_label}", f"¬A: no {a_label}", "Total"],
        )
        st.dataframe(table, width="stretch")
        st.caption("Para leer P(A | B), mira la columna B. Para leer P(B | A), mira la fila A: cambian los denominadores.")
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.bar(["B", "¬B"], [n_ab / max(n_b, 1), n_a_notb / max(n_notb, 1)], color=["#4C72B0", "#DD8452"])
        ax.axhline(p_a, color="#111827", ls=":", lw=1.6, label=f"P(A)={p_a:.2f}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proporción de A dentro de cada grupo")
        ax.set_title("Comparación entre P(A | B) y P(A | no B)")
        ax.legend()
        mia_pyplot(fig)
        plt.close(fig)
        st.latex(
            rf"P(A)=\frac{{{n_ab}+{n_a_notb}}}{{{pop}}}={p_a:.3f}, \qquad P(B\mid A)=\frac{{{n_ab}}}{{{n_ab+n_a_notb}}}={p_b_given_a:.3f}"
        )
    how_to_read(
        "La tabla separa claramente qué universo se usa para cada condicional. La columna B sirve para $P(A\\mid B)$; "
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
        mia_pyplot(fig); plt.close(fig)
    how_to_read(
        "La barra azul y la naranja muestran probabilidades de ganar estimadas por simulación. "
        "Cuando sólo queda una puerta disponible para cambiar, la estrategia de cambio recoge casi toda la probabilidad de que tu primera elección haya sido errónea. "
        "Si quedan varias puertas cerradas y cambias a una sola elegida al azar, esa ventaja se reparte entre esas puertas restantes."
    )

    self_check_header()
    quiz(
        "Si $P(A\\mid B) = P(A)$, entonces...",
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
    plain_language(
        "Idea antes de la fórmula",
        "Bayes sirve para invertir una pregunta. Muchas veces sabemos qué tan probable es una evidencia si una causa fuera cierta "
        "(por ejemplo, 'si alguien está enfermo, qué tan probable es que el test salga positivo'), pero queremos la pregunta inversa: "
        "'si el test salió positivo, qué tan probable es que realmente esté enfermo'."
    )
    latex_aligned([
        r"\text{posterior}=\frac{\text{verosimilitud}\times\text{prior}}{\text{evidencia total}}",
        r"P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)}",
        r"P(E)=\sum_i P(E\mid H_i)\,P(H_i)",
    ])
    notation_box([
        (r"H", "Hipótesis o causa que queremos evaluar: 'tiene enfermedad', 'es spam', 'esta moneda está cargada'."),
        (r"E", "Evidencia observada: 'test positivo', 'aparece la palabra descuento', 'salió CCS'."),
        ("prior", "Tasa base antes de mirar la evidencia."),
        ("posterior", "Probabilidad después de mirar la evidencia."),
        ("evidencia total", "Todos los caminos posibles que pueden producir la evidencia observada."),
    ], expanded=True)
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

    worked_example("test de drogas")
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

    worked_example("moneda cargada: identificar la hipótesis correcta")
    st.markdown(
        "Tenemos 3 monedas: dos honestas y una cargada con $P(C)=2/3$. Se lanzan las tres y se observa $B=CCS$. "
        "Sea $E_i$ el evento de que la moneda cargada sea la $i$-ésima."
    )
    st.latex(
        r"P(E_1\mid B)=\frac{P(B\mid E_1)P(E_1)}{\sum_{i=1}^3 P(B\mid E_i)P(E_i)}"
        r"=\frac{(2/3)(1/2)(1/2)(1/3)}{(1/18)+(1/18)+(1/36)}=\frac{2}{5}"
    )

    worked_example("filtro de spam con la palabra Descuento")
    st.markdown(
        "Si $P(Spam)=0.20$, $P(Descuento\\mid Spam)=0.70$ y $P(Descuento\\mid Ham)=0.05$, entonces:"
    )
    st.latex(
        r"P(Spam\mid Descuento)=\frac{0.70\cdot0.20}{0.70\cdot0.20+0.05\cdot0.80}\approx0.778"
    )

    worked_example("test covid con baja positividad")
    st.markdown(
        "Con prevalencia $P(D)=0.003$, sensibilidad $0.95$ y especificidad $0.78$, el posterior dado positivo es bajo "
        "porque los falsos positivos vienen de una población sana mucho más grande."
    )
    st.latex(
        r"P(D\mid +)=\frac{0.95\cdot0.003}{0.95\cdot0.003+0.22\cdot0.997}\approx0.0128"
    )

    pitfall(
        "Confundir sensibilidad con valor predictivo positivo. Sensibilidad responde 'si realmente hay enfermedad, ¿el test sale positivo?'; "
        "el posterior responde 'si salió positivo, ¿qué tan probable es que haya enfermedad?'. Son preguntas distintas."
    )

    interactive_header("Bayes con frecuencias naturales y análisis paramétrico")
    lab_task(
        predict="cuando la prevalence es baja, decide si los false positives pueden superar a los true positives.",
        manipulate="cambia prevalence, sensitivity, specificity y tamaño poblacional.",
        verify="lee el posterior mediante conteos: true positives dividido por todos los tests positivos.",
    )
    interactive_guide(
        controls=[
            ("Prevalencia P(D)", "qué fracción de la población realmente tiene la condición."),
            ("Sensitivity P(+ | D)", "qué tan a menudo el test detecta correctamente un caso enfermo."),
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
    prior = st.slider("prevalencia: proporción realmente enferma P(D)", 0.001, 0.5, 0.01, step=0.001, format="%.3f", key="bay_prior")
    sens = st.slider("sensitivity: positive if disease P(+ | D)", 0.5, 1.0, 0.99, step=0.01, key="bay_sens")
    spec = st.slider("specificity: negative if no disease P(- | no D)", 0.5, 1.0, 0.99, step=0.01, key="bay_spec")
    pop = st.slider("Población de referencia para frecuencias naturales", 1000, 100000, 10000, step=1000, key="bay_pop")
    fpr = 1 - spec
    num = sens * prior
    den = num + fpr * (1 - prior)
    post = num / den if den > 0 else 0
    post_neg = ((1 - sens) * prior) / (((1 - sens) * prior) + spec * (1 - prior))
    tabs = st.tabs(["Frecuencias naturales", "Posterior vs prevalencia"])
    with tabs[0]:
        col1, col2 = lab_columns()
        with col1:
            n_d = int(round(pop * prior))
            n_notd = pop - n_d
            tp = int(round(n_d * sens))
            fn = n_d - tp
            tn = int(round(n_notd * spec))
            fp = n_notd - tn
            metric_grid([
                ("enfermedad dado test +", f"{post:.4f}"),
                ("enfermedad dado test −", f"{post_neg:.4f}"),
                ("Verdaderos positivos", f"{tp:,}"),
                ("Falsos positivos", f"{fp:,}"),
            ], columns=2)
            st.caption(
                f"Falsos positivos = sanos × tasa de falso positivo = {n_notd:,} × {fpr:.3f}. "
                "Este término suele dominar cuando la prevalencia es baja."
            )
            latex_aligned([
                rf"P(D\mid +)=\frac{{TP}}{{TP+FP}}=\frac{{{tp}}}{{{tp}+{fp}}}={post:.4f}",
                rf"P(+)=P(+\mid D)P(D)+P(+\mid \neg D)P(\neg D)={den:.4f}",
            ])
        with col2:
            freq_table = pd.DataFrame(
                {
                    "D": [tp, fn, n_d],
                    "¬D": [fp, tn, n_notd],
                    "Total": [tp + fp, fn + tn, pop],
                },
                index=["+", "-", "Total"],
            )
            st.dataframe(freq_table, width="stretch")
            fig, ax = plt.subplots(figsize=(7, 3.1))
            ax.bar(["Positivos"], [tp], color="#4C72B0", label="Verdaderos positivos")
            ax.bar(["Positivos"], [fp], bottom=[tp], color="#DD8452", label="Falsos positivos")
            ax.set_ylabel("Número de casos")
            ax.legend()
            mia_pyplot(fig)
            plt.close(fig)
            st.caption("Esta barra muestra sólo los tests positivos. El posterior P(D|+) es la fracción azul dentro de azul+naranja.")
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
        mia_pyplot(fig)
        plt.close(fig)
        st.caption(
            "La curva muestra por qué el valor predictivo positivo es extremadamente sensible a la prevalencia. "
            "Con baja prevalencia, incluso un test muy bueno puede producir muchos falsos positivos. El eje horizontal está en escala logarítmica, por eso comprime prevalencias pequeñas."
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

    real_world_case(
        "detectar spam con palabras simples",
        "Antes de mirar el dataset de vinos, piensa en un correo. Queremos decidir si es `spam` o `no spam` usando palabras observadas. "
        "Si aparecen `gratis` y `urgente`, esas palabras suelen empujar hacia spam. Si aparece `reunión`, puede empujar hacia no spam. "
        "Naïve Bayes convierte cada palabra en una pequeña pieza de evidencia y suma esas piezas para decidir.",
        controls=[
            ("palabras presentes", "son los atributos del correo; cada palabra aporta evidencia a favor o en contra de una clase."),
            ("prior de spam", "qué tan frecuente es el spam antes de leer el contenido."),
            ("score", "puntaje de compatibilidad; más alto significa que la clase explica mejor el correo."),
        ],
        takeaway="La hipótesis naïve es suponer que, una vez fijada la clase, podemos multiplicar aportes palabra por palabra. No es literalmente cierto, pero suele funcionar bien como primer clasificador.",
        expanded=True,
    )
    toy_words = pd.DataFrame(
        {
            "palabra": ["gratis", "urgente", "reunión"],
            "log evidencia para spam": [1.6, 1.1, -1.3],
            "lectura": ["empuja fuerte a spam", "empuja a spam", "empuja contra spam"],
        }
    )
    st.dataframe(toy_words, hide_index=True, width="stretch")
    st.caption("No necesitas conocer logs aún: en esta tabla, positivo suma evidencia a favor de spam y negativo resta evidencia.")

    with advanced_expander("qué calcula Gaussian Naïve Bayes internamente"):
        st.markdown(
            "Para cada clase, el modelo resume los datos con tres piezas: frecuencia de la clase, promedio de cada atributo y varianza de cada atributo. "
            "Luego, para clasificar una nueva observación, compara qué clase hace más compatibles esos valores observados."
        )
        compact_dataframe(pd.DataFrame([
            {"Paso": "1. Separar por clase", "Lectura": "mirar sólo los ejemplos conocidos de cada clase"},
            {"Paso": "2. Estimar promedios y varianzas", "Lectura": "describir cómo suele verse cada atributo dentro de esa clase"},
            {"Paso": "3. Sumar evidencia atributo por atributo", "Lectura": "cada atributo empuja el puntaje hacia una clase u otra"},
            {"Paso": "4. Elegir el mayor puntaje", "Lectura": "la clase ganadora es la que mejor explica la observación completa"},
        ]))
        st.caption("El código no se muestra en pantalla porque esta sección busca lectura conceptual del modelo, no implementación.")
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
    c1.metric("Porcentaje de aciertos sklearn", f"{acc_sk:.3f}")
    c2.metric("Porcentaje de aciertos implementación propia", f"{acc_np:.3f}")
    st.caption(f"Dataset Wine: {len(fnames)} mediciones químicas por vino y {len(tnames)} clases ({', '.join(tnames)}). No son variables intuitivas como precio o sabor.")

    interactive_header("Descomponer una predicción en evidencia atributo por atributo")
    interactive_guide(
        controls=[
            ("Observación del conjunto de test", "elige qué ejemplo real del dataset Wine quieres analizar."),
            ("Atributos más influyentes a mostrar", "cuántos atributos con mayor impacto en la decisión quieres inspeccionar."),
        ],
        procedure=(
            "Para la observación elegida, el laboratorio calcula el puntaje logarítmico de cada clase y luego compara atributo por atributo "
            "qué términos favorecen a la clase ganadora y cuáles a la segunda mejor."
        ),
        observe=(
            "Esto convierte la decisión del modelo en algo auditable: puedes ver qué atributos empujan fuertemente hacia una clase y cuáles generan duda."
        ),
    )
    col1, col2 = lab_columns()
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
        metric_grid([
            ("Clase predicha", tnames[pred_class]),
            ("Clase real", tnames[y_true]),
            ("Confianza relativa del modelo", f"{probs[pred_idx]:.4f}"),
            ("Ventaja sobre segunda clase", f"{scores[pred_idx] - scores[runner_idx]:.3f}"),
        ], columns=2)
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
        axes[1].tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        mia_pyplot(fig)
        plt.close(fig)
    contribution_df = pd.DataFrame(
        {
            "feature": [fnames[i] for i in top_idx],
            "valor observado": [round(float(x0[i]), 4) for i in top_idx],
            f"log p(x_j|{tnames[pred_class]})": [round(float(log_lik[pred_idx, i]), 4) for i in top_idx],
            f"log p(x_j|{tnames[runner_class]})": [round(float(log_lik[runner_idx, i]), 4) for i in top_idx],
            "diferencia": [round(float(feature_margin[i]), 4) for i in top_idx],
            "lectura": [
                "favorece predicha" if feature_margin[i] > 0.25 else "favorece segunda" if feature_margin[i] < -0.25 else "casi neutro"
                for i in top_idx
            ],
        }
    )
    st.dataframe(
        contribution_df,
        hide_index=True,
        width="stretch",
    )
    how_to_read(
        "Una diferencia positiva favorece la clase predicha; una negativa favorece a la segunda mejor clase. "
        "La decisión final es la suma de todas estas contribuciones más el prior."
    )

    interactive_header("Visualizar frontera con 2 features (Wine)")
    lab_task(
        predict="elige dos atributos y anticipa si las regiones coloreadas deberían separar limpiamente las clases.",
        manipulate="cambia los atributos de los ejes horizontal y vertical.",
        verify="compara las decision regions con los puntos reales etiquetados.",
    )
    col1, col2 = lab_columns()
    with col1:
        feat_x = st.selectbox("Atributo eje horizontal (feature X)", options=list(range(len(fnames))),
                              format_func=lambda i: fnames[i], index=0, key="nb_fx")
        feat_y = st.selectbox("Atributo eje vertical (feature Y)", options=list(range(len(fnames))),
                              format_func=lambda i: fnames[i], index=6, key="nb_fy")
        if feat_x == feat_y:
            st.warning("Elige dos atributos distintos para que la frontera 2D sea interpretable.")

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

    if feat_x == feat_y:
        feat_y = next(i for i in range(len(fnames)) if i != feat_x)
    xx, yy, Z, X2, y2 = _nb_decision_boundary(feat_x, feat_y)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
        for c in np.unique(y2):
            ax.scatter(X2[y2==c, 0], X2[y2==c, 1], label=tnames[c], s=22, edgecolor="k", alpha=0.8)
        ax.set_xlabel(fnames[feat_x]); ax.set_ylabel(fnames[feat_y])
        ax.legend()
        mia_pyplot(fig)
    how_to_read("Zonas coloreadas: región donde el clasificador predice cada clase. Puntos: datos reales coloreados por su etiqueta verdadera.")
    st.caption("Esta frontera corresponde a un modelo Naive Bayes reentrenado sólo con estos dos atributos; no es la frontera del modelo completo de 13 atributos proyectada a 2D.")

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
    notation_box([
        (r"X", "La variable aleatoria: el número que observamos, por ejemplo cantidad de caras, tiempo de espera o error de medición."),
        (r"p_X(k)", "Probabilidad puntual de una variable discreta: la probabilidad de que X tome exactamente el valor k."),
        (r"f_X(x)", "Densidad de una variable continua. No es una probabilidad por sí sola; la probabilidad aparece al calcular área en un intervalo."),
        (r"F_X(x)", "Probabilidad acumulada hasta x: responde 'qué proporción cae en x o menos'."),
        (r"[a,b]", "Rango de valores que nos interesa. En aplicaciones suele ser un intervalo aceptable, una ventana de tiempo o un rango de conteos."),
    ], expanded=True)
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
    compact_dataframe(df_moneda)

    worked_example("urna con 3 rojas y 2 azules, extraer 2 sin reemplazo, $X$ = # rojas")
    st.markdown("PMF usando combinatoria (hipergeométrica):")
    st.latex(r"P(X=k) = \frac{\binom{3}{k}\binom{2}{2-k}}{\binom{5}{2}}, \quad k \in \{0,1,2\}")
    vals = [0,1,2]
    pmf = [comb(3, k, exact=True)*comb(2, 2-k, exact=True) / comb(5, 2, exact=True) for k in vals]
    compact_dataframe(pd.DataFrame({"k": vals, "P(X=k)": pmf, "F(k)": np.cumsum(pmf)}))

    interactive_header("Probabilidad de intervalos: masa/densidad y CDF en paralelo")
    lab_task(
        predict="antes de cambiar la distribución, decide si la probabilidad debería ser suma de barras o área bajo una curva.",
        manipulate="elige Binomial, Normal o Exponential y mueve los extremos del intervalo.",
        verify="compara la masa/área destacada con la diferencia de CDF mostrada bajo el gráfico.",
    )
    st.caption("PMF = masa en puntos; PDF = densidad cuya área da probabilidad; CDF = probabilidad acumulada hasta un valor.")
    real_world_case(
        "leer probabilidades como preguntas concretas",
        "Este laboratorio no sólo mueve una curva. Traduce preguntas reales a probabilidades: "
        "'¿cuántas campañas tendrán entre 6 y 10 respuestas?', '¿qué mediciones caen dentro del rango aceptable?', "
        "'¿cuánto tarda un evento entre media y dos horas?'.",
        controls=[
            ("Distribución", "elige la historia probabilística: conteos, mediciones alrededor de un promedio o tiempos de espera."),
            ("Parámetros", "cambian la frecuencia esperada, la dispersión o la velocidad del proceso."),
            ("Intervalo [a,b]", "define la pregunta concreta: qué rango de resultados quieres medir."),
        ],
        takeaway="La probabilidad del intervalo es el área o suma marcada. En discreta se suman barras; en continua se mide área bajo la curva.",
        expanded=False,
    )
    interactive_guide(
        controls=[
            ("Distribución", "elige si quieres estudiar una variable discreta o continua."),
            ("Parámetros", "determinan la forma concreta de la distribución elegida."),
            ("Intervalo [a,b]", "es el rango de valores cuya probabilidad quieres calcular."),
        ],
        procedure=(
            "El laboratorio calcula la probabilidad del intervalo de dos maneras equivalentes: como suma o área en la gráfica de masa/densidad, "
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
            n = st.slider("número de intentos n", 5, 40, 20, key="cdf_bin_n")
            p = st.slider("probabilidad de éxito p", 0.05, 0.95, 0.40, step=0.05, key="cdf_bin_p")
            a_bin, b_bin = st.slider("rango de valores que quieres contar [a,b]", 0, n, (6, 10), key="cdf_bin_int")
        elif dist_type.startswith("Normal"):
            mu = st.slider("media μ", -3.0, 3.0, 0.0, step=0.1, key="cdf_norm_mu")
            sig = st.slider("desviación estándar σ", 0.2, 3.0, 1.0, step=0.1, key="cdf_norm_sig")
            a_cont, b_cont = st.slider("rango de valores que quieres medir [a,b]", mu - 4 * sig, mu + 4 * sig, (mu - sig, mu + sig), key="cdf_norm_int")
        else:
            lam = st.slider("tasa λ", 0.2, 3.0, 1.0, step=0.1, key="cdf_exp_lam")
            a_cont, b_cont = st.slider("rango de valores que quieres medir [a,b]", 0.0, 8.0 / lam, (0.5 / lam, 2.0 / lam), key="cdf_exp_int")
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
            st.metric("Probabilidad dentro del rango", f"{interval_prob:.4f}")
            latex_aligned([
                rf"P({a_bin}\le X \le {b_bin})=\sum_{{k={a_bin}}}^{{{b_bin}}} p_X(k)",
                rf"P({a_bin}\le X \le {b_bin})=F({b_bin})-F({a_bin-1})={interval_prob:.4f}",
            ])
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
            st.metric("Probabilidad dentro del rango", f"{interval_prob:.4f}")
            st.latex(rf"P({a_cont:.2f}\le X \le {b_cont:.2f})=F({b_cont:.2f})-F({a_cont:.2f})={interval_prob:.4f}")
            ax2.annotate("diferencia vertical = probabilidad", xy=(b_cont, stats.norm.cdf(b_cont, mu, sig)), xytext=(a_cont, 0.78),
                         arrowprops=dict(arrowstyle="->", color="#334155"), fontsize=9)
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
            st.metric("Probabilidad dentro del rango", f"{interval_prob:.4f}")
            st.latex(rf"P({a_cont:.2f}\le X \le {b_cont:.2f})=F({b_cont:.2f})-F({a_cont:.2f})={interval_prob:.4f}")
            ax2.annotate("diferencia vertical = probabilidad", xy=(b_cont, stats.expon.cdf(b_cont, scale=1 / lam)), xytext=(a_cont, 0.78),
                         arrowprops=dict(arrowstyle="->", color="#334155"), fontsize=9)
        ax1.set_ylabel("PMF/PDF")
        ax2.set_ylabel("F(x)")
        plt.tight_layout()
        mia_pyplot(fig)
        plt.close(fig)
    how_to_read(
        "En la figura izquierda se marca la masa o el área del intervalo consultado. La figura derecha muestra "
        "la misma probabilidad leída como diferencia de valores de la CDF."
    )
    lab_note("en variables continuas, la altura de la PDF no es probabilidad; la probabilidad del rango es el área sombreada.")

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
    plain_language(
        "Cómo elegir una distribución sin memorizar fórmulas",
        "Primero pregunta qué tipo de resultado observas: ¿sí/no?, ¿cantidad de éxitos?, ¿conteos por minuto?, ¿tiempo hasta que algo ocurre?, "
        "¿una medición continua alrededor de un promedio? La distribución se elige por la historia del dato, no por la forma bonita de la curva."
    )
    notation_box([
        (r"\lambda", "En Poisson suele leerse como cantidad esperada de eventos por unidad. En Exponencial/Gamma suele leerse como tasa: qué tan rápido ocurren eventos."),
        (r"\mu", "Media o centro típico de una distribución normal."),
        (r"\sigma", "Desviación estándar: escala de dispersión alrededor de la media."),
        (r"\propto", "Proporcional a: falta una constante que normaliza para que el área total sea 1."),
        (r"\Gamma(\alpha)", "Función gamma: una extensión del factorial que aparece en distribuciones continuas como Gamma y Beta."),
    ], expanded=False)

    tabs = st.tabs([
        "Discretas", "Continuas", "Aproximaciones", "Falta de memoria (Exp)", "Explorador interactivo"
    ])

    with tabs[0]:
        st.markdown("#### Distribuciones discretas")
        compact_dataframe(pd.DataFrame([
            {"Historia del dato": "Sí/no en un intento", "Distribución": "Bernoulli(p)", "Dominio": "{0,1}", "Uso típico": "clic/no clic, falla/no falla"},
            {"Historia del dato": "Cantidad de éxitos en n intentos", "Distribución": "Binomial(n,p)", "Dominio": "0,1,...,n", "Uso típico": "conversiones en una campaña"},
            {"Historia del dato": "Ensayos hasta el primer éxito", "Distribución": "Geométrica(p)", "Dominio": "1,2,...", "Uso típico": "intentos hasta resolver"},
            {"Historia del dato": "Muestra sin reemplazo", "Distribución": "Hipergeométrica", "Dominio": "enteros posibles", "Uso típico": "control de calidad"},
            {"Historia del dato": "Conteos por unidad", "Distribución": "Poisson(λ)", "Dominio": "0,1,2,...", "Uso típico": "tickets por hora"},
        ]))
        with advanced_expander("tabla formal de distribuciones discretas"):
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

        worked_example("control de calidad con distribución hipergeométrica")
        st.markdown(
            "Muestreo sin reemplazo desde un lote finito: $N$ items totales, $K$ defectuosos, muestra de tamaño $n$, "
            "y $X$ = defectuosos encontrados."
        )
        st.latex(r"P(X=x)=\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}")
        c1, c2 = st.columns([1, 2])
        with c1:
            N_h = st.slider("N lote", 20, 300, 100, key="hyp_N")
            K_h = st.slider("K defectuosos", 1, N_h - 1, min(12, N_h - 1), key="hyp_K")
            n_hyp = st.slider("n muestra", 1, min(60, N_h), min(10, N_h), key="hyp_n")
        xs_h = np.arange(max(0, n_hyp - (N_h - K_h)), min(K_h, n_hyp) + 1)
        pmf_h = stats.hypergeom.pmf(xs_h, N_h, K_h, n_hyp)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(xs_h, pmf_h, color="#4C72B0")
            ax.set_xlabel("x defectuosos en la muestra")
            ax.set_ylabel("P(X=x)")
            ax.set_title(f"Hipergeom(N={N_h}, K={K_h}, n={n_hyp})")
            mia_pyplot(fig); plt.close(fig)
        st.metric("E[X]", f"{n_hyp*K_h/N_h:.3f}")

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

        worked_example("transformada inversa desde Uniforme(0,1)")
        st.markdown(
            "La uniforme es el motor de simulación: si $U\\sim Unif(0,1)$ y $F$ es una CDF invertible, entonces $X=F^{-1}(U)$ tiene CDF $F$."
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            inv_kind = st.radio("Distribución generada", ["Exponencial", "Logística"], key="inv_kind")
            inv_n = st.slider("muestras", 200, 5000, 1000, step=200, key="inv_n")
        rng_inv = np.random.default_rng(314)
        u = rng_inv.uniform(1e-6, 1 - 1e-6, inv_n)
        if inv_kind == "Exponencial":
            x_inv = -np.log(1 - u)
            x_grid = np.linspace(0, np.quantile(x_inv, 0.995), 300)
            pdf_grid = stats.expon.pdf(x_grid)
            formula_inv = r"F^{-1}(u)=-\log(1-u)"
        else:
            x_inv = np.log(u / (1 - u))
            x_grid = np.linspace(np.quantile(x_inv, 0.005), np.quantile(x_inv, 0.995), 300)
            pdf_grid = stats.logistic.pdf(x_grid)
            formula_inv = r"F^{-1}(u)=\log\frac{u}{1-u}"
        with c2:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(x_inv, bins=35, density=True, color="#4C72B0", alpha=0.65, label="muestras")
            ax.plot(x_grid, pdf_grid, color="#DD8452", lw=2, label="densidad teórica")
            ax.legend()
            mia_pyplot(fig); plt.close(fig)
        st.latex(formula_inv)

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
        mia_pyplot(fig); plt.close(fig)
        st.caption("La curva normal es continua y aproxima masas discretas con ancho aproximado 1; para cálculo fino se usa corrección de continuidad.")
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
                "El laboratorio dibuja la PMF o la PDF de la familia elegida con los parámetros seleccionados."
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
        col1, col2 = lab_columns()
        with col1:
            if dist == "Bernoulli":
                st.caption("Bernoulli modela un resultado sí/no, como clic/no clic.")
                p = st.slider("probabilidad de éxito p", 0.0, 1.0, 0.5, key="exp_p")
                xs = np.array([0, 1]); pmf = np.array([1-p, p]); is_disc = True; mean_val, var_val = p, p * (1 - p)
            elif dist == "Binomial":
                st.caption("Binomial cuenta cuántos éxitos aparecen en una cantidad fija de intentos.")
                n = st.slider("cantidad de intentos n", 1, 100, 20, key="exp_n")
                p = st.slider("probabilidad de éxito p", 0.0, 1.0, 0.4, key="exp_p2")
                xs = np.arange(0, n+1); pmf = stats.binom.pmf(xs, n, p); is_disc = True; mean_val, var_val = n * p, n * p * (1 - p)
            elif dist == "Poisson":
                st.caption("Poisson modela conteos por unidad: tickets por hora, llamadas por minuto, errores por lote.")
                lam = st.slider("tasa esperada de eventos λ", 0.1, 20.0, 3.0, key="exp_lam")
                xs = np.arange(0, int(3*lam)+5); pmf = stats.poisson.pmf(xs, lam); is_disc = True; mean_val, var_val = lam, lam
            elif dist == "Geométrica":
                st.caption("Geométrica cuenta intentos hasta el primer éxito.")
                p = st.slider("probabilidad de éxito en cada intento p", 0.01, 1.0, 0.3, key="exp_gp")
                xs = np.arange(1, 30); pmf = stats.geom.pmf(xs, p); is_disc = True; mean_val, var_val = 1 / p, (1 - p) / p**2
            elif dist == "Normal":
                st.caption("Normal modela mediciones continuas alrededor de un centro.")
                mu = st.slider("media o centro μ", -5.0, 5.0, 0.0, key="exp_mu"); sig = st.slider("dispersión σ", 0.1, 4.0, 1.0, key="exp_sig")
                xs = np.linspace(mu-4*sig, mu+4*sig, 400); pmf = stats.norm.pdf(xs, mu, sig); is_disc = False; mean_val, var_val = mu, sig**2
            elif dist == "Exponencial":
                st.caption("Exponencial modela tiempo de espera hasta el próximo evento.")
                lam = st.slider("tasa de ocurrencia λ", 0.1, 3.0, 1.0, key="exp_elam")
                xs = np.linspace(0, 5/lam, 400); pmf = stats.expon.pdf(xs, scale=1/lam); is_disc = False; mean_val, var_val = 1 / lam, 1 / lam**2
            elif dist == "Gamma":
                st.caption("Gamma modela tiempos acumulados hasta varios eventos.")
                a = st.slider("forma α", 0.5, 10.0, 2.0, key="exp_ga"); lam = st.slider("tasa λ", 0.1, 3.0, 1.0, key="exp_glam")
                xs = np.linspace(0, (a+3)/lam, 400); pmf = stats.gamma.pdf(xs, a, scale=1/lam); is_disc = False; mean_val, var_val = a / lam, a / lam**2
            elif dist == "Beta":
                st.caption("Beta modela una proporción desconocida, como una tasa de conversión entre 0 y 1.")
                a = st.slider("forma α", 0.1, 10.0, 2.0, key="exp_ba"); b = st.slider("forma β", 0.1, 10.0, 2.0, key="exp_bb")
                xs = np.linspace(0.001, 0.999, 400); pmf = stats.beta.pdf(xs, a, b); is_disc = False; mean_val, var_val = a / (a + b), a * b / ((a + b)**2 * (a + b + 1))
            elif dist == "Student-t":
                st.caption("Student-t se parece a una normal, pero con colas más pesadas cuando ν es bajo.")
                nu = st.slider("grados de libertad ν", 1, 50, 5, key="exp_nu")
                xs = np.linspace(-6, 6, 400); pmf = stats.t.pdf(xs, nu); is_disc = False; mean_val = 0 if nu > 1 else float("nan"); var_val = nu / (nu - 2) if nu > 2 else float("inf")
            else:
                st.caption("Uniforme reparte la densidad de forma pareja entre dos límites.")
                a = st.slider("límite inferior a", -5.0, 5.0, 0.0, key="exp_ua"); b = st.slider("límite superior b", a+0.1, a+10.0, a+1.0, key="exp_ub")
                xs = np.linspace(a-0.5, b+0.5, 400); pmf = stats.uniform.pdf(xs, a, b-a); is_disc = False; mean_val, var_val = (a + b) / 2, (b - a)**2 / 12
        with col2:
            metric_grid([
                ("promedio esperado E[X]", f"{mean_val:.3f}" if np.isfinite(mean_val) else "no existe"),
                ("variabilidad Var[X]", f"{var_val:.3f}" if np.isfinite(var_val) else "no finita"),
            ], columns=2)
            fig, ax = plt.subplots(figsize=(7.5, 3.3))
            if is_disc:
                ax.bar(xs, pmf, color="#4C72B0")
                ax.set_ylabel("P(X=k)")
            else:
                ax.fill_between(xs, pmf, color="#4C72B0", alpha=0.55)
                ax.plot(xs, pmf, color="#4C72B0")
                ax.set_ylabel("f(x)")
            ax.set_title(dist)
            mia_pyplot(fig); plt.close(fig)
            lab_note("observa si al mover parámetros la distribución se desplaza, se concentra, se aplana o cambia sus colas.")

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
    latex_aligned([
        r"\mathcal{L}(\theta) = \prod_{i=1}^n p(x_i \mid \theta) \quad \text{(verosimilitud)}",
        r"\ell(\theta) = \log \mathcal{L}(\theta) = \sum_i \log p(x_i\mid\theta) \quad \text{(log-verosimilitud)}",
        r"\hat\theta_{MLE} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \big[-\ell(\theta)\big] \quad \text{(NLL)}",
    ])
    notation_box([
        (r"\theta", "El parámetro que queremos aprender. En una moneda sería la probabilidad de cara; en un modelo puede ser un peso."),
        (r"\mathcal L(\theta)", "Verosimilitud: qué tan bien explica ese parámetro los datos que ya observamos."),
        (r"\prod", "Producto: multiplicar muchos términos. Aparece porque asumimos datos independientes."),
        (r"\log", "Logaritmo: convierte productos en sumas y evita números extremadamente pequeños."),
        (r"\arg\max", "El valor de θ donde la función alcanza su máximo; no es el valor máximo, sino el parámetro que lo logra."),
        ("NLL", "Negative log-likelihood: la misma idea, escrita como pérdida para minimizar."),
    ], expanded=True)
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
    latex_aligned([
        r"\ell(p) = k\log p + (n-k)\log(1-p)",
        r"\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0",
        r"\frac{k}{p}=\frac{n-k}{1-p}\Rightarrow k(1-p)=(n-k)p\Rightarrow k=np\Rightarrow \hat p=\frac{k}{n}",
    ])
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
    lab_task(
        predict="el maximum likelihood estimate debería quedar en la tasa observada de éxitos k/n.",
        manipulate="cambia el tamaño muestral n y el porcentaje de éxitos observado.",
        verify="revisa que el máximo de la log-likelihood quede en el MLE.",
    )
    interactive_guide(
        controls=[
            ("n lanzamientos", "cantidad total de observaciones Bernoulli disponibles."),
            ("porcentaje de éxitos observados k/n", "fracción de éxitos en la muestra."),
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
        k_ratio = st.slider("porcentaje de éxitos observados k/n", 0.0, 1.0, 0.6, key="mle_k")
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
        mia_pyplot(fig); plt.close(fig)
    how_to_read("La curva es cóncava con un único máximo — ese máximo es $\\hat p$. Más datos → pico más estrecho → mayor certeza.")
    lab_note("los datos quedan fijos; lo que cambia sobre el eje horizontal es el parámetro p. Más alto significa que ese p explica mejor esos datos.")

    interactive_header("Cómo castiga la entropía cruzada una predicción")
    lab_task(
        predict="una predicción equivocada y segura debería producir una loss mucho mayor que una predicción incierta.",
        manipulate="cambia la etiqueta verdadera y mueve la probabilidad predicha q.",
        verify="lee el punto negro sobre la curva de BCE.",
    )
    real_world_case(
        "clasificar un correo como spam",
        "Supón que clase 1 significa `spam` y clase 0 significa `no spam`. El modelo entrega `q`, su confianza de que el correo sea spam. "
        "La pérdida BCE castiga poco si el modelo asigna alta probabilidad a la clase correcta, y castiga mucho si está seguro pero equivocado.",
        controls=[
            ("Etiqueta observada y", "la verdad conocida: 1 si era spam, 0 si no era spam."),
            ("Predicción q", "confianza del modelo en que el correo sea spam."),
        ],
        takeaway="Mover q permite ver por qué una predicción segura y equivocada es mucho más grave que una predicción dudosa.",
        expanded=False,
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        y_obs = st.radio("Etiqueta observada y", [0, 1], horizontal=True, key="bce_y")
        q_hat = st.slider("probabilidad que predice el modelo para la clase 1: q = P(y=1|x)", 0.001, 0.999, 0.7, step=0.001, key="bce_q")
        loss = -(y_obs * np.log(q_hat) + (1 - y_obs) * np.log(1 - q_hat))
        metric_grid([
            ("Predicción q", f"{q_hat:.3f}"),
            ("Pérdida BCE", f"{loss:.4f}"),
        ], columns=2)
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
        mia_pyplot(fig)
        plt.close(fig)
    how_to_read(
        "Cuando el modelo está seguro y se equivoca, la pérdida crece abruptamente. En cambio, una predicción segura y correcta tiene pérdida cercana a 0."
    )
    lab_note("si la etiqueta real es 1, la zona buena está cerca de q=1; si la etiqueta real es 0, la zona buena está cerca de q=0.")

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
    notation_box([
        (r"E[X]", "Promedio esperado de X. No siempre es el valor más frecuente; es el centro de equilibrio de la distribución."),
        (r"E[h(X)]", "Primero transformas X con una función h, y luego promedias el resultado."),
        (r"\text{Var}(X)", "Promedio de las distancias cuadradas al centro. Mide dispersión."),
        (r"f(E[X])", "Aplicar una función al promedio."),
        (r"E[f(X)]", "Aplicar la función a cada valor posible y luego promediar. Jensen compara estas dos operaciones."),
    ], expanded=True)
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
    real_world_case(
        "por qué la variabilidad importa",
        "Jensen explica una idea muy práctica: cuando una función no es lineal, evaluar el promedio no equivale a promediar los resultados. "
        "En pérdidas de modelos, finanzas o riesgos, dos escenarios con el mismo promedio pueden tener consecuencias distintas si uno es más variable.",
        controls=[
            ("Función", "elige cómo se transforma el resultado: cuadrado, logaritmo o exponencial."),
            ("centro de X", "centro aproximado de los escenarios."),
            ("dispersión", "qué tan variables son los escenarios alrededor del centro."),
        ],
        takeaway="Si la función es convexa, la variabilidad suele subir el promedio transformado. Si es cóncava, lo baja.",
        expanded=False,
    )
    interactive_guide(
        controls=[
            ("Función", "elige si quieres una función convexa o cóncava."),
            ("centro de X", "fija aproximadamente la ubicación central de la variable."),
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
        mu = st.slider("centro de X", 0.5, 4.0, 2.0, key="jen_mu")
        sig = st.slider("dispersión de X σ", 0.1, 2.0, 0.8, key="jen_sig")
    rng = np.random.default_rng(0)
    xs = rng.normal(mu, sig, 5000)
    if "x²" in f_type:
        f = lambda x: x**2; xp = np.linspace(max(0.01, mu-3*sig), mu+3*sig, 200)
    elif "log" in f_type:
        xs = np.clip(xs, 0.05, None)
        f = lambda x: np.log(x); xp = np.linspace(0.05, mu+3*sig, 200)
    else:
        f = lambda x: np.exp(x); xp = np.linspace(mu-3*sig, mu+3*sig, 200)
    empirical_EX = float(np.mean(xs))
    EfX = np.mean(f(xs)); fEX = f(empirical_EX)
    delta = EfX - fEX
    with col1:
        metric_grid([
            ("brecha de Jensen", f"{delta:.4f}", "convexa → ≥0" if "convexa" in f_type else "cóncava → ≤0"),
            ("E[X] empírico", f"{empirical_EX:.3f}"),
        ], columns=2)
        if "log" in f_type:
            st.caption("Para poder evaluar log(x), las simulaciones no permiten valores menores que 0.05.")
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(xp, f(xp), color="#4C72B0", lw=2, label="f(x)")
        ax.scatter([empirical_EX], [fEX], color="#DD8452", s=90, zorder=5, label=f"f(E[X])={fEX:.2f}")
        ax.axhline(EfX, color="#55A868", ls="--", label=f"E[f(X)]={EfX:.2f}")
        ax.axvline(empirical_EX, color="gray", ls=":", alpha=0.5)
        ax.legend(); ax.set_xlabel("x")
        mia_pyplot(fig); plt.close(fig)
        lab_note("compara el punto naranja f(E[X]) con la línea verde E[f(X)]; la distancia vertical entre ambos es la brecha de Jensen.")

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
        "10. Función Generadora de Momentos, Covarianza y Correlación",
        "Herramientas para comparar distribuciones (FGM) y cuantificar relaciones lineales entre variables."
    )
    beginner_bridge(
        "qué problema resuelve esta sección",
        [
            "Una variable aleatoria es una columna de datos incierta: por ejemplo ventas, errores o tiempos de espera.",
            "Un momento es un resumen numérico de esa variable: media, varianza u otros promedios de potencias.",
            "La correlación sólo mira relaciones tipo recta; una relación curva puede ser fuerte aunque la correlación sea cero.",
        ],
    )
    motivation(
        "La **FGM** empaqueta *todos* los momentos ($E[X], E[X^2], ...$) en una sola función: si la FGM existe "
        "en un entorno de 0 y dos VAs tienen la misma FGM en ese entorno, entonces son iguales en distribución. Útil para probar propiedades de sumas. "
        "La **covarianza** y **correlación** miden qué tan juntas se mueven dos VAs."
    )
    prerequisites_box(
        "- $E[X]$, $\\text{Var}(X)$.\n"
        "- Serie de Taylor de $e^{tX}$.\n"
        "- Derivadas parciales."
    )
    st.markdown("### Construcción")
    plain_language(
        "Intuición de la FGM",
        "La Función Generadora de Momentos (FGM) puede leerse como una máquina: le das una distribución y, al derivarla en cero, "
        "te devuelve promedios de potencias como $E[X]$, $E[X^2]$ y así sucesivamente. No hace falta memorizarla como truco; "
        "su valor es que empaqueta muchos resúmenes de la distribución en una sola función."
    )
    st.caption("Pregunta que responde la fórmula: ¿cómo recupero media, varianza y otros momentos desde una sola función?")
    latex_aligned([
        r"M_X(t) = E[e^{tX}] = 1 + tE[X] + \tfrac{t^2}{2!}E[X^2] + \tfrac{t^3}{3!}E[X^3] + \ldots",
        r"M_X^{(k)}(0) = E[X^k]",
    ])
    st.markdown("**Propiedades clave**:")
    st.markdown(
        "- Unicidad: misma FGM en un entorno de 0 ⇒ misma distribución.\n"
        "- Suma de independientes: $M_{X+Y}(t) = M_X(t)\\,M_Y(t)$.\n"
        "- Ejemplo Gamma($\\alpha, \\lambda$): $M(t) = (\\lambda/(\\lambda-t))^\\alpha$ para $t<\\lambda$."
    )
    worked_example("E[X] y Var[X] de Gamma usando FGM")
    latex_aligned([
        r"M'(0) = \alpha/\lambda = E[X]",
        r"M''(0) = \alpha(\alpha+1)/\lambda^2",
        r"\text{Var}(X)=E[X^2]-E[X]^2=\frac{\alpha(\alpha+1)}{\lambda^2}-\left(\frac{\alpha}{\lambda}\right)^2=\frac{\alpha}{\lambda^2}",
    ])

    st.markdown("### Covarianza y correlación")
    st.latex(r"\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]")
    st.latex(r"\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1, 1]")
    st.markdown("- $\\rho=0$: sin relación lineal (NO implica independencia salvo en variables conjuntamente gaussianas).")
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
            "En variables conjuntamente gaussianas, linealidad y dependencia coinciden de forma especial; fuera de ese caso, no."
        ],
    )

    worked_example("Cov(X, X+Y) con X, Y independientes de varianza 1")
    st.latex(r"\text{Cov}(X, X+Y) = \text{Cov}(X,X) + \text{Cov}(X,Y) = \text{Var}(X) + 0 = 1")

    worked_example("Covarianza cero no implica independencia: X uniforme en {-1, 0, 1}, Y = X²")
    st.markdown(
        "Sea $X \\sim \\text{Uniforme}\\{-1, 0, 1\\}$ (cada valor con probabilidad $1/3$) y $Y = X^2$. "
        "Claramente $Y$ depende de $X$ (la función es determinista), pero calculemos la covarianza:"
    )
    latex_aligned([
        r"E[X] = \tfrac{1}{3}(-1 + 0 + 1) = 0",
        r"E[Y] = E[X^2] = \tfrac{1}{3}(1 + 0 + 1) = \tfrac{2}{3}",
        r"E[XY] = E[X^3] = \tfrac{1}{3}((-1)^3 + 0^3 + 1^3) = 0",
        r"\text{Cov}(X,Y) = E[XY] - E[X]E[Y] = 0 - 0 \cdot \tfrac{2}{3} = 0",
    ])
    st.error(
        "**Conclusión:** $X$ e $Y$ están perfectamente relacionadas ($Y=X^2$), pero su covarianza y su correlación "
        "de Pearson son ambas cero. La relación es cuadrática — simétrica respecto al origen — y eso cancela "
        "cualquier señal lineal."
    )
    insight(
        "La correlación de Pearson $\\rho$ solo detecta relaciones que se pueden trazar con una recta. "
        "Cualquier relación simétrica o curvada puede tener $\\rho \\approx 0$ y aún así existir una dependencia fuerte."
    )

    with advanced_expander("Derivación de FGM para distribuciones clave"):
        st.markdown(
            "La FGM permite calcular momentos derivando en cero, y además actúa como 'huella dactilar': "
            "dos variables con la misma FGM en un entorno de cero tienen la misma distribución."
        )
        tabs_fgm = st.tabs(["Poisson", "Normal estándar", "Exponencial"])
        with tabs_fgm[0]:
            st.markdown("**$X \\sim \\text{Poisson}(\\lambda)$**, con $P(X=k) = \\lambda^k e^{-\\lambda}/k!$")
            latex_aligned([
                r"M(t) = E[e^{tX}] = \sum_{k=0}^\infty e^{tk}\frac{\lambda^k e^{-\lambda}}{k!}",
                r"= e^{-\lambda}\sum_{k=0}^\infty \frac{(e^t \lambda)^k}{k!} = e^{-\lambda} e^{\lambda e^t}",
                r"= e^{\lambda(e^t - 1)}",
            ])
            st.markdown("**Verificación de momentos por derivación:**")
            latex_aligned([
                r"M'(t) = e^{\lambda(e^t-1)}\lambda e^t \Rightarrow M'(0) = \lambda = E[X]",
                r"M''(0) = \lambda^2 + \lambda \Rightarrow \text{Var}(X) = \lambda^2+\lambda-\lambda^2 = \lambda",
            ])
        with tabs_fgm[1]:
            st.markdown("**$X \\sim \\mathcal{N}(0,1)$**, con $f(x) = \\frac{1}{\\sqrt{2\\pi}}e^{-x^2/2}$")
            latex_aligned([
                r"M(t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{tx} e^{-x^2/2}\,dx",
                r"= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-\frac{1}{2}(x-t)^2 + \frac{t^2}{2}}\,dx",
                r"= e^{t^2/2}",
            ])
            st.caption("La integral se completa el cuadrado en el exponente; la integral resultante es la integral total de una gaussiana, que vale 1.")
            st.markdown("Los momentos impares son 0 (simetría); $E[X^2] = M''(0) = 1 = \\text{Var}(X)$.")
        with tabs_fgm[2]:
            st.markdown("**$X \\sim \\text{Exp}(\\lambda)$**, con $f(x) = \\lambda e^{-\\lambda x}$ para $x\\ge 0$")
            latex_aligned([
                r"M(s) = \int_0^\infty e^{sx} \lambda e^{-\lambda x}\,dx = \lambda \int_0^\infty e^{-(\lambda - s)x}\,dx",
                r"= \frac{\lambda}{\lambda - s}, \quad s < \lambda",
            ])
            st.markdown("**Momentos:**")
            latex_aligned([
                r"M'(s) = \frac{\lambda}{(\lambda-s)^2} \Rightarrow E[X] = M'(0) = \frac{1}{\lambda}",
                r"M''(s) = \frac{2\lambda}{(\lambda-s)^3} \Rightarrow E[X^2] = \frac{2}{\lambda^2} \Rightarrow \text{Var}(X) = \frac{1}{\lambda^2}",
            ])

    interactive_header("Dependencia lineal y no lineal")
    lab_task(
        predict="antes de mover ρ, decide si esperas una nube inclinada hacia arriba, hacia abajo o sin inclinación.",
        manipulate="cambia ρ, el patrón no lineal y el ruido.",
        verify="compara lo que ves en la nube con la covarianza/correlación que muestra el título.",
    )
    real_world_case(
        "relaciones entre variables de negocio",
        "Piensa en dos columnas de datos: edad y gasto mensual, temperatura y demanda eléctrica, o visitas y compras. "
        "La correlación mide si una tiende a subir cuando la otra sube, pero puede perder relaciones curvas o circulares.",
        controls=[
            ("correlación deseada ρ", "fuerza y dirección de una relación lineal simulada."),
            ("patrón de dependencia", "forma de relación no lineal donde puede haber dependencia aunque la correlación sea cercana a cero."),
        ],
        takeaway="Una correlación cercana a cero no prueba que no haya relación; sólo dice que no se detecta una relación lineal fuerte.",
        expanded=False,
    )
    tabs = st.tabs(["Correlación ajustable", "Dependencia con ρ≈0"])
    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            rho = st.slider("correlación deseada ρ", -1.0, 1.0, 0.7, step=0.05, key="cov_rho")
            st.caption("+1: suben juntas; -1: una sube cuando la otra baja; 0: no detecta relación lineal.")
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
            mia_pyplot(fig)
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
            noise = st.slider("ruido: cuánto se borra el patrón", 0.0, 1.0, 0.15, step=0.05, key="cov_noise")
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
            mia_pyplot(fig)
            plt.close(fig)
            st.caption(
                "Aquí hay dependencia visible aunque la correlación pueda quedar cerca de cero. "
                "La estructura no es lineal."
            )

    self_check_header()
    quiz(
        "Si $\\rho(X,Y)=0$, entonces $X\\perp Y$.",
        ["Verdadero siempre", "Verdadero sólo para variables conjuntamente gaussianas", "Falso siempre"],
        1,
        "Ejemplo: $Y=X^2$ con $X\\sim N(0,1)$: $\\rho=0$ pero $Y$ depende totalmente de $X$.",
        "$\\rho=0$ descarta relación *lineal*, no cualquier dependencia.",
        key="cov_q1"
    )
    ai_bridge(
        "La **matriz de covarianza** $\\Sigma$ es la base de PCA (siguiente sección), whitening y la "
        "Gaussiana multivariada. En deep learning, **batch normalization** normaliza activaciones por media "
        "y desviación estándar de mini-batch; no elimina covarianzas completas como un whitening estricto."
    )
# ==================================================================
# SECCIÓN 11 — GAUSSIANA MULTIVARIADA Y PCA
# ==================================================================
def sec_pca():
    section_title(
        "11. Gaussiana Multivariada y PCA vía SVD",
        "De una variable a muchas correlacionadas: la geometría de elipses y cómo encontrar sus ejes principales."
    )
    beginner_bridge(
        "cómo leer PCA sin álgebra avanzada",
        [
            "Una nube de puntos puede estar más estirada en una dirección que en otra.",
            "PCA gira los ejes para mirar la nube desde la dirección donde se ve más estirada.",
            "Autovector significa dirección principal; autovalor significa cuánta variación hay en esa dirección.",
        ],
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
    plain_language(
        "Lectura por capas antes de la fórmula completa",
        "Una gaussiana multivariada describe datos con varias columnas a la vez. No sólo pregunta si cada columna varía mucho; "
        "también pregunta si dos columnas tienden a moverse juntas. La matriz de covarianza $\\Sigma$ guarda esas escalas y relaciones."
    )
    st.caption("Pregunta que responde la fórmula: ¿qué tan lejos está un punto del centro si las variables tienen escalas y correlaciones distintas?")
    st.latex(r"D^2(\mathbf x,\boldsymbol\mu)=(\mathbf x-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf x-\boldsymbol\mu)")
    st.caption("Esta es la distancia de Mahalanobis: mide distancia al centro considerando escala y correlaciones.")
    st.latex(r"f_{\mathbf X}(\mathbf x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}D^2(\mathbf x,\boldsymbol\mu)\right)")
    notation_box([
        (r"\mathbf x", "Vector observado: una fila con varias variables o features."),
        (r"\boldsymbol\mu", "Vector de medias: centro de la nube."),
        (r"\Sigma", "Matriz de covarianza: escalas en la diagonal y relaciones entre variables fuera de la diagonal."),
        (r"\Sigma^{-1}", "Inversa de la covarianza: ajusta la noción de distancia según la forma de la nube."),
        (r"|\Sigma|", "Determinante: factor ligado al volumen de la nube."),
        (r"d", "Número de dimensiones o columnas."),
    ], expanded=False)
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
    lab_task(
        predict="si la correlación se acerca a 1 o -1, la elipse debería inclinarse y alargarse.",
        manipulate="cambia las desviaciones marginales y la correlación.",
        verify="mira si las flechas rojas (ejes principales) coinciden con la orientación de la elipse.",
    )
    interactive_guide(
        controls=[
            ("dispersión marginal σ₁ y σ₂", "controlan la escala de las coordenadas X e Y antes de rotar la elipse por correlación."),
            ("correlación ρ", "controla la inclinación y la correlación lineal entre ambas variables."),
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
        s1 = st.slider("dispersión marginal σ₁ (coordenada X)", 0.3, 3.0, 1.0, key="pca_s1")
        s2 = st.slider("dispersión marginal σ₂ (coordenada Y)", 0.3, 3.0, 1.5, key="pca_s2")
        rho = st.slider("correlación ρ", -0.95, 0.95, 0.6, key="pca_rho")
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
        ax.set_title("Densidad: elipse de una gaussiana bivariada")
        fig.colorbar(cf, ax=ax, shrink=0.82, label="densidad")
        mia_pyplot(fig); plt.close(fig)
    how_to_read("Las flechas rojas son los **autovectores** de Σ (ejes de la elipse), con longitudes proporcionales a $\\sqrt{\\lambda_i}$.")
    st.caption(f"Σ={np.round(Sigma,2).tolist()}, autovalores={np.round(eigvals,2).tolist()}. σ₁ y σ₂ son dispersiones marginales; los ejes principales reales son las flechas rojas.")

    st.markdown("### PCA vía SVD")
    st.caption("Pipeline conceptual: centrar datos → covarianza/SVD → direcciones principales → varianza explicada.")
    st.markdown(
        "Dado un dataset $X \\in \\mathbb{R}^{N\\times d}$ **centrado**, computamos SVD:"
    )
    st.latex(r"X = U\,S\,V^T")
    st.caption("Aquí $X$ debe estar centrada: a cada columna se le resta su media. Las filas son datos y las columnas son variables.")
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
        stretch = st.slider("qué tan estirada está la nube", 1.0, 6.0, 3.0, key="pca_stretch")
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
        mia_pyplot(fig); plt.close(fig)
    how_to_read("PC1 apunta en la dirección de máxima varianza. Si los datos están muy estirados en una dirección, PC1 la recupera.")
    st.caption("PC1 es la dirección donde los datos varían más; PC2 es la segunda dirección, perpendicular a PC1.")

    interactive_header("Mini-casos PCA/SVD")
    st.markdown(
        "Estos casos muestran una idea central: columnas fuertemente dependientes producen pocos valores singulares dominantes. "
        "La escala de los autovalores cambia si divides por $N$ o por $N-1$, pero las direcciones principales no."
    )
    col1, col2 = lab_columns()
    with col1:
        pca_case = st.selectbox(
            "caso sintético",
            ["X=[x, x+1, x+2]", "X=[x, 2x+1, 3x+2, 4x]", "toy sinusoidal"],
            key="pca_case_notebook",
        )
    x_nb = np.linspace(-3, 3, 120)
    if pca_case.startswith("X=[x, x+1"):
        X_nb = np.column_stack([x_nb, x_nb + 1, x_nb + 2])
    elif pca_case.startswith("X=[x, 2x"):
        X_nb = np.column_stack([x_nb, 2 * x_nb + 1, 3 * x_nb + 2, 4 * x_nb])
    else:
        X_nb = np.column_stack([x_nb, np.sin(2 * x_nb), np.cos(x_nb)])
    Xc_nb = X_nb - X_nb.mean(axis=0, keepdims=True)
    _, S_nb, Vt_nb = np.linalg.svd(Xc_nb, full_matrices=False)
    evr_nb = S_nb**2 / np.sum(S_nb**2)
    proj_nb = Xc_nb @ Vt_nb[:2].T
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].bar(np.arange(1, len(evr_nb) + 1), evr_nb, color="#4C72B0")
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("componente")
        axes[0].set_ylabel("explained variance ratio")
        axes[1].scatter(proj_nb[:, 0], proj_nb[:, 1], s=16, alpha=0.7, color="#DD8452")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("proyección 2D")
        plt.tight_layout()
        mia_pyplot(fig); plt.close(fig)
    st.dataframe(
        pd.DataFrame({"PC": np.arange(1, len(evr_nb) + 1), "explained_variance_ratio": np.round(evr_nb, 5)}),
        hide_index=True,
        width="stretch",
    )

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
    plain_language(
        "Imagen mental antes de la fórmula",
        "En 2D puedes dibujar un círculo dentro de un cuadrado. En 3D, una esfera dentro de un cubo. "
        "La pregunta es: ¿qué fracción del cubo queda cerca del centro? En dimensiones altas, esa fracción cae muy rápido. "
        "Por eso muchos métodos basados en distancia pierden fuerza cuando hay demasiadas variables."
    )
    st.markdown("**Razón volumen esfera / volumen cubo**:")
    st.latex(r"\frac{V_d(r)}{(2r)^d} = \frac{\pi^{d/2}}{2^d\,\Gamma(d/2+1)}")
    notation_box([
        (r"d", "Número de dimensiones o variables que describen cada punto."),
        (r"r", "Radio: qué tan lejos del centro consideramos que un punto sigue estando cerca."),
        (r"V_d(r)", "Volumen de la esfera en d dimensiones."),
        (r"(2r)^d", "Volumen del cubo que contiene a esa esfera."),
        (r"\Gamma", "Función gamma: aparece al extender fórmulas de volumen a cualquier dimensión."),
    ], expanded=False)
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

    interactive_header("Curse of dimensionality: volume collapse and distance concentration")
    lab_task(
        predict="al crecer la dimensión, deberían caer menos puntos cerca del centro y las distancias nearest/farthest deberían distinguirse peor.",
        manipulate="cambia la dimensión, el tamaño muestral y el radio usado para definir un local neighborhood.",
        verify="compara la curva de volume ratio con el histograma de distancias y el nearest-neighbor contrast.",
    )
    real_world_case(
        "búsqueda por similitud en muchos atributos",
        "Imagina que cada punto es un usuario descrito por muchas variables: edad, gasto, frecuencia de compra, categorías vistas, etc. "
        "En pocas dimensiones, 'usuarios cercanos' es una idea clara. En muchas dimensiones, casi todos los usuarios pueden parecer igual de lejos.",
        controls=[
            ("dimensión", "cuántas variables describen a cada punto."),
            ("# puntos", "cuántos usuarios o registros simulamos."),
        ],
        takeaway="Si casi ningún punto cae cerca del centro, los métodos que dependen de distancia necesitan más cuidado, reducción dimensional o mejores representaciones.",
        expanded=False,
    )
    interactive_guide(
        controls=[
            ("dimensión d", "número de coordenadas o features usadas para describir cada punto."),
            ("tamaño muestral n", "número de puntos simulados."),
            ("radio local", "umbral de distancia usado para preguntar si un punto está cerca de la query."),
        ],
        procedure=(
            "The lab samples points uniformly in the unit hypercube $[0,1]^d$, measures distances to the center query, "
            "and compares that empirical distance profile with the theoretical sphere/cube volume ratio."
        ),
        observe=(
            "The important ML signal is not only that central volume disappears; distances also compress into a narrow band. "
            "When nearest and farthest points look similarly far away, distance-based methods need better representations."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        d_sim = st.slider("dimensión d", 1, 100, 10, key="curse_d")
        n_sim = st.slider("puntos simulados n", 1000, 30000, 8000, step=1000, key="curse_n")
        local_radius = st.slider("radio local", 0.05, 1.50, 0.50, step=0.05, key="curse_radius")
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, size=(n_sim, d_sim))
    center = np.full(d_sim, 0.5)
    distances = np.linalg.norm(pts - center, axis=1)
    inside_local = np.mean(distances <= local_radius)
    nearest = float(np.min(distances))
    median_dist = float(np.median(distances))
    farthest = float(np.max(distances))
    contrast = (farthest - nearest) / nearest if nearest > 0 else np.inf
    with col2:
        metric_grid([
            ("dentro del radio local", f"{inside_local:.4f}"),
            ("nearest distance", f"{nearest:.3f}"),
            ("median distance", f"{median_dist:.3f}"),
            ("relative contrast", f"{contrast:.3f}"),
        ], columns=4)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
        ds = np.arange(1, 51)
        rr = [np.exp((d/2)*np.log(np.pi) - gammaln(d/2+1)) / (2.0**d) for d in ds]
        axes[0].plot(ds, rr, "o-", color="#4C72B0", label="sphere/cube ratio")
        if d_sim <= 50:
            log_vs = (d_sim/2)*np.log(np.pi) - gammaln(d_sim/2+1)
            ratio_d = np.exp(log_vs) / (2.0**d_sim)
            axes[0].scatter([d_sim], [ratio_d], color="#DD8452", s=80, zorder=5, label=f"d={d_sim}: {ratio_d:.2e}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("dimensión d")
        axes[0].set_ylabel("central volume ratio (log)")
        axes[0].legend(fontsize=8)
        axes[1].hist(distances, bins=40, color="#4C72B0", alpha=0.72)
        axes[1].axvline(nearest, color="#059669", ls="--", label="nearest")
        axes[1].axvline(median_dist, color="#111827", ls=":", label="median")
        axes[1].axvline(local_radius, color="#DD8452", ls="-.", label="radio local")
        axes[1].set_xlabel("distancia a la query central")
        axes[1].set_ylabel("conteo")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        mia_pyplot(fig); plt.close(fig)
    how_to_read(
        "Izquierda: la curva en escala log muestra qué tan rápido desaparece el volumen central. Derecha: el histograma muestra distancias "
        "desde los puntos simulados hasta la query central. Si el histograma se estrecha y queda lejos de cero, separar 'near' y 'far' se vuelve más difícil."
    )
    lab_note("relative contrast = (farthest - nearest) / nearest. Mientras más bajo, menos señal discriminativa queda en la nearest-neighbor distance.")
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
    beginner_bridge(
        "qué es una cota",
        [
            "Una cota no dice la probabilidad exacta; dice un máximo garantizado.",
            "Más hipótesis permiten cotas más fuertes: no negatividad, varianza conocida o datos acotados.",
            "En aprendizaje automático esto sirve para comparar error observado con error esperado en datos nuevos.",
        ],
    )
    motivation(
        "El problema fundamental del Machine Learning: entrenamos modelos con una muestra finita de $n$ datos "
        "y calculamos el **error empírico** ($\\bar X_n$). Pero lo que importa es el **error real** ($\\mu$) sobre "
        "datos futuros invisibles. ¿Cómo garantizamos matemáticamente que $\\bar X_n \\approx \\mu$? "
        "**Markov, Chebyshev y Hoeffding** dan cotas superiores bajo supuestos explícitos: no negatividad, varianza finita "
        "o variables acotadas e i.i.d. Son la base de la teoría de generalización en ML."
    )
    prerequisites_box(
        "- $E[X]$, $\\text{Var}(X)$.\n"
        "- VAs acotadas (para Hoeffding)."
    )
    st.markdown("### Construcción")
    notation_box([
        (r"\epsilon", "Tolerancia: cuánto error máximo aceptamos entre el promedio observado y la media real."),
        (r"\delta", "Riesgo permitido: probabilidad máxima de que la tolerancia falle."),
        (r"E[X]", "Promedio esperado."),
        (r"\text{Var}(X)", "Dispersión de X alrededor de su media."),
        ("i.i.d.", "Independientes e idénticamente distribuidas: muestras tomadas del mismo proceso sin influirse entre sí."),
    ], expanded=True)
    st.markdown("**Markov** (para $X\\ge 0$):")
    st.latex(r"P(X \geq a) \leq \frac{E[X]}{a}")
    with advanced_expander("Demostración de la desigualdad de Markov"):
        st.markdown(
            "Para $X$ continua con $X \\geq 0$, particionamos la integral de $E[X]$ en dos tramos:"
        )
        latex_aligned([
            r"E[X] = \int_0^\infty x\,f(x)\,dx = \int_0^a x\,f(x)\,dx + \int_a^\infty x\,f(x)\,dx",
            r"\geq \int_a^\infty x\,f(x)\,dx \geq a\int_a^\infty f(x)\,dx = a\,P(X \geq a)",
        ])
        st.markdown(
            "**¿Por qué $x\\,f(x) \\geq a\\,f(x)$ en $[a,\\infty)$?** "
            "En ese tramo, cada $x$ satisface $x \\geq a$, entonces $x \\cdot f(x) \\geq a \\cdot f(x)$ "
            "(como $f(x)\\geq 0$, multiplicar por algo más grande no cambia el signo). "
            "El primer paso (descartar $[0,a]$) hace la cota laxa: hay mucho margen."
        )
        st.markdown(
            "Dividiendo por $a > 0$ obtenemos $P(X \\geq a) \\leq E[X]/a$."
        )
        st.markdown("**Chebyshev como caso especial:** Aplicar Markov a $Y = (X-\\mu)^2 \\geq 0$ con umbral $a = \\epsilon^2$:")
        latex_aligned([
            r"P((X-\mu)^2 \geq \epsilon^2) \leq \frac{E[(X-\mu)^2]}{\epsilon^2} = \frac{\sigma^2}{\epsilon^2}",
            r"\Longrightarrow P(|X-\mu| \geq \epsilon) \leq \frac{\sigma^2}{\epsilon^2}",
        ])
    st.markdown("**Chebyshev** (a partir de Markov aplicado a $(X-\\mu)^2$):")
    st.latex(r"P(|X-\mu| \geq k\sigma) \leq \frac{1}{k^2}")
    st.markdown("**Hoeffding** (si cada $X_i \\in [a,b]$, i.i.d.):")
    st.latex(r"P\!\left(\big|\bar X_n - \mu\big| \geq t\right) \leq 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)")
    compact_dataframe(pd.DataFrame([
        {"Cota": "Markov", "Supuesto mínimo": "X >= 0 y E[X] finita", "Úsala cuando": "sólo conoces el promedio"},
        {"Cota": "Chebyshev", "Supuesto mínimo": "Var(X) finita", "Úsala cuando": "quieres desviaciones respecto de la media"},
        {"Cota": "Hoeffding", "Supuesto mínimo": "X_i acotadas e i.i.d.", "Úsala cuando": "promedias variables en un intervalo fijo"},
    ]))
    st.markdown(
        "A diferencia de Markov o Chebyshev (que decaen **polinomialmente** en $n$), Hoeffding garantiza que la "
        "probabilidad de error decrece **exponencialmente** con $n$. Esta es la base matemática de la teoría **PAC** "
        "(Probably Approximately Correct) en Machine Learning."
    )
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
    formula_walkthrough(
        "Despeje paso a paso de Hoeffding",
        steps=[
            "Partimos de $2\\exp(-2n\\epsilon^2)\\le \\delta$ porque queremos que la probabilidad de fallar sea como máximo $\\delta$.",
            "Dividimos por 2: $\\exp(-2n\\epsilon^2)\\le \\delta/2$.",
            "Aplicamos logaritmo natural: $-2n\\epsilon^2\\le \\log(\\delta/2)$.",
            "Multiplicamos por $-1$ y cambia el signo: $2n\\epsilon^2\\ge \\log(2/\\delta)$.",
            "Despejamos: $n\\ge \\log(2/\\delta)/(2\\epsilon^2)$.",
        ],
        expanded=False,
    )

    worked_example("Markov y Chebyshev en ejercicios clásicos")
    st.markdown("**Caras en $n$ lanzamientos.** Si $X$ cuenta caras en $n$ monedas justas, $E[X]=n/2$ y $Var(X)=n/4$.")
    st.latex(r"P(X\ge 3n/4)\le \frac{E[X]}{3n/4}=\frac{2}{3}\quad\text{(Markov)}")
    st.latex(r"P(X\ge 3n/4)\le P(|X-n/2|\ge n/4)\le \frac{n/4}{(n/4)^2}=\frac{4}{n}\quad\text{(Chebyshev)}")
    st.markdown("**Coleccionador de cupones.** Si $X$ es el tiempo hasta juntar $n$ cupones, $E[X]=nH_n$.")
    st.latex(r"P(X\ge 2nH_n)\le \frac{E[X]}{2nH_n}=\frac{1}{2}")
    st.markdown("**Suma de dos dados repetida $n$ veces.** Cada lanzamiento doble tiene media $7$ y varianza $35/6$.")
    st.latex(r"P\left(|S_n-7n|\ge 10\sqrt{35n/6}\right)\le \frac{(35/6)n}{100(35n/6)}=0.01")

    interactive_header("Comparador de cotas")
    lab_task(
        predict="si bajas el riesgo δ o el error tolerado ε, deberían subir las muestras necesarias.",
        manipulate="mueve ε y δ.",
        verify="compara cuántas muestras pide Hoeffding versus Chebyshev.",
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        eps = st.slider("error máximo tolerado ε (más bajo = más muestras)", 0.01, 0.5, 0.1, step=0.01, key="conc_eps")
        delta = st.slider("riesgo permitido δ (más bajo = más muestras)", 0.001, 0.5, 0.05, step=0.001, key="conc_delta")
        n_h = np.ceil(np.log(2/delta) / (2*eps**2))
        var_unit = 0.25
        n_c = np.ceil(var_unit / (eps**2 * delta))
        metric_grid([
            ("n (Hoeffding)", f"{int(n_h)}"),
            ("n (Chebyshev, σ²≤0.25)", f"{int(n_c)}"),
        ], columns=2)
    with col2:
        ns = np.arange(10, 2000)
        p_hoef = np.minimum(1.0, 2*np.exp(-2*ns*eps**2))
        p_cheb = np.minimum(1.0, var_unit/(ns*eps**2))
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.plot(ns, p_hoef, label="Hoeffding", color="#4C72B0", lw=2)
        ax.plot(ns, p_cheb, label="Chebyshev", color="#DD8452", lw=2)
        ax.axhline(delta, ls=":", color="gray", label=f"δ={delta}")
        ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("garantía superior: P(error ≥ ε)")
        ax.legend()
        mia_pyplot(fig); plt.close(fig)
    how_to_read("Hoeffding (azul) decae exponencialmente, Chebyshev (naranja) sólo polinomialmente. Misma ε, Hoeffding necesita muchas menos muestras.")
    lab_note("estas curvas son garantías pesimistas: valores más bajos no son más 'verdaderos', sólo cotas superiores más informativas.")

    interactive_header("Cotas teóricas versus comportamiento empírico")
    lab_task(
        predict="la barra empírica debería quedar debajo de las cotas, pero no necesariamente cerca de ellas.",
        manipulate="elige distribución, n, ε y repeticiones.",
        verify="revisa si las cotas actúan como techo y no como predicción exacta.",
    )
    interactive_guide(
        controls=[
            ("Distribución generadora", "elige la distribución real desde la que se simulan los datos."),
            ("tamaño de cada muestra n", "tamaño de cada muestra usada para formar el promedio."),
            ("error que quieres vigilar ε", "umbral de desviación respecto de la media."),
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
        source = st.radio("Distribución generadora", ["Bernoulli(0.3)", "Uniforme(0,1)"], key="conc_src")
        n_emp = st.slider("tamaño de cada muestra n", 5, 500, 60, key="conc_nemp")
        eps_emp = st.slider("error que quieres vigilar ε", 0.01, 0.4, 0.10, step=0.01, key="conc_epsemp")
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
        mia_pyplot(fig)
        plt.close(fig)
    metric_grid([
        ("Probabilidad empírica", f"{empirical_tail:.4f}"),
        ("Cota de Chebyshev", f"{cheb_tail:.4f}"),
        ("Cota de Hoeffding", f"{hoef_tail:.4f}"),
    ], columns=3)
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
    beginner_bridge(
        "qué se estima con una muestra",
        [
            "La población completa suele ser inaccesible; una muestra es el subconjunto que sí observamos.",
            "La media muestral estima el promedio real de la población.",
            "En pooled testing se mezclan muestras biológicas para ahorrar tests cuando los positivos son raros.",
        ],
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
    st.caption("Aquí 'muestra' significa datos observados de una población. Más abajo, en testeo agrupado, 'muestra' significa muestra biológica; son usos distintos de la palabra.")
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

    worked_example("testeo agrupado (pooled testing) — conscriptos con enfermedad rara")
    st.markdown(
        "**Problema:** 1000 conscriptos deben ser testeados para una enfermedad con prevalencia $p = 0.002$ (0.2%). "
        "Hacer 1000 tests individuales es costoso. "
        "**Estrategia:** dividir en 10 grupos de 100 personas, mezclar la sangre y testear la mezcla."
    )
    st.markdown(
        "- Si la mezcla es **negativa** → las 100 personas están sanas. Usamos **1 test** para ese grupo.\n"
        "- Si la mezcla es **positiva** → hacemos 100 tests individuales adicionales. Usamos **1+100 = 101 tests**.\n\n"
        "**¿Cuántos tests esperamos realizar?** Sea $Z_i = 1$ si el grupo $i$ da positivo, 0 si da negativo."
    )
    compact_dataframe(pd.DataFrame([
        {"Escenario en 10 grupos de 100": "Todos los grupos negativos", "Tests totales": 10, "Lectura": "mínimo posible"},
        {"Escenario en 10 grupos de 100": "Todos los grupos positivos", "Tests totales": 1010, "Lectura": "máximo posible"},
        {"Escenario en 10 grupos de 100": "Valor esperado con p=0.002", "Tests totales": "se calcula abajo", "Lectura": "promedio teórico"},
    ]))
    p_conscript = 0.002
    k_conscript = 100
    n_groups = 10
    p_group_pos = 1 - (1 - p_conscript)**k_conscript
    ez_conscript = n_groups + k_conscript * n_groups * p_group_pos
    st.latex(
        rf"E[Z_i] = P(\text{{grupo positivo}}) = 1-(1-0.002)^{{100}} \approx {p_group_pos:.3f}"
    )
    st.latex(
        rf"E[Z] = 10 + 100 \cdot \sum_{{i=1}}^{{10}} E[Z_i] = 10 + 100 \cdot 10 \cdot {p_group_pos:.3f} \approx {ez_conscript:.0f}"
    )
    st.success(
        f"Pasamos de **1000 tests → ~{ez_conscript:.0f} tests**: una reducción del {100*(1-ez_conscript/1000):.0f}% "
        f"usando solo probabilidad y agrupamiento."
    )
    st.markdown("---")
    st.markdown("**Fórmula general** (N personas, grupos de tamaño k, prevalencia p):")
    st.latex(r"E[Z] = \frac{N}{k}\Big[1 + k\cdot (1 - (1-p)^k)\Big]")
    st.caption("Esta forma cerrada supone que $N$ es divisible por $k$. En el laboratorio se maneja también el grupo residual si sobra gente.")
    formula_walkthrough(
        "Qué significa cada parte del costo esperado",
        terms={
            r"N/k": "Número de grupos si todos tienen tamaño k.",
            r"1": "El test inicial de la mezcla del grupo.",
            r"1-(1-p)^k": "Probabilidad de que al menos una persona del grupo sea positiva.",
            r"k(1-(1-p)^k)": "Tests individuales esperados después de una mezcla positiva.",
        },
        steps=[
            "Primero siempre haces un test por grupo: 1 test fijo, independiente del resultado.",
            "Sólo si el grupo sale positivo haces los k tests individuales adicionales.",
            "Si la prevalencia es baja, muchos grupos salen negativos y ahorras muchos tests.",
            "Si k es demasiado grande, casi todos los grupos salen positivos y pierdes el ahorro.",
        ],
        expanded=False,
    )
    pitfall("El modelo supone independencia, prevalencia homogénea y test perfecto. En aplicaciones reales habría que considerar sensibilidad, especificidad y logística de laboratorio.")

    interactive_header("Pooled testing: óptimo en k")
    lab_task(
        predict="para prevalencia baja debería existir un tamaño de grupo k que ahorre muchos tests.",
        manipulate="cambia p y N.",
        verify="busca el fondo de la curva en U y compáralo con el costo sin pooling.",
    )
    st.info("Este modelo asume test perfecto e independencia entre personas. En aplicaciones reales también importan sensibilidad, especificidad, logística y tiempo de procesamiento.")
    interactive_guide(
        controls=[
            ("porcentaje esperado de positivos p", "fracción esperada de individuos positivos en la población."),
            ("personas a testear N", "cantidad total de muestras que deseas testear."),
        ],
        procedure=(
            "Para cada tamaño de grupo $k$, el laboratorio calcula el número esperado total de tests: un test inicial por grupo "
            "más los tests individuales adicionales cuando un grupo sale positivo."
        ),
        observe=(
            "Si $k$ es muy pequeño, casi no ahorras tests. Si $k$ es muy grande, muchos grupos salen positivos y obligan a hacer demasiados tests de confirmación. "
            "Por eso aparece una curva en forma de U."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        p_slider = st.slider("porcentaje esperado de positivos p", 0.001, 0.2, 0.01, step=0.001, format="%.3f", key="pool_p")
        N_slider = st.slider("personas a testear N", 100, 10000, 1000, step=100, key="pool_N")
    ks = np.arange(1, 51)
    def _expected_pooled_tests(N, k, p):
        full_groups, remainder = divmod(int(N), int(k))
        expected_full = full_groups * (1 + k * (1 - (1 - p) ** k))
        expected_remainder = 0 if remainder == 0 else 1 + remainder * (1 - (1 - p) ** remainder)
        return expected_full + expected_remainder
    EZ = np.array([_expected_pooled_tests(N_slider, k, p_slider) for k in ks])
    k_opt = ks[np.argmin(EZ)]
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(ks, EZ, color="#4C72B0", lw=2)
        ax.axhline(N_slider, color="gray", ls="--", label=f"Sin pooling = {N_slider}")
        ax.axvline(k_opt, color="#DD8452", ls=":", label=f"k óptimo = {k_opt}")
        ax.scatter([k_opt], [EZ.min()], s=80, color="#DD8452", zorder=5)
        ax.set_xlabel("personas por grupo k"); ax.set_ylabel("número esperado de tests")
        ax.legend()
        mia_pyplot(fig); plt.close(fig)
        ahorro = 100 * (1 - EZ.min() / N_slider)
        metric_grid([
            ("k óptimo", f"{k_opt}"),
            ("E[# tests] óptimo", f"{EZ.min():.1f}"),
            ("ahorro esperado", f"{ahorro:.1f}%"),
        ], columns=3)
    how_to_read("Curva U: k=1 es test individual (costo N); k grande da muchos reruns. El fondo es el óptimo.")
    lab_note("el óptimo minimiza tests esperados; no necesariamente minimiza tiempo, logística, riesgo operativo o costo de seguimiento.")

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
    beginner_bridge(
        "promedios y campanas",
        [
            "LLN dice que el promedio de muchas observaciones se estabiliza cerca de la media real.",
            "CLT dice cómo se distribuye el error de ese promedio cuando n es grande.",
            "Ambos requieren supuestos; aquí el caso base es muestras i.i.d. con varianza finita para CLT.",
        ],
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
    plain_language(
        "Dos ideas distintas",
        "La Ley de los Grandes Números dice que el promedio se estabiliza cuando juntamos muchos datos. "
        "El Teorema Central del Límite dice algo diferente: describe la forma del error del promedio mientras se estabiliza."
    )
    st.markdown("**Débil (LLN débil)** — convergencia en probabilidad:")
    st.latex(r"\forall \epsilon>0: \lim_{n\to\infty} P(|\bar X_n - \mu|>\epsilon) = 0")
    st.markdown("**Fuerte (LLN fuerte)** — convergencia casi segura:")
    st.latex(r"P\!\left(\lim_{n\to\infty} \bar X_n = \mu\right) = 1")
    st.caption(
        "Diferencia intuitiva: la débil dice «es improbable estar lejos»; la fuerte dice «la trayectoria "
        "completa converge». La fuerte implica la débil, no al revés."
    )
    with st.expander("La diferencia en imágenes: Foto vs Película", expanded=True):
        col_w, col_s = st.columns(2)
        with col_w:
            st.markdown("### Ley Débil — La «Foto»")
            st.markdown(
                "**Convergencia en probabilidad.** Si tomas una foto del error en un instante $n$ muy grande, "
                "es casi seguro que estará cerca de $\\mu$. "
                "Sin embargo, *permite* que si sigues observando hasta el infinito, "
                "la trayectoria ocasionalmente dé «saltos» y se aleje de $\\mu$. "
                "La foto está bien; la película podría tener sorpresas."
            )
            st.info("**Uso teórico:** Sirve para demostrar que un estimador estadístico es *consistente*.")
        with col_s:
            st.markdown("### Ley Fuerte — La «Película»")
            st.markdown(
                "**Convergencia casi segura.** Con probabilidad 1, para todo margen $\\epsilon>0$ existe un momento "
                "$N(\\omega)$ desde el cual $\\bar X_n$ queda dentro de la banda $\\mu\\pm\\epsilon$. "
                "Puede fluctuar antes de ese punto; lo fuerte es el control eventual de la trayectoria completa."
            )
            st.success("**Uso en IA:** Justifica estimadores empíricos bajo supuestos i.i.d.; no garantiza por sí sola estabilidad de entrenamiento.")
    insight(
        "La Ley Fuerte es la pieza que faltaba: la Desigualdad de Kolmogorov (abajo) acota el *máximo* de las "
        "sumas parciales, lo que permite pasar de la convergencia en un punto (Débil) a la convergencia de toda la trayectoria (Fuerte)."
    )
    st.markdown("**Desigualdad de Kolmogorov** (máxima de sumas parciales):")
    st.latex(r"P\!\left(\max_{1\leq k\leq n} |S_k - k\mu| \geq \epsilon\right) \leq \frac{\text{Var}(S_n)}{\epsilon^2}")
    st.caption(
        "A diferencia de Chebyshev —que acota el *último punto* de la trayectoria— Kolmogorov acota la probabilidad "
        "de que *cualquier punto* de la trayectoria (del paso 1 al $n$) haya superado el umbral."
    )

    st.markdown("### Teorema Central del Límite (CLT)")
    st.latex(r"\frac{\bar X_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal N(0,1)")
    st.caption("$\\xrightarrow{d}$ significa convergencia en distribución: la forma de la variable de la izquierda se parece cada vez más a una normal estándar.")
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
            "CLT no exige normalidad de la distribución original, pero sí independencia (o condiciones similares) y varianza finita."
        ],
    )

    worked_example("El encuestador: ¿cuántos votos necesito encuestar?")
    st.markdown(
        "Queremos estimar la fracción $p$ de votos en un referéndum con error $\\epsilon=0.01$ y 95% de confianza. "
        "En el peor caso (empate perfecto) $\\sigma^2 = p(1-p) \\leq 0.25$."
    )
    col_enc_a, col_enc_b, col_enc_c = st.columns(3)
    with col_enc_a:
        st.markdown("**Vía Chebyshev** *(garantía exacta)*")
        st.latex(r"\frac{\sigma^2}{n\epsilon^2} \leq 0.05 \Rightarrow n \geq \frac{0.25}{(0.01)^2 \cdot 0.05} = 50{,}000")
        st.caption("Garantía matemática absoluta. Prepara para el peor escenario.")
    with col_enc_b:
        st.markdown("**Vía Hoeffding** *(garantía + acotado)*")
        n_hoef = int(np.ceil(np.log(2/0.05) / (2 * 0.01**2)))
        st.latex(rf"2e^{{-2n(0.01)^2}} \leq 0.05 \Rightarrow n \geq {n_hoef:,}")
        st.caption("Más eficiente que Chebyshev por el decaimiento exponencial.")
    with col_enc_c:
        st.markdown("**Vía CLT** *(aproximación)*")
        n_clt = int(np.ceil((1.96 * 0.5 / 0.01)**2))
        st.latex(rf"z_{{0.975}} = 1.96 \Rightarrow n \geq \left(\frac{{1.96 \cdot 0.5}}{{0.01}}\right)^2 = {n_clt:,}")
        st.caption("Asume convergencia a Gaussiana. Muy eficiente, pero es una aproximación.")
    st.info(
        "Reducir la confianza del 99% al 95% puede reducir sustancialmente el tamaño muestral requerido, según el método usado. "
        "CLT da la muestra más pequeña pero es una aproximación; Chebyshev da garantía dura pero pide más datos."
    )

    worked_example("Chebyshev vs CLT para dimensionar muestras (caso general)")
    st.info("Chebyshev entrega una garantía conservadora. CLT entrega una aproximación muy útil cuando sus condiciones aplican, pero no es una garantía exacta para todo n.")
    st.markdown(
        "$X_i\\in[0,1]$, queremos $P(|\\bar X_n - \\mu|>0.05) \\leq 0.05$.\n\n"
        "**Chebyshev** con la cota genérica $\\sigma^2\\le 0.25$: "
        "$\\sigma^2/(n\\epsilon^2) \\leq 0.05 \\Rightarrow n \\geq 0.25/(0.0025\\cdot 0.05) = 2000$.\n\n"
        "**CLT** (aproximación): si tomamos el peor caso $\\sigma=0.5$, entonces para un intervalo bilateral del 95% necesitamos "
        "$1.96\\,\\sigma/\\sqrt n \\le 0.05$. Eso da $\\sqrt n \\ge 1.96\\cdot 0.5/0.05 = 19.6$ y por tanto "
        "$n \\ge 19.6^2 \\approx 384.2$.\n\n"
        "→ CLT sigue siendo bastante más eficiente que Chebyshev cuando aplica, pero no por un factor tan extremo."
    )

    interactive_header("LLN y CLT en acción")
    lab_task(
        predict="al aumentar n, las trayectorias del promedio deberían acercarse a la media y el histograma estandarizado a N(0,1).",
        manipulate="cambia distribución generadora, n y número de repeticiones.",
        verify="mira por separado estabilidad del promedio (LLN) y forma del error estandarizado (CLT).",
    )
    limit_view = st.radio(
        "Vista",
        ["LLN trajectories (trayectorias)", "CLT histogram (histograma de promedios)"],
        horizontal=True,
        key="limits_view",
    )
    rng = np.random.default_rng(7)
    if limit_view.startswith("LLN"):
        col1, col2 = st.columns([1, 2])
        with col1:
            dist_name = st.radio("Distribución generadora", ["Bernoulli(0.3)", "Exponencial(1)", "Uniforme(0,1)"], key="lln_dist")
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
            mia_pyplot(fig); plt.close(fig)
        how_to_read("Cada línea es una trayectoria distinta. Todas convergen al valor rojo μ conforme n crece. La dispersión baja como $1/\\sqrt n$.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            dist2 = st.radio("Distribución generadora", ["Exponencial(1)", "Uniforme(0,1)", "Bernoulli(0.5)"], key="clt_dist")
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
            mia_pyplot(fig); plt.close(fig)
        how_to_read("Aunque la distribución original sea sesgada (Exponencial) o discreta (Bernoulli), el histograma del promedio estandarizado se parece a la campana estándar cuando n crece.")

    interactive_header("Cobertura de intervalos construidos con CLT")
    lab_task(
        predict="con n más grande, la empirical coverage debería acercarse a 95%.",
        manipulate="cambia la distribución generadora, el tamaño muestral y la cantidad de intervalos simulados.",
        verify="cuenta cuántos intervalos cubren la media verdadera y compara la fracción con 0.95.",
    )
    interactive_guide(
        controls=[
            ("Distribución para intervalos", "elige la distribución original de donde salen las muestras."),
            ("n por muestra", "tamaño de cada muestra usada para construir un intervalo."),
            ("Número de intervalos simulados", "cuántos intervalos independientes quieres generar."),
        ],
        procedure=(
            "Para cada muestra, el laboratorio construye un intervalo del 95% usando la aproximación del CLT y verifica si contiene o no la media verdadera."
        ),
        observe=(
            "Cada línea representa un intervalo distinto. Si la aproximación es buena, la fracción de intervalos que contienen la media verdadera debería acercarse a 0.95."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        source_ci = st.radio("Distribución para intervalos", ["Exponencial(1)", "Uniforme(0,1)", "Bernoulli(0.5)"], key="clt_ci_src")
        n_ci = st.slider("tamaño de cada muestra n", 5, 400, 40, key="clt_ci_n")
        n_intervals = st.slider("cantidad de intervalos simulados", 50, 1000, 200, step=50, key="clt_ci_rep")
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
        mia_pyplot(fig)
        plt.close(fig)
    st.metric("Fracción de intervalos que atraparon la media", f"{covered.mean():.3f}")
    st.caption(
        "Las líneas azules contienen a la media verdadera y las naranjas no. Con $n$ suficientemente grande, "
        "la cobertura debería acercarse al 95%."
    )
    st.caption("Aquí usamos σ poblacional conocida para aislar el efecto del CLT. En datos reales σ suele estimarse con S; para n pequeño aparece el intervalo t.")

    self_check_header()
    quiz(
        "CLT requiere que la distribución original sea...",
        ["Normal", "Simétrica", "Tener varianza finita y ser i.i.d."],
        2,
        "Esa es la hipótesis clave; no hace falta normalidad de los $X_i$.",
        "Justamente lo milagroso del CLT: no pide normalidad de la distribución original.",
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
        "**Minibatch SGD** suele modelarse con una aproximación tipo CLT: bajo independencia aproximada y varianza finita, "
        "el gradiente promedio de un batch fluctúa alrededor del gradiente poblacional con escala cercana a $1/\\sqrt{|B|}$. "
        "Por eso batches más grandes suelen dar updates más estables. **Bootstrap** usa LLN para "
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
    beginner_bridge(
        "por qué un algoritmo usa azar",
        [
            "Un pivote es el elemento que se usa para partir una lista en menores y mayores.",
            "Aleatorizar el pivote evita que una entrada especialmente mala fuerce siempre el peor caso.",
            "La comparación relevante es el trabajo típico esperado, no sólo una corrida aislada.",
        ],
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
    worked_example("array pequeño: cuándo se comparan dos elementos")
    st.markdown(
        "En el arreglo ordenado `[1,2,3,4,5]`, mira los elementos 2 y 5. Se comparan sólo si el primer pivote elegido entre `{2,3,4,5}` "
        "es 2 o 5. Si sale 3 o 4 primero, ese pivote los separa en subproblemas distintos y nunca se comparan directamente."
    )
    st.latex(r"P(\text{comparar 2 y 5})=\frac{2}{4}")
    st.caption("$O(n\\log n)$ se lee como: el costo crece parecido a n multiplicado por el número de veces que puedes partir el problema.")
    with st.expander("Paso a paso mínimo: pivote y partición", expanded=True):
        compact_dataframe(pd.DataFrame([
            {"Lista activa": "[5, 2, 4, 1, 3]", "Pivote": "4", "Menores": "[2, 1, 3]", "Mayores": "[5]", "Qué pasó": "se comparó 4 con los otros 4 elementos"},
            {"Lista activa": "[2, 1, 3]", "Pivote": "2", "Menores": "[1]", "Mayores": "[3]", "Qué pasó": "se comparó 2 con 1 y 3"},
            {"Lista activa": "[1], [3], [5]", "Pivote": "-", "Menores": "-", "Mayores": "-", "Qué pasó": "listas de tamaño 1 ya están ordenadas"},
        ]))
        st.caption("Quicksort repite esta partición recursivamente. El azar está en elegir el pivote.")
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

    worked_example("algoritmo de mediana aleatorizada")
    st.markdown(
        "El algoritmo de mediana aleatorizada toma una muestra aleatoria de tamaño $\\lceil n^{3/4}\\rceil$, "
        "ordena esa muestra, elige dos pivotes alrededor de su centro y sólo ordena el conjunto candidato "
        "$C=\\{x: d\\le x\\le u\\}$. Si los pivotes dejan fuera a la mediana o $C$ sale demasiado grande, declara fallo."
    )
    st.caption("En este laboratorio la muestra aleatoria se toma sin reemplazo: un elemento no puede aparecer dos veces dentro de la misma muestra auxiliar.")
    st.latex(r"P(\text{fallo}) \le n^{-1/4}")

    def _randomized_median_trial(a, rng):
        n = len(a)
        r_size = int(np.ceil(n ** 0.75))
        sample = np.sort(rng.choice(a, size=min(r_size, n), replace=False))
        spread = int(np.ceil(np.sqrt(n)))
        center = r_size // 2
        d = sample[max(0, center - spread)]
        u = sample[min(r_size - 1, center + spread)]
        if d > u:
            d, u = u, d
        C = a[(a >= d) & (a <= u)]
        ld = int(np.sum(a < d))
        lu = int(np.sum(a > u))
        target = (n - 1) // 2
        fail = ld > target or lu > n - target - 1 or len(C) > 4 * r_size or not (ld <= target < ld + len(C))
        if fail:
            return None, len(C)
        return int(np.sort(C)[target - ld]), len(C)

    interactive_header("Mediana aleatorizada: tasa empírica de fallo")
    lab_task(
        predict="al crecer n, la cota n^(-1/4) baja lentamente.",
        manipulate="cambia n y corridas simuladas.",
        verify="compara fallo empírico, tamaño del candidato C y la cota teórica.",
    )
    col1, col2 = lab_columns()
    with col1:
        n_med = st.slider("cantidad de elementos n", 101, 5001, 1001, step=100, key="med_n")
        if n_med % 2 == 0:
            n_med += 1
        med_runs = st.slider("corridas simuladas", 20, 400, 120, step=20, key="med_runs")
    rng_med = np.random.default_rng(404)
    base_med = rng_med.permutation(n_med)
    true_med = int(np.median(base_med))
    failures = 0
    c_sizes = []
    for _ in range(med_runs):
        pred, c_len = _randomized_median_trial(base_med, np.random.default_rng(int(rng_med.integers(1_000_000))))
        failures += int(pred != true_med)
        c_sizes.append(c_len)
    fail_rate = failures / med_runs
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(c_sizes, bins=20, color="#4C72B0", alpha=0.75)
        ax.axvline(4 * np.ceil(n_med ** 0.75), color="#DD8452", ls="--", label=r"$4n^{3/4}$")
        ax.set_xlabel("tamaño del conjunto candidato |C|")
        ax.set_ylabel("frecuencia")
        ax.legend()
        mia_pyplot(fig); plt.close(fig)
    metric_grid([
        ("fallo empírico", f"{fail_rate:.3f}"),
        ("cota n^(-1/4)", f"{n_med**(-0.25):.3f}"),
    ], columns=2)

    interactive_header("Benchmark: Quicksort determinista vs aleatorizado")
    lab_task(
        predict="con entrada ordenada, el pivote fijo debería empeorar mucho frente al pivote aleatorio.",
        manipulate="cambia tamaño de lista y tipo de entrada.",
        verify="compara las barras con las referencias n log n y n^2.",
    )
    interactive_guide(
        controls=[
            ("tamaño del array", "longitud del arreglo a ordenar."),
            ("Entrada", "elige si el arreglo inicial será aleatorio o ya ordenado."),
        ],
        procedure=(
            "El laboratorio ejecuta una versión determinista de Quicksort y otra con pivote aleatorio sobre la misma entrada, y cuenta comparaciones."
        ),
        observe=(
            "La comparación importante no es sólo quién gana en un caso puntual, sino cómo cambia el costo cuando la entrada es adversarial para el pivote fijo."
        ),
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        n_bench = st.slider("tamaño de la lista a ordenar", 100, 3000, 1000, step=100, key="qs_n")
        adv = st.radio("Entrada", ["Aleatoria", "Ya ordenada: peor caso para pivote fijo"], key="qs_adv")

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
            if idx == 0:
                rest = cur[1:]
            elif idx == len(cur) - 1:
                rest = cur[:-1]
            else:
                rest = np.concatenate((cur[:idx], cur[idx + 1:]))
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
        ax.set_ylabel("comparaciones (trabajo realizado)")
        for i, v in enumerate([c_det, c_rand]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        ax.set_title("Comparaciones para ordenar la lista")
        mia_pyplot(fig)
        st.caption(f"Referencia: n·log₂(n) ≈ {n_bench*math.log2(n_bench):.0f}; n² = {n_bench**2:,}. n log n crece de forma manejable; n² crece mucho más rápido.")
    how_to_read("Con entrada ordenada, determinista explota a $O(n^2)$. Aleatorizado se mantiene cerca de $n\\log n$ en esperanza.")

    interactive_header("Variabilidad del costo de Quicksort aleatorizado")
    lab_task(
        predict="los random pivots deberían mantenerse cerca de n log n incluso cuando el orden de entrada es adversarial para un pivote fijo.",
        manipulate="cambia tamaño de lista, número de corridas y tipo de entrada.",
        verify="lee la dispersión del histograma y compara el costo típico con la referencia n log n.",
    )
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
        mia_pyplot(fig)
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
# SECCIÓN 17 — DESCENSO DE GRADIENTE Y BACKTRACKING
# ==================================================================
def sec_gradient_backtracking():
    section_title(
        "17. Descenso de Gradiente, Newton y Backtracking",
        "Cómo el tamaño de paso, la curvatura y el momentum controlan la trayectoria de optimización."
    )
    beginner_bridge(
        "qué hace un optimizador",
        [
            "Una pérdida mide qué tan mal está funcionando un modelo.",
            "El gradiente indica hacia dónde sube más rápido la pérdida; por eso bajamos en la dirección contraria.",
            "El tamaño de paso decide cuánto avanzar: demasiado pequeño es lento, demasiado grande puede oscilar.",
        ],
    )
    motivation(
        "Entrenar un modelo de IA es resolver $\\min_\\theta \\mathcal{L}(\\theta)$. Los tres algoritmos de esta "
        "sección son las piezas fundamentales de cualquier optimizador moderno (SGD, Adam, L-BFGS): "
        "descenso de gradiente como base, backtracking para elegir el paso, Newton para aprovechar curvatura, "
        "y momentum para acelerar el descenso en valles estrechos."
    )
    st.info("Ruta recomendada si partes desde cero: revisa primero la sección 19 (Cálculo Diferencial), luego la 20 (Convexidad), y vuelve aquí. El gradiente define el movimiento; la convexidad elimina mínimos locales falsos cuando se cumplen sus supuestos.")
    prerequisites_box(
        "- Gradiente $\\nabla f(x)$: dirección de mayor crecimiento local.\n"
        "- Hessiana $H(x) = \\nabla^2 f(x)$: matriz de segundas derivadas (curvatura).\n"
        "- Convexidad: ver sección 20."
    )

    st.markdown("### Descenso de gradiente")
    st.latex(r"x_{k+1} = x_k - \gamma_k \nabla f(x_k)")
    notation_box([
        (r"\preceq", "Orden entre matrices: $A \\preceq B$ significa $B - A$ es semidefinida positiva (SDP), es decir, $v^\\top(B-A)v \\geq 0$ para todo $v$. En la práctica: la curvatura de $B$ supera a la de $A$ en todas las direcciones."),
        ("m, M", "Constantes que acotan los valores propios (eigenvalores) de la Hessiana $H(x)$: $m$ es el mínimo y $M$ el máximo. Cuando $m \\approx M$, la función es casi 'redonda' y el descenso converge rápido."),
    ])
    st.markdown(
        "Cuando $f$ es convexa con curvatura acotada $mI \\preceq H(x) \\preceq MI$, el descenso con "
        "paso seguro $\\gamma = 1/M$ cumple esta garantía de convergencia lineal:"
    )
    st.latex(r"f(x_{k+1}) - f(x^\star) \leq \left(1 - \frac{m}{M}\right)\bigl(f(x_k) - f(x^\star)\bigr)")
    st.caption("El ratio $m/M$ (razón de condición inversa) mide qué tan bien condicionado es el problema. Con $m \\approx M$ converge en muy pocas iteraciones; con $m \\ll M$ puede zig-zagear.")

    st.markdown("### Método de Newton")
    plain_language(
        "Idea: usar curvatura local para dar saltos más inteligentes",
        "El descenso de gradiente asume que la función es localmente plana — sólo usa información de primer orden. "
        "Newton aproxima la función con un paraboloide (Taylor de orden 2) y salta directamente a su mínimo. "
        "Bajo hipótesis locales de regularidad, Hessiana no singular y punto inicial suficientemente cercano, eso le da convergencia cuadrática."
    )
    st.latex(r"x_{k+1} = x_k - \bigl(H(x_k)\bigr)^{-1}\nabla f(x_k)")
    formula_walkthrough(
        "Por qué Newton converge cuadráticamente",
        terms={
            r"H(x_k)^{-1}": "Inversa de la Hessiana: 'estira' el paso según la curvatura de cada dirección.",
            r"\nabla f(x_k)": "Gradiente: dirección de máxima pendiente.",
            r"\|x_{k+1}-x^\star\|": "Error en el paso $k+1$.",
        },
        steps=[
            "Newton minimiza el polinomio de Taylor de orden 2 de $f$ en torno a $x_k$.",
            "Si $f$ fuera exactamente cuadrática, llegaría al mínimo exacto en un solo paso.",
            "Para funciones generales, cada iteración cuadra el error: si estás a distancia $\\epsilon$, el siguiente paso te deja a $\\sim C\\epsilon^2$.",
            "Limitación: calcular (e invertir) la Hessiana cuesta $O(n^3)$ — inviable para modelos con millones de parámetros."
        ],
        expanded=True,
    )
    worked_example("Newton en $F(x) = \\frac{1}{3}x^3 - 4x$")
    st.markdown("$\\nabla F(x) = x^2 - 4$, $H(x) = 2x$. La iteración de Newton queda:")
    st.latex(r"x_{k+1} = x_k - \frac{x_k^2 - 4}{2x_k} = \frac{x_k}{2} + \frac{2}{x_k}")
    st.caption("Partiendo de $x_0 = 3$: $x_1 \\approx 2.167$, $x_2 \\approx 2.019$, $x_3 \\approx 2.000$ (converge al punto crítico $x^\\star = 2$, que es mínimo local).")

    st.markdown("### Descenso con Momentum")
    st.markdown(
        "El momentum agrega un término de 'inercia' que acumula la dirección de pasos anteriores. "
        "Reduce las oscilaciones en valles estrechos y acelera la convergencia en pasillos largos."
    )
    st.latex(r"z_k = \nabla f(x_k) + \beta z_{k-1}, \qquad x_{k+1} = x_k - \gamma z_k")
    st.caption("$\\beta \\in [0,1)$ controla cuánto 'recuerda' el optimizador pasos anteriores. Con $\\beta=0$ recuperamos el descenso estándar.")
    insight(
        "El momentum reduce las oscilaciones porque 'promedia' gradientes consecutivos: si el gradiente "
        "zig-zagea, los componentes que se cancelan se anulan y queda la dirección de avance neta. "
        "En la práctica, valores como $\\beta = 0.9$ funcionan bien en redes neuronales."
    )
    with advanced_expander("Paso y momentum óptimos para una cuadrática (requiere álgebra lineal)"):
        st.markdown(
            "Para $f(x) = \\frac{1}{2}x^\\top Sx$, los eigenvalores (valores propios) de $S$ "
            "cuantifican la curvatura en cada dirección: $\\lambda_{\\min}$ en la dirección más 'plana' "
            "y $\\lambda_{\\max}$ en la más 'empinada'. Los parámetros óptimos son:"
        )
        latex_aligned([
            r"\gamma^\star = \left(\frac{2}{\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}}\right)^2",
            r"\beta^\star = \left(\frac{\sqrt{\lambda_{\max}}-\sqrt{\lambda_{\min}}}{\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}}\right)^2",
        ])
        st.caption("Con estos parámetros el número de iteraciones para converger crece como $\\sqrt{\\lambda_{\\max}/\\lambda_{\\min}}$ en lugar de $\\lambda_{\\max}/\\lambda_{\\min}$ — una mejora radical cuando el problema está mal condicionado.")

    st.markdown("### Descenso zig-zag en una cuadrática anisotrópica")
    st.latex(r"f(x,y)=\tfrac12(x^2+b y^2),\qquad \nabla f(x,y)=(x,by)")
    st.markdown(
        "Cuando las curvaturas por eje son muy distintas, un paso fijo puede producir trayectorias en zig-zag. "
        "Usaremos $\\gamma=2/(b+1)$ como paso guiado para visualizar ese efecto."
    )
    real_world_case(
        "ajustar dos parámetros de un modelo",
        "Piensa que $x$ e $y$ son dos parámetros de un modelo y $f$ es la pérdida que queremos bajar. "
        "El parámetro $b$ controla la curvatura en la dirección y: si las curvaturas por eje son muy distintas, el problema queda mal condicionado y el descenso puede oscilar o avanzar lento.",
        controls=[
            ("curvatura del valle b", "curvatura relativa en la dirección y."),
            ("tamaño de paso γ", "cuánto avanzamos contra el gradiente."),
            ("iteraciones", "cuántas actualizaciones hacemos."),
        ],
        takeaway="Un paso grande puede parecer eficiente, pero si la geometría es estrecha puede producir zig-zag o inestabilidad.",
        expanded=False,
    )
    col1, col2 = lab_columns()
    with col1:
        b_gd = st.slider("curvatura del valle b", 0.02, 2.0, 0.05, step=0.01, key="gd_b")
        gamma_mode = st.radio("tamaño de paso γ", ["2/(b+1)", "manual"], horizontal=True, key="gd_mode")
        gamma_gd = 2 / (b_gd + 1) if gamma_mode.startswith("2/") else st.slider("tamaño de paso γ", 0.01, 2.5, 0.3, key="gd_gamma")
        n_iter_gd = st.slider("iteraciones", 5, 80, 30, key="gd_iters")
    x = np.array([2.0, 1.5])
    path = [x.copy()]
    for _ in range(n_iter_gd):
        grad = np.array([x[0], b_gd * x[1]])
        x = x - gamma_gd * grad
        path.append(x.copy())
    path = np.array(path)
    with col2:
        f_final_gd = 0.5 * (path[-1,0]**2 + b_gd * path[-1,1]**2)
        grad_final_gd = np.linalg.norm([path[-1, 0], b_gd * path[-1, 1]])
        metric_grid([
            ("tamaño de paso γ", f"{gamma_gd:.3f}"),
            ("pérdida final f", f"{f_final_gd:.4f}"),
            ("tamaño del gradiente final", f"{grad_final_gd:.4f}"),
        ], columns=3)
        xx, yy = np.meshgrid(np.linspace(-2.3, 2.3, 160), np.linspace(-2.0, 2.0, 160))
        zz = 0.5 * (xx**2 + b_gd * yy**2)
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.contour(xx, yy, zz, levels=18, cmap="viridis")
        ax.plot(path[:, 0], path[:, 1], "o-", color="#DD8452", ms=3)
        ax.scatter([0], [0], color="black", s=30, label="mínimo")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Trayectoria de descenso")
        ax.legend()
        mia_pyplot(fig); plt.close(fig)
    how_to_read("La trayectoria naranja muestra los parámetros. Si el paso es grande respecto de la curvatura, aparece oscilación.")
    lab_note("cada punto naranja es una iteración sucesiva; el punto negro es el mínimo de la pérdida.")

    interactive_header("Backtracking line search")
    lab_task(
        predict="si α exige demasiada mejora o β achica muy agresivamente, los pasos aceptados cambiarán.",
        manipulate="mueve α, β e iteraciones.",
        verify="mira la trayectoria y la gráfica de pasos aceptados.",
    )
    st.markdown(
        "Backtracking parte con un paso candidato y lo reduce por un factor $\\beta$ hasta satisfacer una condición tipo Armijo."
    )
    latex_aligned([
        r"f(x-\gamma\nabla f(x))",
        r"\le f(x)-\alpha\gamma\|\nabla f(x)\|^2",
    ])
    formula_walkthrough(
        "Armijo en palabras",
        terms={
            r"\gamma": "Tamaño de paso candidato.",
            r"\alpha": "Qué tan exigentes somos con la mejora mínima.",
            r"\beta": "Factor por el que reducimos el paso cuando no cumple.",
            r"\|\nabla f(x)\|^2": "Tamaño del gradiente al cuadrado: mide qué tan fuerte es la señal local de descenso.",
        },
        steps=[
            "Proponemos un paso grande.",
            "Calculamos si la pérdida baja lo suficiente.",
            "Si no baja lo suficiente, multiplicamos el paso por β y probamos de nuevo.",
            "Aceptamos el primer paso que cumple la condición.",
        ],
        expanded=True,
    )
    col1, col2 = lab_columns()
    with col1:
        alpha_bt = st.slider("exigencia de mejora α", 0.01, 0.49, 0.1, step=0.01, key="bt_alpha")
        beta_bt = st.slider("factor para achicar el paso β", 0.1, 0.9, 0.5, step=0.05, key="bt_beta")
        n_bt = st.slider("iteraciones backtracking", 1, 40, 10, key="bt_iters")

    def _f_bt(z):
        return 0.5 * (z[0] ** 2 + 10 * z[1] ** 2)

    def _g_bt(z):
        return np.array([z[0], 10 * z[1]])

    z = np.array([2.0, 1.0])
    bt_path = [z.copy()]
    bt_steps = []
    bt_values = [_f_bt(z)]
    for _ in range(n_bt):
        g = _g_bt(z)
        step = 1.0
        while _f_bt(z - step * g) > _f_bt(z) - alpha_bt * step * np.dot(g, g):
            step *= beta_bt
            if step < 1e-8:
                break
        z = z - step * g
        bt_path.append(z.copy())
        bt_steps.append(step)
        bt_values.append(_f_bt(z))
    bt_path = np.array(bt_path)
    with col2:
        metric_grid([
            ("pérdida inicial f", f"{bt_values[0]:.3f}"),
            ("pérdida final f", f"{bt_values[-1]:.3e}"),
            ("último γ", f"{bt_steps[-1]:.3f}" if bt_steps else "-"),
        ], columns=3)
        xx, yy = np.meshgrid(np.linspace(-2.2, 2.2, 160), np.linspace(-1.2, 1.2, 160))
        zz = 0.5 * (xx**2 + 10 * yy**2)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].contour(xx, yy, zz, levels=20, cmap="viridis")
        axes[0].plot(bt_path[:, 0], bt_path[:, 1], "o-", color="#DD8452", ms=3)
        axes[0].set_title("trayectoria")
        axes[0].set_xlabel("x0")
        axes[0].set_ylabel("x1")
        axes[1].plot(bt_steps, "o-", color="#4C72B0")
        axes[1].set_title("pasos aceptados")
        axes[1].set_xlabel("iteración")
        axes[1].set_ylabel("γ")
        axes[1].set_ylim(0, max(bt_steps) * 1.15 if bt_steps else 1)
        plt.tight_layout()
        mia_pyplot(fig); plt.close(fig)
        with st.expander("Últimas iteraciones aceptadas", expanded=False):
            compact_dataframe(pd.DataFrame({
                "iteración": np.arange(len(bt_values)),
                "f(x)": [f"{v:.3e}" for v in bt_values],
                "γ aceptado": ["-"] + [f"{s:.3f}" for s in bt_steps],
            }).tail(8))
    how_to_read(
        "La figura izquierda muestra cómo se mueve el punto hacia el mínimo. La derecha no muestra la pérdida, sino el tamaño de paso aceptado por Armijo. "
        "Si la pérdida final aparece muy pequeña, no significa error: significa que el algoritmo llegó muy cerca del mínimo; por eso se muestra en notación científica."
    )
    lab_note("en 'pasos aceptados', el eje vertical no es precisión ni pérdida: es el tamaño de paso elegido en cada iteración.")

    interactive_header("Descenso de gradiente vs Newton vs Momentum")
    lab_task(
        predict="Newton debería resolver la cuadrática en un paso; momentum debería reducir oscilaciones en casos mal condicionados.",
        manipulate="cambia b, algoritmo, paso γ y momentum β.",
        verify="compara trayectoria y pérdida en escala logarítmica.",
    )
    interactive_guide(
        controls=[
            ("Curvatura b", "controla el condicionamiento de la cuadrática; valores muy distintos de 1 crean direcciones con curvaturas desbalanceadas."),
            ("Algoritmo", "elige entre descenso estándar, Newton o descenso con momentum."),
            ("β (momentum)", "solo activo cuando se elige momentum; controla la inercia."),
            ("Iteraciones", "número de pasos del optimizador."),
        ],
        procedure="Todos optimizan $f(x_1, x_2) = \\frac{1}{2}(x_1^2 + b x_2^2)$ partiendo del mismo punto.",
        observe="Compara la trayectoria y el valor final de la pérdida. Newton llega en 1 paso (cuadrática exacta); "
                "gradient descent puede zig-zagear; momentum reduce las oscilaciones.",
    )
    col1, col2 = lab_columns()
    with col1:
        b_cmp = st.slider("curvatura b", 0.01, 2.0, 0.05, step=0.01, key="cmp_b")
        algo = st.radio("Algoritmo", ["Descenso de gradiente", "Newton", "Momentum"], key="cmp_algo")
        n_cmp = st.slider("Iteraciones", 3, 60, 20, key="cmp_iters")
        gamma_cmp = st.slider("Paso γ", 0.01, 2.5, min(2/(1+b_cmp), 2.4), key="cmp_gamma")
        beta_cmp = st.slider("β (momentum)", 0.0, 0.99, 0.8, step=0.01, key="cmp_beta") if algo == "Momentum" else 0.0

    def _f_cmp(z, b): return 0.5*(z[0]**2 + b*z[1]**2)
    def _g_cmp(z, b): return np.array([z[0], b*z[1]])
    def _h_inv_cmp(z, b): return np.diag([1.0, 1.0/b])

    z0 = np.array([2.0, 1.5])
    path_cmp = [z0.copy()]
    losses_cmp = [_f_cmp(z0, b_cmp)]
    z = z0.copy()
    mom_z = np.zeros(2)
    for _ in range(n_cmp):
        g = _g_cmp(z, b_cmp)
        if algo == "Descenso de gradiente":
            z = z - gamma_cmp * g
        elif algo == "Newton":
            z = z - _h_inv_cmp(z, b_cmp) @ g
        else:
            mom_z = g + beta_cmp * mom_z
            z = z - gamma_cmp * mom_z
        path_cmp.append(z.copy())
        losses_cmp.append(_f_cmp(z, b_cmp))
    path_cmp = np.array(path_cmp)

    with col2:
        metric_grid([
            ("Pérdida inicial", f"{losses_cmp[0]:.3f}"),
            ("Pérdida final", f"{losses_cmp[-1]:.2e}"),
        ], columns=2)
        xx2, yy2 = np.meshgrid(np.linspace(-2.5, 2.5, 160), np.linspace(-2.0, 2.0, 160))
        zz2 = 0.5*(xx2**2 + b_cmp*yy2**2)
        fig2, axes2 = plt.subplots(1, 2, figsize=(10.2, 3.6))
        axes2[0].contour(xx2, yy2, zz2, levels=20, cmap="viridis")
        axes2[0].plot(path_cmp[:, 0], path_cmp[:, 1], "o-", color="#DD8452", ms=3.5, lw=1.4)
        axes2[0].scatter([0], [0], color="black", s=30, zorder=5)
        axes2[0].set_title(f"Trayectoria — {algo}")
        axes2[0].set_xlabel("$x_1$"); axes2[0].set_ylabel("$x_2$")
        losses_cmp_plot = np.maximum(np.asarray(losses_cmp, dtype=float), np.finfo(float).tiny)
        axes2[1].semilogy(losses_cmp_plot, "o-", color="#4C72B0", ms=3)
        axes2[1].set_xlabel("iteración"); axes2[1].set_ylabel("$f(x_k)$")
        axes2[1].set_title("Curva de convergencia (escala log)")
        plt.tight_layout()
        mia_pyplot(fig2); plt.close(fig2)
    how_to_read(
        "Izquierda: trayectoria en el espacio de parámetros. "
        "Derecha: pérdida en escala logarítmica — una recta indica convergencia lineal; "
        "Newton sobre cuadráticas converge en un solo paso."
    )

    self_check_header()
    quiz(
        "¿Por qué Newton no se usa directamente en redes neuronales grandes?",
        ["Porque no converge en funciones no convexas",
         "Porque calcular e invertir la Hessiana ($n\\times n$) es $O(n^3)$, inviable con millones de parámetros",
         "Porque el momentum es siempre mejor"],
        1,
        "Con $n=10^7$ parámetros, la Hessiana tiene $10^{14}$ entradas. Existen aproximaciones como L-BFGS que evitan calcularla explícitamente.",
        "El costo cúbico de la inversión es el obstáculo central.",
        key="gd_q1"
    )
    quiz(
        "El backtracking de Armijo sirve para...",
        ["Calcular el gradiente exacto",
         "Encontrar un tamaño de paso que garantice suficiente disminución de $f$",
         "Eliminar el zig-zag siempre"],
        1,
        "La condición de Armijo: $f(x - \\gamma g) \\leq f(x) - \\alpha\\gamma\\|g\\|^2$. Garantiza que la pérdida sí baja.",
        "Backtracking no calcula gradientes; solo ajusta el tamaño de paso hasta que la función baje 'suficientemente'.",
        key="gd_q2"
    )
    ai_bridge(
        "**Adam** = momentum (primer momento) + escala adaptativa por varianza del gradiente (segundo momento). "
        "**L-BFGS** = Newton aproximado sin calcular la Hessiana completa, usando historial de gradientes. "
        "**Learning rate scheduling** = backtracking global a lo largo de las épocas. "
        "Todo optimizador moderno combina estas tres ideas."
    )

# ==================================================================
# SECCIÓN 18 — MÉTODO DE LOS MOMENTOS
# ==================================================================
def sec_mom():
    section_title(
        "18. Método de los Momentos (MoM)",
        "El enfoque más intuitivo para aprender parámetros de una distribución: igualar momentos muestrales con teóricos."
    )
    beginner_bridge(
        "estimar un parámetro",
        [
            "Un parámetro es un número desconocido que controla una distribución, como la tasa $\\lambda$ en Poisson o la media $\\mu$ en Normal.",
            "MoM iguala resúmenes observados de los datos con resúmenes teóricos de la distribución.",
            "Funciona bien sólo cuando esos momentos realmente identifican el parámetro y existen las condiciones de convergencia.",
        ],
    )
    motivation(
        "Hasta ahora asumíamos que los parámetros ($\\mu$, $\\sigma$, $\\lambda$) eran conocidos. "
        "En IA enfrentamos el **problema inverso**: dado un dataset, ¿qué distribución lo generó? "
        "El Método de los Momentos (MoM) usa la **Ley de los Grandes Números** como puente: "
        "los momentos muestrales $\\hat\\mu_k = \\frac{1}{n}\\sum X_i^k$ convergen a los momentos teóricos $E[X^k]$. "
        "Igualando ambos se despeja el parámetro."
    )
    prerequisites_box(
        "- $E[X]$, $E[X^2]$, varianza.\n"
        "- Ley de los Grandes Números (sección 15).\n"
        "- Distribuciones Poisson y Normal (sección 7)."
    )
    st.markdown("### Construcción general")
    plain_language(
        "Idea central del MoM",
        "Los momentos de una distribución son cantidades como $E[X]$, $E[X^2]$, etc. "
        "El MoM dice: si tienes un parámetro desconocido $\\theta$, exprésalo en términos de momentos teóricos y luego "
        "sustitúyelo por los momentos muestrales. Es el método de estimación más antiguo e intuitivo. "
        "Existe un método alternativo más sofisticado (Máxima Verosimilitud, MLE) que puede ser más preciso "
        "pero requiere conocer la distribución exacta y a veces no tiene solución cerrada — la comparación aparece al final de esta sección."
    )
    pitfall(
        "El MoM sólo funciona si los momentos **identifican** el parámetro, es decir, si parámetros distintos "
        "producen momentos distintos. Si dos distribuciones con $\\theta_1 \\neq \\theta_2$ tienen el mismo $E[X]$, "
        "MoM no puede distinguirlas con un solo momento — se necesitaría un momento de orden superior."
    )
    st.markdown("**Procedimiento en 3 pasos:**")
    latex_aligned([
        r"\text{1. Expresar } \theta \text{ como función de los momentos teóricos: } \mu_k = E[X^k]",
        r"\text{2. Sustituir por momentos muestrales: } \hat\mu_k = \frac{1}{n}\sum_{i=1}^n X_i^k",
        r"\text{3. Despejar } \hat\theta \text{ del sistema de ecuaciones resultante}",
    ])
    insight(
        "Con $p$ parámetros desconocidos, se necesitan $p$ ecuaciones de momentos. "
        "Bajo identificación, momentos finitos y continuidad del mapeo de momentos, los estimadores MoM son consistentes: "
        "$\\hat\\theta \\xrightarrow{P} \\theta$ cuando $n\\to\\infty$."
    )

    worked_example("MoM para Poisson($\\lambda$) — número de clics por minuto")
    st.markdown(
        "Sea $X_1,\\ldots,X_n \\sim \\text{Poisson}(\\lambda)$, donde $\\lambda$ es desconocido. "
        "Para una Poisson, el primer momento teórico es simplemente $E[X] = \\lambda$."
    )
    latex_aligned([
        r"E[X] = \lambda",
        r"\Longrightarrow \hat\lambda_{\text{MoM}} = \hat\mu_1 = \frac{1}{n}\sum_{i=1}^n X_i = \bar X_n",
    ])
    st.success("El estimador MoM de $\\lambda$ es el promedio de los datos observados. Coincide con el MLE.")

    worked_example("MoM para Normal($\\mu$, $\\sigma^2$) — dos parámetros, dos momentos")
    st.markdown(
        "Sea $X_i \\sim \\mathcal{N}(\\mu, \\sigma^2)$ con $\\mu$ y $\\sigma^2$ desconocidos. "
        "Necesitamos dos ecuaciones:"
    )
    latex_aligned([
        r"\text{1er momento: } \mu = E[X] \Longrightarrow \hat\mu = \bar X_n",
        r"\text{2do momento: } \sigma^2 = E[X^2] - (E[X])^2 \Longrightarrow \hat\sigma^2 = \hat\mu_2 - \bar X_n^2",
    ])
    st.markdown("Donde $\\hat\\mu_2 = \\frac{1}{n}\\sum X_i^2$. Sustituyendo:")
    st.latex(r"\hat\sigma^2 = \frac{1}{n}\sum_i X_i^2 - \bar X_n^2 = \frac{1}{n}\sum_i (X_i - \bar X_n)^2")
    pitfall(
        "El estimador MoM de $\\sigma^2$ divide por $n$, no por $n-1$. "
        "Por eso es ligeramente **sesgado** (subestima la varianza), a diferencia de $S^2$ con corrección de Bessel. "
        "El MLE de la Normal también usa $n$; la corrección $n-1$ es un ajuste de insesgadez separado."
    )

    st.markdown("### MoM vs MLE — ¿cuándo preferir uno u otro?")
    col_mom, col_mle = st.columns(2)
    with col_mom:
        st.markdown("**Método de los Momentos**")
        st.markdown(
            "- ✅ Muy simple: basta igualar promedios de potencias.\n"
            "- ✅ Puede producir estimadores consistentes bajo identificación y condiciones regulares.\n"
            "- ✅ Útil cuando la distribución exacta es desconocida (solo se modelan momentos).\n"
            "- ⚠️ Puede ser sesgado (no maximiza la verosimilitud).\n"
            "- ⚠️ Menos eficiente que MLE cuando la distribución es correctamente especificada."
        )
    with col_mle:
        st.markdown("**Máxima Verosimilitud (MLE)**")
        st.markdown(
            "- ✅ Asintóticamente eficiente (menor varianza posible para $n$ grande).\n"
            "- ✅ Invariante bajo transformaciones del parámetro.\n"
            "- ⚠️ Requiere conocer la distribución correcta.\n"
            "- ⚠️ Puede requerir optimización numérica si no hay forma cerrada.\n"
            "- ⚠️ La log-verosimilitud puede ser no cóncava (múltiples máximos)."
        )

    interactive_header("Ajuste de distribución por MoM")
    lab_task(
        predict="al aumentar n, el ajuste MoM debería acercarse a la distribución verdadera.",
        manipulate="elige distribución, tamaño de muestra y simula nuevas muestras.",
        verify="compara parámetros verdaderos, estimadores y la curva ajustada contra el histograma.",
    )
    interactive_guide(
        controls=[
            ("Distribución", "la distribución verdadera que genera los datos."),
            ("Tamaño de muestra n", "cuántas observaciones usarás para estimar el parámetro."),
            ("Simular nueva muestra", "genera una nueva realización para ver la variabilidad del estimador."),
        ],
        procedure=(
            "Se generan $n$ observaciones i.i.d., se calcula el estimador MoM, y se compara la distribución ajustada "
            "con la distribución verdadera y el histograma de los datos."
        ),
        observe=(
            "Con $n$ pequeño, el estimador MoM fluctúa bastante. A medida que crece $n$, la distribución ajustada "
            "se aproxima a la verdadera (consistencia garantizada por LLN)."
        ),
    )
    col1, col2 = lab_columns()
    with col1:
        mom_dist = st.selectbox(
            "Distribución verdadera",
            ["Poisson(λ=3)", "Normal(μ=2, σ=1.5)", "Exponencial(λ=2)"],
            key="mom_dist",
        )
        n_mom = st.slider("Tamaño de muestra n", 10, 2000, 100, step=10, key="mom_n")
        if st.button("Simular nueva muestra", key="mom_sim"):
            st.session_state["mom_seed"] = np.random.randint(0, 100000)
        seed_mom = st.session_state.get("mom_seed", 7)
        rng_mom = np.random.default_rng(seed_mom)

        if mom_dist.startswith("Poisson"):
            lam_true = 3.0
            data_mom = rng_mom.poisson(lam_true, size=n_mom)
            lam_hat = data_mom.mean()
            st.markdown("**Parámetro verdadero:** $\\lambda = 3$")
            st.markdown(f"**Estimador MoM:** $\\hat\\lambda = \\bar X_n = {lam_hat:.4f}$")
            metric_grid([
                ("λ verdadero", "3.000"),
                ("λ̂ MoM", f"{lam_hat:.4f}"),
                ("Error", f"{abs(lam_hat-lam_true):.4f}"),
                ("n", str(n_mom)),
            ], columns=2)
        elif mom_dist.startswith("Normal"):
            mu_true, sigma_true = 2.0, 1.5
            data_mom = rng_mom.normal(mu_true, sigma_true, size=n_mom)
            mu_hat = data_mom.mean()
            sigma2_hat = ((data_mom - mu_hat)**2).mean()
            st.markdown("**Parámetros verdaderos:** $\\mu=2,\\ \\sigma^2=2.25$")
            st.markdown(f"**Estimadores MoM:** $\\hat\\mu = {mu_hat:.4f},\\ \\hat\\sigma^2 = {sigma2_hat:.4f}$")
            metric_grid([
                ("μ verdadero", "2.000"),
                ("μ̂ MoM", f"{mu_hat:.4f}"),
                ("σ² verdadera", "2.250"),
                ("σ̂² MoM", f"{sigma2_hat:.4f}"),
            ], columns=2)
        else:
            lam_true_exp = 2.0
            data_mom = rng_mom.exponential(1/lam_true_exp, size=n_mom)
            lam_hat_exp = 1.0 / data_mom.mean()
            st.markdown("**Parámetro verdadero:** $\\lambda = 2$  (media = 0.5)")
            st.markdown(f"**Estimador MoM:** $\\hat\\lambda = 1/\\bar X_n = {lam_hat_exp:.4f}$")
            metric_grid([
                ("λ verdadero", "2.000"),
                ("λ̂ MoM", f"{lam_hat_exp:.4f}"),
                ("Error", f"{abs(lam_hat_exp-lam_true_exp):.4f}"),
                ("n", str(n_mom)),
            ], columns=2)
    with col2:
        fig_mom, ax_mom = plt.subplots(figsize=(7, 3.5))
        if mom_dist.startswith("Poisson"):
            max_val = max(int(data_mom.max()) + 2, 12)
            xs_p = np.arange(0, max_val)
            ax_mom.bar(xs_p - 0.2, [np.sum(data_mom == k)/n_mom for k in xs_p],
                       width=0.35, label="Histograma (datos)", color="#4C72B0", alpha=0.75)
            ax_mom.bar(xs_p + 0.2, stats.poisson.pmf(xs_p, lam_hat),
                       width=0.35, label=f"MoM ajustado (λ̂={lam_hat:.2f})", color="#DD8452", alpha=0.75)
            ax_mom.bar(xs_p + 0.0, stats.poisson.pmf(xs_p, lam_true),
                       width=0.05, label="Distribución verdadera", color="#059669", lw=2)
            ax_mom.set_xlabel("k"); ax_mom.set_ylabel("Probabilidad")
        elif mom_dist.startswith("Normal"):
            xs_n = np.linspace(data_mom.min() - 1, data_mom.max() + 1, 200)
            ax_mom.hist(data_mom, bins=30, density=True, alpha=0.55, color="#4C72B0", label="Datos")
            ax_mom.plot(xs_n, stats.norm.pdf(xs_n, mu_hat, np.sqrt(sigma2_hat)),
                        color="#DD8452", lw=2, label=f"MoM ajustado (μ̂={mu_hat:.2f}, σ̂={np.sqrt(sigma2_hat):.2f})")
            ax_mom.plot(xs_n, stats.norm.pdf(xs_n, mu_true, sigma_true),
                        color="#059669", lw=2, ls="--", label="Distribución verdadera")
            ax_mom.set_xlabel("x"); ax_mom.set_ylabel("Densidad")
        else:
            xs_e = np.linspace(0, data_mom.max() + 0.5, 200)
            ax_mom.hist(data_mom, bins=30, density=True, alpha=0.55, color="#4C72B0", label="Datos")
            ax_mom.plot(xs_e, stats.expon.pdf(xs_e, scale=1/lam_hat_exp),
                        color="#DD8452", lw=2, label=f"MoM ajustado (λ̂={lam_hat_exp:.2f})")
            ax_mom.plot(xs_e, stats.expon.pdf(xs_e, scale=1/lam_true_exp),
                        color="#059669", lw=2, ls="--", label="Distribución verdadera")
            ax_mom.set_xlabel("x"); ax_mom.set_ylabel("Densidad")
        ax_mom.set_title(f"MoM: n={n_mom}")
        ax_mom.legend(fontsize=8)
        mia_pyplot(fig_mom)
        plt.close(fig_mom)
    how_to_read(
        "Las barras azules (o histograma azul) son los datos simulados. La curva naranja es la distribución "
        "ajustada por MoM; la verde punteada es la distribución verdadera. Con $n$ pequeño, naranja y verde pueden "
        "diferir. Al crecer $n$, el ajuste MoM converge a la verdad (por LLN)."
    )

    self_check_header()
    quiz(
        "¿Por qué el estimador MoM de $\\sigma^2$ es ligeramente sesgado?",
        [
            "Porque usa todos los datos",
            "Porque divide por $n$ en vez de $n-1$; usa $\\bar X$ ya calculado de los mismos datos",
            "Porque solo usa el primer momento",
        ],
        1,
        "Dividir por $n$ en vez de $n-1$ introduce sesgo negativo: en promedio subestima la varianza.",
        "La corrección de Bessel ($n-1$) compensa el haber usado $\\bar X$ como centro de los datos.",
        key="mom_q1"
    )
    quiz(
        "Si los datos son $X_1,...,X_n \\sim \\text{Poisson}(\\lambda)$ desconocida, el estimador MoM de $\\lambda$ es...",
        ["$S^2 = \\frac{1}{n-1}\\sum(X_i-\\bar X)^2$", "$\\bar X_n$", "$\\max_i X_i$"],
        1,
        "Igualamos $E[X] = \\lambda$ con $\\hat\\mu_1 = \\bar X_n$. Sencillo y coincide con MLE.",
        "Para la Poisson, $E[X]=\\lambda$, así que $\\hat\\lambda = \\bar X_n$.",
        key="mom_q2"
    )
    ai_bridge(
        "El MoM es el precursor conceptual de muchos métodos en ML: "
        "**normalización de batch** estima media y varianza de activaciones (momentos de orden 1 y 2); "
        "los **Modelos de Mezcla Gaussiana (GMM)** se ajustan estimando medias, varianzas y pesos de componentes; "
        "en **NLP**, los *embedding* de palabras capturan momentos de co-ocurrencia. "
        "Estimar momentos muestrales es el primer paso antes de ajustar cualquier modelo probabilístico complejo."
    )


# ==================================================================
# SECCIÓN 19 — CÁLCULO DIFERENCIAL PARA IA
# ==================================================================
def sec_calculo():
    section_title(
        "19. Cálculo Diferencial para IA",
        "Derivadas, gradiente, Hessiana y Jacobiana: las herramientas que hacen posible el aprendizaje."
    )
    beginner_bridge(
        "derivadas sin notación",
        [
            "Una derivada mide cuánto cambia una salida cuando mueves un poco una entrada.",
            "El gradiente junta muchas derivadas: una por cada parámetro.",
            "La Hessiana mide curvatura; la Jacobiana organiza derivadas cuando hay varias salidas.",
        ],
    )
    motivation(
        "Entrenar un modelo de IA requiere medir *cómo cambia la pérdida* cuando se perturba cada parámetro. "
        "Eso es exactamente lo que calculan las derivadas. Esta sección construye desde la derivada escalar "
        "hasta el gradiente multivariable y la Hessiana, las mismas herramientas que usa la backpropagation."
    )
    prerequisites_box(
        "- Límites: $\\lim_{h\\to0}$.\n"
        "- Funciones $f: \\mathbb{R} \\to \\mathbb{R}$ y $f: \\mathbb{R}^n \\to \\mathbb{R}$.\n"
        "- Producto interno de vectores."
    )

    st.markdown("### Derivada en una variable")
    plain_language(
        "Razón de cambio instantáneo",
        "La derivada $f'(a)$ es la pendiente de la recta tangente a $f$ en el punto $x=a$. "
        "Mientras la recta secante mide el cambio promedio en un intervalo $[a, a+h]$, "
        "la tangente mide el cambio exactamente en $a$ (límite cuando $h \\to 0$)."
    )
    latex_aligned([
        r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
    ])
    notation_box([
        (r"f'(x)", "Derivada de $f$ en $x$: razón de cambio instantánea."),
        (r"h", "Perturbación infinitesimal; desaparece en el límite."),
        (r"f''(x)", "Segunda derivada: mide cómo cambia la pendiente (curvatura)."),
    ], expanded=False)
    with st.expander("Diferenciabilidad y ejemplos donde falla", expanded=True):
        st.markdown(
            "- Una función es diferenciable en $a$ si la pendiente límite existe desde ambos lados.\n"
            "- Una discontinuidad no es diferenciable.\n"
            "- $|x|$ no es diferenciable en 0: por la izquierda la pendiente es -1 y por la derecha es 1.\n"
            "- $1/(x-1)$ no es diferenciable en $x=1$ porque la función ni siquiera está definida ahí."
        )
    with advanced_expander("Tabla de derivadas esenciales"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(
                "| $f(x)$ | $f'(x)$ |\n"
                "|---|---|\n"
                "| $c$ (constante) | $0$ |\n"
                "| $x^n$ | $n x^{n-1}$ |\n"
                "| $e^x$ | $e^x$ |\n"
                "| $a^x$ | $a^x \\ln a$ |\n"
            )
        with col_d2:
            st.markdown(
                "| $f(x)$ | $f'(x)$ |\n"
                "|---|---|\n"
                "| $\\ln x$ | $1/x$ |\n"
                "| $\\sin x$ | $\\cos x$ |\n"
                "| $\\cos x$ | $-\\sin x$ |\n"
                "| $\\sigma(x) = 1/(1+e^{-x})$ | $\\sigma(x)(1-\\sigma(x))$ |\n"
            )
        st.markdown("**Reglas de combinación:**")
        latex_aligned([
            r"(f \pm g)' = f' \pm g'",
            r"(fg)' = f'g + fg' \quad \text{(regla del producto)}",
            r"\left(\frac{f}{g}\right)' = \frac{f'g-fg'}{g^2} \quad \text{(regla del cociente)}",
            r"(f \circ g)'(x) = f'(g(x))\,g'(x) \quad \text{(regla de la cadena)}",
            r"\log_a(x)' = \frac{1}{x\ln a}",
        ])
        pitfall("La regla de la cadena es la base de backpropagation: la derivada de una función compuesta es el producto de las derivadas en cada capa.")

    st.markdown("### Polinomios de Taylor")
    st.markdown(
        "Un polinomio de Taylor de orden $n$ aproxima $f$ cerca del punto $a$ usando sus primeras $n$ derivadas:"
    )
    st.latex(r"T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k")
    worked_example("Taylor de $\\cos(x)$ en torno a $a=0$ (orden 4)")
    st.latex(r"T_4(x) = 1 - \frac{x^2}{2} + \frac{x^4}{24}")
    st.caption("A orden 2 tenemos la aproximación cuadrática usada en el Método de Newton.")

    st.markdown("### Gradiente: derivada multivariable")
    st.markdown(
        "Para $f: \\mathbb{R}^n \\to \\mathbb{R}$, el gradiente es el vector de derivadas parciales:"
    )
    st.latex(r"\nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top \in \mathbb{R}^n")
    insight(
        "El gradiente apunta en la dirección de **máximo crecimiento** local. Por eso el descenso de gradiente "
        "va en dirección $-\\nabla f$: opuesta al crecimiento, hacia el mínimo."
    )
    st.markdown("**Derivada direccional** en la dirección $v$ (unitario):")
    st.latex(r"D_v f(x) = \nabla f(x) \cdot v")
    st.caption("El máximo de $D_v f$ sobre todos los $v$ unitarios se alcanza cuando $v = \\nabla f(x)/\\|\\nabla f(x)\\|$.")
    worked_example("Gradiente de $f(x_1, x_2, x_3) = (x_1 + e^{x_2})^2 + \\ln(x_2 x_3)$")
    latex_aligned([
        r"\frac{\partial f}{\partial x_1} = 2(x_1 + e^{x_2})",
        r"\frac{\partial f}{\partial x_2} = 2(x_1 + e^{x_2})e^{x_2} + \frac{1}{x_2}",
        r"\frac{\partial f}{\partial x_3} = \frac{1}{x_3}",
    ])

    st.markdown("### Hessiana: curvatura multivariable")
    st.markdown("La **matriz Hessiana** $H \\in \\mathbb{R}^{n \\times n}$ recoge todas las segundas derivadas parciales:")
    st.latex(r"H_{ij}(x) = \frac{\partial^2 f}{\partial x_i \partial x_j}(x)")
    st.markdown("**Aproximación de Taylor de orden 2** (base del Método de Newton):")
    st.latex(r"f(x + \Delta x) \approx f(x) + (\Delta x)^\top \nabla f(x) + \frac{1}{2}(\Delta x)^\top H(x)\,\Delta x")
    st.caption("Si $H(z)$ es semidefinida positiva para todo $z$ de un dominio convexo, entonces $f$ es convexa en ese dominio. Una Hessiana PSD en un punto sólo describe curvatura local.")

    st.markdown("### Matriz Jacobiana: funciones vectoriales")
    st.markdown("Para $f: \\mathbb{R}^n \\to \\mathbb{R}^m$ con $f = [f_1, \\ldots, f_m]^\\top$, la Jacobiana es:")
    st.latex(r"J(x) = \frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}, \qquad J_{ij} = \frac{\partial f_i}{\partial x_j}")
    worked_example("Jacobiana de una función lineal $f(x)=Ax$")
    st.markdown("Si $A\\in\\mathbb{R}^{m\\times n}$ y $f(x)=Ax$, cada salida es combinación lineal de las entradas. Su Jacobiana es constante:")
    st.latex(r"J_f(x)=A")
    formula_walkthrough(
        "Regla de la cadena generalizada",
        formula=r"\frac{d}{dt}(f \circ g)(t) = J_f(g(t)) \cdot J_g(t)",
        terms={
            r"J_f(g(t))": "Jacobiana de $f$ evaluada en $g(t)$.",
            r"J_g(t)": "Jacobiana de $g$ evaluada en $t$.",
        },
        steps=[
            "En 1D, la regla de la cadena es $(f\\circ g)' = f'(g(t))\\cdot g'(t)$.",
            "En dimensión mayor, las derivadas escalares se reemplazan por Jacobianas.",
            "Backpropagation aplica esto repetidamente por las capas de una red neuronal.",
        ],
        expanded=False,
    )
    worked_example("Gradiente de la pérdida de mínimos cuadrados $L = \\|y - \\Phi\\theta\\|^2$")
    st.markdown("Con $e = y - \\Phi\\theta$ (residuo) y $L = \\|e\\|^2$:")
    st.markdown("Aplicando la regla de la cadena (Jacobiana × Jacobiana):")
    latex_aligned([
        r"\frac{\partial L}{\partial e} = 2e^\top \in \mathbb{R}^{1 \times N} \quad \text{(vector fila)}",
        r"\frac{\partial e}{\partial \theta} = -\Phi \in \mathbb{R}^{N \times D}",
        r"\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial e}\frac{\partial e}{\partial \theta} = -2e^\top \Phi \in \mathbb{R}^{1\times D}",
    ])
    st.markdown("Transponiendo para obtener el **gradiente como vector columna** (convención estándar en ML):")
    st.latex(r"\nabla_\theta L = -2\Phi^\top e = -2\Phi^\top(y - \Phi\theta) \in \mathbb{R}^D")
    st.caption("Igualando a cero: $\\Phi^\\top\\Phi\\,\\hat\\theta = \\Phi^\\top y$ — las ecuaciones normales de mínimos cuadrados.")

    interactive_header("Visualización del gradiente y las curvas de nivel")
    lab_task(
        predict="la flecha del gradiente debería apuntar hacia donde sube más rápido la superficie.",
        manipulate="elige función y mueve el punto.",
        verify="observa que la flecha es perpendicular a la curva de nivel local.",
    )
    interactive_guide(
        controls=[
            ("Función $f$", "elige la superficie a explorar."),
            ("Selected point (punto elegido)", "punto donde se calcula el gradiente."),
        ],
        procedure="El laboratorio evalúa el gradiente en el punto elegido y lo dibuja sobre las curvas de nivel de $f$.",
        observe="El gradiente (flecha) es siempre perpendicular a la curva de nivel que pasa por ese punto, "
                "y apunta hacia donde $f$ crece más rápido.",
    )
    col1, col2 = lab_columns()
    with col1:
        f_choice = st.selectbox(
            "Función $f(x_1, x_2)$",
            ["Cuadrática anisotrópica", "Rosenbrock", "Seno 2D"],
            key="calc_f",
        )
        calc_formulas = {
            "Cuadrática anisotrópica": r"f(x_1,x_2)=x_1^2+5x_2^2",
            "Rosenbrock": r"f(x_1,x_2)=(1-x_1)^2+10(x_2-x_1^2)^2",
            "Seno 2D": r"f(x_1,x_2)=\sin(x_1)\cos(x_2)",
        }
        st.latex(calc_formulas[f_choice])
        x0_g = st.slider("coordenada x1", -2.0, 2.0, 0.8, step=0.1, key="calc_x0")
        y0_g = st.slider("coordenada x2", -2.0, 2.0, 0.5, step=0.1, key="calc_y0")
    dx = 1e-5
    if f_choice.startswith("Cuadrática"):
        def _fv(x, y): return x**2 + 5*y**2
    elif f_choice.startswith("Rosenbrock"):
        def _fv(x, y): return (1-x)**2 + 10*(y - x**2)**2
    else:
        def _fv(x, y): return np.sin(x) * np.cos(y)
    gx = (_fv(x0_g+dx, y0_g) - _fv(x0_g-dx, y0_g)) / (2*dx)
    gy = (_fv(x0_g, y0_g+dx) - _fv(x0_g, y0_g-dx)) / (2*dx)
    grad_norm = np.sqrt(gx**2 + gy**2)
    with col1:
        metric_grid([
            ("f at selected point", f"{_fv(x0_g, y0_g):.4f}"),
            ("partial derivative in x1", f"{gx:.4f}"),
            ("partial derivative in x2", f"{gy:.4f}"),
            ("gradient norm", f"{grad_norm:.4f}"),
        ], columns=2)
    with col2:
        xg = np.linspace(-2.2, 2.2, 120)
        yg = np.linspace(-2.2, 2.2, 120)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = _fv(Xg, Yg)
        fig_g, ax_g = plt.subplots(figsize=(6.5, 4.5))
        ax_g.contour(Xg, Yg, Zg, levels=20, cmap="viridis")
        scale = 0.4 / max(grad_norm, 1e-9)
        ax_g.annotate("", xy=(x0_g + gx*scale, y0_g + gy*scale), xytext=(x0_g, y0_g),
                      arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2))
        ax_g.scatter([x0_g], [y0_g], color="#DC2626", s=60, zorder=5)
        ax_g.set_xlabel("$x_1$"); ax_g.set_ylabel("$x_2$")
        ax_g.set_title("Curvas de nivel y gradiente (flecha roja)")
        mia_pyplot(fig_g); plt.close(fig_g)
    how_to_read(
        "Las curvas de nivel son las 'curvas de igual altitud' de $f$. El gradiente (flecha roja) es siempre perpendicular a ellas "
        "y apunta hacia las curvas de valor más alto."
    )

    self_check_header()
    quiz(
        "La derivada parcial $\\partial f/\\partial x_i$ mide...",
        ["El cambio de $f$ en todas las direcciones simultáneamente",
         "El cambio de $f$ cuando solo $x_i$ varía infinitesimalmente, manteniendo el resto fijo",
         "La norma del gradiente"],
        1,
        "Exactamente: 'parcial' porque solo varía una coordenada a la vez.",
        "Piensa en seccionar la superficie con un plano paralelo al eje $x_i$.",
        key="calc_q1"
    )
    ai_bridge(
        "**Backpropagation** es la regla de la cadena aplicada repetidamente hacia atrás en una red neuronal. "
        "La Hessiana no se calcula explícitamente en redes grandes (es $n^2$ coeficientes), pero "
        "**Adam** escala actualizaciones usando medias móviles del primer y segundo momento de los gradientes. "
        "**Batch normalization** normaliza activaciones usando media y varianza del mini-batch. "
        "**Gradient clipping** acota $\\|\\nabla f\\|$ para evitar explosión de gradientes en RNNs."
    )


# ==================================================================
# SECCIÓN 20 — CONVEXIDAD
# ==================================================================
def sec_convexidad():
    section_title(
        "20. Convexidad",
        "La propiedad que convierte un problema de optimización difícil en uno garantizado."
    )
    beginner_bridge(
        "convexidad en una frase",
        [
        "Una función convexa tiene bowl shape (forma de cuenco): no hay pozos falsos separados del mejor punto.",
            "Convexo no significa necesariamente único; único requiere convexidad estricta u otros supuestos.",
            "La Hessiana ayuda a certificar convexidad sólo si es semidefinida positiva en todo el dominio relevante.",
        ],
    )
    motivation(
        "En optimización general, un mínimo local puede no ser global: el algoritmo puede quedar atrapado. "
        "La **convexidad** elimina esa trampa: para funciones convexas, todo mínimo local es automáticamente global. "
        "Muchas pérdidas clásicas en ML son convexas bajo formulaciones adecuadas; con existencia de minimizador, paso apropiado "
        "y condiciones de rango/regularización, los métodos de primer orden pueden converger a un óptimo global."
    )
    prerequisites_box(
        "- Gradiente $\\nabla f$ y Hessiana $H$ (sección 19).\n"
        "- Semidefinida positiva: $A \\succeq 0 \\Leftrightarrow x^\\top A x \\geq 0\\ \\forall x$."
    )

    st.markdown("### Conjuntos convexos")
    st.latex(r"C \text{ es convexo} \iff \forall\, x,y \in C,\; \theta \in [0,1]:\; \theta x + (1-\theta)y \in C")
    st.caption("Interpretación: el segmento que une cualquier par de puntos del conjunto está completamente dentro del conjunto.")

    st.markdown("### Funciones convexas")
    st.latex(r"f(\theta y + (1-\theta)x) \leq \theta f(y) + (1-\theta)f(x) \quad \forall x,y,\; \theta \in [0,1]")
    plain_language(
        "En palabras",
        "La cuerda que une dos puntos del gráfico de $f$ siempre está **por encima** de la función. "
        "La función no tiene 'valles ocultos' entre dos puntos."
    )
    insight(
        "**Teorema clave:** Si $f$ es convexa, todo mínimo local es un mínimo global. "
        "Si además $f$ es diferenciable y existe un punto con $\\nabla f(x^\\star)=0$, ese punto es óptimo global. "
        "La unicidad requiere supuestos adicionales."
    )

    st.markdown("### Condición de primer orden")
    st.markdown("Si $f$ es diferenciable y convexa, la desigualdad de convexidad se puede escribir con el gradiente:")
    st.latex(r"f(y) \geq f(x) + \nabla f(x)^\top (y - x) \quad \forall x, y")
    st.caption("Interpretación: el plano tangente a $f$ en cualquier punto $x$ está **por debajo** (o sobre) la función. La tangente es un soporte global.")
    with advanced_expander("Demostración de la condición de primer orden"):
        st.markdown("**($\\Rightarrow$)** Si $f$ es convexa, sea $z = \\theta y + (1-\\theta)x$. Por convexidad:")
        latex_aligned([
            r"f(x + \theta(y-x)) \leq (1-\theta)f(x) + \theta f(y)",
            r"\Rightarrow \frac{f(x + \theta(y-x)) - f(x)}{\theta} \leq f(y) - f(x)",
        ])
        st.markdown(
            "Tomando $\\theta \\to 0^+$, el lado izquierdo converge a $\\nabla f(x)^\\top(y-x)$ "
            "— la derivada direccional de $f$ en la dirección $(y-x)$, que en dimensión mayor es exactamente "
            "$\\nabla f(x) \\cdot (y-x)$. Así: $f(y) \\geq f(x) + \\nabla f(x)^\\top(y-x)$."
        )
        st.markdown("**($\\Leftarrow$)** Si la condición lineal vale, para $z = \\theta y + (1-\\theta)x$:")
        latex_aligned([
            r"f(y) \geq f(z) + \nabla f(z)^\top(y-z)",
            r"f(x) \geq f(z) + \nabla f(z)^\top(x-z)",
        ])
        st.markdown("Multiplicando por $\\theta$ y $1-\\theta$ y sumando: $\\theta f(y) + (1-\\theta)f(x) \\geq f(z)$.")

    st.markdown("### Condición de segundo orden")
    st.markdown("Si $f$ es dos veces diferenciable:")
    st.latex(r"f \text{ convexa} \iff \text{Dom}(f) \text{ convexo y } H(x) \succeq 0 \text{ en todo punto}")
    st.caption("$H \\succeq 0$ (semidefinida positiva): todos los valores propios de la Hessiana son $\\geq 0$. La curvatura es no negativa en todas las direcciones.")

    st.markdown("### Propiedades de cierre")
    st.markdown(
        "- **Máximo de convexas:** $\\max_{1 \\leq i \\leq n} f_i(x)$ es convexa si cada $f_i$ lo es.\n"
        "- **Combinación no-negativa:** $\\sum_i a_i f_i(x)$ con $a_i \\geq 0$ es convexa.\n"
        "- **Composición afín:** $g(Ax+b)$ es convexa si $g$ es convexa.\n"
        "- **Normas:** $\\|x\\|$ es convexa para cualquier norma (bola unitaria = conjunto convexo)."
    )

    worked_example("¿Es $f(x) = (y \\cdot x + b)^2$ convexa en $\\mathbb{R}^D$?")
    st.markdown("$g(t) = t^2$ es convexa ($g'' = 2 > 0$) y $t = y \\cdot x + b$ es afín en $x$. Por composición afín, $f$ es convexa.")
    worked_example("¿Es $f(x) = \\ln(1 + e^{-(y\\cdot x + b)})$ convexa? (logistic loss)")
    st.markdown("$g(t) = \\ln(1+e^{-t})$ es convexa ($g'' = \\sigma(t)(1-\\sigma(t)) > 0$) y $t=y\\cdot x+b$ es afín. Por composición, la logistic loss es convexa en $x$.")
    st.success("Resultado: no hay mínimos locales falsos. La existencia y unicidad del minimizador requieren condiciones adicionales, como regularización o diseño de rango adecuado; si los datos son separables, puede no existir minimizador finito.")

    interactive_header("Visualización: convexa vs no convexa")
    lab_task(
        predict="en funciones convexas, la cuerda debería quedar sobre la curva.",
        manipulate="mueve los puntos a y b y cambia la función.",
        verify="compara cuerda, tangente y curva para detectar cuándo falla la convexidad.",
    )
    interactive_guide(
        controls=[
            ("Función", "elige entre funciones convexas y no convexas para ver la diferencia."),
            ("Puntos a y b", "mueve los dos puntos para ver si la cuerda queda sobre la función."),
        ],
        procedure="Se dibujan $f$, la cuerda entre dos puntos, y el plano tangente en $a$.",
        observe="Para funciones convexas la cuerda siempre queda sobre $f$ y la tangente siempre por debajo. "
                "Para no convexas, puede haber tramos donde la cuerda corte a la función.",
    )
    col1, col2 = lab_columns()
    with col1:
        fconv_choice = st.selectbox(
            "Función",
            ["Cuadrática convexa", "Exponencial convexa", "Doble pozo no convexo", "Seno no convexo"],
            key="conv_f",
        )
        conv_formulas = {
            "Cuadrática convexa": r"f(x)=x^2",
            "Exponencial convexa": r"f(x)=e^x",
            "Doble pozo no convexo": r"f(x)=x^4-4x^2",
            "Seno no convexo": r"f(x)=\sin(x)",
        }
        st.latex(conv_formulas[fconv_choice])
        a_cv = st.slider("punto a", -3.0, 3.0, -1.5, step=0.1, key="conv_a")
        b_cv = st.slider("punto b", -3.0, 3.0, 1.5, step=0.1, key="conv_b")
    if fconv_choice == "Cuadrática convexa":
        fconv = lambda x: x**2; fconv_label = r"$x^2$"; is_convex = True
    elif fconv_choice == "Exponencial convexa":
        fconv = np.exp; fconv_label = r"$e^x$"; is_convex = True
    elif fconv_choice == "Doble pozo no convexo":
        fconv = lambda x: x**4 - 4*x**2; fconv_label = r"$x^4 - 4x^2$"; is_convex = False
    else:
        fconv = np.sin; fconv_label = r"$\sin(x)$"; is_convex = False
    xs_cv = np.linspace(-3.2, 3.2, 300)
    dx_cv = 1e-6
    slope_a = (fconv(a_cv+dx_cv) - fconv(a_cv-dx_cv)) / (2*dx_cv)
    tangent = fconv(a_cv) + slope_a * (xs_cv - a_cv)
    chord_x = np.array([a_cv, b_cv])
    chord_y = np.array([fconv(a_cv), fconv(b_cv)])
    with col1:
        if is_convex:
            st.success("Esta función **es convexa**: la cuerda queda sobre $f$ y la tangente por debajo.")
        else:
            st.error("Esta función **no es convexa**: hay tramos donde la cuerda corta a $f$.")
        metric_grid([
            ("f(a)", f"{fconv(a_cv):.3f}"),
            ("f(b)", f"{fconv(b_cv):.3f}"),
            ("function at midpoint", f"{fconv((a_cv+b_cv)/2):.3f}"),
            ("chord at midpoint", f"{(fconv(a_cv)+fconv(b_cv))/2:.3f}"),
        ], columns=2)
    with col2:
        fig_cv, ax_cv = plt.subplots(figsize=(7, 4))
        ax_cv.plot(xs_cv, fconv(xs_cv), color="#4C72B0", lw=2.2, label=fconv_label)
        ax_cv.plot(chord_x, chord_y, "o--", color="#DD8452", lw=1.8, ms=7, label="Cuerda $[a,b]$")
        ax_cv.plot(xs_cv, tangent, color="#059669", lw=1.4, ls=":", label=f"Tangente en $a={a_cv:.1f}$")
        ax_cv.set_xlim(-3.3, 3.3)
        y_all = np.concatenate([fconv(xs_cv), tangent])
        ax_cv.set_ylim(np.nanmin(y_all)-0.5, np.nanmax(y_all)+0.5)
        ax_cv.legend(fontsize=9)
        mia_pyplot(fig_cv); plt.close(fig_cv)
    how_to_read(
        "Naranja = cuerda entre $a$ y $b$. Verde punteado = tangente en $a$. "
        "Función convexa: cuerda sobre la curva, tangente por debajo. "
        "No convexa: la cuerda puede cortar a la curva."
    )

    self_check_header()
    quiz(
        "¿Por qué la pérdida de regresión logística puede optimizarse con descenso de gradiente hasta el óptimo global?",
        ["Porque la pérdida logística es cero en el óptimo",
         "Porque es convexa: todo mínimo local es global",
         "Porque la Hessiana es la identidad"],
        1,
        "Convexidad garantiza que no hay trampas locales: todo mínimo local es global. La convergencia y unicidad requieren condiciones adicionales.",
        "La convexidad es la propiedad que elimina los mínimos locales no globales.",
        key="conv_q1"
    )
    quiz(
        "La Hessiana de una función convexa cumple...",
        ["$H(x) \\preceq 0$ (semidefinida negativa)",
         "$H(x) \\succeq 0$ (semidefinida positiva)",
         "$H(x) = I$ (identidad)"],
        1,
        "$H \\succeq 0$ significa curvatura no negativa en todas las direcciones — la función no tiene 'valles' locales.",
        "Semidefinida positiva = todos los valores propios $\\geq 0$.",
        key="conv_q2"
    )
    ai_bridge(
        "**Regresión lineal** (MSE) y **regresión logística** (BCE) son problemas convexos en formulaciones clásicas; la unicidad depende de rango, regularización y existencia del minimizador. "
        "**Redes neuronales profundas** son no convexas → múltiples mínimos locales, pero en la práctica "
        "son suficientemente buenos. **Regularización L2** ($\\lambda\\|\\theta\\|^2$) preserva la convexidad "
        "cuando el problema original es convexo y puede aportar convexidad estricta en parámetros regularizados."
    )


# ==================================================================
# SECCIÓN 21 — LEVENBERG-MARQUARDT Y NLS
# ==================================================================
def sec_levenberg():
    section_title(
        "21. Levenberg-Marquardt y Mínimos Cuadrados No Lineales",
        "El algoritmo que interpola entre descenso de gradiente y Newton para ajustar modelos complejos."
    )
    beginner_bridge(
        "ajustar una curva",
        [
            "Un residuo es dato observado menos predicción: $r = y - \\hat y$.",
            "Mínimos cuadrados busca parámetros que hagan pequeños esos residuos.",
            "LM resuelve un sistema lineal en cada iteración; la computadora calcula el paso, no invertimos matrices a mano.",
        ],
    )
    motivation(
        "Los mínimos cuadrados lineales tienen solución exacta. Pero en ML muchos modelos son no lineales: "
        "redes neuronales, modelos de crecimiento, series exponenciales. El método de **Levenberg-Marquardt** "
        "combina la estabilidad del descenso de gradiente con una aproximación tipo Gauss-Newton, y es el optimizador "
        "detrás de muchos ajustes de curvas en ciencia e ingeniería."
    )
    prerequisites_box(
        "- Jacobiana $J(\\theta)$ (sección 19).\n"
        "- Descenso de gradiente y Newton (sección 17).\n"
        "- Mínimos cuadrados: minimizar $\\|y - \\hat y(\\theta)\\|^2$."
    )

    st.markdown("### Mínimos cuadrados lineales (revisión)")
    st.markdown(
        "Dados $m$ puntos $(t_i, y_i)$ y un modelo **lineal** en los parámetros $\\hat y = \\Phi\\theta$, "
        "minimizar $E(\\theta) = \\|y - \\Phi\\theta\\|^2$ tiene solución cerrada:"
    )
    st.latex(r"\Phi^\top\Phi\,\hat\theta = \Phi^\top y \quad \text{(ecuaciones normales)}")
    st.caption("Si $\\Phi^\\top\\Phi$ es invertible: $\\hat\\theta = (\\Phi^\\top\\Phi)^{-1}\\Phi^\\top y$.")

    st.markdown("### Mínimos cuadrados no lineales (NLS)")
    st.markdown(
        "Cuando $\\hat y(\\theta)$ depende **no linealmente** de $\\theta$, no hay solución cerrada. "
        "Minimizamos $E(\\theta) = \\|y - \\hat y(\\theta)\\|^2$ iterativamente."
    )
    st.latex(r"\nabla E = -2 J(\theta)^\top \bigl(y - \hat y(\theta)\bigr) = 0")
    st.markdown("donde $J(\\theta) = \\partial\\hat y / \\partial\\theta \\in \\mathbb{R}^{m \\times n}$ es la Jacobiana del modelo.")

    st.markdown("### Método Levenberg-Marquardt")
    plain_language(
        "La idea: dos extremos y un interpolador",
        "Si el parámetro de regularización $\\lambda$ es grande, el método se comporta como descenso de gradiente "
        "(pasos pequeños, estable pero lento). Si $\\lambda \\to 0$, se comporta como Gauss-Newton (pasos grandes, rápido "
        "pero puede divergir). LM adapta $\\lambda$ automáticamente durante la iteración."
    )
    latex_aligned([
        r"\text{Descenso:}\quad \Delta\theta = s\,J^\top(y - \hat y(\theta))",
        r"\text{Gauss-Newton:}\quad J^\top J\,\Delta\theta = J^\top(y - \hat y(\theta))",
        r"\text{Levenberg-Marquardt:}\quad (J^\top J + \lambda I)\,\Delta\theta = J^\top(y - \hat y(\theta))",
    ])
    formula_walkthrough(
        "Por qué funciona la regularización de LM",
        terms={
            r"J^\top J": "Aproximación de la Hessiana de $E$ (asume residuos pequeños).",
            r"\lambda I": "Regularización que garantiza que la matriz sea invertible y controla el tamaño del paso.",
            r"\lambda \to \infty": "El término $\\lambda I$ domina → $\\Delta\\theta \\approx \\frac{1}{\\lambda}J^\\top r$ (gradiente escalado).",
            r"\lambda \to 0": "$J^\\top J$ domina → iteración de Gauss-Newton (Newton aproximado para mínimos cuadrados).",
        },
        steps=[
            "En cada iteración, LM propone un paso $\\Delta\\theta$ resolviendo el sistema lineal.",
            "Si el paso reduce $E$, se acepta y $\\lambda$ se reduce (más confianza, más Newton).",
            "Si el paso no reduce $E$, se rechaza y $\\lambda$ se aumenta (menos confianza, más gradiente).",
        ],
        expanded=True,
    )

    interactive_header("Ajuste de modelo no lineal por LM")
    lab_task(
        predict="con poco ruido y suficientes iteraciones, el ajuste naranja debería acercarse a la curva verde.",
        manipulate="cambia modelo, ruido, regularización inicial e iteraciones.",
        verify="compara curva inicial, curva ajustada, pérdida y regularización final.",
    )
    interactive_guide(
        controls=[
            ("Modelo verdadero", "la función que generó los datos."),
            ("Ruido", "varianza del ruido gaussiano añadido."),
            ("$\\lambda$ inicial", "parámetro de regularización de LM."),
            ("Iteraciones", "pasos del optimizador LM."),
        ],
        procedure="Se generan datos de un modelo no lineal con ruido, y LM ajusta los parámetros minimizando la suma de cuadrados.",
        observe="Observa cómo el ajuste mejora con más iteraciones y cómo la pérdida cae. "
                "Con $\\lambda$ grande el algoritmo es estable pero más lento.",
    )
    col1, col2 = lab_columns()
    with col1:
        lm_model = st.selectbox(
            "Modelo verdadero",
            ["Exponencial decreciente", "Logístico sigmoidal"],
            key="lm_model",
        )
        lm_formulas = {
            "Exponencial decreciente": r"\hat y(t)=a e^{-bt}",
            "Logístico sigmoidal": r"\hat y(t)=\frac{L}{1+e^{-k(t-t_0)}}",
        }
        st.latex(lm_formulas[lm_model])
        lm_noise = st.slider("ruido σ", 0.0, 1.0, 0.2, step=0.05, key="lm_noise")
        lm_lambda = st.slider("λ inicial", 0.001, 10.0, 0.1, step=0.01, key="lm_lambda")
        lm_iters = st.slider("Iteraciones LM", 1, 50, 20, key="lm_iters")
        if st.button("Nueva muestra", key="lm_seed_btn"):
            st.session_state["lm_seed"] = np.random.randint(0, 100000)
        lm_seed = st.session_state.get("lm_seed", 42)

    rng_lm = np.random.default_rng(lm_seed)
    t_lm = np.linspace(0, 5, 30)
    if lm_model.startswith("Exponencial"):
        a_true, b_true = 3.0, 0.8
        y_true = a_true * np.exp(-b_true * t_lm)
        y_obs = y_true + lm_noise * rng_lm.normal(size=len(t_lm))
        def model_fn(t, theta): return theta[0] * np.exp(-theta[1] * t)
        def jac_fn(t, theta):
            J = np.zeros((len(t), 2))
            J[:, 0] = np.exp(-theta[1] * t)
            J[:, 1] = -theta[0] * t * np.exp(-theta[1] * t)
            return J
        theta0 = np.array([2.0, 0.4])
        labels = ["a", "b"]
        true_vals = [a_true, b_true]
    else:
        L_true, k_true, t0_true = 5.0, 1.5, 2.5
        y_true = L_true / (1 + np.exp(-k_true*(t_lm - t0_true)))
        y_obs = y_true + lm_noise * rng_lm.normal(size=len(t_lm))
        def model_fn(t, theta): return theta[0] / (1 + np.exp(-theta[1]*(t - theta[2])))
        def jac_fn(t, theta):
            L, k, t0 = theta
            sig = 1 / (1 + np.exp(-k*(t - t0)))
            J = np.zeros((len(t), 3))
            J[:, 0] = sig
            J[:, 1] = L * sig * (1-sig) * (t - t0)
            J[:, 2] = -L * sig * (1-sig) * k
            return J
        theta0 = np.array([3.0, 1.0, 2.0])
        labels = ["L", "k", "t₀"]
        true_vals = [L_true, k_true, t0_true]

    theta = theta0.copy()
    lam = lm_lambda
    losses_lm = []
    for _ in range(lm_iters):
        r = y_obs - model_fn(t_lm, theta)
        J_lm = jac_fn(t_lm, theta)
        loss_now = float(np.dot(r, r))
        losses_lm.append(loss_now)
        A = J_lm.T @ J_lm + lam * np.eye(len(theta))
        b_rhs = J_lm.T @ r
        try:
            delta = np.linalg.solve(A, b_rhs)
        except np.linalg.LinAlgError:
            break
        theta_new = theta + delta
        r_new = y_obs - model_fn(t_lm, theta_new)
        loss_new = float(np.dot(r_new, r_new))
        if loss_new < loss_now:
            theta = theta_new; lam = max(lam / 10, 1e-7)
        else:
            lam = min(lam * 10, 1e7)

    with col1:
        for lbl, tv, fitted in zip(labels, true_vals, theta):
            metric_grid([(f"Verdadero {lbl}", f"{tv:.3f}"), (f"Ajustado {lbl}", f"{fitted:.3f}")], columns=2)
        st.metric("final loss E", f"{losses_lm[-1]:.4f}" if losses_lm else "-")
        st.metric("λ final", f"{lam:.2e}")

    with col2:
        t_fine = np.linspace(0, 5, 200)
        fig_lm, axes_lm = plt.subplots(1, 2, figsize=(10.2, 3.6))
        axes_lm[0].scatter(t_lm, y_obs, color="#4C72B0", s=22, label="Datos", zorder=5)
        axes_lm[0].plot(t_fine, model_fn(t_fine, theta0), color="#CBD5E1", lw=1.5, ls="--", label="Inicio")
        axes_lm[0].plot(t_fine, model_fn(t_fine, theta), color="#DD8452", lw=2, label="LM ajustado")
        axes_lm[0].plot(t_fine, model_fn(t_fine, np.array(true_vals)), color="#059669", lw=1.5, ls=":", label="Verdadero")
        axes_lm[0].set_xlabel("t"); axes_lm[0].set_ylabel("y"); axes_lm[0].legend(fontsize=8)
        axes_lm[0].set_title("Ajuste del modelo")
        losses_lm_plot = np.maximum(np.asarray(losses_lm, dtype=float), np.finfo(float).tiny)
        axes_lm[1].semilogy(losses_lm_plot, "o-", color="#4C72B0", ms=3)
        axes_lm[1].set_xlabel("Iteración LM"); axes_lm[1].set_ylabel("$E(\\theta)$")
        axes_lm[1].set_title("Convergencia de la pérdida")
        plt.tight_layout()
        mia_pyplot(fig_lm); plt.close(fig_lm)
    how_to_read(
        "Izquierda: gris punteado = parámetros iniciales, naranja = ajuste LM final, verde = verdadero. "
        "Derecha: curva de pérdida en escala log. Convergencia rápida suele indicar que LM aceptó pasos tipo Gauss-Newton (λ pequeño)."
    )

    self_check_header()
    quiz(
        "¿Qué ocurre cuando $\\lambda \\to 0$ en Levenberg-Marquardt?",
        ["El algoritmo se convierte en descenso de gradiente puro",
         "El algoritmo se aproxima al método de Newton-Gauss",
         "El paso $\\Delta\\theta$ se hace cero"],
        1,
        "Con $\\lambda=0$: $(J^\\top J)\\Delta\\theta = J^\\top r$ — las ecuaciones normales de Gauss-Newton.",
        "La regularización $\\lambda I$ desaparece; queda la aproximación cuadrática de Newton.",
        key="lm_q1"
    )
    ai_bridge(
        "LM es la base de muchos ajustadores de modelos en **visión por computador** (calibración de cámaras, "
        "ajuste de poses), **física computacional** y **bioinformática**. En deep learning, "
        "**K-FAC** (Kronecker-Factored Approximate Curvature) usa una idea similar: aproximar la curvatura "
        "de la pérdida para dar pasos más inteligentes que SGD puro."
    )


# ==================================================================
#                   NAVEGACIÓN / SIDEBAR
# ==================================================================
st.sidebar.title("Probabilidad para IA")
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
    "17. Descenso de Gradiente, Newton y Backtracking": sec_gradient_backtracking,
    "18. Método de los Momentos (MoM)": sec_mom,
    "19. Cálculo Diferencial para IA": sec_calculo,
    "20. Convexidad": sec_convexidad,
    "21. Levenberg-Marquardt y NLS": sec_levenberg,
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
        "18. Método de los Momentos (MoM)",
    ],
    "Optimización": [
        "19. Cálculo Diferencial para IA",
        "20. Convexidad",
        "17. Descenso de Gradiente, Newton y Backtracking",
        "21. Levenberg-Marquardt y NLS",
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
                width="stretch",
            ):
                if label != st.session_state.choice:
                    st.session_state.choice = label
                    st.query_params["section"] = SECTION_TO_SLUG[label]
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Cada sección integra marco formal, ejemplos, exploración computacional y conexiones con aprendizaje automático.")

SECTION_LABELS = [label for labels in NAV_GROUPS.values() for label in labels]
SECTION_TO_GROUP = {label: group for group, labels in NAV_GROUPS.items() for label in labels}

SECTION_TAKEAWAYS = {
    "1. Espacios y Axiomas de Kolmogorov": (
        "Un probability model necesita outcomes, events y una regla de probabilidad.",
        "Para verificar que los cálculos probabilísticos posteriores sean coherentes.",
        "No tratar las reglas de probabilidad como intuición opcional; son restricciones.",
    ),
    "2. Regla de Laplace y Combinatoria": (
        "El counting depende de order y replacement antes de elegir cualquier fórmula.",
        "Cuando los outcomes son equiprobables y el sample space puede contarse.",
        "No usar combinations ni permutations antes de decidir qué cuenta como distinto.",
    ),
    "3. Probabilidad Condicional": (
        "La conditional probability cambia el universo de referencia.",
        "Cuando la información B ya es conocida antes de preguntar por el evento A.",
        "No leer una conditional probability alta como causalidad por sí sola.",
    ),
    "4. Teorema de Bayes": (
        "Bayes actualiza un prior con evidencia mediante likelihoods.",
        "Para diagnosis, filtering, classification y razonamiento posterior.",
        "No confundir sensitivity con posterior probability.",
    ),
    "5. Clasificador Naïve Bayes": (
        "Naive Bayes combina priors y feature likelihoods bajo una conditional-independence assumption.",
        "Como clasificador probabilístico transparente y como baseline.",
        "No asumir que la independence assumption es literalmente cierta en datos reales.",
    ),
    "6. VA: PMF, PDF y CDF": (
        "La PMF cuenta masa, la PDF describe densidad y la CDF acumula probabilidad.",
        "Para calcular probabilidades de intervalo mediante diferencias de CDF.",
        "No leer la altura de una density continua como probabilidad.",
    ),
    "7. Catálogo de Distribuciones": (
        "Cada distribution codifica una historia generadora de datos distinta.",
        "Para alinear model assumptions con el tipo de outcome.",
        "No elegir una distribution solo porque su curva parece familiar.",
    ),
    "8. MLE y Entropía Cruzada": (
        "Maximum likelihood elige parámetros que vuelven más plausibles los datos observados.",
        "Para conectar statistical estimation con loss minimization.",
        "No interpretar una likelihood alta como prueba de que la familia de modelos es correcta.",
    ),
    "9. Esperanza, Varianza y Jensen": (
        "Expectation resume comportamiento promedio, variance resume dispersión y Jensen captura efectos de curvatura.",
        "Para razonar sobre uncertainty, risk y transformaciones.",
        "No asumir que transformar un promedio equivale a promediar una transformación.",
    ),
    "10. FGM, Covarianza y Correlación": (
        "Generating functions y covariance revelan moments y linear dependence.",
        "Usar correlation para asociación lineal y covariance para variación conjunta.",
        "No concluir independence a partir de zero correlation en general.",
    ),
    "11. Gaussiana Multivariada y PCA": (
        "La covariance geometry determina elipses, principal directions y proyecciones de baja dimensión.",
        "Usar PCA para resumir variación e inspeccionar effective dimension.",
        "No tratar las PCA components como variables causales.",
    ),
    "12. Maldición de la Dimensionalidad": (
        "High dimension vuelve escasos los local neighborhoods y menos discriminativas las distancias.",
        "Para motivar embeddings, feature selection y dimensionality reduction.",
        "No concluir que high-dimensional learning es imposible cuando los datos tienen estructura.",
    ),
    "13. Desigualdades de Concentración": (
        "Las concentration inequalities entregan upper bounds garantizados para grandes desviaciones.",
        "Cuando las probabilidades exactas son desconocidas pero hay assumptions disponibles.",
        "No leer una bound como predicción exacta.",
    ),
    "14. Muestras y Testeo Agrupado": (
        "El sampling design cambia costo, incertidumbre e interpretación.",
        "Usar pooled testing cuando los positivos son raros y los tests pueden combinarse con sentido.",
        "No asumir que pooling ayuda cuando la prevalence es alta.",
    ),
    "15. Leyes Límite: LLN y CLT": (
        "LLN explica estabilidad de promedios; CLT explica la forma de errores escalados.",
        "Para justificar estimation, uncertainty y confidence intervals bajo assumptions.",
        "No asumir que CLT arregla dependencia, varianza infinita o muestras diminutas.",
    ),
    "16. Algoritmos Aleatorizados": (
        "Randomness puede proteger algoritmos contra estructura adversarial y reducir expected cost.",
        "Usar randomized analysis para comparar typical cost con worst-case cost.",
        "No confundir expected performance con garantía para cada corrida individual.",
    ),
    "17. Descenso de Gradiente, Newton y Backtracking": (
        "Los optimization methods difieren en cómo eligen direction y step size.",
        "Usar gradients, curvature y line search para razonar sobre convergence behavior.",
        "No asumir que un paso más grande siempre es más rápido o más seguro.",
    ),
    "18. Método de los Momentos (MoM)": (
        "MoM estima parámetros igualando theoretical moments y empirical moments.",
        "Para estimación transparente cuando los moments identifican los parámetros.",
        "No asumir consistency sin identification y regularity conditions.",
    ),
    "19. Cálculo Diferencial para IA": (
        "Derivatives describen sensibilidad local; gradients apuntan hacia el aumento más rápido.",
        "Para entender learning rules, optimization y local approximation.",
        "No confundir información local de derivatives con comportamiento global.",
    ),
    "20. Convexidad": (
        "Convexity elimina false local minima bajo los assumptions correctos.",
        "Para saber cuándo las optimization guarantees son plausibles.",
        "No asumir que toda ML loss es convexa o tiene minimizador único.",
    ),
    "21. Levenberg-Marquardt y NLS": (
        "LM interpola entre pasos estables tipo gradient y pasos rápidos tipo Gauss-Newton.",
        "Para nonlinear least squares cuando hay Jacobians disponibles.",
        "No esperar que arregle automáticamente mala elección de modelo o inicialización imposible.",
    ),
}

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
        <div class="mia-hero-kicker">Probabilidad para IA · {_current_group} · Sección {_current_idx + 1} de {len(SECTION_LABELS)}</div>
        <h1>Probabilidad para Inteligencia Artificial</h1>
        <p>Desarrollo conceptual, derivaciones paso a paso, laboratorios interactivos y lectura rigurosa de los modelos probabilísticos usados en IA.</p>
        <div class="mia-progress-track"><div class="mia-progress-fill" style="width:{_progress_pct}%;"></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

SECTIONS[st.session_state.choice]()
if st.session_state.choice in SECTION_TAKEAWAYS:
    minimum_takeaway(*SECTION_TAKEAWAYS[st.session_state.choice])

st.markdown("<div class='mia-prevnext-spacer'></div>", unsafe_allow_html=True)
c_prev, c_info, c_next = st.columns([5, 2, 5])
with c_prev:
    if _current_idx > 0:
        prev_label = SECTION_LABELS[_current_idx - 1]
        if st.button(f"← {prev_label}", width="stretch", key=f"prevbtn_{_current_idx}"):
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
        if st.button(f"{next_label} →", width="stretch", key=f"nextbtn_{_current_idx}"):
            _goto_section(next_label)
    else:
        st.markdown("&nbsp;", unsafe_allow_html=True)
