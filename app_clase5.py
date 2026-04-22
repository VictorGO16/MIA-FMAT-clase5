import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import time
from scipy import stats
from scipy.special import comb, gammaln
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

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
    "font.size": 10,
})

# ==================================================================
#                    UTILIDADES PEDAGÓGICAS
# ==================================================================
def motivation(text):
    st.info(f"🧭 **Motivación** — {text}")

def prerequisites_box(prereqs_md):
    with st.expander("📚 Prerrequisitos — ábrelo si algún término no te es familiar"):
        st.markdown(prereqs_md)

def how_to_read(text):
    st.caption(f"📖 **Cómo leer este gráfico** — {text}")

def ai_bridge(text):
    st.markdown("#### 🤖 Puente con Inteligencia Artificial")
    st.markdown(text)

def worked_example(title):
    st.markdown(f"### 🔬 Ejemplo resuelto paso a paso — {title}")

def interactive_header(title):
    st.markdown(f"### 🎛️ Explora con tus propios valores — {title}")

def self_check_header():
    st.markdown("### ✅ Autoevaluación")
    st.caption("Responde antes de mirar arriba. El feedback te explica el porqué.")

def quiz(question, options, correct_idx, feedback_ok, feedback_wrong, key):
    st.markdown(f"**{question}**")
    ans = st.radio("Opciones", options, key=key, index=None, label_visibility="collapsed")
    if ans is None:
        st.caption("_Selecciona una opción para ver el feedback._")
        return
    if options.index(ans) == correct_idx:
        st.success(f"✅ Correcto. {feedback_ok}")
    else:
        st.error(f"❌ No es esa. {feedback_wrong}")

def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")

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

    st.markdown("### 🧱 Construcción formal")
    st.markdown(
        "Un **espacio de probabilidad** es una terna $(\\Omega, \\mathcal{A}, P)$ donde:\n"
        "- $\\Omega$ (Omega) es el **espacio muestral**: todos los resultados posibles del experimento.\n"
        "- $\\mathcal{A}$ es una **σ-álgebra**: la colección de subconjuntos de $\\Omega$ a los que "
        "podemos asignar probabilidad (llamados **eventos**).\n"
        "- $P: \\mathcal{A} \\to [0,1]$ es la función de probabilidad."
    )
    st.markdown("**Los 3 axiomas de Kolmogorov:**")
    st.latex(r"\textbf{A1 (No negatividad):}\quad P(A) \geq 0 \ \ \forall A \in \mathcal{A}")
    st.latex(r"\textbf{A2 (Normalización):}\quad P(\Omega) = 1")
    st.latex(r"\textbf{A3 (Aditividad numerable):}\quad P\Big(\bigcup_{i=1}^{\infty} A_i\Big) = \sum_{i=1}^{\infty} P(A_i) \quad \text{si los } A_i \text{ son disjuntos}")
    st.markdown("**Consecuencias inmediatas** (todo lo demás se deduce de estos 3):")
    st.latex(r"P(\emptyset) = 0, \quad P(A^c) = 1 - P(A), \quad A \subseteq B \Rightarrow P(A) \le P(B)")
    st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B) \quad \text{(inclusión-exclusión)}")

    worked_example("dado justo de 6 caras")
    st.markdown(
        "- $\\Omega = \\{1,2,3,4,5,6\\}$, todos equiprobables.\n"
        "- Evento **Par** = $\\{2,4,6\\}$ → $P(\\text{Par}) = 3/6 = 0.5$.\n"
        "- Evento **Mayor que 4** = $\\{5,6\\}$ → $P = 2/6 = 1/3$.\n"
        "- $P(\\text{Par} \\cup \\text{Mayor4}) = P(\\text{Par}) + P(\\text{Mayor4}) - P(\\text{Par}\\cap\\text{Mayor4})$\n"
        "  $= 1/2 + 1/3 - 1/6 = 2/3$."
    )

    interactive_header("Frecuencia relativa vs probabilidad teórica")
    st.caption("Lanza una moneda o dado muchas veces. La frecuencia empírica converge a la P teórica (LLN, que veremos en la sección 15).")
    col1, col2 = st.columns([1, 2])
    with col1:
        exp_type = st.radio("Experimento", ["Moneda (2 resultados)", "Dado (6 resultados)"], key="kolm_exp")
        n_trials = st.slider("Número de lanzamientos", 10, 5000, 500, step=10, key="kolm_n")
        if st.button("🎲 Simular", key="kolm_btn"):
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
    st.markdown("### 🧱 Construcción")
    st.latex(r"P(A) = \frac{|A|}{|\Omega|} \quad \text{(Regla de Laplace, sólo si todos los resultados son equiprobables)}")
    st.markdown(
        "| Con orden / sin orden | Con reemplazo | Sin reemplazo |\n"
        "|---|---|---|\n"
        "| **Con orden** | $n^k$ | $\\dfrac{n!}{(n-k)!}$ |\n"
        "| **Sin orden** | $\\binom{n+k-1}{k}$ | $\\binom{n}{k}$ |\n"
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
        def birthday_prob(n):
            if n > 365: return 1.0
            return 1 - np.exp(sum(np.log(1 - i/365) for i in range(n)))
        p = birthday_prob(n_people)
        st.metric("P(coincidencia)", f"{p:.4f}", f"{p*100:.2f}%")
    with col2:
        xs = np.arange(2, 101)
        ys = [birthday_prob(i) for i in xs]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(xs, ys, color="#4C72B0", lw=2)
        ax.axvline(n_people, color="#DD8452", ls="--", alpha=0.7)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("n personas"); ax.set_ylabel("P(al menos 2 comparten cumpleaños)")
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
    st.markdown("### 🧱 Construcción")
    st.latex(r"P(A\mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0")
    st.markdown("Interpretación: *de todos los casos donde $B$ ocurrió, qué fracción también tiene $A$*.")
    st.markdown("**Regla del producto** (despejando):")
    st.latex(r"P(A \cap B) = P(A\mid B)\,P(B) = P(B\mid A)\,P(A)")
    st.markdown("**Independencia**: $A$ y $B$ son independientes si saber uno no cambia la probabilidad del otro:")
    st.latex(r"P(A\mid B) = P(A) \iff P(A \cap B) = P(A)\,P(B)")

    worked_example("Monty Hall")
    st.markdown(
        "Tres puertas, tras una hay un auto, tras las otras dos una cabra. Eliges una puerta; el presentador "
        "(que sabe dónde está el auto) abre otra con una cabra. ¿Conviene **cambiar** tu elección?\n\n"
        "Razonamiento condicional: al elegir primero, $P(\\text{auto}) = 1/3$. Esa probabilidad no cambia "
        "cuando abren otra puerta. La puerta restante concentra el $2/3$ complementario.\n\n"
        "**Conclusión**: cambiar gana con probabilidad **2/3**; quedarse con **1/3**."
    )
    interactive_header("Simulador Monty Hall")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_doors = st.slider("Número de puertas", 3, 20, 3, key="mh_doors")
        n_opened = st.slider("Puertas que abre el presentador", 1, max(1, n_doors - 2), 1, key="mh_open")
        n_sim = st.slider("Simulaciones", 500, 20000, 5000, step=500, key="mh_sim")
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
        st.metric("P(ganar quedándose)", f"{p_stay:.3f}")
        st.metric("P(ganar cambiando)", f"{p_switch:.3f}")
        st.latex(rf"P(\text{{switch}}) = \frac{{n-1}}{{n(n-1-k)}}\cdot(n-1-k) = \frac{{n-1}}{{n}}\cdot\frac{{1}}{{n-1-k}}\cdot(n-1-k)")
        theoretical_switch = (n_doors - 1) / (n_doors * (n_doors - 1 - n_opened)) * (n_doors - 1 - n_opened)
        st.caption(f"Teórico P(switch) con n={n_doors}, k={n_opened}: ≈ {theoretical_switch:.3f}")
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Quedarse", "Cambiar"], [p_stay, p_switch], color=["#DD8452", "#4C72B0"])
        ax.set_ylim(0, 1); ax.set_ylabel("P(ganar)")
        for i, v in enumerate([p_stay, p_switch]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        st.pyplot(fig); plt.close(fig)
    how_to_read("Eje y: proporción de simulaciones ganadoras. La barra 'Cambiar' debería estar consistentemente más alta.")

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
    st.markdown("### 🧱 Construcción")
    st.latex(r"P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)} = \frac{P(E\mid H)\,P(H)}{\sum_i P(E\mid H_i)\,P(H_i)}")
    st.markdown(
        "- $P(H)$: **prior** (creencia antes de ver la evidencia).\n"
        "- $P(E\\mid H)$: **verosimilitud** (qué tan bien explica $H$ lo observado).\n"
        "- $P(E)$: **evidencia** (constante de normalización).\n"
        "- $P(H\\mid E)$: **posterior** (creencia actualizada)."
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

    interactive_header("Calculadora bayesiana interactiva")
    col1, col2 = st.columns([1, 2])
    with col1:
        prior = st.slider("Prevalencia P(D)", 0.001, 0.5, 0.01, step=0.001, format="%.3f", key="bay_prior")
        sens = st.slider("Sensibilidad P(+|D)", 0.5, 1.0, 0.99, step=0.01, key="bay_sens")
        spec = st.slider("Especificidad P(-|¬D)", 0.5, 1.0, 0.99, step=0.01, key="bay_spec")
        fpr = 1 - spec
        num = sens * prior; den = num + fpr * (1 - prior)
        post = num / den if den > 0 else 0
        st.metric("P(D | +)", f"{post:.4f}")
        st.metric("P(D | −)", f"{(1-sens)*prior / ((1-sens)*prior + spec*(1-prior)):.4f}")
    with col2:
        priors = np.linspace(0.001, 0.5, 200)
        posts = sens * priors / (sens * priors + fpr * (1 - priors))
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.plot(priors, posts, color="#4C72B0", lw=2)
        ax.axvline(prior, color="#DD8452", ls="--", label=f"prior={prior:.3f}")
        ax.axhline(post, color="#55A868", ls=":", label=f"posterior={post:.3f}")
        ax.set_xlabel("Prevalencia P(D)"); ax.set_ylabel("P(D | +)")
        ax.set_xscale("log"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    how_to_read("Eje x (log): prevalencia. Eje y: posterior tras test positivo. Con baja prevalencia y test imperfecto, el posterior cae rápido.")

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
    st.markdown("### 🧱 Construcción")
    st.latex(r"P(y\mid x) = \frac{P(y)\,P(x\mid y)}{P(x)} \propto P(y)\prod_{j=1}^d P(x_j\mid y)")
    st.markdown("**Decisión MAP**:")
    st.latex(r"\hat y = \arg\max_y\ P(y)\prod_{j=1}^d P(x_j\mid y)")
    st.markdown("**Truco del logaritmo** (evita underflow, convierte producto en suma):")
    st.latex(r"\hat y = \arg\max_y\ \log P(y) + \sum_{j=1}^d \log P(x_j\mid y)")
    st.markdown(
        "En **Gaussian NB**, cada $P(x_j\\mid y)$ es una normal con parámetros $\\mu_{y,j}, \\sigma_{y,j}^2$ "
        "estimados por MLE desde los datos de la clase $y$."
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
    def run_wine_nb(test_size=0.3, seed=42):
        data = load_wine()
        X, y = data.data, data.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        # sklearn
        skn = GaussianNB().fit(X_tr, y_tr)
        acc_sk = accuracy_score(y_te, skn.predict(X_te))
        # numpy from scratch
        classes = np.unique(y_tr)
        mu = np.stack([X_tr[y_tr==c].mean(axis=0) for c in classes])
        var = np.stack([X_tr[y_tr==c].var(axis=0) + 1e-9 for c in classes])
        priors = np.array([(y_tr==c).mean() for c in classes])
        log_prior = np.log(priors)
        # log likelihood per class
        def predict_scratch(X):
            n, d = X.shape
            logprobs = np.zeros((n, len(classes)))
            for ci in range(len(classes)):
                diff = X - mu[ci]
                ll = -0.5 * np.sum(np.log(2*np.pi*var[ci]) + diff**2 / var[ci], axis=1)
                logprobs[:, ci] = log_prior[ci] + ll
            return classes[np.argmax(logprobs, axis=1)]
        acc_np = accuracy_score(y_te, predict_scratch(X_te))
        return acc_sk, acc_np, data.feature_names, data.target_names

    acc_sk, acc_np, fnames, tnames = run_wine_nb()
    c1, c2 = st.columns(2)
    c1.metric("Accuracy sklearn", f"{acc_sk:.3f}")
    c2.metric("Accuracy numpy desde cero", f"{acc_np:.3f}")
    st.caption(f"Dataset: Wine ({len(fnames)} features, {len(tnames)} clases: {', '.join(tnames)}).")

    interactive_header("Visualizar frontera con 2 features (Wine)")
    data = load_wine()
    col1, col2 = st.columns([1, 2])
    with col1:
        feat_x = st.selectbox("Feature X", options=list(range(len(data.feature_names))),
                              format_func=lambda i: data.feature_names[i], index=0, key="nb_fx")
        feat_y = st.selectbox("Feature Y", options=list(range(len(data.feature_names))),
                              format_func=lambda i: data.feature_names[i], index=6, key="nb_fy")
    X2 = data.data[:, [feat_x, feat_y]]; y2 = data.target
    model = GaussianNB().fit(X2, y2)
    x_min, x_max = X2[:, 0].min()-0.5, X2[:, 0].max()+0.5
    y_min, y_max = X2[:, 1].min()-0.5, X2[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
        for c in np.unique(y2):
            ax.scatter(X2[y2==c, 0], X2[y2==c, 1], label=data.target_names[c], s=22, edgecolor="k", alpha=0.8)
        ax.set_xlabel(data.feature_names[feat_x]); ax.set_ylabel(data.feature_names[feat_y])
        ax.legend()
        st.pyplot(fig); plt.close(fig)
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
    st.markdown("### 🧱 Construcción")
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

    interactive_header("PMF/PDF vs CDF lado a lado")
    col1, col2 = st.columns([1, 3])
    with col1:
        dist_type = st.radio("Distribución", ["Binomial (discreta)", "Normal (continua)", "Exponencial (continua)"], key="cdf_kind")
    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))
        if dist_type.startswith("Binomial"):
            n = 20; p = 0.4
            ks = np.arange(0, n+1)
            pmf = stats.binom.pmf(ks, n, p); cdf = stats.binom.cdf(ks, n, p)
            ax1.bar(ks, pmf, color="#4C72B0"); ax1.set_title("PMF Binomial(20, 0.4)"); ax1.set_xlabel("k")
            ax2.step(ks, cdf, where="post", color="#DD8452"); ax2.set_title("CDF"); ax2.set_xlabel("k")
        elif dist_type.startswith("Normal"):
            xs = np.linspace(-4, 4, 400)
            pdf = stats.norm.pdf(xs); cdf = stats.norm.cdf(xs)
            ax1.plot(xs, pdf, color="#4C72B0"); ax1.set_title("PDF N(0,1)")
            ax2.plot(xs, cdf, color="#DD8452"); ax2.set_title("CDF Φ(x)")
        else:
            xs = np.linspace(0, 5, 400)
            pdf = stats.expon.pdf(xs); cdf = stats.expon.cdf(xs)
            ax1.plot(xs, pdf, color="#4C72B0"); ax1.set_title("PDF Exp(1)")
            ax2.plot(xs, cdf, color="#DD8452"); ax2.set_title("CDF 1 − e^(−x)")
        for ax in (ax1, ax2):
            ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    how_to_read("Izquierda: masa/densidad puntual. Derecha: probabilidad acumulada hasta ese punto. La CDF siempre empieza en 0 y termina en 1, nunca baja.")

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
    st.markdown("### 🧱 Construcción")
    st.latex(r"\mathcal{L}(\theta) = \prod_{i=1}^n p(x_i \mid \theta) \quad \text{(verosimilitud)}")
    st.latex(r"\ell(\theta) = \log \mathcal{L}(\theta) = \sum_i \log p(x_i\mid\theta) \quad \text{(log-verosimilitud)}")
    st.latex(r"\hat\theta_{MLE} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \big[-\ell(\theta)\big] \quad \text{(NLL)}")

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
    st.markdown("### 🧱 Construcción")
    st.latex(r"E[X] = \sum_k k\,p_X(k) \quad \text{(discreta)}, \quad E[X] = \int x\,f_X(x)\,dx \quad \text{(continua)}")
    st.markdown("**Linealidad** (la propiedad más útil de $E$):")
    st.latex(r"E[aX+bY+c] = aE[X]+bE[Y]+c \quad \textbf{aunque } X,Y \text{ no sean independientes}")
    st.latex(r"\text{Var}(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2 \ge 0")
    st.latex(r"\text{Var}(aX+b) = a^2\text{Var}(X), \quad \text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y) \text{ si } X\perp Y")

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

    st.markdown("### 📐 Desigualdad de Jensen")
    st.markdown("Si $f$ es **convexa**:")
    st.latex(r"E[f(X)] \geq f(E[X])")
    st.markdown("Si $f$ es **cóncava**:")
    st.latex(r"E[f(X)] \leq f(E[X])")
    st.markdown("Igualdad si y sólo si $f$ es lineal en el rango de $X$, o $X$ es constante.")

    interactive_header("Visualización de Jensen")
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
    st.markdown("### 🧱 Construcción")
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

    st.markdown("### 🔗 Covarianza y correlación")
    st.latex(r"\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]")
    st.latex(r"\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1, 1]")
    st.markdown("- $\\rho=0$: sin relación lineal (NO implica independencia salvo en gaussianas).")
    st.markdown("- $\\text{Var}(X+Y) = \\text{Var}(X) + \\text{Var}(Y) + 2\\text{Cov}(X,Y)$.")

    worked_example("Cov(X, X+Y) con X, Y independientes de varianza 1")
    st.latex(r"\text{Cov}(X, X+Y) = \text{Cov}(X,X) + \text{Cov}(X,Y) = \text{Var}(X) + 0 = 1")

    interactive_header("Scatter de datos con correlación ajustable")
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
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        emp_rho = np.corrcoef(X.T)[0, 1]
        ax.set_title(f"ρ teórica={rho:.2f}, ρ empírica={emp_rho:.2f}")
        st.pyplot(fig); plt.close(fig)

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
    st.markdown("### 🧱 Gaussiana multivariada")
    st.latex(r"f_{\mathbf X}(\mathbf x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf x-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\right)")
    st.markdown("Las curvas de nivel (*isoprobabilidad*) son **elipses** con ejes = autovectores de $\\Sigma$ y longitudes proporcionales a $\\sqrt{\\lambda_i}$.")

    interactive_header("Gaussiana bivariada — heatmap con covarianza ajustable")
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

    st.markdown("### 📐 PCA vía SVD")
    st.markdown(
        "Dado un dataset $X \\in \\mathbb{R}^{N\\times d}$ **centrado**, computamos SVD:"
    )
    st.latex(r"X = U\,S\,V^T")
    st.markdown("Las columnas de $V$ son las **direcciones principales**, y la covarianza muestral se factoriza así:")
    st.latex(r"\hat\Sigma = \frac{X^T X}{N-1} = V\,\frac{S^2}{N-1}\,V^T")
    st.markdown("→ autovalores de $\\hat\\Sigma$ = $S_i^2/(N-1)$. Los primeros $k$ autovectores forman el mejor subespacio $k$-dim en sentido de mínimo error cuadrático.")

    interactive_header("PCA sobre una nube 2D rotada")
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
    st.markdown("### 🧱 Construcción")
    st.markdown("**Razón volumen esfera / volumen cubo**:")
    st.latex(r"\frac{V_d(r)}{(2r)^d} = \frac{\pi^{d/2}}{2^d\,\Gamma(d/2+1)}")
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

    st.markdown("### 📐 Consecuencias para ML")
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
    st.markdown("### 🧱 Construcción")
    st.markdown("**Markov** (para $X\\ge 0$):")
    st.latex(r"P(X \geq a) \leq \frac{E[X]}{a}")
    st.markdown("**Chebyshev** (a partir de Markov aplicado a $(X-\\mu)^2$):")
    st.latex(r"P(|X-\mu| \geq k\sigma) \leq \frac{1}{k^2}")
    st.markdown("**Hoeffding** (si cada $X_i \\in [a,b]$, i.i.d.):")
    st.latex(r"P\!\left(\big|\bar X_n - \mu\big| \geq t\right) \leq 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)")

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
    st.markdown("### 🧱 Estimadores muestrales")
    st.latex(r"\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i, \quad E[\bar X_n]=\mu,\quad \text{Var}[\bar X_n] = \frac{\sigma^2}{n}")
    st.latex(r"S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X_n)^2 \quad \text{(insesgado de } \sigma^2\text{)}")
    st.caption("El $n-1$ (corrección de Bessel) viene de que $\\bar X$ ya 'consumió' un grado de libertad.")

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
    st.markdown("### 🧱 Ley de los Grandes Números")
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

    st.markdown("### 🧱 Teorema Central del Límite (CLT)")
    st.latex(r"\frac{\bar X_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal N(0,1)")
    st.markdown(
        "Para $n$ grande, **sin importar la distribución de $X_i$** (con varianza finita), $\\bar X_n$ es "
        "aproximadamente $\\mathcal N(\\mu, \\sigma^2/n)$."
    )
    st.markdown("**CLT multivariado**:")
    st.latex(r"\sqrt{n}(\bar{\mathbf X}_n - \boldsymbol\mu) \xrightarrow{d} \mathcal N_d(\mathbf 0, \Sigma)")

    worked_example("Chebyshev vs CLT para dimensionar muestras")
    st.markdown(
        "$X_i\\in[0,1]$, queremos $P(|\\bar X_n - \\mu|>0.05) \\leq 0.05$.\n\n"
        "**Chebyshev**: $\\sigma^2/(n\\epsilon^2) \\leq 0.05 \\Rightarrow n \\geq 0.25/(0.0025\\cdot 0.05) = 2000$. "
        "Relajando a $\\sigma^2 \\leq 0.25$ genérico da $n\\geq 2000$; con $\\sigma^2 = 0.0025$ concreto (Bernoulli p=0.5 → $\\sigma^2=0.25$): n≥400.\n\n"
        "**CLT** (aproximación): $z = 0.05\\sqrt n/\\sigma$; con $\\sigma=0.5$ necesitamos $z\\approx 1.96$ y $P(|Z|>1.96)\\approx 0.05 \\Rightarrow \\sqrt n \\geq 19.6 \\Rightarrow n\\approx 27$.\n\n"
        "→ CLT mucho más eficiente cuando aplica (n grande, varianza finita)."
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
    st.markdown("### 🧱 Quicksort aleatorizado")
    st.markdown(
        "Pseudocódigo: elige un **pivote al azar**, particiona, recursiona izquierda y derecha. "
        "Sea $X_{ij}=1$ si los elementos $i$-ésimo y $j$-ésimo del array ordenado fueron comparados durante el algoritmo. "
        "Se compara sólo si uno es pivote *antes* de separarlos."
    )
    st.latex(r"P(X_{ij}=1) = \frac{2}{j-i+1}")
    st.markdown("Por linealidad de esperanza:")
    st.latex(r"E[\text{comp}] = \sum_{i<j}\frac{2}{j-i+1} = O(n\log n)")

    worked_example("Quickselect")
    st.markdown(
        "Pregunta: encontrar el $k$-ésimo más pequeño sin ordenar todo. Quicksort recursa en ambos lados; "
        "**Quickselect** sólo recursa en el lado que contiene el $k$. El análisis análogo da $E[\\#\\text{comp}]=O(n)$."
    )

    interactive_header("Benchmark: Quicksort determinista vs aleatorizado")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_bench = st.slider("tamaño del array", 100, 3000, 1000, step=100, key="qs_n")
        adv = st.radio("Entrada", ["Aleatoria", "Ya ordenada (worst-case deterministic)"], key="qs_adv")
    rng = np.random.default_rng(5)
    if adv.startswith("Aleatoria"):
        arr = rng.permutation(n_bench)
    else:
        arr = np.arange(n_bench)

    def quicksort_det(a, comps):
        if len(a) <= 1:
            return a
        pivot = a[0]
        comps[0] += len(a) - 1
        left = a[1:][a[1:] < pivot]; right = a[1:][a[1:] >= pivot]
        return np.concatenate([quicksort_det(left, comps), [pivot], quicksort_det(right, comps)])

    def quicksort_rand(a, comps, rng):
        if len(a) <= 1:
            return a
        idx = rng.integers(len(a))
        pivot = a[idx]; rest = np.delete(a, idx)
        comps[0] += len(rest)
        left = rest[rest < pivot]; right = rest[rest >= pivot]
        return np.concatenate([quicksort_rand(left, comps, rng), [pivot], quicksort_rand(right, comps, rng)])

    import sys; sys.setrecursionlimit(10000)
    c_det = [0]; _ = quicksort_det(arr.copy(), c_det)
    c_rand = [0]; _ = quicksort_rand(arr.copy(), c_rand, rng)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(["Determinista", "Aleatorizado"], [c_det[0], c_rand[0]], color=["#DD8452", "#4C72B0"])
        ax.set_ylabel("# comparaciones")
        for i, v in enumerate([c_det[0], c_rand[0]]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        ax.set_title(f"n={n_bench}, n·log₂n={n_bench*math.log2(n_bench):.0f}, n²={n_bench**2:,}")
        st.pyplot(fig); plt.close(fig)
    how_to_read("Con entrada ordenada, determinista explota a $O(n^2)$. Aleatorizado se mantiene cerca de $n\\log n$ (entre las líneas de referencia en el título).")

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
st.sidebar.caption("App didáctica con construcción conceptual, ejemplos resueltos, experimentación interactiva, autoevaluación y puentes con IA.")

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

st.sidebar.markdown("**Fundamentos**")
st.sidebar.caption("Secciones 1–2")
st.sidebar.markdown("**Condicional y Bayes**")
st.sidebar.caption("Secciones 3–5")
st.sidebar.markdown("**VA y distribuciones**")
st.sidebar.caption("Secciones 6–8")
st.sidebar.markdown("**Momentos y estructura**")
st.sidebar.caption("Secciones 9–11")
st.sidebar.markdown("**Alta dimensión y concentración**")
st.sidebar.caption("Secciones 12–13")
st.sidebar.markdown("**Muestras y leyes límite**")
st.sidebar.caption("Secciones 14–15")
st.sidebar.markdown("**Aplicaciones algorítmicas**")
st.sidebar.caption("Sección 16")

choice = st.sidebar.radio("Ir a sección:", list(SECTIONS.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Cada sección sigue el mismo patrón: Motivación → Prerrequisitos → Construcción → Ejemplo resuelto → Interactivo → Autoevaluación → Puente con IA.")

st.title("Probabilidad para Inteligencia Artificial")
st.caption("Curso MIA IMT3850 — Pontificia Universidad Católica de Chile")

SECTIONS[choice]()
