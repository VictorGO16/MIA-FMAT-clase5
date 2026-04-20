import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
from scipy import stats
from scipy.special import comb
from sklearn.datasets import load_wine

# ==================================================================
#                       CONFIGURACIÓN GLOBAL
# ==================================================================
st.set_page_config(
    page_title="Clase 5 - Probabilidad para IA",
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

def quiz(question, options, correct_idx, feedback_ok, feedback_wrong, key):
    """Pregunta de autoevaluación con feedback inmediato."""
    st.markdown(f"**{question}**")
    ans = st.radio(
        "Opciones",
        options,
        key=key,
        index=None,
        label_visibility="collapsed",
    )
    if ans is None:
        st.caption("_Selecciona una opción para ver el feedback._")
        return
    if options.index(ans) == correct_idx:
        st.success(f"✅ Correcto. {feedback_ok}")
    else:
        st.error(f"❌ No es esa. {feedback_wrong}")


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
    st.caption("Responde sin mirar arriba. Si fallas, el feedback te dice dónde está la confusión típica.")


# ==================================================================
#                   CABECERA Y NAVEGACIÓN
# ==================================================================

st.title("Fundamentos de Probabilidad e Inferencia para IA")
st.caption("Clase 5 — Curso MIA IMT3850")

st.markdown(
    "Esta aplicación es **autocontenida y progresiva**. No necesitas saber "
    "probabilidad ni matemática avanzada antes de entrar. Cada sección sigue la misma "
    "estructura: **motivación → prerrequisitos opcionales → construcción de la idea → "
    "ejemplo resuelto paso a paso → exploración interactiva → autoevaluación → conexión con IA**. "
    "Si algún término te suena raro, abre el bloque de prerrequisitos de esa sección."
)

menu = [
    "1. Espacios y Axiomas de Kolmogorov",
    "2. Regla de Laplace y Combinatoria",
    "3. Probabilidad Condicional",
    "4. Teorema de Bayes",
    "5. Clasificador Naïve Bayes",
    "6. Catálogo de Distribuciones",
    "7. Máxima Verosimilitud (MLE)",
    "8. Valor Esperado y Varianza",
    "9. Momentos (FGM) y Covarianza",
]
choice = st.sidebar.radio("📚 Contenido de la clase", menu)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Sugerencia.** Lee de forma lineal la primera vez. "
    "Cada sección es autosuficiente, así que puedes volver a consultar cualquiera "
    "sin necesidad de repasar las anteriores."
)
st.sidebar.caption("MIA — IMT3850")


# ==================================================================
# 1. ESPACIOS DE PROBABILIDAD Y AXIOMAS DE KOLMOGOROV
# ==================================================================
if choice == menu[0]:
    st.header("1. Espacios de Probabilidad y Axiomas de Kolmogorov")

    motivation(
        "Cuando decimos 'la probabilidad de que llueva mañana es 30%' o "
        "'la probabilidad de sacar un as es 1/13', estamos usando la palabra "
        "**probabilidad** con una confianza sospechosa: ¿qué objeto matemático "
        "es exactamente? ¿Qué reglas debe obedecer para que los cálculos no se "
        "contradigan? La respuesta, desde 1933, es un conjunto de tres reglas "
        "llamadas **axiomas de Kolmogorov**. Esta sección los construye desde cero."
    )

    prerequisites_box(r"""
- **Conjunto**: una colección de objetos distintos. Por ejemplo $\{1, 2, 3\}$ o $\{\text{cara}, \text{sello}\}$.
- **Elemento**: cada objeto de un conjunto. El número $2$ es un elemento del conjunto $\{1,2,3\}$.
- **Subconjunto**: un conjunto cuyos elementos están todos en otro conjunto. $\{2,4\}$ es subconjunto de $\{1,2,3,4\}$.
- **Unión** $A \cup B$: el conjunto de elementos que están en $A$, en $B$, o en ambos.
- **Intersección** $A \cap B$: elementos que están a la vez en $A$ y en $B$.
- **Complemento** $A^c$: elementos que **no** están en $A$ (pero sí en el universo de referencia).
- **Eventos disjuntos**: dos eventos que no pueden ocurrir a la vez, es decir $A \cap B = \varnothing$.

Con esto basta. Si te confunde, vuelve aquí cuando aparezca un símbolo nuevo.
""")

    st.markdown("---")
    st.markdown("### 🧱 El espacio de probabilidad $(\\Omega, \\mathcal{A}, P)$")
    st.markdown(r"""
Para que la probabilidad sea **matemática** (y no intuición vaga), necesitamos tres ingredientes:

**1. Espacio muestral $\Omega$** — el conjunto de **todos** los resultados posibles del experimento.
- Lanzar una moneda → $\Omega = \{\text{cara}, \text{sello}\}$.
- Lanzar un dado → $\Omega = \{1,2,3,4,5,6\}$.
- Lanzar dos monedas → $\Omega = \{(c,c),(c,s),(s,c),(s,s)\}$.

**2. Conjunto de eventos $\mathcal{A}$** — los subconjuntos de $\Omega$ a los que vamos a asignar probabilidad. Un **evento** es cualquier pregunta de tipo "sí/no" sobre el resultado. Por ejemplo en un dado:
- "sale par" = $\{2,4,6\}$
- "sale mayor que 4" = $\{5,6\}$
- "sale cualquier cosa" = $\Omega$ (el evento seguro)
- "sale nada" = $\varnothing$ (el evento imposible)

**3. Función de probabilidad $P$** — una función que a cada evento $E \in \mathcal{A}$ le asigna un número $P(E) \in [0,1]$.

La pregunta que queda: **¿qué restricciones debe cumplir $P$ para que el sistema sea coherente?** Esas restricciones son los tres axiomas.
""")

    st.markdown("---")
    st.markdown("### 🧱 Los tres axiomas de Kolmogorov")

    col_ax1, col_ax2, col_ax3 = st.columns(3)
    with col_ax1:
        st.markdown("**I. No-negatividad**")
        st.latex(r"P(E) \geq 0")
        st.caption("Una probabilidad nunca es negativa. El mínimo es 0 (imposible).")
    with col_ax2:
        st.markdown("**II. Normalización**")
        st.latex(r"P(\Omega) = 1")
        st.caption("La probabilidad de que ocurra *algo* dentro del universo es 1. Repartimos exactamente 1 unidad de 'confianza'.")
    with col_ax3:
        st.markdown("**III. σ-aditividad**")
        st.latex(r"P\left(\bigcup_{i=1}^\infty E_i\right) = \sum_{i=1}^\infty P(E_i)")
        st.caption("Si los eventos son **disjuntos**, la probabilidad de la unión es la suma.")

    st.markdown(r"""
**Por qué esos tres y no otros.** Estos axiomas son el mínimo indispensable para que:
- las probabilidades nunca sean negativas,
- se sumen a 1 sobre todo el universo,
- y se puedan **sumar** cuando los casos no se solapan.

A partir de estos tres se **deducen** todas las demás reglas (como las que siguen abajo). Son los cimientos.
""")

    st.markdown("---")
    st.markdown("### 🧱 Consecuencias inmediatas de los axiomas")

    st.markdown(r"""
De los axiomas se derivan reglas útiles. No hay que memorizarlas: entenderlas es suficiente porque salen solas al dibujar el diagrama.

**Regla del complemento:** $\ P(A^c) = 1 - P(A)$. Porque $A$ y $A^c$ son disjuntos y su unión es $\Omega$, que vale 1.

**Monotonicidad:** si $A \subseteq B$, entonces $P(A) \leq P(B)$. Un subconjunto no puede tener más probabilidad que el conjunto que lo contiene.

**Regla de la unión general:** si $A$ y $B$ **no** son disjuntos:
""")
    st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)")
    st.markdown(r"""
¿Por qué restar? Porque al sumar $P(A)$ y $P(B)$ contamos la intersección **dos veces**. Restándola una vez, cada zona queda contada exactamente una vez.
""")

    st.markdown("---")
    worked_example("Dado justo, verificación de los tres axiomas")

    st.markdown(r"""
Tomamos un dado de 6 caras equilibrado. Definimos:

$$\Omega = \{1,2,3,4,5,6\}, \qquad P(\{i\}) = \tfrac{1}{6} \text{ para cada cara } i.$$

**¿Cumple el axioma I?** Cada $P(\{i\}) = 1/6 \geq 0$. ✓

**¿Cumple el axioma II?** $P(\Omega) = P(\{1\}) + P(\{2\}) + \dots + P(\{6\}) = 6 \cdot \tfrac{1}{6} = 1$. ✓

**¿Cumple el axioma III?** Definamos los eventos disjuntos $E_1 = \{2\}$ (sale 2) y $E_2 = \{4,6\}$ (sale 4 o 6).
- $E_1 \cap E_2 = \varnothing$ (son disjuntos).
- $P(E_1 \cup E_2) = P(\{2,4,6\}) = 3 \cdot \tfrac{1}{6} = \tfrac{1}{2}$.
- $P(E_1) + P(E_2) = \tfrac{1}{6} + \tfrac{2}{6} = \tfrac{1}{2}$. ✓

**Aplicando la regla de unión cuando NO son disjuntos.** Sea $A = \{1,2,3\}$ (sale $\leq 3$) y $B = \{2,4,6\}$ (sale par).
- $A \cap B = \{2\}$, con $P(A \cap B) = 1/6$.
- $P(A) = 3/6$, $P(B) = 3/6$.
- $P(A \cup B) = 3/6 + 3/6 - 1/6 = 5/6$.
- Verificación directa: $A \cup B = \{1,2,3,4,6\}$, 5 elementos de 6 → $5/6$. ✓
""")

    st.markdown("---")
    interactive_header("Venn con intersección real y verificación de axiomas")

    st.markdown(
        "Ajusta las tres probabilidades básicas. La app calcula todas las regiones, "
        "colorea el diagrama de Venn y avisa si la combinación **viola** algún axioma."
    )

    col_sl, col_plot = st.columns([1, 1.4])
    with col_sl:
        p_a = st.slider("P(A) — probabilidad del evento A", 0.0, 1.0, 0.45, 0.01, key="pa_s1")
        p_b = st.slider("P(B) — probabilidad del evento B", 0.0, 1.0, 0.35, 0.01, key="pb_s1")
        max_inter = min(p_a, p_b)
        min_inter = max(0.0, p_a + p_b - 1.0)
        p_inter = st.slider(
            "P(A ∩ B) — probabilidad de la intersección",
            0.0, 1.0, min(0.15, max_inter), 0.01, key="pinter_s1",
        )

        p_a_only = p_a - p_inter
        p_b_only = p_b - p_inter
        p_union = p_a + p_b - p_inter
        p_neither = 1.0 - p_union

        st.markdown("**Descomposición del universo $\\Omega$:**")
        df = pd.DataFrame({
            "Región": ["Solo A (A \\ B)", "Intersección (A ∩ B)", "Solo B (B \\ A)", "Ninguno ((A ∪ B)ᶜ)"],
            "Probabilidad": [p_a_only, p_inter, p_b_only, p_neither],
        })
        st.dataframe(df.style.format({"Probabilidad": "{:.3f}"}), hide_index=True, use_container_width=True)

        violations = []
        if p_inter > max_inter + 1e-9:
            violations.append(
                f"P(A ∩ B) = {p_inter:.2f} supera a min(P(A), P(B)) = {max_inter:.2f}. "
                "La intersección nunca puede ser mayor que cualquiera de los eventos que la contienen."
            )
        if p_neither < -1e-9:
            violations.append(
                f"La suma P(A) + P(B) − P(A ∩ B) = {p_union:.2f} supera 1. "
                "Viola el axioma de normalización."
            )
        if violations:
            for v in violations:
                st.error("⚠️ " + v)
        else:
            st.success("Los valores son consistentes con los 3 axiomas.")

        st.metric("P(A ∪ B)", f"{p_union:.3f}")
        st.metric("P(Aᶜ) = 1 − P(A)", f"{1 - p_a:.3f}")

    with col_plot:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        rect = mpatches.Rectangle((-1.6, -1.4), 3.2, 2.8, facecolor="#f0f0f0",
                                   edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        r = 0.95
        c1 = plt.Circle((-0.45, 0), r, facecolor="#4C72B0", alpha=0.45,
                         edgecolor="#2a4e7a", linewidth=1.5)
        c2 = plt.Circle((0.45, 0), r, facecolor="#DD8452", alpha=0.45,
                         edgecolor="#914d26", linewidth=1.5)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.text(-0.95, 0, f"Solo A\n{p_a_only:.2f}", ha="center", va="center", fontsize=10, weight="bold")
        ax.text(0.95, 0, f"Solo B\n{p_b_only:.2f}", ha="center", va="center", fontsize=10, weight="bold")
        ax.text(0, 0, f"A ∩ B\n{p_inter:.2f}", ha="center", va="center", fontsize=10, weight="bold")
        ax.text(-1.45, 1.2, f"Ω = 1.00", ha="left", va="top", fontsize=11, weight="bold")
        ax.text(1.45, -1.25, f"Ninguno: {p_neither:.2f}", ha="right", va="bottom", fontsize=10, style="italic")
        ax.text(-0.9, 1.1, "A", fontsize=14, weight="bold", color="#2a4e7a")
        ax.text(0.9, 1.1, "B", fontsize=14, weight="bold", color="#914d26")
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "El rectángulo gris es **Ω** (todo lo que puede pasar, probabilidad 1). "
            "El círculo azul es **A**, el naranja es **B**, y la zona donde se "
            "superponen es **A ∩ B**. Los números dentro de cada región son su probabilidad: "
            "**los 4 números suman 1**. Si mueves los sliders, observa cómo 'solo A' + 'A ∩ B' "
            "siempre da P(A): la intersección pertenece tanto a A como a B."
        )

        st.markdown("**Vista alternativa: barra de probabilidad** (los 4 trozos suman 1)")
        fig2, ax2 = plt.subplots(figsize=(6.5, 1.2))
        segs = [("Solo A", p_a_only, "#4C72B0"),
                ("A ∩ B", p_inter, "#8172B2"),
                ("Solo B", p_b_only, "#DD8452"),
                ("Ninguno", max(p_neither, 0), "#cccccc")]
        left = 0
        for name, val, color in segs:
            if val > 1e-6:
                ax2.barh(0, val, left=left, color=color, edgecolor="white")
                if val > 0.04:
                    ax2.text(left + val/2, 0, f"{name}\n{val:.2f}", ha="center", va="center",
                             fontsize=8, color="white", weight="bold")
                left += val
        ax2.set_xlim(0, 1); ax2.set_ylim(-0.5, 0.5)
        ax2.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")
    self_check_header()

    quiz(
        "Si P(A) = 0.7 y P(B) = 0.5, ¿cuál es el valor **mínimo** posible de P(A ∩ B)?",
        [
            "0 — los eventos podrían ser disjuntos.",
            "0.2 — es el valor más pequeño compatible con los axiomas.",
            "0.35 — es P(A)·P(B) asumiendo independencia.",
            "0.5 — la intersección no puede ser mayor que el menor de los dos.",
        ],
        correct_idx=1,
        feedback_ok="De la regla P(A ∪ B) = P(A) + P(B) − P(A ∩ B) ≤ 1 se despeja P(A ∩ B) ≥ 0.7 + 0.5 − 1 = 0.2. Si fuera menor, la unión superaría 1 y violaría la normalización.",
        feedback_wrong="El razonamiento es: la unión no puede exceder 1 (axioma II). Entonces P(A ∩ B) ≥ P(A) + P(B) − 1 = 0.2. Eventos con probabilidades tan grandes **no pueden** ser disjuntos.",
        key="q1_s1",
    )

    quiz(
        "¿Cuál de estas afirmaciones es una **consecuencia directa** de los axiomas (no un axioma en sí mismo)?",
        [
            "P(E) ≥ 0 para todo evento E.",
            "P(Ω) = 1.",
            "P(Aᶜ) = 1 − P(A).",
            "P(A ∪ B) = P(A) + P(B) cuando A y B son disjuntos.",
        ],
        correct_idx=2,
        feedback_ok="P(Aᶜ) = 1 − P(A) se deduce: A y Aᶜ son disjuntos y su unión es Ω, así que por los axiomas II y III, P(A) + P(Aᶜ) = 1.",
        feedback_wrong="Las otras tres son **exactamente** los axiomas I, II y III (la última es σ-aditividad para dos eventos). La regla del complemento es una consecuencia, no un axioma.",
        key="q2_s1",
    )

    st.markdown("---")
    ai_bridge(
        "En Machine Learning, cuando un modelo de clasificación produce un "
        "vector de **softmax** como $[0.7, 0.2, 0.1]$, está devolviendo una "
        "distribución de probabilidad sobre las clases. Los axiomas garantizan "
        "que los números sean no negativos y sumen 1 — por eso se usa softmax "
        "y no otra función cualquiera. Cuando veas 'probability distribution' "
        "en un paper, los axiomas de Kolmogorov son las reglas que tienen que cumplirse."
    )


# ==================================================================
# 2. REGLA DE LAPLACE Y COMBINATORIA
# ==================================================================
elif choice == menu[1]:
    st.header("2. Regla de Laplace y Análisis Combinatorio")

    motivation(
        "Sacas 5 cartas al azar de una baraja de 52. ¿Cuál es la probabilidad de "
        "obtener un **full** (tres cartas del mismo valor y dos cartas de otro "
        "valor igual)? Tu intuición no tiene idea. En una sala con 23 personas, "
        "¿apostarías a que al menos dos comparten cumpleaños? Tu intuición "
        "probablemente diga 'no, muy difícil'. Se equivoca (la probabilidad real "
        "es >50%). Necesitamos herramientas para **contar con precisión**. "
        "La regla de Laplace y la combinatoria son esas herramientas."
    )

    prerequisites_box(r"""
- **Factorial** $n!$: producto de todos los enteros positivos hasta $n$.
  Ejemplo: $4! = 4 \cdot 3 \cdot 2 \cdot 1 = 24$. Por convención $0! = 1$.
- **Principio fundamental del conteo**: si una elección se compone de varios pasos independientes, el total es el **producto** del número de opciones en cada paso. Si tengo 3 poleras y 2 pantalones, puedo hacer $3 \cdot 2 = 6$ combinaciones distintas.
- **Resultados equiprobables**: todos los resultados individuales del espacio muestral tienen la misma probabilidad. Es el caso típico en juegos de azar "justos".
""")

    st.markdown("---")
    st.markdown("### 🧱 La regla de Laplace")
    st.markdown(r"""
Cuando todos los resultados de un experimento son **igualmente probables** y son un número finito, la probabilidad de un evento $A$ se reduce a contar:
""")
    st.latex(r"P(A) = \frac{|A|}{|\Omega|} = \frac{\text{casos favorables}}{\text{casos totales}}")
    st.markdown(r"""
La dificultad **no** está en la fórmula: está en **contar bien** el numerador y el denominador. Para eso sirve la combinatoria.
""")

    st.markdown("---")
    st.markdown("### 🧱 Tres herramientas de conteo que necesitas")

    st.markdown(r"""
**1. Principio multiplicativo (secuencias con reposición).** Si hay $k$ posiciones y cada una tiene $n$ opciones independientes:
""")
    st.latex(r"\text{Total} = n^k")
    st.caption("Ejemplo: un PIN de 4 dígitos entre 0–9 → $10^4 = 10\\,000$ opciones.")

    st.markdown(r"""
**2. Permutaciones (orden importa, sin repetir).** Elegir $r$ objetos de $n$ distintos, cuando el **orden** con que los eliges cambia el resultado:
""")
    st.latex(r"P(n, r) = \frac{n!}{(n-r)!}")
    st.caption("Ejemplo: ¿de cuántas formas 3 personas pueden recibir oro, plata y bronce entre 8 atletas? $8!/5! = 336$.")

    st.markdown(r"""
**3. Combinaciones (orden NO importa).** Elegir $r$ objetos de $n$ distintos cuando sólo importa **quiénes** fueron elegidos, no en qué orden:
""")
    st.latex(r"\binom{n}{r} = \frac{n!}{r!\,(n-r)!}")
    st.caption("Ejemplo: comités de 3 personas en un grupo de 10 → $\\binom{10}{3} = 120$. Formar un comité {Ana, Beto, Carla} es lo mismo que {Carla, Ana, Beto}, por eso dividimos por $r!$ para eliminar los órdenes repetidos.")

    st.markdown(r"""
**Regla práctica para decidir cuál usar:**
- ¿Los resultados son secuencias con posiciones distinguibles (contraseñas, matrículas, podios)? → permutaciones o $n^k$.
- ¿Es un grupo no ordenado (manos de cartas, comités, subconjuntos)? → combinaciones $\binom{n}{r}$.
""")

    st.markdown("---")
    worked_example("Probabilidad de full en póker (5 cartas de 52)")

    st.markdown(r"""
**Denominador:** de cuántas formas puedo recibir 5 cartas de 52 (el orden no importa, es una mano):
""")
    st.latex(r"|\Omega| = \binom{52}{5} = 2\,598\,960")

    st.markdown(r"""
**Numerador:** ¿cuántas manos son un full? Un full es **tres cartas de un valor** y **dos de otro valor**. Construyámoslo como 4 pasos:

1. Elegir el **valor** del trío: 13 opciones (A, 2, …, K).
2. Elegir **3 palos de los 4** disponibles para ese trío: $\binom{4}{3} = 4$.
3. Elegir el valor del par: 12 opciones (cualquier valor distinto del anterior).
4. Elegir 2 palos de 4 para el par: $\binom{4}{2} = 6$.

Multiplicando los pasos:
""")
    st.latex(r"|A| = 13 \cdot \binom{4}{3} \cdot 12 \cdot \binom{4}{2} = 13 \cdot 4 \cdot 12 \cdot 6 = 3\,744")

    st.latex(r"P(\text{full}) = \frac{3\,744}{2\,598\,960} \approx 0{,}00144 = 0{,}144\%")

    st.info(
        "**Lección clave.** La regla de Laplace es 'favorables / totales', "
        "pero el 90% del trabajo es **saber cómo contar** cada uno sin duplicar "
        "ni omitir casos. La combinatoria es el arte de contar sin equivocarse."
    )

    st.markdown("---")
    worked_example("Paradoja del cumpleaños — de dónde sale la fórmula")

    st.markdown(r"""
**Problema:** en una sala hay $n$ personas. ¿Probabilidad de que **al menos dos** compartan día y mes de cumpleaños? (Ignoramos años bisiestos y asumimos 365 días equiprobables.)

**Truco.** Es mucho más fácil calcular el **complemento**: $P(\text{alguna coincidencia}) = 1 - P(\text{todas las fechas distintas})$.

**¿Por qué?** Contar 'al menos dos coinciden' requiere sumar casos de '2 coinciden', '3 coinciden', etc. Contar 'todas distintas' es un solo caso.

**Construcción del 'todas distintas' como secuencia:**

- La 1ª persona puede tener cualquier fecha: $365/365$.
- La 2ª persona debe diferir de la 1ª: $364/365$.
- La 3ª persona debe diferir de las 2 anteriores: $363/365$.
- …
- La $n$-ésima persona debe diferir de las $n-1$ anteriores: $(365 - n + 1)/365$.

Multiplicando (porque los pasos son 'y', independientes bajo la hipótesis de equiprobabilidad):
""")
    st.latex(r"P(\text{todas distintas}) = \prod_{i=0}^{n-1} \frac{365 - i}{365}")
    st.latex(r"P(\text{al menos una coincidencia}) = 1 - \prod_{i=0}^{n-1} \frac{365 - i}{365}")

    st.markdown(r"""
**Por qué la intuición falla.** Piensas: "¿cuál es la probabilidad de que *alguien* comparta cumpleaños **conmigo**?" Eso son $n-1$ comparaciones. Pero la pregunta real es: "¿alguna pareja, cualquiera, comparte?" Con $n$ personas hay $\binom{n}{2} = n(n-1)/2$ pares. Con 23 personas son **253 pares** — 253 oportunidades de choque, no 22.
""")

    st.markdown("---")
    interactive_header("Cumpleaños y otras combinatorias")

    tab_cum, tab_comb = st.tabs(["🎂 Paradoja del cumpleaños", "🎴 Contador combinatorio"])

    with tab_cum:
        n_personas = st.slider("Número de personas en la sala (n)", 2, 100, 23, key="n_cum_s2")

        p_no = 1.0
        for i in range(n_personas):
            p_no *= (365 - i) / 365
        p_si = 1 - p_no
        pares = int(comb(n_personas, 2))

        c1, c2, c3 = st.columns(3)
        c1.metric(f"P(al menos coincidencia) con n={n_personas}", f"{p_si*100:.2f}%")
        c2.metric("Pares posibles (n choose 2)", f"{pares}")
        c3.metric("P(todas fechas distintas)", f"{p_no*100:.2f}%")

        fig, ax = plt.subplots(figsize=(9, 4))
        ns = np.arange(1, 80)
        ps = [1 - math.prod([(365-i)/365 for i in range(x)]) for x in ns]
        ax.plot(ns, ps, color="#C44E52", linewidth=2.5, label="P(alguna coincidencia)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Barrera del 50%")
        ax.axvline(23, color="gray", linestyle=":", alpha=0.5, label="n=23 (cruza el 50%)")
        ax.scatter([n_personas], [p_si], color="black", s=70, zorder=5, label=f"Tu elección: n={n_personas}")
        ax.set_xlabel("Número de personas (n)")
        ax.set_ylabel("Probabilidad")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "El eje horizontal es cuánta gente hay en la sala. El vertical, la probabilidad "
            "de que al menos dos compartan cumpleaños. La curva **crece muy rápido al "
            "inicio**: con apenas 23 personas ya cruza el 50%, y con 50 personas supera el "
            "97%. La línea punteada vertical marca el valor famoso (23). Este gráfico es la "
            "prueba gráfica de por qué la intuición falla: el número de **pares** crece "
            "cuadráticamente con $n$, no linealmente."
        )

    with tab_comb:
        st.markdown("Calcula permutaciones y combinaciones para ver la diferencia entre 'ordenar' y 'elegir'.")
        n_val = st.slider("n (total de objetos)", 1, 30, 10, key="n_comb_s2")
        r_val = st.slider("r (cuántos eliges)", 1, n_val, min(3, n_val), key="r_comb_s2")

        perm_val = math.perm(n_val, r_val)
        comb_val = int(comb(n_val, r_val))

        c1, c2 = st.columns(2)
        c1.metric(f"Permutaciones P({n_val},{r_val})", f"{perm_val:,}")
        c1.caption("Ordenar r de n, cada orden distinto cuenta.")
        c2.metric(f"Combinaciones C({n_val},{r_val})", f"{comb_val:,}")
        c2.caption("Elegir r de n, sin importar el orden.")

        st.markdown(r"""
**La relación entre ambas.** Cada combinación no ordenada se corresponde con $r!$ permutaciones ordenadas (las distintas maneras de ordenar ese mismo grupo). Por eso:
""")
        st.latex(r"\binom{n}{r} = \frac{P(n,r)}{r!}")
        st.caption(
            f"Con n={n_val}, r={r_val}: P({n_val},{r_val}) / {r_val}! = "
            f"{perm_val:,} / {math.factorial(r_val):,} = {comb_val:,} ✓"
        )

    st.markdown("---")
    self_check_header()

    quiz(
        "Una contraseña tiene 4 caracteres, cada uno una **letra mayúscula (26 opciones)**, y se permite repetir. ¿Cuántas contraseñas distintas hay?",
        [
            "26·25·24·23 = 358 800 (permutaciones sin repetir).",
            "26⁴ = 456 976 (principio multiplicativo con reposición).",
            "C(26, 4) = 14 950 (combinaciones).",
            "26! / 22! = 358 800.",
        ],
        correct_idx=1,
        feedback_ok="Cada una de las 4 posiciones tiene 26 opciones independientes (porque se permite repetir). Por el principio multiplicativo: 26·26·26·26 = 26⁴.",
        feedback_wrong="Si las letras **se pueden repetir**, cada posición tiene 26 opciones libres, independientemente de lo que pusiste antes. 26·26·26·26 = 26⁴. Las otras opciones prohíben repetición, que no es nuestro caso.",
        key="q1_s2",
    )

    quiz(
        "Un comité de 3 personas se elige entre 10 candidatos. ¿Cuál es el conteo correcto de comités distintos?",
        [
            "10³ = 1 000.",
            "10·9·8 = 720 (permutaciones).",
            "C(10, 3) = 120 (combinaciones).",
            "10! = 3 628 800.",
        ],
        correct_idx=2,
        feedback_ok="Un comité es un **grupo no ordenado**: {Ana, Beto, Carla} es el mismo comité que {Carla, Ana, Beto}. Usamos combinaciones: C(10,3) = 120.",
        feedback_wrong="Un comité no distingue orden: da igual en qué posición nombres a cada miembro. Por eso usamos C(10,3) = 120, no P(10,3) = 720 (que contaría cada comité 3! = 6 veces).",
        key="q2_s2",
    )

    st.markdown("---")
    ai_bridge(
        "La combinatoria aparece apenas abajo de la superficie en ML. Cuando "
        "calculas cuántos subconjuntos de features puede elegir un modelo de "
        "selección de variables, o cuántas posibles particiones train/test de "
        "tamaño fijo existen, o cuántas rutas distintas puede tomar un árbol de "
        "decisión con $d$ niveles binarios ($2^d$), estás contando combinatoriamente. "
        "En NLP, el número de **n-gramas** posibles con un vocabulario de $V$ "
        "palabras es $V^n$ — por eso los modelos de lenguaje antiguos sufrían "
        "el 'curse of dimensionality': para $V=10\\,000$ y $n=5$, son $10^{20}$ "
        "n-gramas, imposible de almacenar."
    )


# ==================================================================
# 3. PROBABILIDAD CONDICIONAL
# ==================================================================
elif choice == menu[2]:
    st.header("3. Probabilidad Condicional e Independencia")

    motivation(
        "Sacas una carta al azar de una baraja. ¿Probabilidad de que sea un as? "
        "4/52 ≈ 7.7%. Ahora te digo: **'la carta es de corazones'**. ¿Cambia tu "
        "probabilidad de que sea as? Sí: ahora el universo se redujo de 52 "
        "cartas a solo 13 (las de corazones), y entre ellas hay 1 as. La probabilidad "
        "pasó a ser 1/13 ≈ 7.7%. En este caso no cambió, pero en otros casos "
        "la información nueva sí la mueve drásticamente. La **probabilidad "
        "condicional** formaliza cómo actualizar nuestras creencias cuando "
        "recibimos información."
    )

    prerequisites_box(r"""
- **Evento**: un subconjunto del espacio muestral (una pregunta sí/no sobre el resultado). Si te cuesta esto, revisa la Sección 1.
- **Intersección $A \cap B$**: el evento "ambos ocurren". Ejemplo en un dado: $A = \{\text{par}\}$, $B = \{\text{mayor que 3}\}$, entonces $A \cap B = \{4, 6\}$.
- **Fracciones**: sólo necesitas recordar que $\frac{a/c}{b/c} = \frac{a}{b}$ (los comunes se cancelan).
""")

    st.markdown("---")
    st.markdown("### 🧱 La definición (y por qué esa definición)")

    st.markdown(r"""
La **probabilidad condicional de $E$ dado $F$**, escrita $P(E|F)$, es la probabilidad de que ocurra $E$ **sabiendo que $F$ ya ocurrió**. Se define como:
""")
    st.latex(r"P(E \mid F) = \frac{P(E \cap F)}{P(F)} \qquad \text{siempre que } P(F) > 0")

    st.markdown(r"""
**Intuición visual.** Al saber que $F$ ocurrió, nuestro universo se achica de $\Omega$ a $F$. Dentro de ese nuevo universo, sólo sobreviven los resultados que además están en $E$, es decir $E \cap F$. La fórmula es simplemente "**fracción favorable dentro del nuevo universo**":
""")
    st.latex(r"P(E \mid F) = \frac{\text{parte de } E \text{ que queda dentro de } F}{\text{tamaño del nuevo universo, que es } F}")

    st.markdown(r"""
**Regla del producto.** Reordenando la definición obtenemos una forma muy útil:
""")
    st.latex(r"P(E \cap F) = P(E \mid F) \cdot P(F) = P(F \mid E) \cdot P(E)")
    st.markdown(r"""
Es decir: la probabilidad de que ocurran ambos se puede calcular paso a paso: primero que ocurra uno, y dado eso, que ocurra el otro.
""")

    st.markdown("---")
    st.markdown("### 🧱 Independencia")
    st.markdown(r"""
Dos eventos $E$ y $F$ son **independientes** si saber que ocurrió uno **no cambia** la probabilidad del otro:
""")
    st.latex(r"P(E \mid F) = P(E) \quad \Longleftrightarrow \quad P(E \cap F) = P(E) \cdot P(F)")
    st.markdown(r"""
**Cuidado con la intuición.** Independencia **no** es lo mismo que "disjuntos". De hecho, dos eventos disjuntos con probabilidad positiva **nunca** son independientes (si uno ocurre, el otro es imposible, así que sí se influencian).
""")

    st.markdown("---")
    worked_example("Carta de corazones → probabilidad de as")

    st.markdown(r"""
Baraja estándar de 52 cartas. Definimos:
- $E$ = "la carta es un as" → $|E| = 4$, $P(E) = 4/52 = 1/13$.
- $F$ = "la carta es de corazones" → $|F| = 13$, $P(F) = 13/52 = 1/4$.
- $E \cap F$ = "es el as de corazones" → $|E \cap F| = 1$, $P(E \cap F) = 1/52$.

Aplicando la fórmula:
""")
    st.latex(r"P(E \mid F) = \frac{P(E \cap F)}{P(F)} = \frac{1/52}{13/52} = \frac{1}{13}")

    st.markdown(r"""
En este caso $P(E|F) = P(E)$, lo que significa que **ser as es independiente de ser de corazones** (porque cada palo tiene exactamente 1 as). Ahora veamos un caso donde la información **sí** cambia la probabilidad.
""")

    st.markdown("---")
    worked_example("Monty Hall — la ganancia del razonamiento condicional")

    st.markdown(r"""
**El problema.** Hay 3 puertas. Detrás de una hay un auto; detrás de las otras dos hay cabras.

1. Eliges una puerta (digamos, la Puerta 1). No la abres aún.
2. El presentador, que **sabe dónde está el auto**, abre una de las otras dos puertas y te muestra una cabra (nunca abre la que tiene el auto, nunca abre la que elegiste).
3. Te pregunta: ¿quieres quedarte con tu puerta o cambiar a la otra que queda cerrada?

**La intuición seductora (y equivocada).** "Quedan 2 puertas, una tiene el auto, la otra no. Es 50/50." **No**.

**Análisis caso por caso** (sin pérdida de generalidad, asumimos que elegiste la Puerta 1):

| Dónde está el auto | Probabilidad inicial | Qué hace el presentador | Resultado si te QUEDAS | Resultado si CAMBIAS |
|---|---|---|---|---|
| Puerta 1 (elegiste bien) | 1/3 | Abre 2 o 3 (ambas con cabra) | **GANAS** | pierdes |
| Puerta 2 | 1/3 | Obligado a abrir la 3 (no puede abrir la 2 con el auto ni la 1 que elegiste) | pierdes | **GANAS** (cambias a 2) |
| Puerta 3 | 1/3 | Obligado a abrir la 2 | pierdes | **GANAS** (cambias a 3) |

**Sumando:**
- $P(\text{ganar si te quedas}) = 1/3$.
- $P(\text{ganar si cambias}) = 1/3 + 1/3 = 2/3$.

**El 'truco' mental.** Cambiar hereda la probabilidad de que tu elección inicial **haya sido mala** — que es $2/3$, porque había 2 cabras y 1 auto en 3 puertas. La información del presentador no te ayuda si elegiste bien (1/3), pero te salva si elegiste mal (2/3).
""")

    st.markdown("---")
    interactive_header("Simulador Monty Hall con seguimiento en vivo")

    col_s, col_r = st.columns([1, 1.3])
    with col_s:
        n_sims = st.number_input("Número de partidas a simular", 100, 50000, 2000, step=100, key="nsims_s3")
        go = st.button("🎲 Simular partidas", key="go_s3")

    if go:
        rng = np.random.default_rng()
        auto = rng.integers(0, 3, n_sims)
        eleccion = rng.integers(0, 3, n_sims)
        ganar_quedarse = int(np.sum(auto == eleccion))
        ganar_cambiar = n_sims - ganar_quedarse

        with col_r:
            c1, c2 = st.columns(2)
            c1.metric("Victorias si te QUEDAS", f"{ganar_quedarse/n_sims*100:.2f}%",
                      delta=f"Teoría: 33.33%")
            c2.metric("Victorias si CAMBIAS", f"{ganar_cambiar/n_sims*100:.2f}%",
                      delta=f"Teoría: 66.67%")

            fig, ax = plt.subplots(figsize=(7, 3.2))
            step = max(1, n_sims // 500)
            xs = np.arange(step, n_sims + 1, step)
            rolling_stay = np.cumsum(auto == eleccion)[step-1::step] / xs
            rolling_switch = 1 - rolling_stay
            ax.plot(xs, rolling_stay, label="Quedarse", color="#4C72B0", linewidth=2)
            ax.plot(xs, rolling_switch, label="Cambiar", color="#C44E52", linewidth=2)
            ax.axhline(1/3, color="#4C72B0", linestyle=":", alpha=0.6)
            ax.axhline(2/3, color="#C44E52", linestyle=":", alpha=0.6)
            ax.set_xlabel("Partidas acumuladas")
            ax.set_ylabel("Tasa de victoria acumulada")
            ax.set_ylim(0, 1)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            how_to_read(
                "El eje X es cuántas partidas llevamos simuladas. El eje Y es la "
                "tasa de victorias **acumulada** hasta ese punto. Las líneas "
                "punteadas marcan los valores teóricos (1/3 y 2/3). Con pocas "
                "partidas las curvas oscilan, pero con muchas se estabilizan "
                "exactamente en los valores que predice el análisis condicional. "
                "Esto es la **ley de los grandes números** en acción."
            )

    st.markdown("---")
    self_check_header()

    quiz(
        "En un grupo de estudiantes: 40% son mujeres, y 30% del grupo son mujeres que programan. ¿Cuál es la probabilidad de que una estudiante elegida al azar **programe**, dado que es mujer?",
        [
            "30% — es el porcentaje de mujeres programadoras directamente.",
            "75% — se divide: P(programa y mujer) / P(mujer) = 0.30 / 0.40.",
            "12% — se multiplica 0.40 · 0.30.",
            "70% — se suma 0.30 + 0.40.",
        ],
        correct_idx=1,
        feedback_ok="Aplicando la definición: P(programa | mujer) = P(programa ∩ mujer) / P(mujer) = 0.30 / 0.40 = 0.75.",
        feedback_wrong="La definición pide P(A ∩ B) / P(B). Aquí B = 'mujer' (con P=0.40) y A ∩ B = 'mujer programadora' (con P=0.30). Divides: 0.30/0.40 = 0.75.",
        key="q1_s3",
    )

    quiz(
        "En Monty Hall, ¿por qué **cambiar** gana con probabilidad 2/3 en lugar de 1/2?",
        [
            "Porque el presentador revela información nueva que favorece la puerta no elegida.",
            "Porque al quedar 2 puertas, las probabilidades se redistribuyen proporcionalmente a su probabilidad inicial: la tuya sigue con 1/3, y la otra hereda el 2/3 restante.",
            "Porque el presentador podría engañarte.",
            "Por azar: en ensayos largos converge a 1/2.",
        ],
        correct_idx=1,
        feedback_ok="Exacto. La puerta que elegiste conserva su probabilidad original 1/3 (nada nuevo sobre ella). La puerta que queda cerrada hereda toda la probabilidad 'sobrante' 2/3, porque el presentador eliminó deterministamente la otra cabra.",
        feedback_wrong="La clave: el presentador **no elige al azar** cuál abrir. Siempre abre una cabra. Eso inyecta información asimétrica: tu puerta sigue con 1/3, la otra puerta ahora concentra 2/3. No es 50/50.",
        key="q2_s3",
    )

    st.markdown("---")
    ai_bridge(
        "Casi todo el Machine Learning moderno es razonamiento condicional. "
        "Un clasificador de imágenes calcula $P(\\text{clase} \\mid \\text{píxeles})$. "
        "Un modelo de lenguaje como GPT calcula $P(\\text{palabra}_t \\mid \\text{palabras}_{1{:}t-1})$ — "
        "la probabilidad de la siguiente palabra dada la historia. El mecanismo de "
        "**atención** en transformers puede verse como una forma de ajustar "
        "probabilidades condicionales según qué parte del contexto es más informativa. "
        "Cada vez que escuches 'given' en un paper, piensa en la barra vertical '$|$'."
    )


# ==================================================================
# 4. TEOREMA DE BAYES
# ==================================================================
elif choice == menu[3]:
    st.header("4. Teorema de Bayes")

    motivation(
        "Un test médico detecta una enfermedad con **sensibilidad del 99%** "
        "(si estás enfermo, sale positivo con probabilidad 0.99) y **especificidad "
        "del 99%** (si estás sano, sale negativo con probabilidad 0.99). La "
        "enfermedad afecta a **1 de cada 10 000 personas**. Acabas de dar positivo. "
        "¿Cuál es tu probabilidad real de estar enfermo? La intuición grita "
        "'99%'. La respuesta correcta es **menos del 1%**. El Teorema de Bayes "
        "es la herramienta que explica por qué y te permite calcularlo."
    )

    prerequisites_box(r"""
- **Probabilidad condicional** $P(A|B)$: revisa la Sección 3 si no la tienes fresca. Es el pilar de Bayes.
- **Regla del producto**: $P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$. De aquí sale Bayes por reordenamiento algebraico.
- **Partición del espacio**: dos eventos $E$ y $E^c$ (su complemento) cubren todo $\Omega$ sin solapamiento. Todo lo que ocurra, ocurre en uno u otro.
""")

    st.markdown("---")
    st.markdown("### 🧱 El teorema, con cada pieza nombrada")

    st.markdown(r"""
Partiendo de la regla del producto $P(A \cap B) = P(B|A) \cdot P(A) = P(A|B) \cdot P(B)$ y despejando $P(A|B)$:
""")
    st.latex(r"\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}")

    st.markdown(r"""
Cada término tiene un nombre con significado:

| Término | Nombre | Qué es |
|---|---|---|
| $P(A)$ | **Prior** (a priori) | Tu creencia sobre $A$ **antes** de ver evidencia. |
| $P(B \mid A)$ | **Likelihood** (verosimilitud) | Qué tan probable es la evidencia $B$ **si** $A$ fuera cierto. |
| $P(B)$ | **Evidencia** (marginal) | Qué tan probable es $B$ en general, sumando sobre todas las hipótesis. |
| $P(A \mid B)$ | **Posterior** (a posteriori) | Tu creencia actualizada sobre $A$ **después** de observar $B$. |

**Lectura en una frase:** "posterior = (likelihood × prior) / evidencia".
""")

    st.markdown("---")
    st.markdown("### 🧱 El denominador en detalle: regla de probabilidad total")

    st.markdown(r"""
El término $P(B)$ casi nunca se conoce directamente. Se calcula sumando sobre las hipótesis posibles. Si $A$ y $A^c$ parten el universo:
""")
    st.latex(r"P(B) = P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)")

    st.markdown(r"""
**Por qué funciona.** Todo resultado en $B$ o bien pasó "vía $A$" o bien "vía $A^c$" — no hay otra opción. Sumamos la 'masa' de $B$ en cada rama.

La forma **completa y aplicable** de Bayes, entonces, es:
""")
    st.latex(r"P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)}")

    st.markdown("---")
    worked_example("Urnas — Bayes en su forma más limpia")

    st.markdown(r"""
Tienes **dos urnas**. Urna A contiene 3 bolas rojas y 7 azules. Urna B contiene 8 rojas y 2 azules.
Eliges una urna al azar (50/50), sacas una bola **sin mirar**, y resulta ser roja. **¿Probabilidad de que venga de la Urna A?**

**Paso 1 — identificar las piezas.**
- $A$ = "vino de Urna A", $B^*$ = "la bola es roja". (Usamos $B^*$ para no confundir con la Urna B.)
- Prior: $P(A) = 0.5$, $P(A^c) = 0.5$.
- Likelihoods: $P(B^* \mid A) = 3/10 = 0.3$, $P(B^* \mid A^c) = 8/10 = 0.8$.

**Paso 2 — calcular la evidencia.**
""")
    st.latex(r"P(B^*) = 0.3 \cdot 0.5 + 0.8 \cdot 0.5 = 0.15 + 0.40 = 0.55")

    st.markdown(r"**Paso 3 — aplicar Bayes.**")
    st.latex(r"P(A \mid B^*) = \frac{0.3 \cdot 0.5}{0.55} = \frac{0.15}{0.55} = \frac{3}{11} \approx 0{,}273")

    st.markdown(r"""
**Interpretación.** El prior era 0.5 (urnas igualmente probables). Pero la evidencia 'roja' es **más típica** de la Urna B que de la A, así que el posterior se corre a favor de B: $P(A | \text{roja}) = 0{,}273$ (y por tanto $P(B | \text{roja}) = 0{,}727$). Bayes reasignó la creencia según la evidencia.
""")

    st.markdown("---")
    worked_example("Filtro de spam — actualización con una palabra clave")

    st.markdown(r"""
Un servicio de correo estima que:
- el 30% de los emails son spam → $P(S) = 0{,}3$, $P(\bar S) = 0{,}7$.
- el 80% de los spams contienen la palabra 'gratis' → $P(G \mid S) = 0{,}8$.
- el 10% de los no-spams también la contienen → $P(G \mid \bar S) = 0{,}1$.

Llega un email que **contiene** 'gratis'. ¿Probabilidad de que sea spam?

**Evidencia:**
""")
    st.latex(r"P(G) = 0{,}8 \cdot 0{,}3 + 0{,}1 \cdot 0{,}7 = 0{,}24 + 0{,}07 = 0{,}31")
    st.markdown(r"**Posterior:**")
    st.latex(r"P(S \mid G) = \frac{0{,}8 \cdot 0{,}3}{0{,}31} = \frac{0{,}24}{0{,}31} \approx 0{,}774")

    st.markdown(r"""
Pasamos de creer 30% a creer **~77%** que es spam. La palabra 'gratis' fue evidencia fuerte. Este es literalmente el mecanismo del **filtro bayesiano de spam** (Paul Graham, "A Plan for Spam", 2002).
""")

    st.markdown("---")
    worked_example("Test médico — la falacia de la tasa base")

    st.markdown(r"""
Enfermedad con prevalencia **1 en 10 000** ($P(E) = 0{,}0001$). Test con sensibilidad 99% ($P(+|E) = 0{,}99$) y especificidad 99% ($P(- \mid \bar E) = 0{,}99$, equivalentemente $P(+|\bar E) = 0{,}01$). Das positivo. ¿Probabilidad real de estar enfermo?

**Evidencia:**
""")
    st.latex(r"P(+) = 0{,}99 \cdot 0{,}0001 + 0{,}01 \cdot 0{,}9999 = 0{,}000099 + 0{,}009999 = 0{,}010098")

    st.markdown(r"**Posterior:**")
    st.latex(r"P(E \mid +) = \frac{0{,}99 \cdot 0{,}0001}{0{,}010098} \approx 0{,}0098 = 0{,}98\%")

    st.markdown(r"""
**¿Por qué este resultado tan contraintuitivo?** Imagina 10 000 personas con esta enfermedad rara:
- 1 está enferma. El test le da positivo (≈1 verdadero positivo).
- 9 999 están sanas. El test se equivoca en el 1% de ellas → **~100 falsos positivos**.

Entre los ~101 positivos, sólo 1 está realmente enfermo. Por eso $P(E|+) \approx 1/101 \approx 1\%$.

**Moraleja.** Cuando la enfermedad es **muy rara**, aunque el test sea muy preciso, la mayoría de los positivos son falsas alarmas. Este fenómeno se llama **falacia de la tasa base**: la gente (incluidos médicos) ignora el prior y se fija sólo en la sensibilidad.
""")

    st.markdown("---")
    interactive_header("Calculadora de Bayes con visualización por población")

    col_in, col_out = st.columns([1, 1.3])
    with col_in:
        prev = st.number_input(
            "Prior P(E) — prevalencia de la enfermedad",
            0.00001, 0.5, 0.003, format="%.5f", key="prev_s4",
        )
        sens = st.slider("Sensibilidad P(+|E)", 0.50, 1.0, 0.95, 0.01, key="sens_s4")
        spec = st.slider("Especificidad P(−|sano)", 0.50, 1.0, 0.90, 0.01, key="spec_s4")

        fpr = 1 - spec
        p_pos = sens * prev + fpr * (1 - prev)
        post = (sens * prev) / p_pos if p_pos > 0 else 0.0

        st.metric("P(Enfermo | positivo) — posterior", f"{post*100:.3f}%")
        st.metric("P(positivo) — evidencia total", f"{p_pos*100:.3f}%")

        st.markdown("**Cálculo numérico paso a paso:**")
        st.latex(
            r"P(+) = " + f"{sens:.2f} \\cdot {prev:.5f} + {fpr:.2f} \\cdot {1-prev:.5f}"
            r" = " + f"{p_pos:.5f}"
        )
        st.latex(
            r"P(E \mid +) = \frac{" + f"{sens:.2f} \\cdot {prev:.5f}"
            r"}{" + f"{p_pos:.5f}" + r"} = " + f"{post:.4f}"
        )

    with col_out:
        N = 100_000
        enfermos = int(round(N * prev))
        sanos = N - enfermos
        TP = int(round(enfermos * sens))
        FN = enfermos - TP
        FP = int(round(sanos * fpr))
        TN = sanos - FP

        st.markdown(f"**Ejemplo concreto: una población de {N:,} personas**")
        df = pd.DataFrame({
            "": ["Enfermos (E)", "Sanos (Ē)", "**Total**"],
            "Test + ": [TP, FP, TP + FP],
            "Test −": [FN, TN, FN + TN],
            "Total": [enfermos, sanos, N],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown(
            f"De los **{TP + FP:,}** positivos, **{TP:,}** son verdaderos enfermos "
            f"y **{FP:,}** son falsas alarmas. "
            f"La fracción de enfermos entre los positivos es "
            f"**{TP}/{TP+FP} = {TP/(TP+FP)*100:.2f}%** — coincide con el posterior de Bayes."
        )

        fig, ax = plt.subplots(figsize=(7, 3.2))
        categorias = ["Verdaderos\npositivos", "Falsos\npositivos",
                      "Verdaderos\nnegativos", "Falsos\nnegativos"]
        valores = [TP, FP, TN, FN]
        colores = ["#55a868", "#c44e52", "#4c72b0", "#dd8452"]
        bars = ax.bar(categorias, valores, color=colores)
        ax.set_yscale("log")
        ax.set_ylabel("Personas (escala log)")
        for bar, v in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                    f"{v:,}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "Las 4 barras muestran cómo se distribuyen las personas de la "
            "población según la verdad (enfermo/sano) y el resultado del test "
            "(+/-). El **eje vertical es logarítmico** porque con enfermedades "
            "raras las barras tendrían escalas muy distintas. Busca visualmente "
            "qué tan grande es la barra verde (verdaderos +) comparada con la "
            "roja (falsos +): esa relación **es** el posterior."
        )

    st.markdown("---")
    self_check_header()

    quiz(
        "Un abogado argumenta: 'La probabilidad de que el ADN del acusado coincida con la escena del crimen, siendo inocente, es 1 en 1 millón. Por lo tanto, la probabilidad de que sea inocente es 1 en 1 millón'. ¿Es correcto?",
        [
            "Sí, el dato es directo.",
            "No: confunde P(coincidencia | inocente) con P(inocente | coincidencia). Sin un prior razonable sobre culpabilidad no se puede concluir nada.",
            "No: el ADN nunca es fiable al 100%.",
            "Sí, pero el juez puede decidir otra cosa.",
        ],
        correct_idx=1,
        feedback_ok="Es la **falacia del fiscal**, caso clásico de mal uso bayesiano. Bayes dice que P(inocente|coincidencia) depende también del prior P(inocente) y de P(coincidencia) — la evidencia sola no basta.",
        feedback_wrong="P(B|A) y P(A|B) son **distintas**: sólo se relacionan vía Bayes, usando el prior. Sin prior, no hay conclusión. Confundirlas se llama 'falacia del fiscal' y ha causado condenas erróneas reales.",
        key="q1_s4",
    )

    quiz(
        "Si mantienes sensibilidad y especificidad constantes y **aumentas** la prevalencia (prior), ¿qué pasa con el posterior P(E|+)?",
        [
            "Se mantiene igual: depende sólo del test.",
            "Disminuye, porque hay más falsos positivos.",
            "Aumenta: el prior mayor hace que más casos positivos sean realmente enfermos.",
            "Es impredecible sin más datos.",
        ],
        correct_idx=2,
        feedback_ok="Correcto. El numerador P(+|E)·P(E) crece con el prior, y aunque el denominador también crece, la fracción aumenta. Por eso en poblaciones de alto riesgo los tests tienen más valor predictivo.",
        feedback_wrong="El numerador contiene el prior directamente: si P(E) sube, el numerador sube más rápido que el denominador. Por eso un test positivo en un paciente 'de alto riesgo' es mucho más informativo que en uno sano de la población general.",
        key="q2_s4",
    )

    st.markdown("---")
    ai_bridge(
        "Bayes es la columna vertebral del **aprendizaje bayesiano**: en lugar "
        "de encontrar un único 'mejor' parámetro, se mantiene una distribución "
        "sobre todos los parámetros plausibles, que se actualiza con cada nuevo "
        "dato. Aplicaciones concretas: redes bayesianas, inferencia variacional, "
        "procesos gaussianos, RL con exploración bayesiana (Thompson sampling), "
        "y la interpretación bayesiana del dropout. En la próxima sección "
        "construimos el clasificador **Naïve Bayes** que es Bayes aplicado "
        "directamente como algoritmo de ML."
    )


# ==================================================================
# 5. CLASIFICADOR NAÏVE BAYES
# ==================================================================
elif choice == menu[4]:
    st.header("5. Clasificador Naïve Bayes")

    motivation(
        "Ya sabes aplicar Bayes a un evento contra otro. Pero ¿y si queremos "
        "clasificar una muestra con **13 atributos** simultáneos (como un vino "
        "con su acidez, alcohol, fenoles, magnesio, etc.) entre 3 clases? "
        "La fórmula de Bayes pide $P(x_1, x_2, \\dots, x_{13} \\mid C)$ — una "
        "distribución conjunta sobre 13 variables. Estimarla requiere una "
        "cantidad astronómica de datos. **Naïve Bayes** hace una simplificación "
        "descarada (asume que los atributos son independientes dada la clase) "
        "que **no es cierta** pero funciona sorprendentemente bien. Vamos a "
        "entender por qué y a aplicarla a clasificar vinos reales."
    )

    prerequisites_box(r"""
- **Teorema de Bayes** (Sección 4) — aquí es el corazón del método.
- **Independencia** (Sección 3) — la asunción 'naïve' es $P(x_1, \dots, x_n | C) = \prod_i P(x_i | C)$.
- **Distribución Normal** (Sección 6 la explica con detalle): campana simétrica con media $\mu$ y desviación $\sigma$. Su densidad es $\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. Sólo necesitas saber que la uso para modelar cada atributo continuo.
- **Logaritmo**: $\log(ab) = \log a + \log b$. Lo usamos para convertir productos de probabilidades pequeñas en sumas, y evitar que se vayan a cero numérico.
""")

    st.markdown("---")
    st.markdown("### 🧱 De Bayes al clasificador")

    st.markdown(r"""
Queremos: dado un vector de atributos $\mathbf{x} = (x_1, \dots, x_n)$, predecir la clase $C$ más probable:
""")
    st.latex(r"\hat C = \arg\max_k \; P(C_k \mid \mathbf{x})")
    st.markdown(r"""
Aplicando Bayes:
""")
    st.latex(r"P(C_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C_k) \cdot P(C_k)}{P(\mathbf{x})}")

    st.markdown(r"""
Como $P(\mathbf{x})$ es igual para todas las clases (no depende de $k$), no afecta cuál es el máximo y lo podemos **ignorar**:
""")
    st.latex(r"\hat C = \arg\max_k \; P(\mathbf{x} \mid C_k) \cdot P(C_k)")

    st.markdown("---")
    st.markdown("### 🧱 La asunción 'ingenua'")

    st.markdown(r"""
Calcular $P(x_1, \dots, x_n \mid C)$ directamente es **inviable**. Si cada atributo continuo lo discretizáramos en 10 cubetas y tuviéramos 13 atributos, la tabla conjunta tendría $10^{13}$ entradas. Un dataset con 200 vinos nunca podría estimar eso.

**Solución naïve**: asumir que, dada la clase, los atributos son **independientes** entre sí:
""")
    st.latex(r"P(x_1, \dots, x_n \mid C_k) = \prod_{i=1}^n P(x_i \mid C_k)")
    st.markdown(r"""
Ahora sólo necesitamos estimar $P(x_i \mid C_k)$ por atributo, no toda la conjunta. De $10^{13}$ parámetros pasamos a $13 \cdot 10$ = 130.

**¿Por qué se llama 'naïve'?** Porque la hipótesis suele ser **falsa** (en vinos, alcohol y acidez se correlacionan; en texto, palabras como 'nueva' y 'york' aparecen juntas). Sin embargo, el clasificador suele funcionar bien: las decisiones sólo necesitan ordenar correctamente las clases, no estimar las probabilidades con exactitud.
""")

    st.markdown(r"""
**Para atributos continuos**, se suele suponer que $P(x_i \mid C_k)$ es una Normal con parámetros estimados del entrenamiento:
""")
    st.latex(r"P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} \exp\!\left(-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}\right)")
    st.markdown(r"""
donde $\mu_{k,i}, \sigma_{k,i}$ son media y desviación del atributo $i$ **calculadas usando solo las muestras de la clase $k$**.

**Pasar a logaritmos**: multiplicar 13 densidades pequeñas puede colapsar a 0 numéricamente. Tomamos log:
""")
    st.latex(r"\log P(C_k \mid \mathbf{x}) \propto \log P(C_k) + \sum_{i=1}^n \log P(x_i \mid C_k)")
    st.markdown(r"""
Sumar logs es numéricamente estable y el $\arg\max$ no cambia (log es monótono creciente).
""")

    st.markdown("---")
    worked_example("Clasificar un vino — cálculo paso a paso")

    st.markdown("""
Usamos el **Wine Dataset** clásico (sklearn): 178 vinos italianos de 3 cultivares, cada uno descrito por 13 atributos químicos.
Vamos a:
1. Estimar $\\mu, \\sigma$ por clase para cada atributo a partir de los datos.
2. Tomar un vino (puedes elegir cuál) y calcular el log-score por clase.
3. Ver quién gana el argmax.
""")

    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = [f"Clase {i} ({wine.target_names[i]})" for i in range(3)]

    mus = np.array([X[y == k].mean(axis=0) for k in range(3)])
    sigmas = np.array([X[y == k].std(axis=0, ddof=1) for k in range(3)])
    priors = np.array([(y == k).mean() for k in range(3)])

    col_sel, col_calc = st.columns([1, 1.3])
    with col_sel:
        st.markdown("**Elige un vino del dataset:**")
        idx = st.slider("Índice del vino (0 a 177)", 0, 177, 12, key="wine_idx_s5")
        x_sel = X[idx]
        y_true = int(y[idx])
        st.caption(f"Clase real: **{class_names[y_true]}**")

        df_vino = pd.DataFrame({
            "Atributo": feature_names,
            "Valor": x_sel,
        })
        st.dataframe(df_vino.style.format({"Valor": "{:.3f}"}), hide_index=True,
                     use_container_width=True, height=260)

    with col_calc:
        st.markdown("**Cálculo de log-score por clase:**")
        logpx_given_k = np.zeros((3, len(feature_names)))
        for k in range(3):
            logpx_given_k[k] = stats.norm.logpdf(x_sel, loc=mus[k], scale=sigmas[k])

        log_posterior = np.log(priors) + logpx_given_k.sum(axis=1)
        pred = int(np.argmax(log_posterior))

        rows = []
        for k in range(3):
            rows.append({
                "Clase": class_names[k],
                "log P(Cₖ)": np.log(priors[k]),
                "Σ log P(xᵢ|Cₖ)": logpx_given_k[k].sum(),
                "log-score total": log_posterior[k],
            })
        df_scores = pd.DataFrame(rows)
        st.dataframe(
            df_scores.style.format({
                "log P(Cₖ)": "{:.3f}",
                "Σ log P(xᵢ|Cₖ)": "{:.2f}",
                "log-score total": "{:.2f}",
            }).highlight_max(subset=["log-score total"], color="#d4edda"),
            hide_index=True, use_container_width=True,
        )

        emoji = "🎯" if pred == y_true else "❌"
        st.markdown(
            f"**Clase predicha:** {class_names[pred]} {emoji} "
            f"(real: {class_names[y_true]})"
        )

        post_norm = np.exp(log_posterior - log_posterior.max())
        post_norm = post_norm / post_norm.sum()
        st.markdown("**Probabilidades posteriores normalizadas** (softmax de los log-scores):")
        for k in range(3):
            st.progress(float(post_norm[k]), text=f"{class_names[k]}: {post_norm[k]*100:.2f}%")

    st.markdown(
        "**Lectura:** el log-score total por clase combina el prior (ligeramente "
        "distinto entre clases) y la suma de log-verosimilitudes de los 13 atributos. "
        "La clase con mayor log-score gana. Convertimos a probabilidades con softmax "
        "para lectura humana, pero internamente todo se hace en logs."
    )

    st.markdown("---")
    interactive_header("Visualizando la verosimilitud de un atributo")

    feat_idx = st.selectbox(
        "Atributo a visualizar:",
        range(len(feature_names)),
        format_func=lambda i: feature_names[i],
        key="feat_s5",
    )

    x_plot = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 400)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for k in range(3):
        subset = X[y == k, feat_idx]
        ax.hist(subset, bins=15, density=True, alpha=0.3, color=colors[k])
        pdf = stats.norm.pdf(x_plot, mus[k, feat_idx], sigmas[k, feat_idx])
        ax.plot(x_plot, pdf, color=colors[k], linewidth=2.3, label=class_names[k])

    val_sel = x_sel[feat_idx]
    ax.axvline(val_sel, color="black", linestyle="--", linewidth=1.5,
               label=f"Vino elegido: {val_sel:.2f}")
    for k in range(3):
        like_k = stats.norm.pdf(val_sel, mus[k, feat_idx], sigmas[k, feat_idx])
        ax.scatter([val_sel], [like_k], color=colors[k], s=90, zorder=5,
                   edgecolor="black", linewidth=1.2)
    ax.set_xlabel(feature_names[feat_idx])
    ax.set_ylabel("Densidad de probabilidad")
    ax.legend(loc="upper right", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

    how_to_read(
        "Cada campana es la **verosimilitud** P(atributo|clase) asumida Normal, "
        "ajustada a los datos reales de esa clase (que aparecen como histograma "
        "semi-transparente detrás). La línea negra vertical marca el valor del "
        "vino que seleccionaste. Los puntos grandes son los valores de "
        "verosimilitud que ese vino obtiene para cada clase — **la clase cuya "
        "curva es más alta en esa posición** gana 'puntos' para este atributo. "
        "Naïve Bayes suma (en logs) estos puntos sobre los 13 atributos."
    )

    st.markdown("---")
    self_check_header()

    quiz(
        "¿Por qué el clasificador se llama 'ingenuo' (naïve)?",
        [
            "Porque usa una fórmula muy simple que cualquiera entiende.",
            "Porque asume que los atributos son **independientes** dada la clase, aunque en la realidad casi siempre están correlacionados.",
            "Porque no usa redes neuronales.",
            "Porque sólo funciona con datos de entrenamiento pequeños.",
        ],
        correct_idx=1,
        feedback_ok="Exacto. La independencia condicional es la asunción 'ingenua'. Sorprendentemente, aun siendo falsa, la clasificación final suele ser correcta porque sólo importa el argmax, no las probabilidades exactas.",
        feedback_wrong="El nombre viene de la **hipótesis de independencia condicional** entre atributos, que simplifica enormemente el cálculo pero rara vez es cierta. Lo asombroso es que el modelo es robusto a esa simplificación.",
        key="q1_s5",
    )

    quiz(
        "¿Por qué sumamos **log-probabilidades** en vez de multiplicar probabilidades directamente?",
        [
            "Porque es más rápido computacionalmente.",
            "Para evitar **underflow numérico** al multiplicar muchos valores pequeños, y porque log es monótono (no cambia el argmax).",
            "Porque las probabilidades logarítmicas son las 'reales'.",
            "Porque scikit-learn lo exige.",
        ],
        correct_idx=1,
        feedback_ok="Sí. Multiplicar 13 densidades ~0.1 da ~10⁻¹³, y con más atributos se va a 0. Tomar log convierte productos en sumas y trabaja en escala manejable. El argmax no cambia porque log es creciente.",
        feedback_wrong="El motivo real es **numérico**: multiplicar muchos números pequeños colapsa a 0 en punto flotante (underflow). La suma de logs es equivalente al log del producto, pero estable. La Sección 7 lo demuestra con un slider.",
        key="q2_s5",
    )

    st.markdown("---")
    ai_bridge(
        "Naïve Bayes fue durante años el clasificador **por defecto para texto** "
        "(spam, categorización de noticias, análisis de sentimiento), porque la "
        "asunción de independencia entre palabras, siendo falsa, produce modelos "
        "extremadamente rápidos de entrenar e inferir. Hoy los transformers lo "
        "superan en precisión, pero Naïve Bayes sigue siendo un baseline honesto: "
        "si tu modelo sofisticado no le gana por un margen claro, algo anda mal. "
        "El patrón 'asumir independencia para hacer tratable la conjunta' aparece "
        "también en modelos gráficos, Markov Random Fields y VAEs."
    )


# ==================================================================
# 6. CATÁLOGO DE DISTRIBUCIONES
# ==================================================================
elif choice == menu[5]:
    st.header("6. Catálogo de Distribuciones de Probabilidad")

    motivation(
        "Los fenómenos del mundo real siguen **patrones** reconocibles: la "
        "cantidad de llamadas que recibe un call center por hora, las alturas de "
        "los estudiantes de una universidad, el tiempo hasta que falla un "
        "componente electrónico. Cada patrón tiene un 'molde matemático' — una "
        "**distribución de probabilidad** — que lo describe. Conocer el catálogo "
        "y cuándo usar cada uno te ahorra reinventar la rueda: si sabes que tu "
        "dato es 'número de eventos en un intervalo', de inmediato piensas en "
        "Poisson y ya tienes fórmula para $E[X]$, $\\mathrm{Var}(X)$, etc."
    )

    prerequisites_box(r"""
- **Variable aleatoria $X$**: una función que convierte el resultado de un experimento en un número. Si lanzo moneda: $X=1$ si cara, $X=0$ si sello.
- **Distribución discreta** — sus valores son puntos separados (enteros, típicamente). Se describe con la **PMF** (Probability Mass Function): $p(x) = P(X = x)$. Las alturas de la PMF suman 1.
- **Distribución continua** — sus valores son un continuo (reales). Se describe con la **PDF** (Probability Density Function): $f(x)$ tal que $P(a \leq X \leq b) = \int_a^b f(x)\,dx$. **Atención**: $f(x)$ **no** es la probabilidad de que $X = x$ — esa probabilidad es cero. $f(x)$ es densidad: probabilidad por unidad de longitud.
- **Parámetros**: valores que controlan la forma de la distribución (la $\mu, \sigma$ de la Normal; la $\lambda$ de la Poisson, etc.).
""")

    st.markdown("---")
    st.markdown("### 🧱 Discretas vs continuas en una tabla")

    st.markdown(r"""
| Propiedad | Discreta | Continua |
|---|---|---|
| Valores posibles | Contables (0, 1, 2, …) | Continuo (un intervalo de $\mathbb{R}$) |
| Descripción | PMF: $p(x) = P(X=x)$ | PDF: $f(x)$ (densidad) |
| Suma/integral | $\sum_x p(x) = 1$ | $\int_{-\infty}^{\infty} f(x)\,dx = 1$ |
| $P(X = x)$ | $p(x)$ | **0** (para cualquier punto concreto) |
| $P(a \le X \le b)$ | $\sum_{x=a}^{b} p(x)$ | $\int_a^b f(x)\,dx$ = área bajo la curva |
| Gráfico típico | Barras | Curva |
""")

    st.markdown("---")
    st.markdown("### 🎛️ Explora el catálogo")

    tipo = st.radio("Tipo de distribución:", ["Discretas", "Continuas"],
                    key="tipo_s6", horizontal=True)

    if tipo == "Discretas":
        dist = st.selectbox(
            "Distribución:",
            ["Bernoulli", "Binomial", "Poisson", "Geométrica", "Hipergeométrica", "Uniforme discreta"],
            key="distD_s6",
        )

        if dist == "Bernoulli":
            st.markdown("**Qué modela:** un único experimento con dos resultados, 'éxito' (1) o 'fracaso' (0). Ejemplo: un lanzamiento de moneda, un paciente que responde sí/no al tratamiento.")
            p = st.slider("Probabilidad de éxito p", 0.0, 1.0, 0.5, 0.01, key="p_bern_s6")
            st.latex(r"P(X = x) = p^x(1-p)^{1-x}, \quad x \in \{0,1\}")
            st.latex(fr"E[X] = p = {p:.2f}, \quad \mathrm{{Var}}(X) = p(1-p) = {p*(1-p):.4f}")
            x = np.array([0, 1]); y = np.array([1-p, p])

        elif dist == "Binomial":
            st.markdown("**Qué modela:** número de éxitos en $n$ ensayos de Bernoulli **independientes** con la misma $p$. Ejemplo: cantidad de caras en 10 lanzamientos; cantidad de clics en 100 impresiones.")
            n = st.slider("Número de ensayos n", 1, 100, 20, key="n_bin_s6")
            p = st.slider("Probabilidad de éxito p", 0.0, 1.0, 0.5, 0.01, key="p_bin_s6")
            st.latex(r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k \in \{0, \dots, n\}")
            st.latex(fr"E[X] = np = {n*p:.2f}, \quad \mathrm{{Var}}(X) = np(1-p) = {n*p*(1-p):.3f}")
            x = np.arange(n+1); y = stats.binom.pmf(x, n, p)

        elif dist == "Poisson":
            st.markdown("**Qué modela:** número de eventos que ocurren en un intervalo fijo (tiempo, espacio, longitud), cuando la tasa promedio es $\\lambda$ eventos por intervalo. Ejemplo: emails que llegan en 1 hora; partículas radiactivas por segundo; goles por partido.")
            lam = st.slider("Tasa promedio λ", 0.1, 30.0, 5.0, 0.1, key="lam_poi_s6")
            st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \dots\}")
            st.latex(fr"E[X] = \lambda = {lam:.2f}, \quad \mathrm{{Var}}(X) = \lambda = {lam:.2f}")
            x = np.arange(int(max(40, lam*3)))
            y = stats.poisson.pmf(x, lam)

        elif dist == "Geométrica":
            st.markdown("**Qué modela:** número de ensayos hasta obtener el **primer éxito** (incluyéndolo). Ejemplo: cuántos lanzamientos hasta la primera cara; cuántos intentos hasta prender el motor.")
            p = st.slider("Probabilidad de éxito por intento", 0.01, 1.0, 0.3, 0.01, key="p_geo_s6")
            st.latex(r"P(X = k) = (1-p)^{k-1} p, \quad k \in \{1, 2, 3, \dots\}")
            st.latex(fr"E[X] = 1/p = {1/p:.2f}, \quad \mathrm{{Var}}(X) = (1-p)/p^2 = {(1-p)/p**2:.2f}")
            x = np.arange(1, min(50, int(5/p) + 10))
            y = stats.geom.pmf(x, p)

        elif dist == "Hipergeométrica":
            st.markdown("**Qué modela:** número de éxitos al sacar $n$ objetos **sin reposición** de una población de $N$ con $K$ éxitos. Ejemplo: cartas de un mismo palo al sacar 5 de una baraja; piezas defectuosas en una muestra.")
            N = st.slider("Tamaño de población N", 10, 200, 50, key="N_hyp_s6")
            K = st.slider("Éxitos en la población K", 1, N, 20, key="K_hyp_s6")
            n = st.slider("Tamaño de muestra n", 1, N, 10, key="n_hyp_s6")
            st.latex(r"P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}")
            E = n * K / N
            Var = n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))
            st.latex(fr"E[X] = n \cdot \frac{{K}}{{N}} = {E:.2f}, \quad \mathrm{{Var}}(X) = {Var:.3f}")
            x = np.arange(max(0, n-(N-K)), min(n, K) + 1)
            y = stats.hypergeom.pmf(x, N, K, n)

        else:  # Uniforme discreta
            st.markdown("**Qué modela:** todos los valores enteros en un rango $\\{a, a+1, \\dots, b\\}$ son igual de probables. Ejemplo: lanzar un dado justo (a=1, b=6).")
            a = st.slider("a (mínimo)", 1, 20, 1, key="a_unif_s6")
            b = st.slider("b (máximo)", a, 30, 6, key="b_unif_s6")
            n_vals = b - a + 1
            st.latex(fr"P(X = k) = 1/{n_vals} \text{{ para }} k \in \{{{a}, \dots, {b}\}}")
            st.latex(fr"E[X] = \frac{{a+b}}{{2}} = {(a+b)/2:.1f}, \quad \mathrm{{Var}}(X) = \frac{{(b-a+1)^2-1}}{{12}} = {((b-a+1)**2-1)/12:.3f}")
            x = np.arange(a, b+1); y = np.ones_like(x, dtype=float) / n_vals

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.bar(x, y, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Valor k")
        ax.set_ylabel("P(X = k)")
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "Cada barra es la probabilidad **exacta** de que X tome ese valor. "
            "Las alturas **suman 1**. Fíjate cómo la forma cambia al mover los "
            "sliders: la Binomial con p=0.5 es simétrica; la Poisson se alarga "
            "hacia la derecha (skew positivo); la Geométrica decae "
            "exponencialmente."
        )

    else:  # Continuas
        dist = st.selectbox(
            "Distribución:",
            ["Uniforme", "Normal (Gaussiana)", "Exponencial", "Gamma", "Beta", "Student-t"],
            key="distC_s6",
        )

        if dist == "Uniforme":
            st.markdown("**Qué modela:** todos los valores en un intervalo $[a, b]$ son igualmente probables (densidad constante). Ejemplo: un número aleatorio entre 0 y 1; ángulo aleatorio de una rueda.")
            a = st.slider("a (mínimo)", -5.0, 5.0, 0.0, 0.1, key="a_uc_s6")
            b = st.slider("b (máximo)", a + 0.1, 10.0, a + 3.0, 0.1, key="b_uc_s6")
            st.latex(fr"f(x) = \frac{{1}}{{b-a}} = {1/(b-a):.3f} \text{{ para }} x \in [{a:.1f}, {b:.1f}]")
            st.latex(fr"E[X] = \frac{{a+b}}{{2}} = {(a+b)/2:.2f}, \quad \mathrm{{Var}}(X) = \frac{{(b-a)^2}}{{12}} = {(b-a)**2/12:.3f}")
            x = np.linspace(a - 1, b + 1, 600)
            y = stats.uniform.pdf(x, loc=a, scale=b-a)

        elif dist == "Normal (Gaussiana)":
            st.markdown("**Qué modela:** fenómenos con muchos aportes pequeños e independientes que se suman — por el **Teorema Central del Límite**, el resultado tiende a ser Normal. Ejemplo: altura humana, errores de medición, IQ.")
            mu = st.slider("Media μ", -5.0, 5.0, 0.0, 0.1, key="mu_nor_s6")
            sigma = st.slider("Desviación σ", 0.1, 5.0, 1.0, 0.1, key="sig_nor_s6")
            st.latex(r"f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)")
            st.latex(fr"E[X] = \mu = {mu:.2f}, \quad \mathrm{{Var}}(X) = \sigma^2 = {sigma**2:.3f}")
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
            y = stats.norm.pdf(x, mu, sigma)

        elif dist == "Exponencial":
            st.markdown("**Qué modela:** tiempo hasta el **primer evento** en un proceso Poisson de tasa $\\lambda$. Equivalentemente: tiempos de vida con 'falta de memoria'. Ejemplo: tiempo entre llamadas a un call center, vida útil de un componente electrónico.")
            lam = st.slider("Tasa λ", 0.1, 5.0, 1.0, 0.1, key="lam_exp_s6")
            st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
            st.latex(fr"E[X] = 1/\lambda = {1/lam:.3f}, \quad \mathrm{{Var}}(X) = 1/\lambda^2 = {1/lam**2:.3f}")
            x = np.linspace(0, 10/lam, 600)
            y = stats.expon.pdf(x, scale=1/lam)

        elif dist == "Gamma":
            st.markdown("**Qué modela:** tiempo hasta que ocurran $\\alpha$ eventos en un proceso Poisson de tasa $\\beta$. Generaliza la Exponencial ($\\alpha = 1$ recupera Exp). Ejemplo: tiempo total en una fila con varios trámites.")
            a = st.slider("Forma α", 0.1, 10.0, 2.0, 0.1, key="a_gam_s6")
            b = st.slider("Tasa β", 0.1, 5.0, 1.0, 0.1, key="b_gam_s6")
            st.latex(r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x \geq 0")
            st.latex(fr"E[X] = \alpha/\beta = {a/b:.3f}, \quad \mathrm{{Var}}(X) = \alpha/\beta^2 = {a/b**2:.3f}")
            x = np.linspace(0, (a/b) * 5 + 2, 600)
            y = stats.gamma.pdf(x, a, scale=1/b)

        elif dist == "Beta":
            st.markdown("**Qué modela:** proporciones o probabilidades, es decir valores en $[0,1]$. Muy usada como **prior** bayesiano sobre una probabilidad desconocida. Con $\\alpha = \\beta = 1$ es la Uniforme; con ambos grandes e iguales se concentra en 0.5.")
            a = st.slider("α", 0.1, 10.0, 2.0, 0.1, key="a_bet_s6")
            b = st.slider("β", 0.1, 10.0, 2.0, 0.1, key="b_bet_s6")
            st.latex(r"f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0,1]")
            mean_beta = a/(a+b)
            var_beta = a*b / ((a+b)**2 * (a+b+1))
            st.latex(fr"E[X] = \frac{{\alpha}}{{\alpha+\beta}} = {mean_beta:.3f}, \quad \mathrm{{Var}}(X) = {var_beta:.4f}")
            x = np.linspace(0, 1, 600)
            y = stats.beta.pdf(x, a, b)

        else:  # Student-t
            st.markdown("**Qué modela:** como la Normal, pero con **colas más gruesas** (eventos extremos más probables). Aparece al estimar la media de una muestra pequeña con varianza desconocida. Converge a la Normal cuando los grados de libertad $\\nu \\to \\infty$.")
            nu = st.slider("Grados de libertad ν", 1, 50, 3, key="nu_t_s6")
            st.latex(r"f(x) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}")
            mean_t = 0.0 if nu > 1 else np.nan
            var_t = nu/(nu-2) if nu > 2 else np.inf
            st.latex(fr"E[X] = 0 \text{{ (si }}\nu>1\text{{)}}, \quad \mathrm{{Var}}(X) = \frac{{\nu}}{{\nu-2}} = {var_t:.3f} \text{{ (si }}\nu>2\text{{)}}")
            x = np.linspace(-6, 6, 600)
            y = stats.t.pdf(x, nu)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(x, y, color="#4C72B0", linewidth=2.2)
        ax.fill_between(x, y, alpha=0.25, color="#4C72B0")
        ax.set_xlabel("Valor x")
        ax.set_ylabel("f(x) — densidad")
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "La curva es la **densidad** $f(x)$. Cuidado: el valor $f(x)$ puntual "
            "**no** es la probabilidad de que $X = x$ (esa es 0). La probabilidad "
            "es el **área bajo la curva** sobre un intervalo. El área total es 1. "
            "Donde la curva es más alta, los valores de $X$ son 'más densos' (más "
            "frecuentes al muestrear). La desviación σ controla el ancho; la "
            "media μ controla la posición horizontal."
        )

    st.markdown("---")
    st.markdown("### 🧱 Relaciones entre distribuciones (oro pedagógico)")

    st.markdown(r"""
Las distribuciones no son silos aislados: se **transforman unas en otras** bajo límites o casos particulares. Conocer estas conexiones es lo que convierte un catálogo en un mapa.

- **Bernoulli → Binomial**: si $X_i$ son Bernoulli$(p)$ independientes, $\sum_{i=1}^n X_i$ es Binomial$(n,p)$.
- **Binomial → Poisson**: si $n \to \infty$ y $p \to 0$ con $np = \lambda$ fijo, Binomial$(n,p)$ se aproxima a Poisson$(\lambda)$. Útil para eventos **raros**.
- **Exponencial es un caso de Gamma**: Gamma$(\alpha = 1, \beta = \lambda)$ = Exponencial$(\lambda)$.
- **Suma de Exponenciales es Gamma**: si $X_1, \dots, X_k$ son Exp$(\lambda)$ iid, entonces $\sum X_i$ es Gamma$(k, \lambda)$. Por eso Gamma modela 'tiempo hasta el $k$-ésimo evento'.
- **Uniforme(0,1) = Beta(1,1)**: caso particular con densidad constante.
- **Student-t → Normal**: cuando los grados de libertad $\nu \to \infty$.
- **TCL — Teorema Central del Límite**: si $X_1, \dots, X_n$ son iid con media $\mu$ y varianza $\sigma^2 < \infty$, entonces $\bar X = \frac{1}{n}\sum X_i$ se aproxima a Normal$(\mu, \sigma^2/n)$ cuando $n$ crece. **Explica por qué la Normal aparece en todas partes**.
""")

    st.markdown("---")
    interactive_header("Demostración: Binomial → Poisson al crecer n")

    st.markdown(
        "Fija $\\lambda = np$ y haz crecer $n$ (con $p$ ajustado para mantener el "
        "producto constante). Verás que la Binomial se 'acerca' a la Poisson."
    )
    lam_conv = st.slider("λ = np", 0.5, 10.0, 3.0, 0.5, key="lam_conv_s6")
    n_conv = st.slider("n (tamaño de la Binomial)", 5, 500, 20, key="n_conv_s6")
    p_conv = lam_conv / n_conv

    k_max = int(lam_conv + 4 * math.sqrt(lam_conv)) + 2
    k = np.arange(0, k_max)
    bin_pmf = stats.binom.pmf(k, n_conv, p_conv)
    poi_pmf = stats.poisson.pmf(k, lam_conv)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.4
    ax.bar(k - width/2, bin_pmf, width=width, color="#4C72B0", label=f"Binomial(n={n_conv}, p={p_conv:.3f})")
    ax.bar(k + width/2, poi_pmf, width=width, color="#C44E52", label=f"Poisson(λ={lam_conv:.1f})")
    ax.set_xlabel("k")
    ax.set_ylabel("Probabilidad")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    diff = np.abs(bin_pmf - poi_pmf).sum()
    st.caption(f"Distancia total |Binomial − Poisson| = **{diff:.4f}**. "
               f"Al subir $n$ manteniendo $\\lambda$ fijo, esta distancia tiende a cero.")

    st.markdown("---")
    self_check_header()

    quiz(
        "Recibes en promedio 12 correos por hora. ¿Qué distribución usarías para modelar la cantidad de correos en los próximos 30 minutos?",
        [
            "Binomial con n=30, p=12/60.",
            "Poisson con λ=6 (tasa por 30 min).",
            "Exponencial con λ=12.",
            "Normal con μ=12.",
        ],
        correct_idx=1,
        feedback_ok="Poisson modela conteos de eventos en intervalos. La tasa es 12/hora, así que en 30 min el λ efectivo es 6.",
        feedback_wrong="Para **contar eventos** en un intervalo fijo con tasa constante, la herramienta es **Poisson**. La tasa se escala al intervalo: 12/hora → 6/media hora. Exponencial mide el *tiempo* entre eventos, no el conteo.",
        key="q1_s6",
    )

    quiz(
        "Tienes que modelar 'probabilidad subyacente de clic en un anuncio', que es un número desconocido en $[0,1]$. ¿Qué distribución es natural como prior?",
        [
            "Normal centrada en 0.5.",
            "Poisson con λ=0.5.",
            "Beta con parámetros α y β.",
            "Exponencial.",
        ],
        correct_idx=2,
        feedback_ok="La Beta vive exactamente en [0,1] y es la prior conjugada natural para una probabilidad de Bernoulli/Binomial. Ajustando α, β puedes codificar tu creencia previa.",
        feedback_wrong="Una **probabilidad** vive en [0,1]. Normal y Poisson pueden dar valores fuera de ahí. Exponencial es positiva pero no acotada. La Beta sí vive en [0,1] y se ajusta con α, β.",
        key="q2_s6",
    )

    st.markdown("---")
    ai_bridge(
        "En deep learning las distribuciones aparecen por todas partes: la "
        "**Normal** es la prior de los pesos en regularización L2 (weight decay) "
        "y la distribución latente de los VAEs. La **Bernoulli** aparece en la "
        "última capa de un clasificador binario (sigmoid → Bernoulli). La "
        "**Multinomial** / **Categórica** es la salida de softmax. La inicialización "
        "**Xavier/He** usa Normal o Uniforme escaladas. Cuando un modelo genera "
        "texto con 'temperatura', está muestreando de una Categórica modificada. "
        "Entender las distribuciones es entender **qué está haciendo** el modelo, "
        "no sólo sus resultados."
    )


# ==================================================================
# 7. MÁXIMA VEROSIMILITUD (MLE)
# ==================================================================
elif choice == menu[6]:
    st.header("7. Estimación de Máxima Verosimilitud (MLE)")

    motivation(
        "Lanzas una moneda 10 veces y obtienes 7 caras. ¿Cuál es el 'mejor' "
        "estimado para la probabilidad $p$ de que salga cara? Tu intuición "
        "dice 7/10. **¿Podemos justificarlo formalmente?** Sí. **MLE** "
        "(Máxima Verosimilitud) es el principio: elige el parámetro que hace "
        "que los datos observados sean **lo más probables posible**. "
        "Este principio es la columna vertebral de cómo 'aprende' casi todo "
        "modelo estadístico — incluidas las redes neuronales, cuyo **cross-entropy** "
        "es exactamente MLE con otro nombre."
    )

    prerequisites_box(r"""
- **Verosimilitud vs probabilidad**: son el mismo número, vistos al revés. **Probabilidad** fija el parámetro y pregunta por los datos ("dado p, ¿qué tan probable es obtener 7 caras?"). **Verosimilitud** fija los datos y la ve como función del parámetro ("dadas las 7 caras, ¿qué $p$ las hace más plausibles?").
- **Logaritmo natural** $\ln$ (o $\log$): transforma productos en sumas, $\ln(ab) = \ln a + \ln b$. Es monótono: donde $f(x)$ alcanza su máximo, $\ln f(x)$ también. Por eso maximizar $\ln \mathcal{L}$ es equivalente a maximizar $\mathcal{L}$.
- **Derivada como 'pendiente'**: la derivada $\frac{df}{dx}$ es la pendiente de la curva en cada punto. En un máximo, la pendiente es **cero** (la curva deja de subir). Así se encuentra analíticamente.
""")

    st.markdown("---")
    st.markdown("### 🧱 La definición, con palabras")

    st.markdown(r"""
Tienes datos $D = \{x_1, \dots, x_n\}$ y un modelo que depende de un parámetro $\theta$ (por ejemplo, una moneda con probabilidad de cara $\theta$). La **verosimilitud** es:
""")
    st.latex(r"\mathcal{L}(\theta) = P(D \mid \theta) = \prod_{i=1}^n P(x_i \mid \theta)")
    st.markdown(r"""
(El producto supone que los datos son independientes dado $\theta$.)

**El MLE** es el parámetro que maximiza esta función:
""")
    st.latex(r"\hat\theta_{\text{MLE}} = \arg\max_\theta \; \mathcal{L}(\theta)")

    st.markdown(r"""
**Paso 1 — log-verosimilitud.** Trabajar con un producto de $n$ números es incómodo y numéricamente inestable. Tomamos logaritmo:
""")
    st.latex(r"\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^n \log P(x_i \mid \theta)")
    st.markdown(r"""
Maximizar $\ell$ es equivalente a maximizar $\mathcal{L}$ (log es monótono).

**Paso 2 — NLL (Negative Log-Likelihood).** En optimización se suele **minimizar**. Definimos:
""")
    st.latex(r"\text{NLL}(\theta) = -\ell(\theta) = -\sum_{i=1}^n \log P(x_i \mid \theta)")
    st.markdown(r"""
Minimizar NLL = maximizar verosimilitud. Este número es exactamente el **loss** que optimiza un clasificador neuronal.
""")

    st.markdown("---")
    worked_example("MLE para una moneda — todo el cálculo a mano")

    st.markdown(r"""
**Setup.** Lanzamos una moneda $n$ veces. Obtenemos $k$ caras y $n-k$ sellos. El parámetro desconocido es $p$, la probabilidad de cara. Queremos $\hat p_{\text{MLE}}$.

**Verosimilitud.** Cada lanzamiento es Bernoulli$(p)$:
""")
    st.latex(r"\mathcal{L}(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^k (1-p)^{n-k}")

    st.markdown(r"**Log-verosimilitud.**")
    st.latex(r"\ell(p) = k \log p + (n-k)\log(1-p)")

    st.markdown(r"**Derivada respecto a $p$ e igualamos a cero** (condición de máximo):")
    st.latex(r"\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0")
    st.latex(r"\Rightarrow k(1-p) = (n-k)p \;\Rightarrow\; k - kp = np - kp \;\Rightarrow\; k = np")
    st.latex(r"\boxed{\hat p_{\text{MLE}} = \frac{k}{n}}")

    st.markdown(r"""
**¡La frecuencia empírica!** Con 7 caras en 10 lanzamientos, el MLE de $p$ es $0{,}7$. La matemática confirma la intuición.

**Conexión con Cross-Entropy.** Si escribimos $y_i \in \{0,1\}$ (el dato) y llamamos $p_i$ a la probabilidad que nuestro modelo asigna a "es 1", la NLL toma la forma:
""")
    st.latex(r"\text{NLL} = -\sum_{i=1}^n \left[y_i \log p_i + (1-y_i) \log(1-p_i)\right]")
    st.markdown(r"""
Esta es exactamente la **Binary Cross-Entropy Loss** usada en clasificación binaria neuronal. No hay 'otra' loss; es MLE.
""")

    st.markdown("---")
    interactive_header("Visualizador de la verosimilitud de la moneda")

    col_sim, col_plot = st.columns([1, 1.4])
    with col_sim:
        n_flips = st.slider("Número de lanzamientos n", 5, 500, 30, key="n_flips_s7")
        p_true = st.slider("p verdadero (para generar datos)", 0.0, 1.0, 0.65, 0.01, key="pt_s7")
        seed = st.number_input("Semilla aleatoria", 0, 10000, 42, key="seed_s7")
        rng = np.random.default_rng(int(seed))
        data = rng.binomial(1, p_true, n_flips)
        k_obs = int(data.sum())
        p_hat = k_obs / n_flips
        st.metric("Caras observadas (k)", f"{k_obs}/{n_flips}")
        st.metric("p̂ MLE = k/n", f"{p_hat:.4f}")
        p_guess = st.slider("Si adivinaras p = …", 0.01, 0.99, 0.5, 0.01, key="pguess_s7")
        ll_guess = k_obs * math.log(p_guess) + (n_flips - k_obs) * math.log(1 - p_guess)
        ll_max = k_obs * math.log(max(p_hat, 1e-9)) + (n_flips - k_obs) * math.log(max(1 - p_hat, 1e-9))
        st.metric("log-verosimilitud de tu adivinanza",
                  f"{ll_guess:.3f}",
                  delta=f"{ll_guess - ll_max:.3f} vs MLE")

    with col_plot:
        ps = np.linspace(0.001, 0.999, 400)
        ll = k_obs * np.log(ps) + (n_flips - k_obs) * np.log(1 - ps)
        lik = np.exp(ll - ll.max())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(ps, lik, color="#4C72B0", linewidth=2.2, label=r"$\mathcal{L}(p)$ (normalizada)")
        ax1.axvline(p_hat, color="#C44E52", linestyle="--", linewidth=2, label=fr"MLE $\hat p = {p_hat:.3f}$")
        ax1.axvline(p_guess, color="gray", linestyle=":", linewidth=1.5, label=fr"Tu adivinanza $p={p_guess:.2f}$")
        ax1.set_ylabel("Verosimilitud")
        ax1.legend(fontsize=9)

        ax2.plot(ps, ll, color="#4C72B0", linewidth=2.2, label=r"$\ell(p) = \log \mathcal{L}(p)$")
        ax2.axvline(p_hat, color="#C44E52", linestyle="--", linewidth=2)
        ax2.axvline(p_guess, color="gray", linestyle=":", linewidth=1.5)
        ax2.scatter([p_hat], [ll_max], color="#C44E52", s=80, zorder=5)
        ax2.scatter([p_guess], [ll_guess], color="black", s=80, zorder=5)
        ax2.set_xlabel("p (parámetro)")
        ax2.set_ylabel("log-verosimilitud")
        ax2.legend(fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "El eje X es el parámetro $p$ (qué valor adivinarías para la moneda). "
            "Arriba: la verosimilitud $\\mathcal{L}(p)$ normalizada (se ve que "
            "tiene un **único máximo**, que es el MLE). Abajo: su logaritmo "
            "$\\ell(p)$, una curva más manejable con el mismo máximo. La línea "
            "roja punteada marca el MLE $\\hat p = k/n$. La línea gris es tu "
            "adivinanza: cualquier $p \\neq \\hat p$ produce una log-verosimilitud "
            "**más baja** — es decir, hace los datos menos probables."
        )

    st.markdown("---")
    st.markdown("### 🧱 ¿Por qué usar log? El underflow numérico")

    st.markdown(
        "La razón práctica para trabajar en logs no es elegancia: es que multiplicar "
        "muchas probabilidades pequeñas **colapsa a cero** en punto flotante. Veamos el efecto."
    )

    N_uf = st.slider("Número de observaciones independientes N", 10, 5000, 300, key="N_uf_s7")
    p_uf = st.slider("Probabilidad individual p", 0.01, 0.99, 0.5, 0.01, key="p_uf_s7")
    prod_val = p_uf ** N_uf
    log_val = N_uf * math.log(p_uf)

    col_a, col_b = st.columns(2)
    col_a.metric("Producto directo p^N", f"{prod_val:.3e}" if prod_val > 1e-300 else "0.0 (underflow)")
    col_b.metric("Suma de logs N·log(p)", f"{log_val:.3f}")

    if prod_val < 1e-300:
        st.error(
            "⚠️ **Underflow**: el producto es tan pequeño que `float64` lo "
            "representa como 0 exacto. Cualquier cálculo posterior (un argmax, "
            "una división) quedaría arruinado. La suma de logs, en cambio, "
            "es un número perfectamente manejable."
        )
    else:
        st.success(
            "Por ahora el producto cabe en `float64`, pero fíjate qué tan "
            "rápido se acerca a cero al subir N. En redes neuronales procesamos "
            "millones de pasos — el log es **obligatorio**, no un adorno."
        )

    st.markdown("---")
    self_check_header()

    quiz(
        "Lanzas una moneda 100 veces y obtienes 38 caras. ¿Cuál es el MLE de la probabilidad de cara?",
        [
            "0.5 — lo más razonable sin más info.",
            "0.38 — la frecuencia empírica k/n.",
            "0.62 — la frecuencia de sellos.",
            "No se puede estimar sin un prior.",
        ],
        correct_idx=1,
        feedback_ok="El MLE para Bernoulli es exactamente k/n = 38/100 = 0.38. Sale de derivar la log-verosimilitud e igualar a cero.",
        feedback_wrong="Para datos Bernoulli, el MLE es k/n — la frecuencia observada. Lo demostramos derivando la log-verosimilitud. MLE no asume prior: 'deja que los datos hablen'.",
        key="q1_s7",
    )

    quiz(
        "¿Por qué el término 'log-verosimilitud' aparece en casi todo algoritmo de aprendizaje?",
        [
            "Porque los logs se ven más elegantes en papers.",
            "Porque convierte el producto de probabilidades en suma (estable numéricamente) y preserva el máximo. Minimizar −log-verosimilitud es lo mismo que maximizar la verosimilitud.",
            "Porque log es la función de activación en redes neuronales.",
            "Porque los logaritmos eliminan los parámetros de la Normal.",
        ],
        correct_idx=1,
        feedback_ok="Exacto. Producto → suma (estable), log es monótono (mismo argmax), y negar lo convierte en un loss a minimizar. Esa es literalmente la cross-entropy.",
        feedback_wrong="Razón real: es la única forma práctica de trabajar con la verosimilitud de millones de datos sin underflow, preservando el argmax. Y minimizar su negativo es el 'loss' estándar en ML.",
        key="q2_s7",
    )

    st.markdown("---")
    ai_bridge(
        "Cuando entrenas un clasificador con PyTorch o TensorFlow y usas "
        "`nn.CrossEntropyLoss` o `nn.BCELoss`, estás **literalmente** calculando "
        "la NLL del modelo respecto a las etiquetas. El **backpropagation** "
        "computa el gradiente de esta NLL respecto a los pesos; el **optimizador** "
        "ajusta los pesos para reducirla. 'Entrenar una red' = MLE en un modelo "
        "muy flexible parametrizado por los pesos. Este principio es "
        "asombrosamente general: se aplica igual a regresión logística, LDA, "
        "HMMs, GANs (en su formulación original), modelos de lenguaje, etc. "
        "Si entiendes MLE, entiendes el esqueleto de casi todo aprendizaje supervisado."
    )


# ==================================================================
# 8. VALOR ESPERADO Y VARIANZA
# ==================================================================
elif choice == menu[7]:
    st.header("8. Valor Esperado, Varianza y Variables Indicadoras")

    motivation(
        "Apuestas $1 dólar a un dado: ganas el valor que salga. ¿Cuánto deberías "
        "esperar ganar en promedio a largo plazo? Intuitivamente, algún tipo de "
        "'promedio ponderado'. El **valor esperado** $E[X]$ formaliza exactamente "
        "eso. Su compañera, la **varianza**, mide cuán dispersos quedan los "
        "resultados alrededor de ese promedio. Y las **variables indicadoras** "
        "son un truco fenomenal que reduce problemas aparentemente imposibles "
        "(como el 'guardarropa': ¿cuántos recuperan su abrigo si los devuelven "
        "al azar?) a sumas sencillas."
    )

    prerequisites_box(r"""
- **Variable aleatoria $X$**: función que asigna un número al resultado de un experimento (vista en la Sección 6).
- **PMF o PDF**: cómo se reparten las probabilidades sobre los valores posibles.
- **Sumatoria $\sum$ e integral $\int$**: en esta sección aparecen, pero los ejemplos los hacemos paso a paso — no necesitas manejarlas fluidamente.
""")

    st.markdown("---")
    st.markdown("### 🧱 Valor esperado — definición")

    st.markdown(r"""
Para una variable aleatoria **discreta** que toma valores $x_1, x_2, \dots$ con probabilidades $p_1, p_2, \dots$:
""")
    st.latex(r"E[X] = \sum_i x_i \cdot P(X = x_i)")
    st.markdown(r"""
Para una variable **continua** con densidad $f$:
""")
    st.latex(r"E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\,dx")

    st.markdown(r"""
**Interpretación.** Es un **promedio ponderado**: cada valor posible se pesa por su probabilidad. A largo plazo, si repitieras el experimento infinitas veces y promediaras los resultados, obtendrías $E[X]$. Esto no es metáfora: es el teorema de **la ley de los grandes números**.

**Notación.** También se escribe $\mu$ o $\mu_X$ ('la media').
""")

    st.markdown("---")
    worked_example("E[dado justo]")

    st.markdown(r"""
$X$ = valor de un dado justo. Valores posibles 1 a 6, cada uno con probabilidad $1/6$.
""")
    st.latex(r"E[X] = 1 \cdot \tfrac{1}{6} + 2 \cdot \tfrac{1}{6} + 3 \cdot \tfrac{1}{6} + 4 \cdot \tfrac{1}{6} + 5 \cdot \tfrac{1}{6} + 6 \cdot \tfrac{1}{6} = \tfrac{21}{6} = 3{,}5")
    st.markdown(r"""
**Observación importante:** $E[X] = 3{,}5$ es un valor que el dado **nunca** produce. El valor esperado **no es el valor más probable**; es el centro de masa de la distribución.
""")

    st.markdown("---")
    st.markdown("### 🧱 Propiedades: linealidad de la esperanza")

    st.markdown(r"""
La propiedad más útil del valor esperado:
""")
    st.latex(r"E[aX + bY + c] = a\,E[X] + b\,E[Y] + c")
    st.markdown(r"""
**Lo sorprendente:** esto vale **aunque $X$ e $Y$ sean dependientes**. Es una consecuencia directa de que la sumatoria/integral es lineal — nada más.

¿Por qué esto es útil? Porque descompone problemas complicados en trocitos simples. El ejemplo del guardarropa lo aprovecha al máximo.
""")

    st.markdown("---")
    st.markdown("### 🧱 Variables indicadoras — el truco mágico")

    st.markdown(r"""
Para un evento $A$, definimos la **variable indicadora**:
""")
    st.latex(r"I_A = \begin{cases} 1 & \text{si } A \text{ ocurre} \\ 0 & \text{si } A \text{ no ocurre} \end{cases}")
    st.markdown(r"""
Su valor esperado es notablemente simple:
""")
    st.latex(r"E[I_A] = 1 \cdot P(A) + 0 \cdot P(A^c) = P(A)")

    st.markdown(r"""
**Técnica.** Si quieres contar 'cuántos eventos de una lista ocurren', defines una indicadora por cada evento y sumas:
""")
    st.latex(r"N = I_{A_1} + I_{A_2} + \dots + I_{A_n}")
    st.latex(r"E[N] = P(A_1) + P(A_2) + \dots + P(A_n)")
    st.markdown(r"""
Incluso si los eventos $A_i$ son **fuertemente dependientes**. La linealidad no pregunta si son independientes.
""")

    st.markdown("---")
    worked_example("Guardarropa — cuántos recuperan su abrigo")

    st.markdown(r"""
$n$ personas dejan sus abrigos en un guardarropa. El asistente pierde los tickets y devuelve los abrigos **al azar** (una permutación uniforme). ¿Cuántas personas, en promedio, reciben su propio abrigo?

**Intuición.** Uno podría pensar 'con $n$ grande, casi nadie acertará'. Veamos qué dice la matemática.

**Paso 1 — indicadora por persona.** Para cada persona $i$, sea $X_i = 1$ si recibe su abrigo y $0$ si no.

**Paso 2 — $P(X_i = 1)$.** De las $n!$ permutaciones posibles, $(n-1)!$ tienen el abrigo $i$ en la posición $i$. Por tanto:
""")
    st.latex(r"P(X_i = 1) = \frac{(n-1)!}{n!} = \frac{1}{n}")

    st.markdown(r"**Paso 3 — esperanza del total.** Sea $N = X_1 + \dots + X_n$ el número de aciertos. Por linealidad:")
    st.latex(r"E[N] = E[X_1] + E[X_2] + \dots + E[X_n] = n \cdot \frac{1}{n} = 1")

    st.markdown(r"""
**Resultado impresionante:** sin importar si son **3 personas o 3 millones**, el número esperado de aciertos es **exactamente 1**. Las $X_i$ son fuertemente dependientes (si muchos aciertan, pocas combinaciones posibles para el resto), pero la linealidad de la esperanza no exige independencia y funciona igual.
""")

    st.markdown("---")
    st.markdown("### 🧱 Varianza — cómo de disperso")

    st.markdown(r"""
El valor esperado te dice el centro; la **varianza** te dice cuán lejos del centro suelen caer los valores:
""")
    st.latex(r"\mathrm{Var}(X) = E\left[(X - \mu)^2\right]")
    st.markdown(r"""
Se eleva al cuadrado para que desviaciones positivas y negativas no se cancelen y para penalizar más las grandes. Una **fórmula alternativa útil** (sale expandiendo el cuadrado):
""")
    st.latex(r"\mathrm{Var}(X) = E[X^2] - \big(E[X]\big)^2")

    st.markdown(r"""
La raíz cuadrada de la varianza es la **desviación estándar** $\sigma = \sqrt{\mathrm{Var}(X)}$ — tiene las mismas unidades que $X$ (por eso a menudo se reporta en vez de la varianza).
""")

    st.markdown("---")
    worked_example("Varianza de una Bernoulli(p)")

    st.markdown(r"""
$X \sim$ Bernoulli$(p)$, así que $X \in \{0, 1\}$ con $P(X=1) = p$.

$E[X] = 1 \cdot p + 0 \cdot (1-p) = p$.

$E[X^2] = 1^2 \cdot p + 0^2 \cdot (1-p) = p$ (¡igual a $E[X]$ porque $0^2=0$ y $1^2=1$!).

$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$.
""")

    st.markdown(r"""
**Interpretación.** La varianza es máxima cuando $p = 1/2$ (máxima incertidumbre) y cero cuando $p = 0$ o $p = 1$ (resultado determinista). Esta forma 'parábola invertida' aparecerá en la sección interactiva.
""")

    st.markdown("---")
    interactive_header("Ley de los grandes números y varianza de Bernoulli")

    tab_lgn, tab_guarda, tab_var = st.tabs(
        ["📉 Ley de grandes números", "🧥 Guardarropa simulado", "📊 Var(Bernoulli) vs p"],
    )

    with tab_lgn:
        st.markdown(
            "Promedia muchas realizaciones de una variable y mira cómo el "
            "promedio converge al valor esperado teórico."
        )
        dist_lgn = st.selectbox("Distribución:", ["Dado justo (E=3.5)", "Bernoulli(0.3) (E=0.3)", "Exponencial(λ=2) (E=0.5)"], key="distlgn_s8")
        n_lgn = st.slider("Número total de muestras", 100, 20000, 3000, step=100, key="nlgn_s8")
        rng = np.random.default_rng(7)
        if dist_lgn.startswith("Dado"):
            samples = rng.integers(1, 7, n_lgn); E_teo = 3.5
        elif dist_lgn.startswith("Bernoulli"):
            samples = rng.binomial(1, 0.3, n_lgn); E_teo = 0.3
        else:
            samples = rng.exponential(1/2, n_lgn); E_teo = 0.5

        running_mean = np.cumsum(samples) / np.arange(1, n_lgn + 1)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(running_mean, color="#4C72B0", linewidth=1.5, label="Promedio acumulado")
        ax.axhline(E_teo, color="#C44E52", linestyle="--", linewidth=2, label=f"E[X] teórico = {E_teo}")
        ax.set_xlabel("Número de muestras")
        ax.set_ylabel("Promedio")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "El eje X es cuántas muestras se llevan acumuladas. El eje Y es el "
            "**promedio** de todas las muestras hasta ese punto. La línea roja "
            "es el valor esperado teórico. **Observación clave**: al principio "
            "el promedio es muy ruidoso; con muchas muestras se 'clava' en el "
            "valor teórico. Esto es la Ley de los Grandes Números."
        )

    with tab_guarda:
        n_g = st.slider("Personas en el guardarropa (n)", 2, 200, 20, key="n_g_s8")
        n_sims_g = st.slider("Simulaciones", 500, 10000, 2000, step=500, key="nsims_g_s8")
        rng = np.random.default_rng(42)
        aciertos = np.array([
            int(np.sum(rng.permutation(n_g) == np.arange(n_g)))
            for _ in range(n_sims_g)
        ])

        col_m, col_p = st.columns([1, 1.5])
        with col_m:
            st.metric("Promedio empírico de aciertos", f"{aciertos.mean():.4f}")
            st.metric("Predicción teórica", "1.0000")
            st.metric("% de noches con 0 aciertos", f"{(aciertos == 0).mean()*100:.1f}%")
            st.caption(
                "Para $n$ grande, la probabilidad de 0 aciertos tiende a $1/e \\approx 36.8\\%$. "
                "Esto lleva al 'problema de los desarreglos' (derangements)."
            )

        with col_p:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            max_a = max(6, aciertos.max())
            bins = np.arange(-0.5, max_a + 1.5, 1)
            ax.hist(aciertos, bins=bins, color="#4C72B0", edgecolor="white")
            ax.axvline(aciertos.mean(), color="#C44E52", linestyle="--", linewidth=2,
                        label=f"promedio = {aciertos.mean():.3f}")
            ax.set_xlabel("Aciertos por noche")
            ax.set_ylabel("Frecuencia")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            how_to_read(
                "Cada barra es cuántas noches simuladas tuvieron ese número de "
                "aciertos. La línea roja marca el promedio empírico. Sin "
                "importar cuán grande pongas $n$, ese promedio gravita al **1.0** "
                "teórico — el resultado del análisis con indicadoras."
            )

    with tab_var:
        ps = np.linspace(0, 1, 200)
        vars_ = ps * (1 - ps)
        p_highlight = st.slider("Resalta un valor de p", 0.0, 1.0, 0.5, 0.01, key="p_var_s8")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(ps, vars_, color="#4C72B0", linewidth=2.2)
        ax.fill_between(ps, vars_, alpha=0.2, color="#4C72B0")
        ax.scatter([p_highlight], [p_highlight*(1-p_highlight)], color="#C44E52",
                   s=120, zorder=5, label=f"p={p_highlight:.2f} → Var={p_highlight*(1-p_highlight):.3f}")
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("p")
        ax.set_ylabel("Var(Bernoulli(p)) = p(1-p)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "Curva parabólica $p(1-p)$. Se anula en $p=0$ y $p=1$ "
            "(resultado determinista → sin incertidumbre) y alcanza el máximo "
            "$0{,}25$ en $p=0{,}5$ (máxima indecisión). Esta intuición se "
            "generaliza: **la varianza mide incertidumbre**."
        )

    st.markdown("---")
    self_check_header()

    quiz(
        "Si E[X] = 5 y E[Y] = 3, ¿cuánto vale E[2X − Y + 4]?",
        [
            "11 — por linealidad: 2·5 − 3 + 4 = 11.",
            "Depende de si X e Y son independientes.",
            "No se puede calcular sin más datos.",
            "8.",
        ],
        correct_idx=0,
        feedback_ok="Linealidad. E[2X − Y + 4] = 2·E[X] − E[Y] + 4 = 10 − 3 + 4 = 11. **No** se necesita independencia.",
        feedback_wrong="Linealidad de la esperanza: E[aX+bY+c] = a·E[X]+b·E[Y]+c. Vale sin suposiciones sobre independencia. 2·5−3+4 = 11.",
        key="q1_s8",
    )

    quiz(
        "Una Bernoulli(p) tiene la varianza **más alta** cuando…",
        [
            "p = 0 — no pasa nada, sin variación.",
            "p = 1 — pasa siempre, sin variación.",
            "p = 0.5 — máxima incertidumbre.",
            "p = 1/3.",
        ],
        correct_idx=2,
        feedback_ok="La varianza es p(1-p), parábola con máximo en p=0.5 donde vale 0.25. Es la situación de máxima impredecibilidad.",
        feedback_wrong="Var(Bernoulli) = p(1-p). Esta parábola alcanza su máximo en p=0.5 (0.25) y se anula en p=0 y p=1 (resultados deterministas).",
        key="q2_s8",
    )

    st.markdown("---")
    ai_bridge(
        "En RL, el valor esperado es la pieza central: el **valor** $V(s)$ de "
        "un estado es $E[\\text{retorno futuro} \\mid s]$, y los algoritmos "
        "(Q-learning, policy gradient) están construidos para estimarlo con "
        "muestreo. En deep learning, el **gradiente estocástico (SGD)** "
        "funciona porque $E[\\nabla L \\text{ en un minibatch}] = \\nabla E[L]$ "
        "(insesgado). La varianza del gradiente controla el ruido del "
        "entrenamiento: técnicas como Adam, momentum, batch norm, reducen varianza. "
        "Entender E y Var es entender **cuándo y por qué** un estimador es confiable."
    )


# ==================================================================
# 9. FGM Y COVARIANZA
# ==================================================================
elif choice == menu[8]:
    st.header("9. Función Generadora de Momentos y Covarianza")

    motivation(
        "Dos preguntas motivan esta sección. **Primera:** calcular $E[X]$ requiere "
        "una suma o integral; $E[X^2]$, otra; $E[X^3]$, otra… ¿hay una función "
        "'mágica' que los genere todos de una vez? Sí: la **FGM** (Función "
        "Generadora de Momentos). **Segunda:** ¿cómo cuantificamos si dos "
        "variables se mueven 'juntas'? Con la **covarianza** (y su prima "
        "estandarizada, la correlación). Juntas, estas dos herramientas son "
        "la base del análisis multivariado y aparecen por todas partes en ML."
    )

    prerequisites_box(r"""
- **Derivada**: la 'pendiente instantánea' de una función. Notación: $\frac{df}{dx}$ o $f'(x)$. Sólo la usamos conceptualmente; los cálculos los hacemos paso a paso.
- **Serie de Taylor de $e^x$**: la exponencial tiene la expansión
$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots$$
Esta identidad es la razón por la que la FGM funciona.
- **Valor esperado** (Sección 8): lo usamos como operador. Recuerda que $E[aX] = a\,E[X]$ y $E[X + Y] = E[X] + E[Y]$.
""")

    st.markdown("---")
    st.markdown("### 🧱 FGM — definición")

    st.latex(r"M_X(s) = E\left[e^{sX}\right]")
    st.markdown(r"""
Es una función de una nueva variable $s$ (no confundir con $X$). La definición usa la exponencial de $sX$, que parece arbitraria, **pero hay una razón profunda**.

**El 'truco' — por qué genera todos los momentos.** Expandamos $e^{sX}$ en serie de Taylor:
""")
    st.latex(r"e^{sX} = 1 + sX + \frac{s^2 X^2}{2!} + \frac{s^3 X^3}{3!} + \dots")
    st.markdown(r"Tomando valor esperado (usando linealidad en cada término):")
    st.latex(r"M_X(s) = 1 + s\,E[X] + \frac{s^2}{2!}E[X^2] + \frac{s^3}{3!}E[X^3] + \dots")
    st.markdown(r"""
¡Todos los momentos $E[X^n]$ están escondidos como coeficientes! ¿Cómo los extraemos? Derivando respecto a $s$ y evaluando en $s=0$:
""")
    st.latex(r"M'_X(0) = E[X], \quad M''_X(0) = E[X^2], \quad M^{(n)}_X(0) = E[X^n]")
    st.markdown(r"""
**Receta práctica:**
1. Calcula $M_X(s) = E[e^{sX}]$ (una sola integral/suma).
2. Deriva $n$ veces.
3. Evalúa en $0$.
4. Tienes $E[X^n]$ sin tener que calcular una integral nueva.

**Bonus**: dos variables con la misma FGM tienen la misma distribución (teorema de unicidad). La FGM 'codifica' la distribución por completo.
""")

    st.markdown("---")
    worked_example("FGM de una Exponencial(λ) y extracción de E[X], Var(X)")

    st.markdown(r"""
$X \sim$ Exp$(\lambda)$, con $f(x) = \lambda e^{-\lambda x}$ para $x \geq 0$.

**Paso 1 — calcular $M(s)$:**
""")
    st.latex(r"M(s) = \int_0^\infty e^{sx} \lambda e^{-\lambda x}\,dx = \lambda \int_0^\infty e^{-(\lambda - s)x}\,dx")
    st.latex(r"= \lambda \cdot \frac{1}{\lambda - s} = \frac{\lambda}{\lambda - s} \quad (\text{válido si } s < \lambda)")

    st.markdown(r"**Paso 2 — derivar respecto a $s$:**")
    st.latex(r"M'(s) = \frac{\lambda}{(\lambda - s)^2}, \qquad M''(s) = \frac{2\lambda}{(\lambda - s)^3}")

    st.markdown(r"**Paso 3 — evaluar en 0:**")
    st.latex(r"E[X] = M'(0) = \frac{\lambda}{\lambda^2} = \frac{1}{\lambda}")
    st.latex(r"E[X^2] = M''(0) = \frac{2\lambda}{\lambda^3} = \frac{2}{\lambda^2}")

    st.markdown(r"**Paso 4 — varianza:**")
    st.latex(r"\mathrm{Var}(X) = E[X^2] - \big(E[X]\big)^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}")

    st.markdown(r"""
**Todo sin hacer la integral $\int x^2 f(x)\,dx$.** Una sola integral para $M(s)$, y a partir de ahí los momentos salen derivando — un trámite algebraico, no analítico.
""")

    st.markdown("---")
    st.markdown("### 🧱 Covarianza — cuando se mueven juntas")

    st.latex(r"\mathrm{Cov}(X, Y) = E\left[(X - \mu_X)(Y - \mu_Y)\right] = E[XY] - E[X]\,E[Y]")

    st.markdown(r"""
**Interpretación visual.** Imagina el punto $(X, Y)$ respecto al centro $(\mu_X, \mu_Y)$:
- Si cuando $X$ está 'arriba' de $\mu_X$, $Y$ también está 'arriba' → producto positivo → Cov > 0.
- Si cuando $X$ está 'arriba', $Y$ está 'abajo' → producto negativo → Cov < 0.
- Si los signos se cancelan en promedio → Cov = 0.

**Advertencia sobre escalas.** La covarianza depende de las **unidades** de $X$ e $Y$. Si mides $X$ en metros y lo cambias a milímetros, la covarianza se multiplica por 1000. Por eso se usa una versión **estandarizada**:
""")
    st.latex(r"\rho_{X,Y} = \frac{\mathrm{Cov}(X, Y)}{\sigma_X\,\sigma_Y} \quad \in [-1, 1]")
    st.markdown(r"""
$\rho$ es la **correlación (de Pearson)**. Siempre cae en $[-1, 1]$:
- $\rho = 1$: perfecta relación lineal creciente.
- $\rho = -1$: perfecta relación lineal decreciente.
- $\rho = 0$: sin relación **lineal** (¡ojo con esta palabra!).

**Cuidado — error común.** Si $X$ e $Y$ son independientes, entonces $\mathrm{Cov}(X, Y) = 0$. **Lo inverso no es cierto**: puede haber $\mathrm{Cov} = 0$ con $X$ e $Y$ fuertemente dependientes si la relación es **no lineal**. Ejemplo: $X$ uniforme en $[-1,1]$ y $Y = X^2$. Cov $= 0$ pero $Y$ es determinada por $X$.
""")

    st.markdown("---")
    interactive_header("Explora la FGM y la covarianza")

    tab_fgm, tab_cov = st.tabs(["🔧 FGM de distribuciones simples", "🎯 Covarianza y correlación 2D"])

    with tab_fgm:
        st.markdown(
            "Elige una distribución, y la app te muestra su FGM cerrada y los "
            "momentos $E[X]$ y $E[X^2]$ obtenidos derivando."
        )
        dist_fgm = st.selectbox("Distribución:",
                                ["Bernoulli(p)", "Exponencial(λ)", "Poisson(λ)", "Normal(μ, σ²)"],
                                key="distfgm_s9")
        s_eval = st.slider("Evalúa M(s) en s =", -1.0, 1.0, 0.3, 0.05, key="s_eval_s9")

        if dist_fgm.startswith("Bernoulli"):
            p = st.slider("p", 0.01, 0.99, 0.4, 0.01, key="p_fgm_s9")
            st.latex(r"M(s) = (1 - p) + p e^s")
            val = (1 - p) + p * math.exp(s_eval)
            E, E2 = p, p
        elif dist_fgm.startswith("Exponencial"):
            lam = st.slider("λ", 0.1, 5.0, 1.5, 0.1, key="lam_fgm_s9")
            st.latex(r"M(s) = \frac{\lambda}{\lambda - s}, \quad s < \lambda")
            val = lam / (lam - s_eval) if s_eval < lam else np.inf
            E, E2 = 1/lam, 2/lam**2
        elif dist_fgm.startswith("Poisson"):
            lam = st.slider("λ", 0.1, 10.0, 3.0, 0.1, key="lam2_fgm_s9")
            st.latex(r"M(s) = \exp\!\big(\lambda(e^s - 1)\big)")
            val = math.exp(lam * (math.exp(s_eval) - 1))
            E, E2 = lam, lam + lam**2
        else:
            mu = st.slider("μ", -3.0, 3.0, 0.0, 0.1, key="mu_fgm_s9")
            sig = st.slider("σ", 0.1, 3.0, 1.0, 0.1, key="sig_fgm_s9")
            st.latex(r"M(s) = \exp\!\Big(\mu s + \tfrac{1}{2}\sigma^2 s^2\Big)")
            val = math.exp(mu * s_eval + 0.5 * sig**2 * s_eval**2)
            E, E2 = mu, mu**2 + sig**2

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("M(s) evaluada", f"{val:.4f}" if np.isfinite(val) else "∞")
        col_b.metric("E[X] = M'(0)", f"{E:.4f}")
        col_c.metric("E[X²] = M''(0)", f"{E2:.4f}")

        ss = np.linspace(-1, 1, 200)
        if dist_fgm.startswith("Bernoulli"):
            ms = (1 - p) + p * np.exp(ss)
        elif dist_fgm.startswith("Exponencial"):
            ms = np.where(ss < lam, lam / (lam - ss), np.nan)
        elif dist_fgm.startswith("Poisson"):
            ms = np.exp(lam * (np.exp(ss) - 1))
        else:
            ms = np.exp(mu * ss + 0.5 * sig**2 * ss**2)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(ss, ms, color="#4C72B0", linewidth=2.2, label="M(s)")
        ax.axvline(0, color="gray", linestyle=":")
        ax.scatter([s_eval], [val], color="#C44E52", s=80, zorder=5, label=f"M({s_eval:.2f})")
        ax.scatter([0], [1.0], color="black", s=80, zorder=5, label="M(0)=1")
        ax.set_xlabel("s")
        ax.set_ylabel("M(s)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        how_to_read(
            "La FGM $M(s)$ es una función de $s$ — **no de $X$**. Siempre pasa por "
            "$M(0) = 1$ (porque $e^0 = 1$). La **pendiente** en $s=0$ es $E[X]$. "
            "La **curvatura** en $s=0$ codifica $E[X^2]$. Derivando y evaluando "
            "en cero obtienes todos los momentos sin integrales adicionales."
        )

    with tab_cov:
        st.markdown(
            "Ajusta la correlación $\\rho$ y observa cómo cambia la forma de la "
            "nube de puntos de una Normal bivariada con medias 0 y varianzas 1."
        )
        rho = st.slider("Correlación ρ", -0.99, 0.99, 0.6, 0.01, key="rho_s9")
        n_pts = st.slider("Número de puntos", 200, 5000, 1500, step=100, key="npts_s9")
        rng = np.random.default_rng(3)
        cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
        data_2d = rng.multivariate_normal([0, 0], cov_matrix, n_pts)

        cov_emp = np.cov(data_2d.T)[0, 1]
        corr_emp = np.corrcoef(data_2d.T)[0, 1]

        col_metrics, col_plot = st.columns([1, 1.3])
        with col_metrics:
            st.metric("Cov(X,Y) empírica", f"{cov_emp:.3f}")
            st.metric("Correlación ρ empírica", f"{corr_emp:.3f}")
            st.markdown("**Matriz de covarianza teórica:**")
            st.latex(
                r"\Sigma = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix} = "
                fr"\begin{{pmatrix}} 1 & {rho:.2f} \\ {rho:.2f} & 1 \end{{pmatrix}}"
            )
            if abs(rho) < 0.05:
                caso = "**Sin relación lineal.** Nube circular, $\\rho \\approx 0$."
            elif rho > 0:
                caso = "**Relación lineal positiva.** Al crecer $X$, tiende a crecer $Y$."
            else:
                caso = "**Relación lineal negativa.** Al crecer $X$, tiende a decrecer $Y$."
            st.info(caso)

        with col_plot:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            ax.scatter(data_2d[:, 0], data_2d[:, 1], s=8, alpha=0.35, color="#4C72B0")
            ax.axhline(0, color="gray", linewidth=0.7, alpha=0.6)
            ax.axvline(0, color="gray", linewidth=0.7, alpha=0.6)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            st.pyplot(fig)
            plt.close(fig)

            how_to_read(
                "Cada punto es una muestra $(x, y)$ de la Normal bivariada. "
                "Si la nube parece un **círculo** → $\\rho \\approx 0$. "
                "Si se **inclina a 45°** → $\\rho > 0$. Si se **inclina a -45°** "
                "→ $\\rho < 0$. Cuanto más angosta la elipse, más cerca de $\\pm 1$. "
                "La correlación captura la fuerza de la tendencia **lineal** — "
                "relaciones curvas (por ejemplo $Y = X^2$) pueden dar $\\rho = 0$ "
                "aunque $X$ e $Y$ no sean independientes."
            )

    st.markdown("---")
    self_check_header()

    quiz(
        "Si $X \\sim$ Poisson($\\lambda$), entonces $M(s) = \\exp(\\lambda(e^s - 1))$. ¿Cuánto es $E[X]$?",
        [
            "$\\lambda^2$.",
            "$\\lambda$ — se obtiene derivando M(s) una vez y evaluando en 0.",
            "$e^\\lambda$.",
            "$\\lambda(e - 1)$.",
        ],
        correct_idx=1,
        feedback_ok="M'(s) = λ·e^s·exp(λ(e^s−1)). En s=0: λ·1·exp(0) = λ. Así que E[X] = λ, como es bien conocido para Poisson.",
        feedback_wrong="Deriva M respecto a s (aplicando regla de la cadena) y evalúa en s=0. Queda M'(0) = λ. Es la forma rápida de ver que E[Poisson(λ)] = λ.",
        key="q1_s9",
    )

    quiz(
        "$X$ es uniforme en $[-1, 1]$ y $Y = X^2$. ¿Cuál es $\\mathrm{Cov}(X, Y)$?",
        [
            "Positiva y alta — $Y$ depende completamente de $X$.",
            "Cero — y sin embargo $Y$ está completamente determinada por $X$.",
            "Negativa — valores grandes de $X$ dan $Y$ chico.",
            "No se puede calcular sin datos.",
        ],
        correct_idx=1,
        feedback_ok="Cov(X, X²) = E[X³] − E[X]E[X²]. Por simetría E[X] = E[X³] = 0, así que Cov = 0. Pero Y = X² es totalmente predecible desde X: Cov=0 NO implica independencia.",
        feedback_wrong="Cov(X, X²) = E[X³] − E[X]·E[X²]. Como X es simétrico en [-1,1], E[X] = E[X³] = 0 y Cov = 0. Sin embargo Y = X² es dependiente de X. Este es el ejemplo clásico de que 'no correlación ≠ independencia'.",
        key="q2_s9",
    )

    st.markdown("---")
    ai_bridge(
        "La **matriz de covarianza** de los features es el objeto central de PCA "
        "(Análisis de Componentes Principales): sus autovectores dan las "
        "direcciones de máxima varianza, que se usan para reducción de "
        "dimensionalidad. En estadística multivariada y en modelos gaussianos "
        "(como los Gaussian Processes), $\\Sigma$ es literalmente el objeto que "
        "parametriza la distribución. La FGM, por su parte, aparece cuando se "
        "demuestran propiedades asintóticas (por ejemplo, una prueba corta del "
        "Teorema Central del Límite pasa por ver que la FGM de la suma "
        "normalizada converge a la FGM de la Normal). Y en ML moderno, "
        "'correlación' sigue siendo el diagnóstico más común para detectar "
        "redundancia entre features o fugas de etiqueta a datos."
    )




