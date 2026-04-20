# Clase 5: Fundamentos de Probabilidad para IA

Este folder contiene una aplicación interactiva desarrollada en Streamlit para visualizar y experimentar con todos los contenidos de la Clase 5 del curso MIA.

## Contenidos Cubiertos
- **Eventos y Probabilidades**: Simulaciones de monedas, dados y el problema del cumpleaños.
- **Probabilidad Condicional**: Simulador del problema de Monty Hall y calculadora de Bayes para diagnósticos médicos o filtros de spam.
- **Naïve Bayes**: Implementación interactiva con el *Wine Dataset* para clasificar tipos de vino usando atributos químicos.
- **Distribuciones**: Visualizador de PMF y PDF para las distribuciones clásicas (Bernoulli, Normal, Poisson, etc).
- **Inferencia (MLE)**: Visualización de la función de verosimilitud y pérdida de entropía cruzada.
- **Momentos y Esperanza**: Simulación de la ley de los grandes números y cálculo de momentos vía FGM.
- **Distribuciones Conjuntas**: Mapa de calor de una normal bivariada variando la covarianza.

## Instrucciones de Uso

1. **Instalar dependencias**:
   Asegúrate de tener instalado Python y las librerías necesarias:
   ```bash
   pip install streamlit numpy pandas matplotlib seaborn scipy scikit-learn
   ```

2. **Ejecutar la App**:
   Desde la terminal, sitúate en este directorio y corre:
   ```bash
   streamlit run app_clase5.py
   ```

3. **Interacción**:
   Usa el menú lateral para navegar por los distintos temas y ajusta los sliders para ver cómo cambian los resultados y las gráficas en tiempo real.
