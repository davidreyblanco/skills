# Algoritmos de Clustering
## Máster en Ciencia de Datos — Syllabus del Módulo

---

| | |
|---|---|
| **Módulo** | Algoritmos de Clustering |
| **Sesiones** | 2 sesiones presenciales (con alumnos remotos) |
| **Duración total** | 10 horas (2 × 5 h) |
| **Nivel** | Avanzado — prerrequisito: fundamentos de ML y Python |
| **Lenguaje** | Python (scikit-learn, scipy, pandas, matplotlib, seaborn) |
| **Entorno de trabajo** | Jupyter Notebooks |
| **Modalidad** | Presencial híbrido |

---

## 1. Descripción del módulo

Este módulo proporciona una formación intensiva y aplicada sobre los principales algoritmos de clustering utilizados en ciencia de datos. A lo largo de las dos sesiones los participantes comprenderán los fundamentos matemáticos de los métodos de agrupamiento, aprenderán a seleccionar el algoritmo adecuado según el problema y desarrollarán la capacidad de implementar, evaluar y comunicar soluciones de clustering en contextos de negocio reales.

El módulo está diseñado para ingenieros y programadores con experiencia previa en aprendizaje automático supervisado, por lo que la curva de aprendizaje hace énfasis en la profundidad algorítmica, los criterios de selección y la aplicabilidad práctica. Toda la práctica se realiza en **Jupyter Notebooks**, facilitando la experimentación interactiva y la entrega de ejercicios.

---

## 2. Objetivos de aprendizaje

Al finalizar el módulo, el alumno será capaz de:

1. **Comprender** las diferencias entre aprendizaje no supervisado y supervisado, y contextualizar el clustering dentro del ecosistema del ML.
2. **Implementar** los algoritmos K-Means, K-Medoids, Clustering Jerárquico, DBSCAN, Gaussian Mixture Models y Mapas Auto-Organizados de Kohonen en Python.
3. **Comparar** K-Means y K-Medoids, identificando cuándo la robustez ante outliers justifica el mayor coste computacional.
4. **Seleccionar** el algoritmo de clustering más adecuado en función de la naturaleza de los datos, el tamaño del dataset y los objetivos del negocio.
5. **Evaluar** la calidad de una solución de clustering utilizando métricas cuantitativas (Silhouette, Davies-Bouldin, Calinski-Harabasz) y métodos visuales.
6. **Interpretar** y comunicar los resultados de clustering a audiencias técnicas y no técnicas.
7. **Aplicar** técnicas de reducción de dimensionalidad (PCA, t-SNE) como paso previo para mejorar el clustering en datos de alta dimensionalidad.
8. **Resolver** casos prácticos de segmentación de clientes, análisis de mercado y agrupación genérica de datos.

---

## 3. Prerrequisitos

Los alumnos deberán tener conocimiento de los siguientes conceptos antes de comenzar el módulo:

- Python intermedio (NumPy, Pandas, Matplotlib)
- Fundamentos de álgebra lineal (distancias, vectores, matrices de covarianza)
- Conceptos básicos de estadística descriptiva
- Experiencia con scikit-learn (pipelines, train/test split, métricas básicas)
- Aprendizaje supervisado (clasificación y regresión)
- Manejo básico de Jupyter Notebooks

---

## 4. Estructura del módulo

### SESIÓN 1 — Fundamentos y Algoritmos Clásicos (5 horas)

> **Objetivo de sesión:** Construir una base sólida sobre el problema del clustering, dominar los algoritmos más utilizados en la industria y comprender sus supuestos, limitaciones y casos de uso.

| Bloque | Tema | Duración |
|---|---|---|
| 1.1 | Introducción al Clustering y al Espacio de Distancias | 55 min |
| 1.2 | K-Means y K-Medoids | 110 min |
| 1.3 | Clustering Jerárquico | 65 min |
| 1.4 | DBSCAN: Clustering Basado en Densidad | 60 min |
| — | Recapitulación + Q&A | 10 min |
| | **Total** | **300 min (5 h)** |

---

#### Bloque 1.1 — Introducción al Clustering y al Espacio de Distancias (55 min)

**Teoría (25 min)**
- ¿Qué es el aprendizaje no supervisado? Diferencias con aprendizaje supervisado y semi-supervisado.
- El problema del clustering: definición formal, tipos (particional, jerárquico, basado en densidad, probabilístico, neuronal).
- Aplicaciones reales: segmentación de clientes, detección de anomalías, compresión de imágenes, agrupación de documentos.
- Métricas de distancia y similitud: Euclidiana, Manhattan, Chebyshev, coseno. ¿Cuándo usar cada una?
- La maldición de la dimensionalidad y su impacto en clustering.

**Práctica guiada en Jupyter Notebook (30 min)**
- Exploración visual de datasets sintéticos con `make_blobs`, `make_moons`, `make_circles`.
- Cálculo y visualización de matrices de distancia con `scipy.spatial.distance`.
- Discusión: ¿qué hace que un buen cluster sea "bueno"?

---

#### Bloque 1.2 — K-Means y K-Medoids (110 min)

> Este bloque estudia los dos algoritmos particionales más importantes: K-Means como referencia estándar y K-Medoids como su extensión robusta ante outliers.

**Teoría K-Means (30 min)**
- Algoritmo Lloyd: asignación, actualización, convergencia. Complejidad computacional O(n·k·d·i).
- Sensibilidad a la inicialización: K-Means++ como mejora estándar.
- Elección de K: método del codo (WCSS), regla empírica √(n/2).
- Limitaciones: clusters esféricos, sensibilidad a outliers, escala de variables.
- Variantes: Mini-Batch K-Means para grandes volúmenes de datos.

**Práctica K-Means en Jupyter Notebook (35 min)**
- Implementación paso a paso con `sklearn.cluster.KMeans`.
- **Caso práctico:** Segmentación de clientes con el dataset *Mall Customers* (ingresos vs. gasto). Construcción del gráfico del codo e interpretación de segmentos.
- Comparación visual: K-Means vs. K-Means++ con distintas semillas.
- Normalización de variables y su impacto en los resultados.

**Teoría K-Medoids (20 min)**
- De centroide a medoide: ¿por qué usar puntos reales del dataset?
- Algoritmo PAM (Partitioning Around Medoids): inicialización, fase BUILD y fase SWAP.
- Complejidad computacional O(k·(n−k)²) por iteración vs. K-Means: cuándo merece la pena.
- Robustez ante outliers: demostración intuitiva. Ventaja en espacios donde la media no tiene sentido (datos categóricos, distancias no-euclidianas).
- Variantes escalables: CLARA (muestreo), CLARANS (búsqueda aleatoria de vecinos).

**Práctica K-Medoids en Jupyter Notebook (25 min)**
- Implementación con `sklearn_extra.cluster.KMedoids` (pip: `scikit-learn-extra`).
- **Ejercicio comparativo:** Aplicar K-Means y K-Medoids sobre el mismo dataset con outliers artificiales. Visualizar y medir el impacto de cada outlier en centroide vs. medoide.
- Discusión: ¿cuándo elegirías K-Medoids sobre K-Means en un proyecto real?

---

#### Bloque 1.3 — Clustering Jerárquico (65 min)

**Teoría (25 min)**
- Enfoque aglomerativo vs. divisivo.
- Criterios de enlace: simple, completo, promedio, Ward. Efectos en la forma de los clusters.
- Dendrogramas: lectura, corte óptimo, umbral de distancia.
- Complejidad computacional: O(n²) en tiempo y espacio. Escalabilidad.

**Práctica guiada en Jupyter Notebook (40 min)**
- Implementación con `scipy.cluster.hierarchy` y `sklearn.cluster.AgglomerativeClustering`.
- Visualización de dendrogramas y selección del número de clusters.
- **Caso práctico:** Agrupación de países según indicadores macroeconómicos (dataset World Bank simplificado).
- Comparación de linkage: Ward vs. complete vs. average en el mismo dataset.

---

#### Bloque 1.4 — DBSCAN: Clustering Basado en Densidad (60 min)

**Teoría (25 min)**
- Conceptos clave: punto núcleo, punto frontera, ruido (outlier). Parámetros ε (eps) y MinPts.
- Algoritmo: alcanzabilidad por densidad, conectividad. Complejidad O(n log n) con árbol kd.
- Ventajas sobre K-Means y K-Medoids: clusters de forma arbitraria, detección nativa de outliers, no requiere especificar K.
- Desafíos: clusters de densidad variable, elección de hiperparámetros.
- HDBSCAN como extensión moderna.

**Práctica guiada en Jupyter Notebook (35 min)**
- Implementación con `sklearn.cluster.DBSCAN`.
- Técnica del gráfico k-distancia para seleccionar ε.
- **Caso práctico:** Detección de patrones de compra anómalos en datos de transacciones de e-commerce.
- Comparación DBSCAN vs. K-Means en datasets con ruido y formas no convexas.

---

#### Recapitulación Sesión 1 + Q&A (10 min)

Tabla comparativa de los cuatro algoritmos:

| Algoritmo | Tipo | Clusters | Outliers | Complejidad | Ideal para |
|---|---|---|---|---|---|
| K-Means | Particional | Esféricos | Sensible | O(n·k·d·i) | Datos grandes, bien separados |
| K-Medoids | Particional | Esféricos | **Robusto** | O(k·(n−k)²) | Outliers presentes, distancias no-euclidianas |
| Jerárquico | Jerárquico | Cualquier forma | Moderado | O(n²) | Exploración, dendrogramas, n pequeño |
| DBSCAN | Densidad | **Forma arbitraria** | **Nativo** | O(n log n) | Datos geoespaciales, anomalías |

- Preguntas abiertas y discusión sobre casos propios de los alumnos.
- Preview de la Sesión 2.

---

### SESIÓN 2 — Métodos Avanzados, Evaluación y Proyectos Reales (5 horas)

> **Objetivo de sesión:** Profundizar en el enfoque probabilístico y neuronal del clustering, dominar las métricas de evaluación, aplicar reducción de dimensionalidad y completar un proyecto integrador de principio a fin.

| Bloque | Tema | Duración |
|---|---|---|
| 2.1 | Gaussian Mixture Models (GMM) y el Algoritmo EM | 70 min |
| 2.2 | Mapas Auto-Organizados de Kohonen (SOM) | 60 min |
| 2.3 | Métricas de Evaluación de Clustering | 65 min |
| 2.4 | Reducción de Dimensionalidad para Clustering | 35 min |
| 2.5 | Proyecto Integrador | 55 min |
| 2.6 | Buenas Prácticas, Escalabilidad y Cierre | 15 min |
| | **Total** | **300 min (5 h)** |

---

#### Bloque 2.1 — Gaussian Mixture Models (GMM) y el Algoritmo EM (70 min)

**Teoría (35 min)**
- Limitaciones del clustering duro: por qué necesitamos probabilidades.
- Modelo de mezcla de gaussianas: formulación matemática, parámetros (μ, Σ, π).
- Algoritmo Expectation-Maximization (EM): paso E (responsabilidades), paso M (actualización de parámetros). Convergencia.
- Selección del número de componentes: BIC y AIC.
- Comparación K-Means vs. GMM: clusters esféricos vs. elípticos, asignaciones duras vs. suaves.

**Práctica guiada en Jupyter Notebook (35 min)**
- Implementación con `sklearn.mixture.GaussianMixture`.
- Visualización de elipses de covarianza y probabilidades de pertenencia.
- **Caso práctico:** Perfilado probabilístico de clientes de telecomunicaciones (churn dataset).
- Selección de número de componentes mediante curvas BIC/AIC.

---

#### Bloque 2.2 — Mapas Auto-Organizados de Kohonen (SOM) (60 min)

> Los SOM son una familia de redes neuronales no supervisadas que combinan clustering y reducción de dimensionalidad. Producen una representación topológica de los datos, preservando las relaciones de vecindad del espacio original en una cuadrícula de baja dimensión.

**Teoría (30 min)**
- Arquitectura de un SOM: capa de entrada, capa de mapa (cuadrícula 2D), pesos sinápticos.
- Entrenamiento: selección de la BMU (*Best Matching Unit*), función de vecindad (gaussiana, burbuja), tasa de aprendizaje y su decaimiento.
- Convergencia y preservación topológica: ¿qué significa que el mapa "aprende" la estructura de los datos?
- Interpretación del mapa: U-Matrix (distancias entre neuronas), mapas de componentes (component planes), mapas de activación.
- Ventajas: visualización intuitiva en 2D, escalabilidad, sin necesidad de especificar K a priori.
- Limitaciones: hiperparámetros sensibles (tamaño del mapa, tasa de aprendizaje, épocas), no probabilístico, difícil comparación con métricas estándar.
- Aplicaciones reales: análisis de riesgo crediticio, perfiles de comportamiento de usuarios, visualización de datos genómicos.

**Práctica guiada en Jupyter Notebook (30 min)**
- Implementación con la librería `minisom` (instalación: `pip install minisom`).
- Entrenamiento de un SOM sobre el dataset de segmentación de clientes.
- Visualización de la U-Matrix y los component planes.
- **Caso práctico:** Mapa de perfiles de clientes de e-commerce — identificar zonas del mapa con comportamiento similar e interpretarlas en términos de negocio.
- Comparación: ¿los clusters del SOM coinciden con los obtenidos por K-Means y GMM?

---

#### Bloque 2.3 — Métricas de Evaluación de Clustering (65 min)

**Teoría (25 min)**

*Métricas internas (sin ground truth):*
- **Coeficiente Silhouette:** interpretación [-1, 1], análisis por muestra individual y promedio global.
- **Índice Davies-Bouldin:** compacidad vs. separación, valores menores son mejores.
- **Índice Calinski-Harabasz (Variance Ratio Criterion):** dispersión inter/intra cluster, valores mayores son mejores.
- **Inercia / WCSS:** útil para el método del codo pero no comparable entre distintos K.

*Métricas externas (con ground truth, para evaluación supervisada):*
- Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Fowlkes-Mallows.

*Criterios de selección de métrica según contexto. Nota sobre SOMs: las métricas estándar son aplicables sobre las asignaciones del BMU.*

**Práctica guiada en Jupyter Notebook (40 min)**
- Implementación completa de un pipeline de evaluación con `sklearn.metrics`.
- Comparación sistemática: K-Means, K-Medoids, Jerárquico, DBSCAN y GMM sobre el mismo dataset con las tres métricas internas.
- **Ejercicio:** Dashboard de evaluación — selección automática del mejor algoritmo y K óptimo para un dataset desconocido.
- Discusión: ¿pueden las métricas mentir? Casos donde el mejor score no es la mejor solución.

---

#### Bloque 2.4 — Reducción de Dimensionalidad para Clustering (35 min)

**Teoría (15 min)**
- ¿Por qué reducir dimensiones antes de clusterizar? La maldición de la dimensionalidad revisitada.
- PCA como preprocesado: proyección a componentes principales, varianza explicada acumulada.
- t-SNE para visualización: parámetro perplexity, stochastic vs. determinístico. Advertencias de uso.
- UMAP como alternativa moderna: velocidad y preservación de estructura global.

**Práctica guiada en Jupyter Notebook (20 min)**
- Pipeline: PCA → K-Means vs. K-Means directo en dataset de alta dimensionalidad.
- Visualización de clusters con t-SNE en 2D. Comparación con U-Matrix del SOM.
- **Demo rápida:** Clustering de productos de retail con descripción textual (bag-of-words + PCA + K-Means).

---

#### Bloque 2.5 — Proyecto Integrador (55 min)

**Enunciado del proyecto:**

Los alumnos trabajarán en grupos de 2-3 personas sobre un dataset real de segmentación de clientes de e-commerce (basado en RFM: Recency, Frequency, Monetary). El objetivo es identificar segmentos accionables de clientes y proponer estrategias de negocio para cada segmento.

El trabajo se entrega como un **Jupyter Notebook** estructurado con celdas de código, visualizaciones y celdas de texto explicativo en Markdown.

**Pasos guiados:**

1. **Exploración y preprocesado** (10 min): análisis exploratorio, tratamiento de outliers, escalado.
2. **Modelado** (20 min): aplicar al menos 2 algoritmos de la sesión con justificación de elección.
3. **Evaluación** (10 min): comparar soluciones con métricas cuantitativas y visualizaciones.
4. **Interpretación de negocio** (10 min): nombrar los segmentos, caracterizar comportamientos y proponer acciones de marketing.
5. **Presentación** (5 min): cada grupo comparte hallazgos principales (1-2 min/grupo).

---

#### Bloque 2.6 — Buenas Prácticas, Escalabilidad y Cierre (15 min)

- Checklist pre-clustering: escalado, outliers, correlación entre variables.
- Errores frecuentes: no escalar los datos, ignorar outliers, asumir que el score más alto es la mejor solución, interpretar un SOM sin validar con métricas.
- Mapa visual final: todos los algoritmos estudiados y criterios de selección.
- Lecturas y recursos recomendados.
- Preguntas finales.

---

## 5. Evaluación

| Componente | Peso | Descripción |
|---|---|---|
| Proyecto integrador (Sesión 2) | 50% | Entrega del Jupyter Notebook con análisis completo y presentación oral |
| Ejercicios de práctica guiada | 30% | Jupyter Notebooks completados y entregados tras cada sesión |
| Participación activa | 20% | Contribución a discusiones, preguntas y debates |

---

## 6. Recursos y materiales

### Libros de referencia

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.) — Capítulos 9-10.
- Aggarwal, C. C. (2015). *Data Mining: The Textbook* — Capítulo 6 (Cluster Analysis).
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning* — Capítulo 9 (Mixture Models and EM).
- Kohonen, T. (2001). *Self-Organizing Maps* (3rd ed.). Springer. — Referencia original del algoritmo.

### Cursos MOOC de referencia

| Curso | Plataforma | Enfoque |
|---|---|---|
| Machine Learning: Clustering & Retrieval — U. Washington | Coursera | Teórico-matemático |
| ML Specialization — Andrew Ng / Stanford | Coursera | Práctico |
| Cluster Analysis in Data Mining — U. Illinois (UIUC) | Coursera | Data Mining |
| Unsupervised Machine Learning — IBM | Coursera | Aplicado |

### Documentación Python

- [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn-extra — KMedoids](https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html)
- [MiniSom — Self-Organizing Maps](https://github.com/JustGlowing/minisom)
- [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [scikit-learn — Clustering metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)

### Datasets utilizados en las sesiones

| Dataset | Fuente | Sesión | Uso |
|---|---|---|---|
| Mall Customers | Kaggle | 1 | K-Means / K-Medoids — segmentación |
| World Bank Indicators (simplificado) | World Bank / Kaggle | 1 | Jerárquico |
| E-commerce Transactions | UCI / Kaggle | 1 | DBSCAN — anomalías |
| Telecom Churn | Kaggle / IBM | 2 | GMM — perfilado probabilístico |
| E-commerce RFM (comportamiento clientes) | UCI / Kaggle | 2 | SOM — mapa de perfiles |
| Online Retail RFM | UCI ML Repository | 2 | Proyecto integrador |

---

## 7. Herramientas y entorno

### Setup recomendado

```bash
# Crear entorno virtual
python -m venv venv-clustering
source venv-clustering/bin/activate  # Linux/Mac
venv-clustering\Scripts\activate      # Windows

# Instalar dependencias base
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter

# Algoritmos adicionales
pip install scikit-learn-extra  # K-Medoids (KMedoids)
pip install minisom              # Self-Organizing Maps de Kohonen

# Herramientas de visualización y clustering avanzado
pip install umap-learn hdbscan yellowbrick

# Lanzar Jupyter
jupyter notebook
```

### Librerías principales

| Librería | Uso principal |
|---|---|
| `scikit-learn` | KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, métricas |
| `scikit-learn-extra` | **KMedoids** (PAM, CLARA) |
| `minisom` | **Self-Organizing Maps de Kohonen** |
| `scipy` | Dendrogramas, métricas de distancia |
| `pandas` | Manipulación de datos |
| `matplotlib` / `seaborn` | Visualización |
| `yellowbrick` | Visualizaciones especializadas para ML (elbow, silhouette plots) |
| `umap-learn` | Reducción de dimensionalidad moderna |
| `hdbscan` | Extensión jerárquica de DBSCAN |
| `jupyter` | Entorno de trabajo interactivo (Notebooks) |

---

## 8. Notas para el instructor

### Gestión del formato híbrido (presencial + remoto)

- Usar **Miro o Jamboard** para dinámicas de grupo abiertas (ej. categorizar algoritmos, mapas conceptuales).
- Los Jupyter Notebooks de práctica deben estar disponibles en un repositorio GitHub o Google Colab antes de cada sesión.
- Las secciones de práctica guiada funcionan bien en formato *live coding*: el instructor programa en directo en el Notebook y los alumnos replican o modifican en su propia copia.
- Para el proyecto integrador: usar salas de breakout (Zoom/Teams) para los grupos remotos y rotación de presencia para los grupos presenciales.

### Adaptaciones según el ritmo del grupo

- Si el grupo avanza rápido en la Sesión 1: profundizar en CLARA/CLARANS (variantes de K-Medoids) y HDBSCAN.
- Si el grupo necesita más tiempo en el Bloque 1.2: reducir la práctica de K-Medoids a solo la demo comparativa y omitir CLARA.
- Si el grupo necesita más tiempo en el Bloque 2.2 (SOM): reducir el Bloque 2.4 (reducción de dimensionalidad) a una demo visual sin código propio.
- El Bloque 2.3 (métricas) suele generar mucha discusión — reservar tiempo adicional si es necesario.

### Consejos para fomentar la aplicabilidad

- Pedir a los alumnos que durante la Sesión 1 identifiquen un problema de clustering de su empresa o sector.
- En el proyecto integrador (Bloque 2.5), permitir que cada grupo use su propio dataset si lo tienen disponible.
- Para los SOMs: invitar a los alumnos a pensar en qué dimensiones del mapa representarían y cómo lo comunicarían a un equipo de marketing.
- Cerrar siempre los bloques con la pregunta: *"¿Cuándo usarías esto en tu trabajo?"*

---

*Syllabus desarrollado con referencia a programas de Coursera (U. Washington, Stanford/DeepLearning.AI, IBM, UIUC) y documentación oficial de scikit-learn. Algoritmo SOM basado en Kohonen (2001).*
