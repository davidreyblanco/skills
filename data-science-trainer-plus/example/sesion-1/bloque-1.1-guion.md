# Bloque 1.1 — Introducción al Clustering y al Espacio de Distancias
## Guión detallado del instructor

**Duración:** 55 minutos (25 min teoría + 30 min práctica en Jupyter Notebook)
**Posición en la sesión:** Apertura de la Sesión 1

---

## PARTE TEÓRICA (25 min)

---

### [00:00 – 00:05] Apertura y encuadre de las dos sesiones

> *Nota para el instructor: Antes de arrancar con teoría, dedica 5 minutos a establecer el contrato de aprendizaje del módulo. Es fundamental que los alumnos entiendan desde el principio qué van a hacer, por qué importa y cómo se van a evaluar.*

**Script de apertura:**

*"Bienvenidos al módulo de clustering. Vamos a trabajar juntos durante 10 horas repartidas en dos sesiones. Mi objetivo no es que seáis capaces de recitar definiciones, sino que al final de la segunda sesión podáis coger un dataset real, aplicar el algoritmo correcto, evaluar si vuestro resultado es bueno y explicárselo a alguien de negocio."*

*"Una advertencia antes de empezar: el clustering es el área del Machine Learning donde es más fácil obtener resultados que parezcan bonitos pero que no signifiquen nada. Parte de lo que vamos a aprender es a detectar cuándo un modelo de clustering está mintiendo."*

**Slide sugerida:** Índice de ambas sesiones con los 10 bloques y duración. Resaltar que cada bloque de teoría va siempre seguido de práctica en Jupyter Notebook.

---

### [00:05 – 00:12] ¿Qué es el aprendizaje no supervisado?

**Concepto central:**

En el aprendizaje supervisado, cada muestra del dataset tiene una etiqueta `y` que le dice al modelo qué es correcto. El modelo aprende a mapear `X → y`. En el aprendizaje no supervisado no hay `y`. Solo tenemos `X`, y el objetivo es encontrar estructura que ya existe en los datos pero que nadie ha etiquetado todavía.

**Analogía para explicarlo:**

*"Imaginad que os doy mil fotos de animales sin nombres. Naturalmente, vuestro cerebro empezaría a agrupar: estos tienen orejas grandes, estos tienen plumas, estos tienen escamas. No os digo nada, pero encontráis grupos que tienen sentido. Eso es exactamente lo que hace un algoritmo de clustering."*

**Los tres grandes tipos de aprendizaje:**

| Tipo | ¿Tiene etiquetas? | Objetivo | Ejemplo |
|---|---|---|---|
| Supervisado | Sí (todas) | Predecir `y` | Clasificar emails spam |
| Semi-supervisado | Parcialmente | Predecir `y` con pocos datos etiquetados | Clasificar con 10% etiquetado |
| No supervisado | No | Descubrir estructura en `X` | Agrupar clientes por comportamiento |

**Punto de discusión rápido (1 min):**
*"¿Cuándo en vuestro trabajo no tendríais etiquetas? Ejemplos: logs de servidor, transacciones, clickstream de usuarios..."*

---

### [00:12 – 00:17] El problema del clustering: definición y tipos

**Definición formal:**

Dado un conjunto de `n` observaciones `{x₁, x₂, ..., xₙ}` en un espacio de características `ℝᵈ`, el clustering busca una partición `{C₁, C₂, ..., Cₖ}` tal que las observaciones dentro de cada cluster sean más similares entre sí que con las observaciones de otros clusters.

*Nota: esta "similitud" es lo que cada algoritmo define de forma diferente. Por eso existen tantos algoritmos distintos.*

**Los cuatro grandes paradigmas:**

**1. Clustering particional** — Asigna cada punto a exactamente un cluster. Produce una partición plana. Ejemplo: K-Means. Pregunta que responde: *"¿En qué grupo está este cliente?"*

**2. Clustering jerárquico** — Produce un árbol de agrupamientos anidados (dendrograma). Se puede cortar a distintos niveles de granularidad. No requiere especificar K de antemano. Ejemplo: Ward linkage. Pregunta: *"¿Qué relación jerárquica hay entre estos elementos?"*

**3. Clustering basado en densidad** — Define clusters como regiones de alta densidad separadas por regiones de baja densidad. Detecta clusters de forma arbitraria y ruido de forma nativa. Ejemplo: DBSCAN. Pregunta: *"¿Dónde se concentran los datos?"*

**4. Clustering probabilístico / por modelos** — Asume que los datos han sido generados por una mezcla de distribuciones. Asigna probabilidades de pertenencia. Ejemplo: GMM. Pregunta: *"¿Con qué probabilidad pertenece este punto a cada grupo?"*

**5. Clustering neuronal** — Redes neuronales no supervisadas que aprenden una representación topológica de los datos. Ejemplo: SOM de Kohonen. Pregunta: *"¿Cómo se organiza la estructura de los datos en un mapa de baja dimensión?"*

**Slide sugerida:** Diagrama visual con los 5 paradigmas y un ejemplo de output de cada uno (imagen de clusters de colores, dendrograma, clusters de formas irregulares, elipses de probabilidad, cuadrícula 2D del SOM).

---

### [00:17 – 00:21] Aplicaciones reales (contextualizar con los alumnos)

*"Antes de meternos en matemáticas, quiero que tengáis en mente para qué sirve esto en la práctica."*

**Casos de uso con contexto de negocio:**

**Segmentación de clientes** — Una empresa de e-commerce agrupa a sus millones de clientes en perfiles (ej: "compradores impulsivos", "cazadores de ofertas", "clientes fieles de alto valor"). Esto permite personalizar campañas de marketing sin necesidad de etiquetar manualmente a ningún cliente. *Herramienta habitual: K-Means, K-Medoids, SOM.*

**Detección de anomalías** — En transacciones bancarias, el 99.9% de las operaciones forman clusters bien definidos de comportamiento normal. Las transacciones que no encajan en ningún cluster son candidatas a fraude. *Herramienta: DBSCAN (ruido nativo), Isolation Forest.*

**Compresión de imágenes** — Una imagen de 24 bits por píxel tiene millones de colores distintos. K-Means puede reducir esos colores a 256 representativos, comprimiendo la imagen de forma lossy pero eficiente.

**Agrupación de documentos / temas** — Dado un corpus de miles de artículos, el clustering agrupa automáticamente textos similares sin necesidad de categorías predefinidas. *Útil para: motores de recomendación, análisis de noticias, moderación de contenido.*

**Bioinformática** — Agrupar genes con perfiles de expresión similares, o pacientes con patrones clínicos parecidos, para identificar subtipos de enfermedades.

**Pregunta de apertura para los alumnos (30 segundos):**
*"¿Alguno de vosotros ha trabajado con clustering antes, aunque no lo llamara así? Pensad en cualquier situación donde hayáis agrupado elementos manualmente."*

---

### [00:21 – 00:25] Métricas de distancia y similitud

*"Para que un algoritmo de clustering sepa qué puntos están 'cerca', necesita una función que mida distancia. La elección de esta función tiene más impacto en el resultado que el propio algoritmo."*

**Notación:** La distancia entre dos puntos `p = (p₁, p₂, ..., pₙ)` y `q = (q₁, q₂, ..., qₙ)` en ℝⁿ.

**Distancia Euclidiana (L2)**
```
d(p, q) = √(Σᵢ (pᵢ − qᵢ)²)
```
La más intuitiva: "línea recta entre dos puntos". Sensible a la escala de las variables. Asume que todas las dimensiones tienen el mismo peso. Ideal cuando las variables son comparables en escala y el espacio es isotrópico.

**Distancia Manhattan (L1)**
```
d(p, q) = Σᵢ |pᵢ − qᵢ|
```
"Distancia de cuadrícula urbana". Menos sensible a outliers que la euclidiana porque no eleva al cuadrado. Útil en espacios de alta dimensionalidad (menor efecto de la maldición de la dimensionalidad). Buena opción cuando los outliers en alguna dimensión no deben penalizar tanto.

**Distancia Chebyshev (L∞)**
```
d(p, q) = max(|pᵢ − qᵢ|)
```
La dimensión con mayor diferencia domina. Útil en logística (movimiento de piezas en un tablero de ajedrez) o cuando una sola dimensión extrema define la disimilitud.

**Similitud coseno**
```
cos(θ) = (p · q) / (||p|| · ||q||)
```
Mide el ángulo entre dos vectores, ignorando la magnitud. Esencial en texto y NLP: dos documentos son similares si hablan de los mismos temas en proporciones parecidas, independientemente de su longitud. Rango [−1, 1], donde 1 = idéntico, 0 = ortogonal, −1 = opuesto.

**Tabla resumen:**

| Métrica | Sensible a escala | Sensible a outliers | Ideal para |
|---|---|---|---|
| Euclidiana (L2) | Sí | Sí | Variables numéricas comparables |
| Manhattan (L1) | Sí | Menos | Alta dimensionalidad, outliers |
| Chebyshev (L∞) | Sí | Sí | Cuando el peor caso define similitud |
| Coseno | No (ignora magnitud) | No | Texto, vectores de frecuencia |

**Slide sugerida:** Visualización geométrica de las tres métricas Lp con círculos unitarios: L1 (rombo), L2 (círculo), L∞ (cuadrado). Muy visual e impactante.

**Nota crítica para alumnos:**
*"Si vuestras variables tienen escalas muy distintas —por ejemplo, edad (0-100) junto a ingresos (0-100,000)— la distancia euclidiana estará dominada por la variable de mayor escala. Siempre hay que normalizar antes de clusterizar. Lo veremos en la práctica."*

**La maldición de la dimensionalidad (1 min)**

En espacios de alta dimensión (d >> 2), las distancias entre puntos tienden a converger: el punto más cercano y el más lejano tienen distancias parecidas. Intuitivamente, todos los puntos quedan "igual de lejos". Esto hace que los algoritmos basados en distancia pierdan significado. Soluciones: reducción de dimensionalidad (PCA, t-SNE, UMAP) antes de clusterizar — lo veremos en la Sesión 2.

---

## PARTE PRÁCTICA — Jupyter Notebook (30 min)

---

### [00:25 – 00:55] Práctica guiada: Exploración del espacio de datos

> *Nota para el instructor: Abre el notebook `sesion1_bloque1_distancias.ipynb` y comparte pantalla. Los alumnos remotos deben tener el notebook abierto en su propio entorno (GitHub/Colab). El objetivo no es que terminen todos los ejercicios, sino que experimenten con los datos y las distancias.*

---

#### Celda 1 — Imports y configuración

```python
# ============================================================
# BLOQUE 1.1 — Introducción al Clustering y Espacio de Datos
# Máster en Ciencia de Datos
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

# Generadores de datos sintéticos
from sklearn.datasets import make_blobs, make_moons, make_circles

# Configuración visual
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
np.random.seed(42)

print("✓ Librerías cargadas correctamente")
```

**Nota al instructor:** Aprovechar esta celda para verificar que todos los alumnos tienen el entorno funcionando. Pedir que ejecuten y confirmen el mensaje de OK.

---

#### Celda 2 — Datasets sintéticos: los tres escenarios clásicos

```python
# Generamos tres tipos de distribuciones de datos
# Cada una plantea un reto diferente para los algoritmos de clustering

# Escenario A: Blobs bien separados (el caso "fácil")
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.8)

# Escenario B: Lunas entrelazadas (clusters no convexos)
X_moons, y_moons = make_moons(n_samples=300, noise=0.05)

# Escenario C: Círculos concéntricos (clusters anidados)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5)

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [
    (X_blobs,   y_blobs,   "A — Blobs separados\n(caso ideal para K-Means)"),
    (X_moons,   y_moons,   "B — Lunas entrelazadas\n(requiere densidad o kernels)"),
    (X_circles, y_circles, "C — Círculos concéntricos\n(K-Means fallará aquí)")
]

for ax, (X, y, title) in zip(axes, datasets):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")

plt.suptitle("Tres morfologías de datos — ¿Un solo algoritmo puede con todos?",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_datasets_sinteticos.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación para el instructor:**

*"Fijaos en estos tres escenarios. El escenario A es el sueño de cualquier algoritmo: los grupos son compactos, esféricos y bien separados. Aquí cualquier método funciona. El B y el C son más interesantes: tienen estructura clara (los humanos los identificamos perfectamente), pero esa estructura no es esférica. En los próximos bloques vamos a ver qué algoritmos fallan aquí y por qué."*

*"Cuando trabajéis con datos reales nunca sabréis en qué escenario estáis. Por eso el primer paso siempre es visualizar —si la dimensionalidad lo permite— y entender la forma de los datos."*

---

#### Celda 3 — El impacto de la escala en las distancias

```python
# Demostración: por qué hay que normalizar antes de clusterizar

# Dataset artificial: dos clientes descritos por edad e ingresos anuales
# Cliente A: 25 años, 25.000€ anuales
# Cliente B: 26 años, 80.000€ anuales
# Cliente C: 55 años, 27.000€ anuales
# ¿Quién está más "cerca" de A?

clientes = np.array([
    [25,  25000],   # Cliente A (referencia)
    [26,  80000],   # Cliente B (1 año más, mucho más rico)
    [55,  27000],   # Cliente C (30 años más, ingresos similares)
])

nombres = ["A (referencia)", "B", "C"]

# Calculamos distancias euclidianas sin normalizar
print("=== Distancias EUCLIDIANAS (sin normalizar) ===")
d_AB = distance.euclidean(clientes[0], clientes[1])
d_AC = distance.euclidean(clientes[0], clientes[2])
print(f"  d(A, B) = {d_AB:,.0f}  ← 1 año de diferencia, 55k€ de diferencia")
print(f"  d(A, C) = {d_AC:,.0f}  ← 30 años de diferencia, 2k€ de diferencia")
print(f"\n  → Según distancia euclidiana, B está más cerca de A que C")
print(f"  → ¿Tiene sentido para negocio? B tiene ingresos 3x mayores que A...")

# Normalizamos con StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
clientes_norm = scaler.fit_transform(clientes)

print("\n=== Distancias EUCLIDIANAS (tras normalización z-score) ===")
d_AB_n = distance.euclidean(clientes_norm[0], clientes_norm[1])
d_AC_n = distance.euclidean(clientes_norm[0], clientes_norm[1])
d_AB_n = distance.euclidean(clientes_norm[0], clientes_norm[1])
d_AC_n = distance.euclidean(clientes_norm[0], clientes_norm[2])
print(f"  d(A, B) normalizada = {d_AB_n:.3f}")
print(f"  d(A, C) normalizada = {d_AC_n:.3f}")
print(f"\n  → Ahora C está más cerca de A (comparten nivel de ingresos)")
print(f"  → La normalización restaura el balance entre variables")
```

**Script de explicación:**

*"Este ejemplo parece trivial pero es uno de los errores más frecuentes en proyectos reales. La distancia euclidiana sin normalizar está completamente dominada por los ingresos porque tienen una escala 1000 veces mayor que la edad. Al normalizar, ambas variables contribuyen de forma equilibrada."*

*"La pregunta que debéis haceros siempre antes de clusterizar: '¿Mis variables están en la misma escala? ¿Quiero que contribuyan por igual?' Si la respuesta es sí, normalizad. Si quereis que una variable tenga más peso, podéis escalarla con un factor diferente."*

---

#### Celda 4 — Visualización de matrices de distancia

```python
# Las matrices de distancia revelan la estructura de los datos
# antes de aplicar ningún algoritmo

# Usamos los blobs (caso simple) para ver qué aspecto tiene una matriz "buena"
from sklearn.preprocessing import StandardScaler

X_blobs_norm = StandardScaler().fit_transform(X_blobs)

# Calculamos la matriz de distancias
dist_matrix = distance.cdist(X_blobs_norm[:80], X_blobs_norm[:80], metric='euclidean')

# Ordenamos por etiqueta real para ver la estructura de bloque
idx_sorted = np.argsort(y_blobs[:80])
dist_sorted = dist_matrix[np.ix_(idx_sorted, idx_sorted)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sin ordenar
im1 = axes[0].imshow(dist_matrix, cmap='viridis_r', aspect='auto')
axes[0].set_title("Matriz de distancias\n(puntos en orden original)", fontsize=11)
axes[0].set_xlabel("Índice de punto")
axes[0].set_ylabel("Índice de punto")
plt.colorbar(im1, ax=axes[0], label="Distancia euclidiana")

# Ordenada por cluster real
im2 = axes[1].imshow(dist_sorted, cmap='viridis_r', aspect='auto')
axes[1].set_title("Matriz de distancias\n(puntos ordenados por cluster real)", fontsize=11)
axes[1].set_xlabel("Índice de punto (ordenado)")
axes[1].set_ylabel("Índice de punto (ordenado)")
plt.colorbar(im2, ax=axes[1], label="Distancia euclidiana")

plt.suptitle("Estructura de bloque diagonal = clusters bien separados",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("Interpretación:")
print("  Colores oscuros = distancia corta (puntos similares)")
print("  Colores claros = distancia larga (puntos distintos)")
print("  Bloques diagonales oscuros = clusters compactos y separados")
```

**Script de explicación:**

*"La imagen de la derecha es lo que queremos ver: bloques oscuros en la diagonal. Cada bloque representa un cluster donde los puntos están cerca entre sí, y la zona clara fuera de los bloques indica que los clusters están bien separados entre ellos. En datos reales, esta estructura perfecta raramente aparece — veremos señales más ambiguas."*

---

#### Celda 5 — Comparación de métricas de distancia (ejercicio interactivo)

```python
# ¿La métrica de distancia cambia la estructura que percibimos?

from scipy.spatial.distance import cdist

# Mismo dataset, tres métricas distintas
metricas = {
    'Euclidiana (L2)': 'euclidean',
    'Manhattan (L1)':  'cityblock',
    'Coseno':          'cosine'
}

# Tomamos una muestra pequeña de los datos de lunas para visualizar
sample_idx = np.random.choice(len(X_moons), 60, replace=False)
X_sample = StandardScaler().fit_transform(X_moons[sample_idx])
y_sample = y_moons[sample_idx]
idx_sorted = np.argsort(y_sample)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (nombre, metrica) in zip(axes, metricas.items()):
    D = cdist(X_sample, X_sample, metric=metrica)
    D_sorted = D[np.ix_(idx_sorted, idx_sorted)]
    im = ax.imshow(D_sorted, cmap='viridis_r', aspect='auto')
    ax.set_title(f"Distancia: {nombre}", fontsize=11)
    ax.set_xlabel("Punto (ordenado por cluster)")
    ax.set_ylabel("Punto (ordenado por cluster)")
    plt.colorbar(im, ax=ax)

plt.suptitle("Dataset 'Lunas' — La misma estructura, vista con tres métricas distintas",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("Reflexión:")
print("  ¿Las tres métricas revelan la misma estructura de bloques?")
print("  ¿Con cuál se ve más claramente la separación entre los dos grupos?")
```

**Script de discusión:**

*"Mirad las tres matrices. Las tres muestran la misma estructura subyacente —dos grupos— pero con claridad distinta según la métrica. Este es el punto de partida de cualquier análisis de clustering: antes de elegir un algoritmo, explorad qué métrica de distancia captura mejor la estructura que os importa."*

---

#### Celda 6 — Discusión de cierre + pregunta reflexiva

```python
# RESUMEN DEL BLOQUE 1.1
# =======================
print("=" * 55)
print("PUNTOS CLAVE DEL BLOQUE 1.1")
print("=" * 55)
print("""
1. El clustering busca estructura no etiquetada en los datos.
   No hay 'respuesta correcta' — hay soluciones más o menos útiles.

2. Existen 5 paradigmas principales: particional, jerárquico,
   basado en densidad, probabilístico y neuronal.
   Cada sesión cubrirá representantes de cada familia.

3. La elección de la métrica de distancia es tan importante
   como la elección del algoritmo. Siempre normalizar primero.

4. En alta dimensionalidad, las distancias pierden significado.
   Reducción de dimensionalidad (Sesión 2) es la solución.

5. Visualizar los datos ANTES de modelar es obligatorio.
   Las matrices de distancia son una herramienta infrautilizada.
""")
print("=" * 55)
```

**Pregunta final de discusión (3 min):**

*"Antes de que pasemos al siguiente bloque, quiero que penséis en un dataset de vuestro trabajo o sector. ¿Qué columnas usaríais para clusterizar? ¿Tendrían todas la misma escala? ¿Qué métrica de distancia tendría sentido?"*

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada del bloque con título, duración y objetivo.
- **Slide 2:** Comparativa supervisado / no supervisado con tabla y ejemplos.
- **Slide 3:** Los 5 paradigmas de clustering con iconos y ejemplo de output visual de cada uno.
- **Slide 4:** Aplicaciones reales — usar iconos de sector (compras, banco, genética). Mínimo texto, máximo imagen.
- **Slide 5:** Fórmulas de las 4 métricas de distancia + círculos unitarios L1/L2/L∞.
- **Slide 6:** Tabla resumen de métricas (cuándo usar cada una).
- **Slide 7:** Advertencia de escalado — el ejemplo de edad vs. ingresos.

### Para el handout (papel o PDF)

El handout de este bloque debe incluir:
- Tabla resumen de los 5 paradigmas de clustering.
- Tabla de métricas de distancia con las fórmulas y el contexto de uso.
- Los tres gráficos de los datasets sintéticos (blobs, lunas, círculos).
- La imagen de la matriz de distancias ordenada.
- Checklist pre-clustering: *(1) ¿Datos normalizados? (2) ¿Métrica adecuada? (3) ¿Datos visualizados?*

### Para el Jupyter Notebook (entrega a alumnos)

El notebook de este bloque se distribuye con las celdas de código completas pero con **celdas de ejercicio** intercaladas con `# TODO:` marcando qué deben completar los alumnos:

**Ejercicio 1:** Añadir la distancia de Minkowski generalizada con p=3 a la comparativa de la Celda 5 e interpretar los resultados.

**Ejercicio 2:** Generar un cuarto dataset con `make_blobs` pero con `cluster_std` muy alto (=3.0). ¿Se siguen viendo bloques diagonales en la matriz de distancias?

**Ejercicio 3 (opcional/avanzado):** Implementar la función de distancia euclidiana desde cero usando solo NumPy (sin scipy) y verificar que produce los mismos resultados.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador de progreso |
|---|---|---|
| Apertura y contrato de aprendizaje | 5 min | Los alumnos tienen el notebook abierto |
| ¿Qué es unsupervised learning? | 7 min | Tabla supervisado/no supervisado en pantalla |
| Tipos de clustering + aplicaciones | 9 min | Diagrama de 5 paradigmas en pantalla |
| Métricas de distancia | 4 min | Tabla de métricas en pantalla |
| Práctica Celda 1-2 (setup + datasets) | 8 min | Todos ejecutan sin errores |
| Práctica Celda 3 (escala) | 7 min | Discusión del output |
| Práctica Celda 4-5 (matrices) | 10 min | Visualizaciones generadas |
| Cierre y discusión | 5 min | Pregunta reflexiva respondida |
| **Total** | **55 min** | |

---

*Bloque 1.1 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
