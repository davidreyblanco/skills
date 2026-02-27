# Bloque 1.4 — DBSCAN: Clustering Basado en Densidad
## Guión detallado del instructor

**Duración:** 60 minutos (25 min teoría + 35 min práctica en Jupyter Notebook)
**Posición en la sesión:** Cuarto y último bloque de la Sesión 1, antes de la recapitulación final

---

## PARTE TEÓRICA (25 min)

---

### [00:00 – 00:04] Transición y motivación: el problema que K-Means no puede resolver

**Script de transición:**

*"Llevamos tres algoritmos estudiados: K-Means, K-Medoids y el jerárquico. Los tres tienen algo en común que todavía no os he dicho explícitamente: los tres asumen, de una forma u otra, que los clusters son convexos —básicamente, esféricos o elípticos. Si los datos tienen una forma distinta, fallan."*

*"Vamos a hacer un experimento mental. Imaginad que tenéis datos de geolocalización de clientes en una ciudad: los puntos se concentran a lo largo de las calles, que son líneas curvas y ramificadas. O imaginad datos de transacciones fraudulentas: el fraude no forma un 'blob' separado, sino que se distribuye en los márgenes y los huecos del comportamiento normal. Ninguno de los algoritmos anteriores puede capturar esa estructura."*

*"DBSCAN —Density-Based Spatial Clustering of Applications with Noise— es el algoritmo diseñado exactamente para estos casos. No busca blobs: busca regiones densas. Y como regalo adicional, detecta outliers de forma nativa sin que tengáis que hacer nada especial."*

**Slide sugerida:** Tres imágenes lado a lado — K-Means fallando en lunas, K-Means fallando en círculos, DBSCAN resolviéndolos. El contraste visual es el mejor gancho.

---

### [00:04 – 00:13] Los tres tipos de puntos y los dos parámetros

**DBSCAN trabaja con dos hiperparámetros y clasifica cada punto en una de tres categorías. Esta clasificación es todo el algoritmo.**

#### Los dos hiperparámetros

**ε (epsilon) — Radio de vecindad:**
Define qué significa "estar cerca". Para cada punto `p`, su **ε-vecindad** es el conjunto de todos los puntos a distancia ≤ ε de `p`:
```
N_ε(p) = { q ∈ D : d(p, q) ≤ ε }
```
Un ε demasiado pequeño hará que casi todo sea ruido. Un ε demasiado grande fusionará clusters distintos. Es el parámetro más sensible.

**MinPts — Mínimo de vecinos:**
El número mínimo de puntos que debe haber en la ε-vecindad de un punto (incluyéndose a sí mismo) para que ese punto sea considerado "núcleo". Controla qué tan densa debe ser una región para ser considerada cluster. Regla empírica: `MinPts ≥ dimensiones + 1`, típicamente entre 3 y 10.

#### Los tres tipos de puntos

*"Cada punto del dataset recibe exactamente una de estas tres etiquetas. Una vez que entendáis la diferencia, el algoritmo es trivial."*

**Punto núcleo (core point):**
Un punto `p` es núcleo si `|N_ε(p)| ≥ MinPts`: tiene al menos MinPts vecinos a distancia ε. Es un punto en el interior de una región densa. Forma la "columna vertebral" del cluster.

**Punto frontera (border point):**
Un punto `q` es frontera si no es núcleo (tiene menos de MinPts vecinos) pero está dentro de la ε-vecindad de algún punto núcleo. Está en el borde del cluster, incluido pero no denso. Un punto frontera puede estar en la vecindad de varios puntos núcleo de clusters distintos.

**Punto ruido (noise / outlier):**
Un punto que no es núcleo ni frontera: no está en ninguna región densa. DBSCAN lo etiqueta como `-1`. Esta es la detección nativa de outliers.

**Diagrama explicativo para la slide:**

```
  ·  ·  ·  ·                    · = punto cualquiera
   · [N] ·  ·    ←— punto núcleo [N]: tiene ≥ MinPts vecinos en radio ε
    ·   [F]  ·   ←— punto frontera [F]: en vecindad de núcleo, pero no núcleo
                  ·              ←— punto ruido: aislado, ni núcleo ni frontera
```

---

### [00:13 – 00:18] El algoritmo: alcanzabilidad y conectividad

**Dos conceptos clave para entender cómo DBSCAN forma clusters:**

**Alcanzabilidad directa por densidad (directly density-reachable):**
Un punto `q` es directamente alcanzable desde `p` si:
1. `p` es un punto núcleo, Y
2. `q` está en la ε-vecindad de `p` (`d(p,q) ≤ ε`)

*La alcanzabilidad directa no es simétrica: `q` puede ser alcanzable desde `p` pero no al revés (si `q` no es núcleo).*

**Alcanzabilidad por densidad (density-reachable):**
`q` es alcanzable por densidad desde `p` si existe una cadena de puntos núcleo `p₁, p₂, ..., pₙ` tal que cada uno alcanza directamente al siguiente y `p₁ = p`, `pₙ = q`.

**Conectividad por densidad (density-connected):**
Dos puntos `p` y `q` son conectados por densidad si existe un punto `o` desde el que ambos son alcanzables por densidad. Esta sí es simétrica.

**El algoritmo:**

```
Para cada punto p no visitado:
  1. Marcar p como visitado
  2. Calcular N_ε(p)
  3. Si |N_ε(p)| < MinPts:
       → Marcar p como RUIDO (provisionalmente)
  4. Si |N_ε(p)| ≥ MinPts:
       → Crear nuevo cluster C
       → Añadir p a C
       → Para cada q en N_ε(p):
           - Si q era RUIDO: reclasificar como FRONTERA, añadir a C
           - Si q no visitado: marcar como visitado
             - Calcular N_ε(q)
             - Si |N_ε(q)| ≥ MinPts: añadir N_ε(q) a la cola de expansión
             - Añadir q a C
       → Continuar expandiendo C hasta agotar la cola
```

**Complejidad:**
- Con árbol kd o ball tree: `O(n log n)` para datasets en baja dimensionalidad (d ≤ 10-15).
- Sin índice espacial: `O(n²)`.
- scikit-learn construye el índice automáticamente con `algorithm='auto'`.

**Script de explicación intuitiva:**

*"El algoritmo es como una epidemia. Parte de un punto núcleo —el 'paciente cero'— y contagia a todos sus vecinos. Cada vecino que también es núcleo a su vez contagia a los suyos. El cluster crece hasta que no hay más vecinos núcleo que expandir. Luego el algoritmo busca otro punto no visitado y empieza una nueva epidemia —un nuevo cluster. Los puntos que nunca fueron contagiados son el ruido."*

---

### [00:18 – 00:23] La pregunta del millón: cómo elegir ε y MinPts

*"DBSCAN tiene una limitación importante que hay que ser honesto con los alumnos: la elección de ε y MinPts es difícil y no hay una respuesta única. Sin embargo, hay una técnica sistemática que funciona bien."*

**Regla empírica para MinPts:**
- Para datos de baja dimensión (d ≤ 2): `MinPts = 4`
- Regla general: `MinPts ≥ d + 1` donde `d` es el número de dimensiones
- Para datos ruidosos: aumentar MinPts (más exigencia de densidad)
- Para datasets pequeños o con muy poca estructura: `MinPts = 3`

**Técnica del gráfico k-distancia (k-distance graph) para elegir ε:**

1. Para cada punto del dataset, calcula la distancia a su k-ésimo vecino más cercano (con `k = MinPts - 1`).
2. Ordena estas distancias de mayor a menor y grafícalas.
3. Busca el "codo" de la curva: el punto donde la curva cambia de pendiente abruptamente.
4. El valor de ε en ese codo es el candidato óptimo.

**Intuición:** Los puntos núcleo tienen distancias pequeñas a su k-ésimo vecino (están rodeados de vecinos). Los puntos frontera tienen distancias algo mayores. Los puntos de ruido tienen distancias muy grandes (están aislados). El codo en el gráfico separa "puntos en regiones densas" de "puntos aislados".

---

### [00:23 – 00:25] DBSCAN vs. los algoritmos anteriores — Posicionamiento definitivo

**Ventajas únicas de DBSCAN:**

1. **No requiere especificar k**: el número de clusters emerge de los datos.
2. **Clusters de forma arbitraria**: puede descubrir clusters en forma de luna, anillo, espiral, o cualquier forma conectada.
3. **Detección nativa de outliers**: los puntos ruido son identificados sin configuración adicional. Es el único algoritmo de los vistos que hace esto de forma explícita.
4. **Robusto ante la distribución**: no asume ninguna forma específica de cluster.

**Limitaciones de DBSCAN:**

1. **Clusters de densidad variable**: si el dataset tiene clusters con densidades muy distintas, un único ε no puede capturarlos bien. *Solución: HDBSCAN.*
2. **Sensibilidad al escalado**: como todos los algoritmos basados en distancia, requiere normalización.
3. **Dificultad para elegir ε**: en alta dimensionalidad, el gráfico k-distancia pierde claridad.
4. **No produce representantes**: no hay centroide ni medoide. Más difícil de interpretar para negocio.

**HDBSCAN — La evolución natural:**
HDBSCAN (Hierarchical DBSCAN) elimina la necesidad de especificar ε. Construye una jerarquía de densidad y extrae los clusters más estables. Es más robusto ante densidades variables. Instalación: `pip install hdbscan`. Merece mención en clase aunque no se implemente en detalle.

---

## PARTE PRÁCTICA — Jupyter Notebook (35 min)

---

### [00:25 – 01:00] Práctica guiada

> *Nota para el instructor: Este es el último bloque de la Sesión 1. El nivel de energía del grupo puede haber bajado — mantener el ritmo con la promesa de que este bloque tiene la demostración visual más impactante de toda la sesión: DBSCAN resolviendo lo que K-Means no puede.*

---

#### Celda 1 — Imports

```python
# ============================================================
# BLOQUE 1.4 — DBSCAN: Clustering Basado en Densidad
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neighbors import NearestNeighbors

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
np.random.seed(42)

print("✓ Imports correctos")
```

---

#### Celda 2 — Demostración visual de los tres tipos de puntos

```python
# -------------------------------------------------------
# Visualización pedagógica: núcleo, frontera, ruido
# con un dataset mínimo y ε visible
# -------------------------------------------------------

np.random.seed(1)

# Dataset pequeño con estructura clara para demostración
X_demo = np.array([
    # Región densa izquierda (cluster 1)
    [1.0, 2.0], [1.3, 2.1], [0.9, 1.8], [1.1, 2.3], [1.4, 1.9],
    [0.8, 2.2], [1.2, 1.7], [1.5, 2.4],
    # Región densa derecha (cluster 2)
    [5.0, 2.0], [5.2, 2.1], [4.8, 1.9], [5.1, 2.3], [4.9, 1.8],
    # Punto frontera entre clusters (no llega a ser núcleo)
    [3.0, 2.0],
    # Outliers aislados
    [0.0, 5.0], [6.5, 0.5],
])

eps    = 0.8
minpts = 3

# Calculamos tipo de cada punto manualmente para ilustración
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(radius=eps).fit(X_demo)
vecindades = nbrs.radius_neighbors(X_demo, return_distance=False)
n_vecinos  = np.array([len(v) for v in vecindades])  # incluye el propio punto

es_nucleo   = n_vecinos >= minpts
es_ruido    = np.zeros(len(X_demo), dtype=bool)
es_frontera = np.zeros(len(X_demo), dtype=bool)

# Un punto es frontera si no es núcleo pero está en la vecindad de un núcleo
for i, vecinos in enumerate(vecindades):
    if not es_nucleo[i]:
        if any(es_nucleo[v] for v in vecinos if v != i):
            es_frontera[i] = True
        else:
            es_ruido[i] = True

# Visualización
fig, ax = plt.subplots(figsize=(10, 7))

# Círculos ε alrededor de los puntos núcleo (muestra solo algunos)
for i in np.where(es_nucleo)[0][:4]:
    circulo = plt.Circle(X_demo[i], eps, color='steelblue',
                         fill=True, alpha=0.08, linestyle='--', linewidth=1)
    ax.add_patch(circulo)
    circulo_borde = plt.Circle(X_demo[i], eps, color='steelblue',
                               fill=False, linestyle='--', linewidth=1)
    ax.add_patch(circulo_borde)

# Puntos coloreados por tipo
ax.scatter(X_demo[es_nucleo, 0],   X_demo[es_nucleo, 1],
           c='steelblue', s=120, zorder=5, label=f'Núcleo (≥{minpts} vecinos en ε)')
ax.scatter(X_demo[es_frontera, 0], X_demo[es_frontera, 1],
           c='orange', s=120, zorder=5, label='Frontera (en vecindad de núcleo)')
ax.scatter(X_demo[es_ruido, 0],    X_demo[es_ruido, 1],
           c='red', marker='x', s=200, zorder=5, linewidths=2.5,
           label='Ruido / Outlier')

# Anotaciones
for i, (x, y) in enumerate(X_demo):
    n = n_vecinos[i]
    ax.annotate(f'{n}v', (x, y),
                textcoords="offset points", xytext=(6, 6), fontsize=8, alpha=0.7)

ax.set_title(f"Los tres tipos de puntos en DBSCAN\n(ε={eps}, MinPts={minpts},"
             f" 'Nv' = nº vecinos en radio ε)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(0.5, 6.0)
ax.set_xlabel("Característica 1")
ax.set_ylabel("Característica 2")

# Etiqueta ε
ax.annotate('', xy=(X_demo[0, 0] + eps, X_demo[0, 1]),
            xytext=(X_demo[0, 0], X_demo[0, 1]),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1.5))
ax.text(X_demo[0, 0] + eps/2, X_demo[0, 1] - 0.18, 'ε',
        fontsize=11, color='steelblue', ha='center')

plt.tight_layout()
plt.savefig("img_dbscan_tipos_puntos.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Puntos núcleo:    {es_nucleo.sum()}")
print(f"Puntos frontera:  {es_frontera.sum()}")
print(f"Puntos ruido:     {es_ruido.sum()}")
```

**Script de explicación:**

*"Los círculos azules son los radios ε alrededor de algunos puntos núcleo. El número anotado junto a cada punto indica cuántos vecinos tiene dentro de ese radio —incluyéndose a sí mismo. Los puntos azules tienen ≥ MinPts vecinos: son núcleo. El naranja está en la vecindad de un núcleo pero no tiene suficientes vecinos propios: es frontera. Las X rojas no pertenecen a ninguna vecindad densa: son outliers."*

*"Nótese que el punto naranja en la posición (3,2) está entre los dos clusters pero tiene muy pocos vecinos —no llega a ser núcleo— y solo está en el borde de un cluster. Esta es la zona gris de DBSCAN."*

---

#### Celda 3 — DBSCAN vs. K-Means en datasets no convexos

```python
# -------------------------------------------------------
# LA DEMOSTRACIÓN CLAVE: lo que K-Means no puede hacer
# -------------------------------------------------------

datasets = {
    'Lunas': make_moons(n_samples=300, noise=0.05, random_state=42),
    'Círculos': make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
    'Blobs': make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42),
}

params_dbscan = {
    'Lunas':     {'eps': 0.15, 'min_samples': 5},
    'Círculos':  {'eps': 0.15, 'min_samples': 5},
    'Blobs':     {'eps': 0.5,  'min_samples': 5},
}
k_kmeans = {'Lunas': 2, 'Círculos': 2, 'Blobs': 3}

fig, axes = plt.subplots(3, 3, figsize=(15, 13))

for row, (nombre, (X, y_real)) in enumerate(datasets.items()):
    X_norm = StandardScaler().fit_transform(X)

    # Columna 0: datos reales
    axes[row, 0].scatter(X_norm[:, 0], X_norm[:, 1],
                         c=y_real, cmap='tab10', s=20, alpha=0.7)
    axes[row, 0].set_title(f"{nombre}\n(etiquetas reales)", fontsize=10)

    # Columna 1: K-Means
    km = KMeans(n_clusters=k_kmeans[nombre], n_init=10, random_state=42)
    labels_km = km.fit_predict(X_norm)
    axes[row, 1].scatter(X_norm[:, 0], X_norm[:, 1],
                         c=labels_km, cmap='tab10', s=20, alpha=0.7)
    axes[row, 1].scatter(km.cluster_centers_[:, 0],
                         km.cluster_centers_[:, 1],
                         c='red', marker='X', s=150, zorder=5)
    axes[row, 1].set_title(f"K-Means k={k_kmeans[nombre]}", fontsize=10)

    # Columna 2: DBSCAN
    p = params_dbscan[nombre]
    db = DBSCAN(eps=p['eps'], min_samples=p['min_samples'])
    labels_db = db.fit_predict(X_norm)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise    = (labels_db == -1).sum()

    # Outliers con estilo especial
    mask_noise = labels_db == -1
    axes[row, 2].scatter(X_norm[~mask_noise, 0], X_norm[~mask_noise, 1],
                         c=labels_db[~mask_noise], cmap='tab10', s=20, alpha=0.8)
    axes[row, 2].scatter(X_norm[mask_noise, 0], X_norm[mask_noise, 1],
                         c='black', marker='x', s=60, linewidths=1.5,
                         label=f'Ruido ({n_noise})', zorder=5)
    if n_noise > 0:
        axes[row, 2].legend(fontsize=8)
    axes[row, 2].set_title(
        f"DBSCAN ε={p['eps']} MinPts={p['min_samples']}\n"
        f"→ {n_clusters} clusters, {n_noise} outliers", fontsize=10
    )

# Encabezados de columna
for ax, titulo in zip(axes[0], ['Datos reales', 'K-Means', 'DBSCAN']):
    ax.set_title(titulo + '\n' + ax.get_title(), fontsize=11, fontweight='bold')

plt.suptitle("K-Means vs. DBSCAN en tres morfologías de datos",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("img_dbscan_vs_kmeans_morfologias.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación — momento clave del bloque:**

*"Esta es la imagen que quiero que os llevéis grabada. Fila superior: dataset de lunas. K-Means corta por la mitad ambas lunas —no puede hacer nada mejor porque los clusters no son esféricos. DBSCAN las identifica perfectamente siguiendo la densidad. Fila media: lo mismo con círculos concéntricos. K-Means falla completamente. DBSCAN perfecto."*

*"La fila inferior es el dataset de blobs: aquí los dos algoritmos dan resultados equivalentes porque los clusters SÍ son convexos y esféricos. Cuando los datos encajan con los supuestos de K-Means, ambos funcionan bien. La diferencia solo aparece cuando esos supuestos se violan."*

---

#### Celda 4 — El gráfico k-distancia para elegir ε

```python
# -------------------------------------------------------
# Técnica sistemática para elegir ε
# -------------------------------------------------------

# Usamos el dataset de lunas con ruido moderado
X_lunas, _ = make_moons(n_samples=400, noise=0.08, random_state=42)
X_lunas_norm = StandardScaler().fit_transform(X_lunas)

minpts = 5  # nuestro MinPts elegido

# Calculamos la distancia al k-ésimo vecino más cercano (k = MinPts - 1)
nbrs = NearestNeighbors(n_neighbors=minpts).fit(X_lunas_norm)
distancias, _ = nbrs.kneighbors(X_lunas_norm)
k_dist = np.sort(distancias[:, -1])[::-1]  # distancia al vecino más lejano, ordenada

# Detectamos el codo automáticamente
# (máxima curvatura en la curva k-distancia)
from numpy.linalg import norm

def encontrar_codo(y):
    """Detecta el codo de una curva usando el método de la línea recta."""
    n = len(y)
    x = np.arange(n)
    # Vector desde el primer al último punto
    inicio = np.array([x[0], y[0]])
    fin    = np.array([x[-1], y[-1]])
    linea  = fin - inicio
    linea_norm = linea / norm(linea)
    # Distancia perpendicular de cada punto a la línea
    dists_perp = np.array([
        norm(np.cross(linea_norm, np.array([x[i], y[i]]) - inicio))
        for i in range(n)
    ])
    return np.argmax(dists_perp)

idx_codo = encontrar_codo(k_dist)
eps_optimo = k_dist[idx_codo]

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico k-distancia
ax1 = axes[0]
ax1.plot(range(len(k_dist)), k_dist, color='steelblue', linewidth=2)
ax1.axhline(y=eps_optimo, color='red', linestyle='--', linewidth=2,
            label=f'ε sugerido = {eps_optimo:.3f}')
ax1.axvline(x=idx_codo, color='orange', linestyle=':', linewidth=2,
            label=f'Codo en índice {idx_codo}')
ax1.scatter([idx_codo], [eps_optimo], c='red', s=100, zorder=5)
ax1.set_xlabel(f"Puntos ordenados por distancia al {minpts}-ésimo vecino")
ax1.set_ylabel(f"Distancia al {minpts}-ésimo vecino más cercano")
ax1.set_title(f"Gráfico k-distancia (MinPts={minpts})\n→ ε ≈ {eps_optimo:.3f}",
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)

# Resultado de DBSCAN con ε automático
db_auto = DBSCAN(eps=eps_optimo, min_samples=minpts)
labels_auto = db_auto.fit_predict(X_lunas_norm)
n_cls = len(set(labels_auto)) - (1 if -1 in labels_auto else 0)
n_nse = (labels_auto == -1).sum()

ax2 = axes[1]
mask_noise = labels_auto == -1
ax2.scatter(X_lunas_norm[~mask_noise, 0], X_lunas_norm[~mask_noise, 1],
            c=labels_auto[~mask_noise], cmap='tab10', s=25, alpha=0.8)
ax2.scatter(X_lunas_norm[mask_noise, 0], X_lunas_norm[mask_noise, 1],
            c='black', marker='x', s=60, linewidths=1.5,
            label=f'Ruido: {n_nse} puntos')
ax2.set_title(f"DBSCAN con ε={eps_optimo:.3f}, MinPts={minpts}\n"
              f"→ {n_cls} clusters, {n_nse} outliers detectados",
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlabel("Característica 1 (norm.)")
ax2.set_ylabel("Característica 2 (norm.)")

plt.suptitle("Selección sistemática de ε mediante gráfico k-distancia",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_dbscan_kdist.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"ε sugerido: {eps_optimo:.4f}")
print(f"Resultado: {n_cls} clusters, {n_nse} outliers")
```

**Script de explicación:**

*"El gráfico de la izquierda es vuestra brújula para elegir ε. El eje X son los puntos ordenados por su distancia al quinto vecino más cercano. Al principio la curva es plana y baja —son los puntos núcleo, bien rodeados de vecinos—. Luego hay un codo donde la curva se dispara hacia arriba: ahí están los puntos frontera y los outliers, que tienen vecinos más lejanos. El ε óptimo está en ese codo."*

*"La línea roja marca el ε sugerido automáticamente. El resultado de la derecha muestra que con ese ε, DBSCAN encuentra correctamente los dos clusters y un puñado de outliers —los puntos que el propio dataset generó con ruido excesivo."*

---

#### Celda 5 — Caso práctico: detección de anomalías en e-commerce

```python
# -------------------------------------------------------
# CASO PRÁCTICO: Detección de comportamiento anómalo
# en transacciones de e-commerce
# -------------------------------------------------------

np.random.seed(42)
n_normal = 400

# Comportamiento normal: correlación entre sesiones y compras
sesiones   = np.random.normal(50, 10, n_normal)
compras    = sesiones * 0.3 + np.random.normal(0, 4, n_normal)
ticket_med = np.random.normal(45, 8, n_normal)

# Patrones anómalos
# Tipo 1: muchas sesiones, pocas compras (bots de scraping)
s_bot  = np.random.uniform(150, 200, 12)
c_bot  = np.random.uniform(0, 3, 12)
t_bot  = np.random.uniform(5, 15, 12)

# Tipo 2: pocas sesiones, ticket altísimo (fraude de tarjeta)
s_frau = np.random.uniform(1, 5, 8)
c_frau = np.random.uniform(8, 15, 8)
t_frau = np.random.uniform(300, 500, 8)

# Tipo 3: comportamiento de usuario VIP extremo (legítimo pero outlier)
s_vip  = np.random.uniform(80, 100, 5)
c_vip  = np.random.uniform(40, 55, 5)
t_vip  = np.random.uniform(200, 280, 5)

# Combinamos
sesiones_all = np.concatenate([sesiones,   s_bot,  s_frau, s_vip])
compras_all  = np.concatenate([compras,    c_bot,  c_frau, c_vip])
ticket_all   = np.concatenate([ticket_med, t_bot,  t_frau, t_vip])
tipo_real    = np.concatenate([
    ['Normal'] * n_normal,
    ['Bot (scraping)'] * 12,
    ['Fraude tarjeta'] * 8,
    ['VIP extremo'] * 5
])

df_ecom = pd.DataFrame({
    'sesiones_mes':  sesiones_all,
    'compras_mes':   compras_all,
    'ticket_medio':  ticket_all,
    'tipo_real':     tipo_real
})

print(f"Dataset: {len(df_ecom)} usuarios")
print(df_ecom['tipo_real'].value_counts())
```

---

#### Celda 6 — Aplicar DBSCAN y visualizar anomalías detectadas

```python
# Escalamos y aplicamos DBSCAN
features = ['sesiones_mes', 'compras_mes', 'ticket_medio']
scaler_ecom = StandardScaler()
X_ecom = scaler_ecom.fit_transform(df_ecom[features])

# Elegimos parámetros con el gráfico k-distancia
nbrs_ecom = NearestNeighbors(n_neighbors=5).fit(X_ecom)
dist_ecom, _ = nbrs_ecom.kneighbors(X_ecom)
k_dist_ecom  = np.sort(dist_ecom[:, -1])[::-1]
eps_ecom     = k_dist_ecom[encontrar_codo(k_dist_ecom)]

db_ecom = DBSCAN(eps=eps_ecom, min_samples=5)
df_ecom['cluster_dbscan'] = db_ecom.fit_predict(X_ecom)

n_clusters_ecom = len(set(df_ecom['cluster_dbscan'])) - \
                  (1 if -1 in df_ecom['cluster_dbscan'].values else 0)
n_outliers_ecom = (df_ecom['cluster_dbscan'] == -1).sum()

print(f"ε utilizado: {eps_ecom:.3f}")
print(f"Clusters encontrados: {n_clusters_ecom}")
print(f"Outliers detectados:  {n_outliers_ecom}")
print()

# ¿Qué tipo real tienen los outliers detectados?
outliers_detectados = df_ecom[df_ecom['cluster_dbscan'] == -1]
print("Composición de los outliers detectados por DBSCAN:")
print(outliers_detectados['tipo_real'].value_counts())
print()
print("Tasa de detección por tipo anómalo:")
for tipo in ['Bot (scraping)', 'Fraude tarjeta', 'VIP extremo']:
    total = (df_ecom['tipo_real'] == tipo).sum()
    detectados = ((df_ecom['tipo_real'] == tipo) &
                  (df_ecom['cluster_dbscan'] == -1)).sum()
    tasa = detectados / total * 100
    print(f"  {tipo}: {detectados}/{total} detectados ({tasa:.0f}%)")
```

---

#### Celda 7 — Visualización 3D de los outliers

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# Vista 2D: Sesiones vs. Ticket Medio
ax1 = fig.add_subplot(121)
colores_tipo = {
    'Normal':        '#377eb8',
    'Bot (scraping)':'#ff7f00',
    'Fraude tarjeta':'#e41a1c',
    'VIP extremo':   '#4daf4a'
}

# Puntos normales (cluster != -1)
mask_normal_db = df_ecom['cluster_dbscan'] != -1
ax1.scatter(df_ecom.loc[mask_normal_db, 'sesiones_mes'],
            df_ecom.loc[mask_normal_db, 'ticket_medio'],
            c='#377eb8', alpha=0.3, s=20, label='Comportamiento normal')

# Outliers coloreados por tipo real
for tipo, color in colores_tipo.items():
    if tipo == 'Normal':
        continue
    mask = (df_ecom['cluster_dbscan'] == -1) & (df_ecom['tipo_real'] == tipo)
    if mask.sum() > 0:
        ax1.scatter(df_ecom.loc[mask, 'sesiones_mes'],
                    df_ecom.loc[mask, 'ticket_medio'],
                    c=color, s=100, marker='*', zorder=5,
                    edgecolors='black', linewidths=0.7,
                    label=f'{tipo} (outlier DBSCAN)')

ax1.set_xlabel("Sesiones / mes")
ax1.set_ylabel("Ticket medio (€)")
ax1.set_title("Outliers detectados por DBSCAN\n(coloreados por tipo real)",
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')

# Vista 2D: Sesiones vs. Compras
ax2 = fig.add_subplot(122)
ax2.scatter(df_ecom.loc[mask_normal_db, 'sesiones_mes'],
            df_ecom.loc[mask_normal_db, 'compras_mes'],
            c='#377eb8', alpha=0.3, s=20, label='Comportamiento normal')
for tipo, color in colores_tipo.items():
    if tipo == 'Normal':
        continue
    mask = (df_ecom['cluster_dbscan'] == -1) & (df_ecom['tipo_real'] == tipo)
    if mask.sum() > 0:
        ax2.scatter(df_ecom.loc[mask, 'sesiones_mes'],
                    df_ecom.loc[mask, 'compras_mes'],
                    c=color, s=100, marker='*', zorder=5,
                    edgecolors='black', linewidths=0.7,
                    label=f'{tipo}')

ax2.set_xlabel("Sesiones / mes")
ax2.set_ylabel("Compras / mes")
ax2.set_title("Vista sesiones vs. compras\n(mismo coloreado)",
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)

plt.suptitle("DBSCAN como detector de anomalías en e-commerce",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_dbscan_ecommerce_anomalias.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de interpretación del caso práctico:**

*"Aquí está la aplicación real. Los puntos azules son el comportamiento normal —sesiones correlacionadas con compras y ticket medio dentro de rango—. Las estrellas naranjas son los bots: muchas sesiones, casi ninguna compra —un ratio que ningún humano tiene—. Las estrellas rojas son el fraude: pocas sesiones pero ticket altísimo —alguien que entra, compra algo carísimo y no vuelve—. Las verdes son los VIP legítimos pero extremos."*

*"DBSCAN los detecta todos sin que le hayamos dicho qué buscar. Solo le dijimos 'encuéntrame las regiones densas' y todo lo que no encaja en esas regiones aparece como ruido. En producción, ese ruido es vuestra lista de casos a revisar por el equipo de fraude."*

---

#### Celda 8 — Sensibilidad a los parámetros: análisis de variabilidad

```python
# -------------------------------------------------------
# ¿Qué pasa si cambiamos ε y MinPts?
# Mapa de calor de resultados
# -------------------------------------------------------

X_lunas2, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
X_lunas2_norm = StandardScaler().fit_transform(X_lunas2)

eps_vals     = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
minpts_vals  = [3, 5, 8, 12]

resultados_grid = np.zeros((len(minpts_vals), len(eps_vals), 2))  # [clusters, ruido%]

for i, mp in enumerate(minpts_vals):
    for j, ep in enumerate(eps_vals):
        db = DBSCAN(eps=ep, min_samples=mp)
        lbl = db.fit_predict(X_lunas2_norm)
        n_cls = len(set(lbl)) - (1 if -1 in lbl else 0)
        pct_ruido = (lbl == -1).mean() * 100
        resultados_grid[i, j, 0] = n_cls
        resultados_grid[i, j, 1] = pct_ruido

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap: número de clusters
im1 = axes[0].imshow(resultados_grid[:, :, 0], cmap='Blues', aspect='auto')
axes[0].set_xticks(range(len(eps_vals)))
axes[0].set_xticklabels([str(e) for e in eps_vals])
axes[0].set_yticks(range(len(minpts_vals)))
axes[0].set_yticklabels([str(m) for m in minpts_vals])
axes[0].set_xlabel("ε (radio de vecindad)")
axes[0].set_ylabel("MinPts")
axes[0].set_title("Número de clusters", fontsize=11, fontweight='bold')
plt.colorbar(im1, ax=axes[0])
for i in range(len(minpts_vals)):
    for j in range(len(eps_vals)):
        axes[0].text(j, i, int(resultados_grid[i, j, 0]),
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     color='white' if resultados_grid[i, j, 0] > 5 else 'black')

# Heatmap: % de ruido
im2 = axes[1].imshow(resultados_grid[:, :, 1], cmap='Reds', aspect='auto')
axes[1].set_xticks(range(len(eps_vals)))
axes[1].set_xticklabels([str(e) for e in eps_vals])
axes[1].set_yticks(range(len(minpts_vals)))
axes[1].set_yticklabels([str(m) for m in minpts_vals])
axes[1].set_xlabel("ε (radio de vecindad)")
axes[1].set_ylabel("MinPts")
axes[1].set_title("% de puntos clasificados como ruido", fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=axes[1], label="%")
for i in range(len(minpts_vals)):
    for j in range(len(eps_vals)):
        axes[1].text(j, i, f"{resultados_grid[i, j, 1]:.0f}%",
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if resultados_grid[i, j, 1] > 40 else 'black')

plt.suptitle("Sensibilidad de DBSCAN a los hiperparámetros — Dataset 'Lunas'",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_dbscan_sensibilidad.png", dpi=150, bbox_inches='tight')
plt.show()

print("Interpretación:")
print("  ε pequeño + MinPts alto  → muchos clusters pequeños, alto % ruido")
print("  ε grande + MinPts bajo   → pocos clusters grandes, bajo % ruido (o 1 cluster)")
print("  Zona intermedia          → resultado útil (2 clusters, ~5-15% ruido)")
```

**Script de explicación:**

*"Este mapa de calor os permite ver de un vistazo cómo cambia el comportamiento de DBSCAN al variar sus parámetros. Con ε muy pequeño —columna izquierda— casi todo es ruido porque los radios son demasiado pequeños. Con ε muy grande —columna derecha— todo se fusiona en un único cluster enorme. La zona útil para este dataset está en el centro: ε entre 0.10 y 0.20, MinPts entre 3 y 8."*

*"Usad este tipo de análisis de sensibilidad cuando no estéis seguros de vuestros parámetros. Especialmente si vuestros datos tienen ruido variable o densidad no uniforme."*

---

#### Celda 9 — Mención a HDBSCAN

```python
# -------------------------------------------------------
# HDBSCAN: la evolución natural de DBSCAN
# (Demo rápida, sin profundizar)
# -------------------------------------------------------

try:
    import hdbscan

    # Dataset con clusters de densidad variable
    np.random.seed(5)
    X_var = np.vstack([
        np.random.normal([0, 0], [0.3, 0.3], (150,)),   # cluster denso
        np.random.normal([4, 4], [1.2, 1.2], (150,)),   # cluster disperso
        np.random.normal([8, 0], [0.4, 0.4], (100,)),   # cluster denso
        np.random.uniform(-3, 11, (20, 2))               # ruido uniforme
    ])
    X_var_norm = StandardScaler().fit_transform(X_var)

    # DBSCAN clásico (difícil calibrar para densidades distintas)
    db_var = DBSCAN(eps=0.35, min_samples=5).fit_predict(X_var_norm)

    # HDBSCAN (sin necesidad de ε)
    hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    labels_hdb = hdb.fit_predict(X_var_norm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, labels, titulo in zip(
        axes,
        [np.arange(len(X_var)) // (len(X_var)//3),  # grupos reales aprox.
         db_var, labels_hdb],
        ['Datos (grupos aproximados)', f'DBSCAN ε=0.35', 'HDBSCAN (sin ε)']
    ):
        mask_noise = labels == -1
        if mask_noise.sum() > 0:
            ax.scatter(X_var_norm[mask_noise, 0], X_var_norm[mask_noise, 1],
                       c='black', marker='x', s=40, alpha=0.5)
        ax.scatter(X_var_norm[~mask_noise, 0], X_var_norm[~mask_noise, 1],
                   c=labels[~mask_noise], cmap='tab10', s=25, alpha=0.8)
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        n_n = mask_noise.sum()
        ax.set_title(f"{titulo}\n{n_c} clusters, {n_n} outliers",
                     fontsize=10, fontweight='bold')

    plt.suptitle("Densidades variables: DBSCAN vs. HDBSCAN",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("img_hdbscan_vs_dbscan.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("HDBSCAN disponible. Instalación: pip install hdbscan")

except ImportError:
    print("HDBSCAN no instalado. Ejecutar: pip install hdbscan")
    print("Concepto clave: HDBSCAN elimina la necesidad de ε construyendo")
    print("una jerarquía de densidad y extrayendo los clusters más estables.")
```

**Script de explicación:**

*"HDBSCAN es la evolución directa de DBSCAN. El problema de DBSCAN con densidades variables es real: si un dataset tiene un cluster muy denso junto a uno más disperso, un único ε no puede capturar bien los dos. HDBSCAN construye internamente una jerarquía de densidades —como un dendrograma del clustering jerárquico pero basado en densidad— y extrae los clusters más estables en esa jerarquía. Solo necesita `min_cluster_size`. En datasets reales con estructura irregular, HDBSCAN suele superar a DBSCAN."*

---

#### Celda 10 — Resumen comparativo de los cuatro algoritmos de la Sesión 1

```python
print("=" * 65)
print("TABLA COMPARATIVA FINAL — SESIÓN 1")
print("=" * 65)

tabla = pd.DataFrame({
    'K-Means': {
        'Tipo':             'Particional',
        'Especifica k':     'Sí (obligatorio)',
        'Forma clusters':   'Esférica',
        'Outliers':         'Sensible',
        'Representante':    'Centroide (ficticio)',
        'Escalabilidad':    'Excelente (O(n·k·d))',
        'Mejor para':       'n grande, clusters bien separados',
    },
    'K-Medoids': {
        'Tipo':             'Particional',
        'Especifica k':     'Sí (obligatorio)',
        'Forma clusters':   'Esférica',
        'Outliers':         'Robusto',
        'Representante':    'Medoide (punto real)',
        'Escalabilidad':    'Limitada (O(k(n-k)²))',
        'Mejor para':       'Outliers presentes, representantes reales',
    },
    'Jerárquico': {
        'Tipo':             'Jerárquico',
        'Especifica k':     'No (corte flexible)',
        'Forma clusters':   'Depende del enlace',
        'Outliers':         'Moderado',
        'Representante':    'Ninguno (árbol)',
        'Escalabilidad':    'Pobre (O(n²))',
        'Mejor para':       'Exploración, n pequeño, estructura anidada',
    },
    'DBSCAN': {
        'Tipo':             'Densidad',
        'Especifica k':     'No (emerge de datos)',
        'Forma clusters':   'Arbitraria',
        'Outliers':         'Nativo',
        'Representante':    'Ninguno',
        'Escalabilidad':    'Buena (O(n log n))',
        'Mejor para':       'Formas arbitrarias, detección de anomalías',
    },
}).T

print(tabla.to_string())
print()
print("Regla de selección rápida:")
print("  ¿Forma arbitraria o necesito detectar outliers?    → DBSCAN")
print("  ¿Quiero explorar k sin decidirlo a priori?         → Jerárquico")
print("  ¿Hay outliers y necesito representantes reales?    → K-Medoids")
print("  ¿Dataset grande, datos limpios, k conocido?        → K-Means")
```

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada del bloque. Las tres imágenes del fracaso de K-Means en lunas y círculos vs. DBSCAN resolviéndolos.
- **Slide 2:** Los dos parámetros ε y MinPts con diagrama geométrico mostrando la ε-vecindad.
- **Slide 3:** Los tres tipos de puntos — diagrama con puntos coloreados, círculos ε y etiquetas.
- **Slide 4:** Pseudocódigo del algoritmo con la metáfora de la epidemia.
- **Slide 5:** Técnica k-distancia — gráfico con el codo señalado.
- **Slide 6:** Tabla resumen de DBSCAN vs. los tres algoritmos anteriores.
- **Slide 7:** Tarjeta de presentación de HDBSCAN — cuándo y por qué usarlo.

### Para el handout

- Tabla comparativa de los 4 algoritmos (criterios de selección).
- Diagrama de los 3 tipos de puntos (Celda 2).
- Gráfico comparativo K-Means vs. DBSCAN en las tres morfologías (Celda 3).
- Guía para elegir ε: pasos del gráfico k-distancia.
- Mapa de calor de sensibilidad a parámetros (Celda 8).
- Checklist de decisión: *¿Forma arbitraria? → DBSCAN. ¿Densidades variables? → HDBSCAN.*

### Para el Jupyter Notebook (ejercicios a completar por los alumnos)

**Ejercicio 1 (Celda 4 ampliada):** Repetir el análisis del gráfico k-distancia con MinPts = 3, 5, 8 y 12. ¿El ε sugerido cambia mucho? ¿Cuál produce el mejor resultado visual?

**Ejercicio 2 (Celda 6 ampliada):** Modificar el dataset de e-commerce añadiendo un nuevo tipo de anomalía: usuarios con exactamente 1 sesión y 1 compra con ticket muy alto (posible compra impulsiva de producto caro). ¿DBSCAN los detecta como outliers o los incluye en el cluster normal?

**Ejercicio 3 (Celda 8 ampliada):** Añadir al mapa de calor una tercera métrica: el Silhouette Score (solo para los puntos no-ruido). ¿Los parámetros con mejor Silhouette coinciden con los que producen el resultado visual más limpio?

**Ejercicio 4 (avanzado):** Implementar el algoritmo DBSCAN desde cero usando solo NumPy. El resultado debe coincidir con `sklearn.cluster.DBSCAN` en asignaciones de núcleo/frontera/ruido. Verificar con `adjusted_rand_score`.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador de progreso |
|---|---|---|
| Transición y motivación visual | 4 min | Las tres imágenes de fallo de K-Means en pantalla |
| Los tres tipos de puntos + dos parámetros | 9 min | Diagrama geométrico en pantalla |
| El algoritmo y la metáfora de la epidemia | 5 min | Pseudocódigo en pantalla |
| Gráfico k-distancia para elegir ε | 4 min | Gráfico anotado en pantalla |
| Posicionamiento vs. algoritmos anteriores | 3 min | Tabla comparativa en pantalla |
| Celda 1-2 (imports + tipos de puntos) | 8 min | Diagrama generado |
| Celda 3 (K-Means vs. DBSCAN morfologías) | 8 min | Los 9 subplots generados |
| Celda 4 (k-distancia) | 6 min | Gráfico de codo generado |
| Celda 5-7 (caso e-commerce) | 8 min | Tasas de detección impresas |
| Celda 8 (sensibilidad parámetros) | 5 min | Mapa de calor generado |
| Celda 9-10 (HDBSCAN + tabla final) | 3 min | Tabla comparativa impresa |
| Discusión de cierre | 3 min buffer | — |
| **Total** | **66 min** *(+6 min de margen)* | |

> *Nota: Si el tiempo aprieta, la Celda 9 (HDBSCAN) es prescindible y puede quedar como lectura opcional. La Celda 10 (tabla comparativa) es crítica — no omitir porque conecta con la recapitulación final de la Sesión 1.*

---

*Bloque 1.4 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
