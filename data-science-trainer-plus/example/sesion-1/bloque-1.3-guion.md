# Bloque 1.3 — Clustering Jerárquico
## Guión detallado del instructor

**Duración:** 65 minutos (25 min teoría + 40 min práctica en Jupyter Notebook)
**Posición en la sesión:** Tercer bloque de la Sesión 1, tras K-Means y K-Medoids

---

## PARTE TEÓRICA (25 min)

---

### [00:00 – 00:04] Transición y motivación

**Script de transición:**

*"Hasta ahora hemos trabajado con algoritmos particionales: K-Means y K-Medoids. Ambos producen una partición plana —cada punto pertenece a exactamente un cluster y no hay relación entre clusters. Pero en muchos dominios reales, la estructura de los datos es jerárquica por naturaleza. Los productos se agrupan en subcategorías, que a su vez forman categorías, que forman departamentos. Los organismos se agrupan en especies, géneros, familias, órdenes. Los clientes se agrupan en microsegmentos dentro de segmentos dentro de perfiles generales."*

*"El clustering jerárquico no produce una solución única: produce un árbol completo de agrupamientos anidados que podemos cortar a cualquier nivel de granularidad. Es la herramienta de exploración por excelencia cuando no sabemos cuántos clusters hay y queremos entender la estructura a múltiples escalas."*

**Pregunta de enganche (30 segundos):**

*"¿Alguien ha oído hablar de un dendrograma antes? No hace falta que sepáis qué es exactamente — en diez minutos vais a saber leerlos e interpretarlos."*

---

### [00:04 – 00:10] Aglomerativo vs. divisivo: dos filosofías opuestas

**Hay dos familias de clustering jerárquico, con lógicas opuestas:**

**Enfoque aglomerativo (bottom-up) — El más utilizado:**

Parte de `n` clusters (uno por punto) y en cada paso fusiona los dos clusters más similares. El proceso continúa hasta que todos los puntos forman un único cluster. El resultado es un árbol de fusiones llamado **dendrograma**.

*"Es como ver cómo se forman alianzas: al principio cada nación es independiente; gradualmente se fusionan las más afines hasta que el mundo entero es una confederación."*

**Enfoque divisivo (top-down) — Menos frecuente:**

Parte de 1 cluster (todos los puntos juntos) y en cada paso divide el cluster más heterogéneo. Producen el mismo árbol en sentido inverso pero son computacionalmente más costosos y menos usados en la práctica. El algoritmo DIANA (DIvisive ANAlysis) es el representante más conocido.

**¿Cuál usar?** En la práctica, casi siempre aglomerativo. scikit-learn solo implementa `AgglomerativeClustering`. El enfoque divisivo tiene sentido cuando se sospecha que la división inicial en dos grandes grupos es muy significativa (ej.: análisis filogenético, segmentación binaria sucesiva).

---

### [00:10 – 00:18] El algoritmo aglomerativo paso a paso

**Pseudocódigo del algoritmo:**

```
1. Inicialización: cada punto es su propio cluster → n clusters
2. Calcular la matriz de distancias D entre todos los pares de clusters
3. REPETIR hasta que quede 1 solo cluster:
   a. Encontrar los dos clusters Ci, Cj con menor distancia d(Ci, Cj)
   b. Fusionar Ci y Cj en un nuevo cluster Cij
   c. Actualizar D: calcular distancias del nuevo cluster Cij al resto
   d. Eliminar Ci y Cj de D, añadir Cij
4. Registrar cada fusión: (Ci, Cj, distancia, tamaño del nuevo cluster)
   → Este registro ES el dendrograma
```

**¿Cómo se mide la distancia entre dos clusters? — Los criterios de enlace**

Esta pregunta es el corazón del clustering jerárquico. Dados dos clusters `A` y `B`, hay varias formas de definir su distancia:

**Enlace simple (single linkage):**
```
d(A, B) = min { d(a, b) : a ∈ A, b ∈ B }
```
La distancia entre los dos puntos más cercanos de cada cluster. Muy sensible a outliers. Puede producir el efecto "cadena" (chaining): clusters elongados y mal definidos porque un solo punto puente une clusters muy distintos. Útil para detectar clusters de forma arbitraria, pero rara vez en aplicaciones de negocio.

**Enlace completo (complete linkage):**
```
d(A, B) = max { d(a, b) : a ∈ A, b ∈ B }
```
La distancia entre los dos puntos más lejanos. Produce clusters compactos y aproximadamente del mismo diámetro. Más robusto ante outliers que el simple. Penaliza las fusiones donde hay puntos muy alejados, aunque la mayoría sean cercanos.

**Enlace promedio (average linkage / UPGMA):**
```
d(A, B) = (1/|A|·|B|) Σ_{a∈A} Σ_{b∈B} d(a, b)
```
La media de todas las distancias entre pares de puntos de A y B. Balance entre simple y completo. Menos sensible a outliers que el simple. Buen punto de partida cuando no hay conocimiento previo.

**Criterio de Ward:**
```
d(A, B) = ΔWCSS = WCSS(A∪B) − WCSS(A) − WCSS(B)
```
En lugar de medir distancia entre clusters, mide el **incremento de inercia** que resultaría de fusionarlos. Ward fusiona siempre los dos clusters cuya unión provoca el menor aumento de varianza interna. Produce clusters compactos y aproximadamente del mismo tamaño. Es el criterio más utilizado en la práctica porque minimiza directamente la misma función objetivo que K-Means. **Default recomendado para la mayoría de casos.**

**Tabla resumen de criterios de enlace:**

| Criterio | Basado en | Clusters resultantes | Sensible a outliers | Uso típico |
|---|---|---|---|---|
| Simple | Par más cercano | Elongados, cadena | Muy alto | Detección de forma arbitraria |
| Completo | Par más lejano | Compactos, esféricos | Moderado | Cuando tamaño uniforme importa |
| Promedio | Media de pares | Balanceados | Bajo | Uso general |
| **Ward** | **Incremento WCSS** | **Compactos, iguales** | **Bajo** | **Recomendado por defecto** |

---

### [00:18 – 00:23] El dendrograma: leerlo e interpretarlo

**¿Qué es un dendrograma?**

Un dendrograma es un árbol binario que representa el historial completo de fusiones del algoritmo aglomerativo. El eje horizontal (o vertical, según la orientación) muestra los puntos o clusters; el eje vertical muestra la **distancia (o disimilitud) a la que se produce cada fusión**.

**Cómo leerlo:**

- Cada hoja del árbol es un punto original.
- Cada nodo interno representa una fusión de dos clusters.
- La **altura** de un nodo es la distancia a la que se fusionaron esos dos clusters.
- Cuanto más alta la fusión, más diferentes eran los clusters al unirse.

**Cómo elegir el número de clusters:**

El procedimiento estándar es buscar el **mayor salto de altura** en el dendrograma: la zona donde hay un "hueco" grande entre dos fusiones consecutivas. Ese hueco indica que los clusters que existían antes de ese hueco eran muy distintos entre sí —era un buen momento para "cortar" el árbol.

*"Imaginad que vais mirando el árbol de abajo hacia arriba. Al principio se fusionan puntos muy parecidos —las alturas son pequeñas. En algún momento hay un salto grande: esa fusión une dos grupos que en realidad eran bastante distintos. Ese es el corte óptimo. El número de ramas que quedan debajo del corte es vuestro k."*

**Complejidad computacional:**

- Tiempo: `O(n²)` para calcular la matriz de distancias inicial + `O(n² log n)` para el proceso de fusión.
- Memoria: `O(n²)` para almacenar la matriz de distancias.

Consecuencia práctica: el clustering jerárquico no escala bien. Para `n > 10.000` se vuelve lento y para `n > 100.000` es inviable en memoria. En esos casos, usar K-Means o DBSCAN.

---

### [00:23 – 00:25] Posicionamiento: cuándo usar clustering jerárquico

**Cuándo SÍ es la herramienta correcta:**

- Cuando se necesita **exploración sin saber k a priori**: el dendrograma sugiere el k de forma visual.
- Cuando los datos tienen **estructura jerárquica real**: taxonomías, filogenias, agrupaciones anidadas.
- Cuando el dataset es **pequeño o mediano** (n < 5.000–10.000) y la interpretación importa más que la velocidad.
- Cuando se quiere un **análisis exploratorio reproducible**: el dendrograma es el mismo en cada ejecución (no hay inicialización aleatoria).
- Cuando se necesita comparar soluciones a **distintos niveles de granularidad** sin re-ejecutar.

**Cuándo NO:**

- Datasets muy grandes (n > 10.000): usar K-Means, Mini-Batch K-Means o DBSCAN.
- Cuando la velocidad es crítica y n es grande.
- Cuando ya se sabe k y los datos son numéricos y bien comportados: K-Means es más eficiente.

---

## PARTE PRÁCTICA — Jupyter Notebook (40 min)

---

### [00:25 – 01:05] Práctica guiada

> *Nota para el instructor: Continuar en el mismo notebook `sesion1_bloque3_jerarquico.ipynb`. Los alumnos deben tener el entorno listo. La práctica de este bloque tiene un componente visual muy fuerte — dedicar tiempo a que los alumnos interpreten los dendrogramas antes de ver la solución.*

---

#### Celda 1 — Imports

```python
# ============================================================
# BLOQUE 1.3 — Clustering Jerárquico
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
np.random.seed(42)

print("✓ Imports correctos")
```

---

#### Celda 2 — Visualización del algoritmo aglomerativo paso a paso (dataset mínimo)

```python
# -------------------------------------------------------
# Demostración con un dataset MUY pequeño (8 puntos)
# para ver cada fusión de forma explícita
# -------------------------------------------------------

# 8 puntos en 2D con estructura de 3 grupos
X_mini = np.array([
    [1.0, 1.0],  # grupo A
    [1.5, 1.2],
    [1.2, 0.8],
    [5.0, 5.0],  # grupo B
    [5.3, 4.8],
    [4.8, 5.2],
    [9.0, 1.0],  # grupo C
    [9.2, 1.3],
])
etiquetas_reales = ['A1','A2','A3','B1','B2','B3','C1','C2']

# Calculamos la matriz de linkage con Ward
Z = linkage(X_mini, method='ward')

print("Historial de fusiones (método Ward):")
print(f"{'Paso':>4}  {'Cluster i':>10}  {'Cluster j':>10}  {'Distancia':>10}  {'Tamaño':>7}")
print("-" * 50)
n = len(X_mini)
for i, (ci, cj, dist, size) in enumerate(Z):
    label_i = etiquetas_reales[int(ci)] if ci < n else f"Cluster-{int(ci)-n+1}"
    label_j = etiquetas_reales[int(cj)] if cj < n else f"Cluster-{int(cj)-n+1}"
    print(f"{i+1:>4}  {label_i:>10}  {label_j:>10}  {dist:>10.3f}  {int(size):>7}")
```

**Script de explicación:**

*"Este es el historial completo de fusiones. En el paso 1 se fusionan los puntos más cercanos —probablemente dos puntos dentro del mismo grupo real. Después se van formando grupos más grandes. En los últimos pasos, la distancia da un salto grande: esos son los grupos 'reales' fusionándose forzosamente."*

*"Ahora vamos a visualizar exactamente este historial como dendrograma."*

---

#### Celda 3 — El dendrograma explicado capa a capa

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Izquierda: los puntos en 2D ---
ax = axes[0]
colores_reales = ['#e41a1c','#e41a1c','#e41a1c',
                   '#377eb8','#377eb8','#377eb8',
                   '#4daf4a','#4daf4a']
for i, (x, y) in enumerate(X_mini):
    ax.scatter(x, y, c=colores_reales[i], s=150, zorder=5)
    ax.annotate(etiquetas_reales[i], (x, y),
                textcoords="offset points", xytext=(8, 5), fontsize=11)
ax.set_title("Puntos en el espacio original", fontsize=12, fontweight='bold')
ax.set_xlabel("Característica 1")
ax.set_ylabel("Característica 2")
ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.5, 7)

# --- Derecha: el dendrograma ---
ax2 = axes[1]
dendrogram(
    Z,
    labels=etiquetas_reales,
    ax=ax2,
    color_threshold=4.0,    # umbral visual para colorear ramas
    leaf_font_size=11,
    above_threshold_color='gray'
)
ax2.set_title("Dendrograma (método Ward)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Puntos")
ax2.set_ylabel("Distancia de fusión (Ward)")

# Línea de corte sugerida
ax2.axhline(y=4.0, color='red', linestyle='--', linewidth=2,
            label='Corte → 3 clusters')
ax2.legend(fontsize=10)

plt.suptitle("Del espacio 2D al dendrograma — lectura directa",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_dendrograma_mini.png", dpi=150, bbox_inches='tight')
plt.show()

print("Lectura del dendrograma:")
print("  Eje X → puntos individuales (hojas del árbol)")
print("  Eje Y → altura de la fusión (mayor = más distintos al fusionarse)")
print("  Línea roja → corte que produce k=3 clusters")
print("  Grupos formados al cortar: {A1,A2,A3}, {B1,B2,B3}, {C1,C2}")
```

**Script de explicación del dendrograma:**

*"Mirad la estructura del árbol. Abajo están los puntos individuales. Las ramas que se unen bajas son fusiones de puntos muy cercanos —dentro del mismo grupo real. Conforme subimos, vemos cómo se consolidan los grupos. El salto más grande está justo antes de que los tres grupos se fusionen en uno. La línea roja corta el árbol en ese punto y nos da tres clusters."*

*"La altura de cada fusión en el eje Y es vuestra información más valiosa. Un salto grande indica una discontinuidad real en los datos."*

---

#### Celda 4 — Comparación de criterios de enlace

```python
# -------------------------------------------------------
# ¿Cómo cambia el dendrograma según el criterio de enlace?
# -------------------------------------------------------

# Dataset con estructura más compleja para ver diferencias
np.random.seed(7)
X_comp, _ = make_blobs(n_samples=80,
                        centers=[[-4,0],[0,0],[4,0],[0,4]],
                        cluster_std=[0.5, 0.5, 0.5, 1.5])

metodos = ['single', 'complete', 'average', 'ward']
titulos = [
    'Single linkage\n(par más cercano)',
    'Complete linkage\n(par más lejano)',
    'Average linkage\n(media de pares)',
    'Ward\n(mínimo WCSS)'
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for ax, metodo, titulo in zip(axes, metodos, titulos):
    Z_m = linkage(X_comp, method=metodo)
    dendrogram(Z_m, ax=ax, no_labels=True,
               color_threshold=0.6 * max(Z_m[:, 2]))
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_ylabel("Distancia de fusión")
    ax.set_xlabel("Puntos")

plt.suptitle("Mismo dataset — cuatro criterios de enlace, cuatro dendrogramas",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_linkage_comparacion.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación de cada criterio:**

*"Fijémonos en Single linkage —el primero. La estructura es muy 'plana' en los niveles bajos: muchas fusiones pequeñas antes del gran salto. Este es el efecto cadena: los puntos se van encadenando de uno en uno. Funciona bien para clusters de forma arbitraria pero es muy sensible a outliers."*

*"Complete linkage fuerza clusters más compactos. Aquí el árbol es más 'equilibrado'."*

*"Ward es el que produce los saltos más claros y la estructura más limpia. Si buscáis un criterio por defecto, Ward es casi siempre el mejor punto de partida."*

---

#### Celda 5 — Cómo leer el codo del dendrograma para elegir k

```python
# -------------------------------------------------------
# Técnica del "mayor salto" para elegir k
# -------------------------------------------------------

Z_ward = linkage(X_comp, method='ward')

# Las alturas de fusión en orden (las últimas n-1 fusiones son las más relevantes)
alturas = Z_ward[:, 2]
alturas_ordenadas = np.sort(alturas)[::-1]  # de mayor a menor

# Aceleraciones: diferencia entre fusiones consecutivas
aceleraciones = np.diff(alturas_ordenadas)
k_sugerido = np.argmax(aceleraciones) + 2  # +2 porque diff reduce en 1 y empezamos en k=2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dendrograma con línea de corte automática
ax1 = axes[0]
umbral = (alturas_ordenadas[k_sugerido - 2] + alturas_ordenadas[k_sugerido - 1]) / 2
dendrogram(Z_ward, ax=ax1, no_labels=True,
           color_threshold=umbral)
ax1.axhline(y=umbral, color='red', linestyle='--', linewidth=2,
            label=f'Corte automático → k={k_sugerido}')
ax1.set_title(f"Dendrograma Ward — Corte sugerido: k={k_sugerido}",
              fontsize=11, fontweight='bold')
ax1.set_ylabel("Distancia de fusión")
ax1.legend(fontsize=10)

# Gráfico de aceleraciones (análogo al codo de K-Means)
ax2 = axes[1]
ks = range(2, len(aceleraciones) + 2)
ax2.bar(ks, aceleraciones[:len(ks)], color='steelblue', alpha=0.8)
ax2.axvline(x=k_sugerido, color='red', linestyle='--', linewidth=2,
            label=f'k sugerido = {k_sugerido}')
ax2.set_xlabel("Número de clusters (k)")
ax2.set_ylabel("Aceleración de la distancia de fusión")
ax2.set_title("Mayor salto → k óptimo sugerido", fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)

plt.suptitle("Selección automática de k desde el dendrograma",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("img_dendrograma_corte.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"K sugerido por el criterio del mayor salto: k = {k_sugerido}")
```

**Script de explicación:**

*"El gráfico de la derecha es el equivalente del 'método del codo' para clustering jerárquico. En lugar de graficar la WCSS, graficamos la aceleración de las distancias de fusión: cuánto sube el umbral de un paso al siguiente. La barra más alta indica la mayor discontinuidad — ahí está el 'corte natural'."*

---

#### Celda 6 — Caso práctico: Agrupación de países por indicadores económicos

```python
# -------------------------------------------------------
# CASO PRÁCTICO: Países agrupados por indicadores
# económicos (dataset simplificado tipo World Bank)
# -------------------------------------------------------

# Dataset sintético que replica la estructura de datos reales
# de indicadores macroeconómicos por país (escala 0-100 normalizada)
np.random.seed(0)

paises_data = {
    'País': [
        'Alemania','Francia','Italia','España','Países Bajos',
        'Polonia','Hungría','Rumanía','Bulgaria','Eslovaquia',
        'Nigeria','Ghana','Kenia','Sudáfrica','Etiopía',
        'Brasil','México','Argentina','Colombia','Chile',
        'China','India','Indonesia','Vietnam','Tailandia',
        'EEUU','Canadá','Australia','Japón','Corea del Sur'
    ],
    'PIB_per_capita_idx': [
        88,84,74,72,90, 52,48,38,35,50,
        22,25,28,45,15, 45,40,38,35,50,
        55,32,38,35,42, 95,92,88,85,80
    ],
    'IDH': [
        93,90,88,88,93, 77,77,74,70,77,
        52,55,55,68,45, 74,74,79,72,80,
        74,64,68,68,70, 92,92,92,91,90
    ],
    'Gini_inv': [  # invertido: mayor = más equitativo
        60,59,56,52,55, 54,52,56,58,50,
        48,50,52,48,62, 42,48,45,48,50,
        52,60,55,60,56, 58,65,68,70,65
    ],
    'Esperanza_vida': [
        80,82,83,83,82, 77,75,74,72,76,
        54,60,61,62,64, 73,75,76,73,79,
        75,68,69,73,75, 79,82,83,84,82
    ]
}

df_paises = pd.DataFrame(paises_data).set_index('País')
print(f"Dataset: {df_paises.shape[0]} países, {df_paises.shape[1]} indicadores")
print(df_paises.head(5))
```

---

#### Celda 7 — Dendrograma de países + corte e interpretación

```python
# Escalamos los datos
scaler = StandardScaler()
X_paises = scaler.fit_transform(df_paises)

# Calculamos el linkage con Ward
Z_paises = linkage(X_paises, method='ward', metric='euclidean')

# ---- Dendrograma anotado ----
fig, ax = plt.subplots(figsize=(14, 7))

dend = dendrogram(
    Z_paises,
    labels=df_paises.index.tolist(),
    ax=ax,
    orientation='top',
    color_threshold=3.5,
    leaf_font_size=10,
    leaf_rotation=45
)

# Línea de corte para k=4
ax.axhline(y=3.5, color='red', linestyle='--', linewidth=2,
           label='Corte → 4 grupos')
ax.set_title("Clustering Jerárquico de países — Indicadores económicos (Ward)",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Distancia Ward (disimilitud)", fontsize=11)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("img_dendrograma_paises.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación:**

*"Este dendrograma ya habla por sí solo. Las hojas del árbol son los países. Los que se fusionan en niveles bajos son los más parecidos según los cuatro indicadores. Mirad cómo Alemania y Países Bajos se fusionan muy pronto, igual que EEUU, Canadá y Australia. En cambio, el grupo africano se fusiona con los demás en niveles muy altos, lo que indica una gran diferencia estructural."*

*"La línea roja corta el árbol en cuatro grupos. Pero podría cortarse en tres o en cinco — dependiendo de qué granularidad tiene sentido para vuestro análisis."*

---

#### Celda 8 — Extracción de clusters y visualización con perfiles

```python
# Extraemos las etiquetas para k=4
from scipy.cluster.hierarchy import fcluster

labels_paises = fcluster(Z_paises, t=4, criterion='maxclust') - 1  # 0-indexed
df_paises['Cluster'] = labels_paises

# ---- Perfil medio de cada cluster ----
perfil = df_paises.groupby('Cluster')[
    ['PIB_per_capita_idx','IDH','Gini_inv','Esperanza_vida']
].mean().round(1)

print("Perfil medio de cada cluster:")
print(perfil)
print()

# ---- Lista de países por cluster ----
for c in sorted(df_paises['Cluster'].unique()):
    miembros = df_paises[df_paises['Cluster'] == c].index.tolist()
    print(f"Cluster {c}: {', '.join(miembros)}")
```

---

#### Celda 9 — Heatmap de perfiles para comunicar resultados

```python
# El heatmap es una forma muy efectiva de comunicar los clusters
# a una audiencia no técnica

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap de perfiles medios por cluster
ax1 = axes[0]
perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min())
sns.heatmap(perfil_norm, annot=perfil, fmt='.0f',
            cmap='RdYlGn', ax=ax1,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Nivel relativo (0=mínimo, 1=máximo)'})
ax1.set_title("Perfil de cada cluster\n(valor real anotado, color = nivel relativo)",
              fontsize=11, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha='right')
ax1.set_yticklabels([f'Cluster {c}' for c in perfil.index], rotation=0)

# Scatter: PIB vs IDH coloreado por cluster
ax2 = axes[1]
colores_c = ['#e41a1c','#377eb8','#4daf4a','#ff7f00']
for c in sorted(df_paises['Cluster'].unique()):
    mask = df_paises['Cluster'] == c
    ax2.scatter(
        df_paises.loc[mask,'PIB_per_capita_idx'],
        df_paises.loc[mask,'IDH'],
        c=colores_c[c], s=100, label=f'Cluster {c}', alpha=0.85
    )
    for pais in df_paises[mask].index:
        ax2.annotate(pais,
                     (df_paises.loc[pais,'PIB_per_capita_idx'],
                      df_paises.loc[pais,'IDH']),
                     fontsize=7, alpha=0.8,
                     textcoords="offset points", xytext=(4, 3))
ax2.set_xlabel("PIB per cápita (índice)", fontsize=11)
ax2.set_ylabel("IDH", fontsize=11)
ax2.set_title("Países en el espacio PIB-IDH\ncoloreados por cluster jerárquico",
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)

plt.suptitle("Interpretación de los clusters jerárquicos de países",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_paises_clusters_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de interpretación:**

*"El heatmap es vuestra herramienta de comunicación con negocio. El color verde indica nivel alto, el rojo nivel bajo. Mirad el patrón: un cluster tiene verde en todo —los países desarrollados de alto PIB, IDH y equidad—. Otro tiene rojo en casi todo —los países de bajo desarrollo—. Los otros dos son posiciones intermedias diferenciadas."*

*"Notad que este resultado no requirió especificar k de antemano. Lo descubrimos del propio dendrograma. Esa es la ventaja de la exploración jerárquica."*

---

#### Celda 10 — Uso de scikit-learn: AgglomerativeClustering

```python
# -------------------------------------------------------
# Interfaz de scikit-learn — más integrada con pipelines
# -------------------------------------------------------

from sklearn.cluster import AgglomerativeClustering

# Equivalente al scipy + fcluster anterior, pero con API sklearn
hc = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward',       # 'ward', 'complete', 'average', 'single'
    metric='euclidean'    # Ward solo funciona con euclidiana
)
labels_sklearn = hc.fit_predict(X_paises)

# Verificar que produce los mismos clusters (pueden tener numeración distinta)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(labels_paises, labels_sklearn)
print(f"Adjusted Rand Index scipy vs sklearn: {ari:.4f}")
print("(1.0 = asignaciones idénticas, ajustado por numeración)")

# Para distancias no-euclidianas, usar connectivity o precomputed
# Ejemplo conceptual (no ejecutar sin un dataset adecuado):
# hc_coseno = AgglomerativeClustering(
#     n_clusters=4, linkage='average', metric='cosine'
# )
print("\nNota: Para métricas distintas a euclidiana, usar linkage='average'")
print("Ward solo está definido para distancia euclidiana.")
```

---

#### Celda 11 — Comparación final: Jerárquico vs K-Means en el mismo dataset

```python
from sklearn.cluster import KMeans

km_paises = KMeans(n_clusters=4, n_init=20, random_state=42)
labels_km_paises = km_paises.fit_predict(X_paises)

ari_comparacion = adjusted_rand_score(labels_paises, labels_km_paises)
print(f"Coincidencia K-Means vs Jerárquico (ARI): {ari_comparacion:.4f}")
print()

# Mostrar diferencias
df_comp = pd.DataFrame({
    'Jerárquico (Ward)': labels_paises,
    'K-Means': labels_km_paises
}, index=df_paises.index)

# Países donde difieren
diferencias = df_comp[df_comp.iloc[:,0] != df_comp.iloc[:,1]]
if len(diferencias) > 0:
    print("Países con asignación diferente entre ambos métodos:")
    print(diferencias)
else:
    print("Ambos métodos producen la misma agrupación (tras ajuste de numeración).")

print("""
Reflexión:
  Si K-Means y Jerárquico coinciden → la estructura es robusta.
  Si difieren → explorar con el dendrograma para entender por qué.
  El jerárquico siempre aporta más información (el árbol completo).
""")
```

**Script de discusión de cierre:**

*"En datasets pequeños como este, K-Means y el clustering jerárquico suelen dar resultados muy parecidos si los datos tienen estructura clara. La ventaja del jerárquico no está en que dé mejores clusters: está en que da un árbol, que es información adicional. Podéis ver qué países están 'a punto de' pertenecer a otro cluster, qué nivel de granularidad tiene sentido, y cómo se estructuran las relaciones entre grupos."*

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada del bloque. Pregunta retórica: *"¿Cuántos clusters tiene este dataset?"* con un dendrograma como imagen de fondo.
- **Slide 2:** Bottom-up vs. top-down — dos diagramas de árbol en espejo. Una flecha sube, otra baja.
- **Slide 3:** Los 4 pasos del algoritmo aglomerativo en pseudocódigo visual.
- **Slide 4:** Los 4 criterios de enlace — fórmula + diagrama geométrico mostrando qué distancia mide cada uno en un par de clusters.
- **Slide 5:** Cómo leer un dendrograma — diagrama anotado con flechas explicando: hojas, altura de fusión, cómo cortar.
- **Slide 6:** Tabla resumen de criterios de enlace.
- **Slide 7:** Comparación de 4 dendrogramas del mismo dataset con distintos criterios de enlace (de la Celda 4).
- **Slide 8:** Cuándo sí / cuándo no usar jerárquico — dos columnas con iconos.

### Para el handout

- Tabla de criterios de enlace con fórmulas y características.
- Imagen del dendrograma anotado (Celda 3) con guía de lectura.
- Imagen comparativa de los 4 criterios de enlace (Celda 4).
- Heatmap de perfiles de países (Celda 9) como ejemplo de output interpretable.
- Tabla de decisión: Jerárquico vs. K-Means vs. K-Medoids.

### Para el Jupyter Notebook (ejercicios a completar por los alumnos)

**Ejercicio 1 (Celda 7 ampliada):** Probar cortes a k=3, k=4 y k=5 sobre el dataset de países. Para cada uno, listar los países de cada cluster e interpretar qué agrupación tiene más sentido geopolíticamente.

**Ejercicio 2 (Celda 4 ampliada):** Añadir el criterio de Ward al dataset de lunas (`make_moons`). ¿Qué sucede? ¿El jerárquico puede resolver clusters no convexos con algún criterio de enlace? (Respuesta esperada: single linkage puede, Ward no.)

**Ejercicio 3 (Celda 10 ampliada):** Repetir el análisis de países usando `metric='cosine'` y `linkage='average'`. ¿Los grupos cambian? ¿Por qué la similitud coseno podría tener sentido para comparar perfiles de países?

**Ejercicio 4 (avanzado):** Implementar el algoritmo de Single Linkage desde cero usando solo NumPy y una matriz de distancias. Verificar que produce el mismo historial de fusiones que `scipy.cluster.hierarchy.linkage(method='single')`.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador de progreso |
|---|---|---|
| Transición y motivación | 4 min | Pregunta de enganche respondida |
| Aglomerativo vs. divisivo | 6 min | Diagrama en pantalla |
| Algoritmo y criterios de enlace | 8 min | Tabla de criterios en pantalla |
| Dendrograma: lectura e interpretación | 7 min | Dendrograma anotado en pantalla |
| Celda 1-2 (imports + pasos manuales) | 8 min | Tabla de fusiones en pantalla |
| Celda 3 (dendrograma mini) | 7 min | Dendrograma generado e interpretado |
| Celda 4-5 (comparación criterios + corte) | 10 min | Los 4 dendrogramas comparados |
| Celda 6-9 (caso países) | 12 min | Heatmap de perfiles generado |
| Celda 10-11 (sklearn + comparación K-Means) | 3 min | ARI calculado |
| Discusión de cierre | 3 min + 2 buffer | Pregunta respondida |
| **Total** | **65 min** | |

---

*Bloque 1.3 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
