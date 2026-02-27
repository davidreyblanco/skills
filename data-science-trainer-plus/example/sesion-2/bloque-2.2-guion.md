# Bloque 2.2 — Mapas Auto-Organizados de Kohonen (SOM)
## Guión detallado del instructor

**Duración:** 60 minutos (30 min teoría + 30 min práctica en Jupyter Notebook)
**Posición en la sesión:** Segundo bloque de la Sesión 2

---

## PARTE TEÓRICA (30 min)

---

### [00:00 – 00:04] Transición: del espacio probabilístico al espacio topológico

**Script de transición:**

*"Con GMM aprendimos a hacer clustering suave: en lugar de asignar etiquetas duras, asignamos probabilidades. Ahora vamos a dar otro salto conceptual. Hasta ahora todos los algoritmos que hemos visto responden a la pregunta: '¿A qué grupo pertenece este punto?' Los Mapas Auto-Organizados de Kohonen responden a una pregunta distinta: '¿Cómo se organiza la estructura de los datos en un mapa de baja dimensión que preserve las relaciones de vecindad?'"*

*"Es decir, los SOM hacen clustering y reducción de dimensionalidad a la vez. Y el resultado no es una lista de etiquetas, sino un mapa —literalmente una cuadrícula 2D— donde los puntos similares están cerca y los distintos están lejos. Eso los convierte en una herramienta de exploración y visualización enormemente potente."*

**Contexto histórico:**

*"Los SOM fueron propuestos por Teuvo Kohonen en 1982 en la Universidad de Helsinki y publicados en su forma definitiva en 1990. A día de hoy siguen siendo ampliamente usados en análisis de riesgo crediticio, visualización de datos genómicos, segmentación de usuarios en plataformas digitales y análisis de series temporales financieras. Su longevidad se debe a su interpretabilidad única: el mapa es autoexplicativo."*

---

### [00:04 – 00:12] Arquitectura de un SOM

**Estructura:**

Un SOM consta de dos capas:

1. **Capa de entrada:** `n` neuronas, una por dimensión del dato de entrada. Cada dato `x ∈ ℝⁿ` se presenta a esta capa.

2. **Capa de mapa (competitiva):** Una cuadrícula 2D de `m₁ × m₂` neuronas. Cada neurona `j` tiene:
   - Una **posición fija** en la cuadrícula: coordenadas `(rⱼ₁, rⱼ₂)` que no cambian.
   - Un **vector de pesos** `wⱼ ∈ ℝⁿ` de la misma dimensión que el dato de entrada. Estos sí cambian durante el entrenamiento.

**Analogía geográfica:**

*"Imaginad un mapa físico del mundo. Cada país tiene una posición fija en el mapa. Pero lo que 'contiene' ese país —sus características climáticas, económicas, culturales— puede ir cambiando. Durante el entrenamiento del SOM, los países ajustan sus 'características' (pesos) para parecerse lo más posible a los datos que les tocan. Al final, países vecinos en el mapa tienen características similares, igual que en la realidad los países vecinos suelen compartir características."*

**Tamaño del mapa:**

La cuadrícula puede ser rectangular o hexagonal. La hexagonal es más habitual porque reduce los efectos de borde. El tamaño `m₁ × m₂` determina la resolución del mapa:
- Mapa pequeño (ej. 5×5): pocos clusters, visión general.
- Mapa grande (ej. 20×20): alta resolución, clusters finos.
- Regla empírica: `m₁ × m₂ ≈ 5√n` donde `n` es el número de datos.

---

### [00:12 – 00:22] El algoritmo de entrenamiento

**El proceso de aprendizaje competitivo — tres conceptos clave:**

**1. BMU (Best Matching Unit) — La neurona ganadora:**

Para cada dato de entrada `x`, se busca la neurona cuyo vector de pesos `wⱼ` es más similar a `x`. La similaridad se mide con distancia euclidiana:

```
BMU(x) = argmin_j ||x - wⱼ||
```

La BMU es la neurona que "representa mejor" al dato en el mapa. Es el análogo al centroide más cercano en K-Means, con la diferencia de que en el SOM la BMU tiene una posición en el mapa 2D.

**2. Función de vecindad — El aprendizaje se propaga:**

A diferencia de K-Means, en el SOM no solo la neurona ganadora actualiza sus pesos: sus vecinas en el mapa también lo hacen, en menor medida. La influencia decrece con la distancia en el mapa.

La función de vecindad gaussiana es la más habitual:

```
h(j, BMU, σ) = exp(-||rⱼ - r_BMU||² / (2σ²))
```

donde `rⱼ` y `r_BMU` son las posiciones 2D de la neurona `j` y de la BMU respectivamente, y `σ` es el radio de vecindad (un hiperparámetro que decrece a lo largo del entrenamiento).

*"Este es el mecanismo que hace que el mapa sea topológico. Si muchos datos similares activan la misma zona del mapa, las neuronas de esa zona acaban teniendo pesos parecidos. Los datos similares quedan mapeados cerca en el mapa 2D."*

**3. Actualización de pesos:**

En cada paso del entrenamiento, para cada dato `x`:

```
wⱼ_nuevo = wⱼ + α · h(j, BMU, σ) · (x - wⱼ)
```

donde `α` es la **tasa de aprendizaje** (learning rate). Interpretación: cada neurona mueve su vector de pesos hacia `x`, con una fuerza proporcional a `α · h`. Las neuronas lejanas a la BMU apenas se mueven.

**Decaimiento temporal:**

Tanto `α` como `σ` decrecen a lo largo del entrenamiento:

```
α(t) = α₀ · exp(-t / λ_α)
σ(t) = σ₀ · exp(-t / λ_σ)
```

Al principio del entrenamiento: `α` grande, `σ` grande → el mapa se organiza a gran escala (topología global). Al final: `α` pequeño, `σ` pequeño → ajuste fino local.

*"Hay dos fases de entrenamiento claramente distintas. La fase inicial con vecindad amplia y aprendizaje rápido es la 'fase de ordenación': el mapa se organiza topológicamente. La fase final con vecindad estrecha es la 'fase de convergencia': el mapa se afina. Si solo entrenáis en la segunda fase sin la primera, el mapa puede quedar topológicamente desordenado."*

---

### [00:22 – 00:28] Interpretación del mapa: U-Matrix y component planes

**Una vez entrenado el SOM, ¿cómo extraemos información?**

**U-Matrix (Unified Distance Matrix):**

Para cada neurona `j`, calcula la distancia media a sus neuronas vecinas en el mapa. El resultado es una imagen donde:
- **Zonas oscuras** (distancias altas) = fronteras entre clusters: las neuronas vecinas son muy distintas entre sí.
- **Zonas claras** (distancias bajas) = interior de clusters: las neuronas vecinas son similares.

*"La U-Matrix es el dendrograma del SOM: te da la estructura de clusters sin necesidad de especificar k. Las fronteras oscuras separan los grupos; las zonas claras son los grupos. Cuántos grupos hay se lee directamente del mapa."*

**Component planes:**

Para cada dimensión `d` del dato de entrada, se puede visualizar el valor del peso `wⱼd` de cada neurona como una imagen de calor. Esto permite:
- Ver qué zonas del mapa tienen valores altos o bajos en cada variable.
- Entender qué variables definen cada región del mapa.
- Detectar correlaciones entre variables (si dos component planes tienen patrón similar, las variables están correladas).

**Mapa de hits (activaciones):**

Para cada neurona, cuenta cuántos datos del dataset la activaron como BMU. Una neurona con muchos hits representa un patrón muy frecuente; una con pocos o cero hits puede ser una zona de transición o una región poco poblada.

---

### [00:28 – 00:30] Ventajas, limitaciones y cuándo usar SOM

**Ventajas:**

- Clustering y reducción de dimensionalidad simultáneos.
- El mapa preserva topología: puntos similares quedan cerca en 2D.
- Muy visual e interpretable, incluso para audiencias no técnicas.
- No requiere especificar k de antemano (la U-Matrix revela la estructura).
- Funciona bien con datos de alta dimensionalidad.

**Limitaciones:**

- Hiperparámetros sensibles: tamaño del mapa, `α₀`, `σ₀`, número de épocas.
- No hay garantía de convergencia global.
- No produce probabilidades (a diferencia de GMM).
- Las métricas estándar de evaluación (Silhouette, Davies-Bouldin) se aplican sobre las asignaciones BMU, no directamente sobre el mapa.
- Para datasets muy grandes puede ser lento si no se usa la variante online eficiente.

**Cuándo usar SOM:**

- Exploración de datos de alta dimensionalidad donde la visualización 2D importa.
- Cuando se necesita un mapa interpretable para presentar a negocio.
- Análisis de perfiles de comportamiento (clientes, usuarios, pacientes).
- Cuando se quiere una vista "panorámica" del espacio de datos antes de aplicar un clustering más fino.

---

## PARTE PRÁCTICA — Jupyter Notebook (30 min)

---

### [00:30 – 01:00] Práctica guiada

---

#### Celda 1 — Instalación y verificación de MiniSom

```python
# ============================================================
# BLOQUE 2.2 — Mapas Auto-Organizados de Kohonen (SOM)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style("white")
np.random.seed(42)

# Verificar/instalar MiniSom
try:
    from minisom import MiniSom
    print("✓ MiniSom disponible")
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "minisom", "-q"])
    from minisom import MiniSom
    print("✓ MiniSom instalado y cargado")

print(f"  API principal: MiniSom(x, y, input_len, sigma, learning_rate)")
```

---

#### Celda 2 — Entrenamiento de un SOM básico con visualización de convergencia

```python
# -------------------------------------------------------
# Entrenamiento paso a paso con dataset sintético
# para entender qué hace el algoritmo
# -------------------------------------------------------

# Dataset: 3 grupos bien definidos en 2D
np.random.seed(0)
X_base, y_base = make_blobs(n_samples=300, centers=3,
                             cluster_std=0.8, random_state=42)
X_base_norm = StandardScaler().fit_transform(X_base)

# Parámetros del SOM
m1, m2    = 8, 8       # tamaño del mapa
sigma0    = 3.0        # radio de vecindad inicial
alpha0    = 0.5        # tasa de aprendizaje inicial
epocas    = 1000

som = MiniSom(
    x=m1, y=m2,
    input_len=X_base_norm.shape[1],
    sigma=sigma0,
    learning_rate=alpha0,
    neighborhood_function='gaussian',
    random_seed=42
)

# Inicialización con PCA (más estable que aleatoria)
som.pca_weights_init(X_base_norm)

# Entrenamiento con registro del error de cuantización
errores = []
n_check = 50
for ep in range(0, epocas, n_check):
    som.train(X_base_norm, n_check, verbose=False)
    errores.append(som.quantization_error(X_base_norm))

# Visualización del error de cuantización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(range(0, epocas, n_check), errores, 'b-o', linewidth=2, markersize=5)
ax1.set_xlabel("Época")
ax1.set_ylabel("Error de cuantización")
ax1.set_title("Convergencia del SOM\n(error decrece → mapa se organiza)",
              fontsize=11, fontweight='bold')
ax1.axhline(y=min(errores), color='red', linestyle='--', alpha=0.7,
            label=f'Mínimo: {min(errores):.4f}')
ax1.legend(fontsize=9)

# U-Matrix
ax2 = axes[1]
umatrix = som.distance_map()
im = ax2.imshow(umatrix.T, cmap='bone_r', origin='lower',
                interpolation='nearest')
plt.colorbar(im, ax=ax2, label='Distancia media a vecinos')
ax2.set_title(f"U-Matrix ({m1}×{m2})\n(oscuro = frontera, claro = interior de cluster)",
              fontsize=11, fontweight='bold')
ax2.set_xlabel("Neurona x")
ax2.set_ylabel("Neurona y")

plt.suptitle("Entrenamiento y convergencia del SOM",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_som_convergencia.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Error de cuantización final: {errores[-1]:.4f}")
print("Interpretación: cuanto menor, más fielmente el mapa representa los datos.")
```

**Script de explicación:**

*"El error de cuantización mide la distancia media de cada punto a su BMU. Si decrece y se estabiliza, el mapa ha convergido: las neuronas han encontrado posiciones estables que representan bien los datos. Si no decrece suficiente, necesitáis más épocas o ajustar la tasa de aprendizaje."*

*"La U-Matrix es la imagen del mapa coloreada por la distancia entre neuronas vecinas. Las zonas oscuras son fronteras —las neuronas a cada lado son muy distintas—. Las zonas claras son el interior de los clusters. ¿Cuántos clusters hay? Contad las zonas claras separadas por fronteras oscuras."*

---

#### Celda 3 — Mapa de hits y asignación de datos al mapa

```python
# -------------------------------------------------------
# ¿Dónde se activan los datos en el mapa?
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) Mapa de hits
ax1 = axes[0]
hits = np.zeros((m1, m2))
for x in X_base_norm:
    bmu = som.winner(x)
    hits[bmu] += 1

im_hits = ax1.imshow(hits.T, cmap='YlOrRd', origin='lower',
                     interpolation='nearest')
plt.colorbar(im_hits, ax=ax1, label='Nº de activaciones')
ax1.set_title("Mapa de hits\n(cuántas veces se activa cada neurona)",
              fontsize=10, fontweight='bold')
ax1.set_xlabel("Neurona x")
ax1.set_ylabel("Neurona y")

# 2) Proyección de datos coloreados por etiqueta real
ax2 = axes[1]
im_u2 = ax2.imshow(umatrix.T, cmap='bone_r', origin='lower',
                   interpolation='nearest', alpha=0.6)
colores_base = plt.cm.tab10(np.linspace(0, 0.3, 3))
marcadores = ['o', 's', '^']
for i, x in enumerate(X_base_norm):
    bmu = som.winner(x)
    ax2.plot(bmu[0], bmu[1],
             marker=marcadores[y_base[i]],
             color=colores_base[y_base[i]],
             markersize=5, alpha=0.7, markeredgewidth=0)
ax2.set_title("Datos proyectados en el mapa\n(forma = cluster real)",
              fontsize=10, fontweight='bold')
ax2.set_xlabel("Neurona x")
ax2.set_ylabel("Neurona y")

# Leyenda manual
for c, (m, col) in enumerate(zip(marcadores, colores_base)):
    ax2.plot([], [], marker=m, color=col, markersize=8,
             label=f'Cluster real {c}', linestyle='None')
ax2.legend(fontsize=8, loc='upper right')

# 3) U-Matrix sola para referencia
ax3 = axes[2]
im_u3 = ax3.imshow(umatrix.T, cmap='bone_r', origin='lower',
                   interpolation='nearest')
plt.colorbar(im_u3, ax=ax3, label='Distancia a vecinos')
ax3.set_title("U-Matrix — referencia\n(las fronteras delimitan los clusters)",
              fontsize=10, fontweight='bold')

plt.suptitle("SOM: hits, proyección de datos y U-Matrix",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_som_hits_proyeccion.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación:**

*"El mapa de la izquierda muestra dónde se concentran los datos en el mapa. Las zonas más calientes son las más frecuentes. El mapa central proyecta cada dato como un punto coloreado según su cluster real —y vemos que los tres clusters ocupan zonas distintas del mapa, exactamente donde esperábamos según la U-Matrix. Esto confirma que el SOM ha aprendido correctamente la estructura."*

---

#### Celda 4 — Caso práctico: SOM sobre datos de e-commerce

```python
# -------------------------------------------------------
# CASO PRÁCTICO: Mapa de perfiles de clientes de e-commerce
# -------------------------------------------------------

np.random.seed(1)
n_clientes = 400

# 5 perfiles de clientes
recencia    = np.concatenate([
    np.random.normal(5,   2,  80),   # recientes activos
    np.random.normal(60,  15, 80),   # dormidos
    np.random.normal(15,  5,  80),   # regulares
    np.random.normal(30,  10, 80),   # ocasionales
    np.random.normal(3,   1,  80),   # VIP muy activos
])
frecuencia  = np.concatenate([
    np.random.normal(8,  2,  80),
    np.random.normal(1,  0.5,80),
    np.random.normal(4,  1,  80),
    np.random.normal(2,  0.8,80),
    np.random.normal(20, 4,  80),
])
monetario   = np.concatenate([
    np.random.normal(120, 30, 80),
    np.random.normal(40,  20, 80),
    np.random.normal(80,  20, 80),
    np.random.normal(60,  25, 80),
    np.random.normal(400, 80, 80),
])
n_productos = np.concatenate([
    np.random.normal(3,  1,  80),
    np.random.normal(1,  0.5,80),
    np.random.normal(2,  0.8,80),
    np.random.normal(1.5,0.7,80),
    np.random.normal(8,  2,  80),
])

perfil_real = np.concatenate([
    ['Activos medios'] * 80,
    ['Dormidos']       * 80,
    ['Regulares']      * 80,
    ['Ocasionales']    * 80,
    ['VIP']            * 80,
])

df_ecom_som = pd.DataFrame({
    'recencia_dias':  recencia.clip(0),
    'frecuencia':     frecuencia.clip(0),
    'monetario':      monetario.clip(0),
    'n_productos':    n_productos.clip(0),
    'perfil_real':    perfil_real,
})

features_som = ['recencia_dias','frecuencia','monetario','n_productos']
scaler_som   = StandardScaler()
X_som = scaler_som.fit_transform(df_ecom_som[features_som])

print(f"Dataset e-commerce: {len(df_ecom_som)} clientes, "
      f"{len(features_som)} features RFM+")
print(df_ecom_som[features_som].describe().round(1))
```

---

#### Celda 5 — Entrenamiento del SOM y U-Matrix del caso real

```python
# Entrenamos el SOM
m1_ecom, m2_ecom = 10, 10
som_ecom = MiniSom(
    x=m1_ecom, y=m2_ecom,
    input_len=len(features_som),
    sigma=2.5,
    learning_rate=0.5,
    neighborhood_function='gaussian',
    random_seed=42
)
som_ecom.pca_weights_init(X_som)
som_ecom.train(X_som, num_iteration=5000, verbose=False)

print(f"Error de cuantización: {som_ecom.quantization_error(X_som):.4f}")

# U-Matrix anotada con los perfiles reales
umatrix_ecom = som_ecom.distance_map()

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# U-Matrix con datos proyectados
ax1 = axes[0]
ax1.imshow(umatrix_ecom.T, cmap='bone_r', origin='lower',
           interpolation='nearest', alpha=0.8)

colores_perfil = {
    'Activos medios': '#377eb8',
    'Dormidos':       '#e41a1c',
    'Regulares':      '#4daf4a',
    'Ocasionales':    '#ff7f00',
    'VIP':            '#984ea3',
}
marcadores_perfil = {
    'Activos medios': 'o',
    'Dormidos':       'x',
    'Regulares':      's',
    'Ocasionales':    '^',
    'VIP':            '*',
}

for perfil in df_ecom_som['perfil_real'].unique():
    mask = df_ecom_som['perfil_real'] == perfil
    for x in X_som[mask.values]:
        bmu = som_ecom.winner(x)
        ax1.plot(bmu[0], bmu[1],
                 marker=marcadores_perfil[perfil],
                 color=colores_perfil[perfil],
                 markersize=5 if perfil != 'VIP' else 9,
                 alpha=0.6, markeredgewidth=0.5)

# Leyenda
for perfil in colores_perfil:
    ax1.plot([], [], marker=marcadores_perfil[perfil],
             color=colores_perfil[perfil], markersize=8,
             label=perfil, linestyle='None')
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax1.set_title(f"SOM {m1_ecom}×{m2_ecom} — Clientes e-commerce proyectados\n"
              f"(U-Matrix de fondo, forma = perfil real)",
              fontsize=11, fontweight='bold')
ax1.set_xlabel("Neurona x")
ax1.set_ylabel("Neurona y")

# Component planes para las 4 variables
ax2 = axes[1]
ax2.set_visible(False)

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
for ax, feature in zip(axes2, features_som):
    idx = features_som.index(feature)
    plane = som_ecom.get_weights()[:, :, idx]
    im = ax.imshow(plane.T, cmap='RdYlGn_r' if 'recencia' in feature else 'RdYlGn',
                   origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Component plane\n'{feature}'",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Neurona x")
    ax.set_ylabel("Neurona y")

plt.suptitle("Component planes — Valor de cada feature por neurona",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("img_som_component_planes.png", dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 7))
plt.imshow(umatrix_ecom.T, cmap='bone_r', origin='lower', interpolation='nearest')
for perfil in df_ecom_som['perfil_real'].unique():
    mask = df_ecom_som['perfil_real'] == perfil
    for x in X_som[mask.values]:
        bmu = som_ecom.winner(x)
        plt.plot(bmu[0], bmu[1],
                 marker=marcadores_perfil[perfil],
                 color=colores_perfil[perfil],
                 markersize=5 if perfil != 'VIP' else 9,
                 alpha=0.6, markeredgewidth=0.5)
for perfil in colores_perfil:
    plt.plot([], [], marker=marcadores_perfil[perfil],
             color=colores_perfil[perfil], markersize=8,
             label=perfil, linestyle='None')
plt.legend(fontsize=9, loc='upper right')
plt.title("U-Matrix con clientes proyectados", fontsize=12, fontweight='bold')
plt.colorbar(label='Distancia a vecinos')
plt.tight_layout()
plt.savefig("img_som_umatrix_ecom.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de interpretación:**

*"Aquí está la potencia del SOM. Los component planes nos dicen qué zona del mapa tiene valores altos o bajos en cada variable. Combinándolos podemos leer el perfil de cada región: la esquina donde 'recencia' es baja (compras recientes) y 'monetario' es alto (gasto elevado) —esa es la zona VIP. La región donde recencia es alta (llevan tiempo sin comprar) y frecuencia es baja —esos son los dormidos."*

*"Y lo más potente: sin haberle dicho al algoritmo cuántos perfiles hay, el mapa nos los muestra visualmente. Las fronteras oscuras de la U-Matrix separan naturalmente los grupos."*

---

#### Celda 6 — Extracción de etiquetas y comparación con K-Means

```python
# -------------------------------------------------------
# Extraer etiquetas de cluster del SOM y comparar
# -------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Asignamos cada cliente a su BMU y luego clusterizamos los pesos del SOM
bmus = np.array([som_ecom.winner(x) for x in X_som])
bmu_idx = bmus[:, 0] * m2_ecom + bmus[:, 1]  # índice lineal de la BMU

# Clusterizar los pesos del mapa con K-Means (k = número de regiones vistas)
pesos_som = som_ecom.get_weights().reshape(m1_ecom * m2_ecom, len(features_som))
km_som = KMeans(n_clusters=5, n_init=20, random_state=42)
cluster_neuronas = km_som.fit_predict(pesos_som)

# Asignar cluster a cada cliente según el cluster de su BMU
df_ecom_som['cluster_som'] = cluster_neuronas[bmu_idx]

# Comparar con los perfiles reales
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_real_enc = le.fit_transform(df_ecom_som['perfil_real'])

ari_som = adjusted_rand_score(y_real_enc, df_ecom_som['cluster_som'])
print(f"ARI entre SOM y perfiles reales: {ari_som:.3f}")

# Comparación con K-Means directo
km_directo = KMeans(n_clusters=5, n_init=20, random_state=42)
labels_km  = km_directo.fit_predict(X_som)
ari_km     = adjusted_rand_score(y_real_enc, labels_km)
print(f"ARI entre K-Means directo y perfiles reales: {ari_km:.3f}")

print(f"\n{'SOM supera K-Means' if ari_som > ari_km else 'K-Means supera SOM'} "
      f"en recuperación de estructura real.")
print("\nCruce SOM vs. perfil real:")
print(pd.crosstab(df_ecom_som['cluster_som'],
                  df_ecom_som['perfil_real'],
                  margins=True).to_string())
```

---

#### Celda 7 — Resumen y cierre del bloque

```python
print("=" * 60)
print("RESUMEN BLOQUE 2.2 — Mapas Auto-Organizados de Kohonen")
print("=" * 60)
print("""
Qué hace un SOM:
  → Aprende un mapa 2D que preserva la topología de los datos.
  → Cada neurona tiene pesos que representan un 'prototipo'.
  → Datos similares activan neuronas vecinas en el mapa.

Cómo interpretar el resultado:
  → U-Matrix:         fronteras oscuras = separación entre clusters.
  → Component planes: valor de cada variable por región del mapa.
  → Mapa de hits:     qué zonas concentran más datos.

Comparativa rápida:
  ┌─────────────────┬─────────────────┬──────────────────┐
  │                 │ K-Means         │ SOM              │
  ├─────────────────┼─────────────────┼──────────────────┤
  │ Salida          │ Etiquetas       │ Mapa + etiquetas │
  │ Visualización   │ Scatter (2D)    │ Mapa topológico  │
  │ Especifica k    │ Sí              │ No (emerge)      │
  │ Dim. reduction  │ No              │ Sí (implícita)   │
  │ Interpretación  │ Centroides      │ Component planes │
  └─────────────────┴─────────────────┴──────────────────┘

Cuándo usar SOM:
  ✓ Exploración de datos de alta dimensionalidad
  ✓ Necesitáis un mapa visual para presentar a negocio
  ✓ No sabéis cuántos clusters hay
  ✓ Queréis ver correlaciones entre variables en el mapa
""")
```

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada. Imagen de un mapa geográfico como analogía visual del SOM.
- **Slide 2:** Arquitectura — diagrama de dos capas: entrada y mapa 2D con conexiones.
- **Slide 3:** El proceso de aprendizaje en 3 pasos: (1) presentar dato, (2) encontrar BMU, (3) actualizar vecindad. Animación o secuencia de 3 imágenes.
- **Slide 4:** La función de vecindad gaussiana — gráfico del decaimiento con la distancia en el mapa.
- **Slide 5:** La U-Matrix — ejemplo con anotación de fronteras y clusters.
- **Slide 6:** Los component planes — los 4 paneles del caso de e-commerce.
- **Slide 7:** Tabla comparativa SOM vs. K-Means vs. GMM.

### Para el handout

- Diagrama de arquitectura del SOM anotado.
- Fórmulas: BMU, función de vecindad, actualización de pesos, decaimiento de α y σ.
- Guía de interpretación de la U-Matrix y los component planes.
- Tabla comparativa con los algoritmos anteriores.
- Regla empírica de tamaño del mapa: `m₁ × m₂ ≈ 5√n`.

### Para el Jupyter Notebook (ejercicios a completar)

**Ejercicio 1:** Variar el tamaño del mapa (5×5, 10×10, 15×15) manteniendo el mismo dataset. ¿Cómo cambia la U-Matrix? ¿Y el error de cuantización?

**Ejercicio 2:** Entrenar el SOM sin la inicialización PCA (`som.random_weights_init(X_som)`). Comparar la U-Matrix resultante con la versión inicializada por PCA. ¿Tarda más en converger?

**Ejercicio 3:** Superponer en el mapa de hits las etiquetas del cluster asignado por K-Means. ¿Las regiones del SOM coinciden con los clusters de K-Means? ¿Dónde difieren?

**Ejercicio 4 (avanzado):** Implementar una iteración del algoritmo de entrenamiento desde cero con NumPy: (1) calcular BMU, (2) calcular función de vecindad gaussiana, (3) actualizar pesos. Verificar que el vector de pesos de la BMU se acerca al dato de entrada.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador |
|---|---|---|
| Transición y contexto histórico | 4 min | Pregunta de enganche respondida |
| Arquitectura del SOM | 8 min | Diagrama de dos capas en pantalla |
| El algoritmo: BMU + vecindad + actualización | 10 min | Fórmulas de actualización en pantalla |
| U-Matrix y component planes | 6 min | Ejemplos visuales en pantalla |
| Ventajas, limitaciones, cuándo usar | 2 min | Tabla en pantalla |
| Celda 1-2 (instalación + convergencia) | 7 min | Gráfico de convergencia generado |
| Celda 3 (mapa de hits + proyección) | 6 min | Los 3 paneles generados |
| Celda 4-5 (caso e-commerce + component planes) | 12 min | U-Matrix con clientes proyectados |
| Celda 6-7 (etiquetas + comparación K-Means) | 5 min | ARI calculado |
| **Total** | **60 min** | |

---

*Bloque 2.2 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
