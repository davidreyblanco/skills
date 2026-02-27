# Bloque 2.4 — Reducción de Dimensionalidad para Clustering
**Sesión 2 | Duración: 35 minutos**

---

## Tabla de tiempos

| Segmento | Contenido | Tiempo |
|---|---|---|
| Teoría 1 | El problema: alta dimensionalidad | 8 min |
| Teoría 2 | PCA como preprocesado de clustering | 7 min |
| Práctica 1 | PCA + K-Means pipeline | 8 min |
| Teoría 3 | t-SNE y UMAP para visualización | 5 min |
| Práctica 2 | Visualización comparativa | 7 min |
| **Total** | | **35 min** |

---

## Teoría 1 — El problema de la alta dimensionalidad (8 min)

### Guión del instructor

> "Hasta ahora hemos trabajado con datos de 2 a 6 dimensiones. En la realidad, es común encontrar datasets con decenas, cientos o miles de características. ¿Qué le ocurre al clustering cuando la dimensionalidad escala?"

**La maldición de la dimensionalidad (revisión)**

Recordad del Bloque 1.1 que en dimensiones altas:
- Las distancias euclidianas se vuelven casi iguales entre todos los puntos
- El volumen se concentra en los vértices del hipercubo, no en el interior
- Los vecinos cercanos dejan de ser significativamente más cercanos que los lejanos

**Consecuencias concretas para clustering:**

| Algoritmo | Problema en alta dim. |
|---|---|
| K-Means / K-Medoids | Distancias euclidianas pierden contraste → centroides mal definidos |
| DBSCAN | Imposible calibrar ε: todas las densidades locales colapsan |
| Jerárquico | Dendrogramas espurios, cortes sin sentido real |
| GMM | Matrices de covarianza sobredimensionadas → overfitting severo |
| SOM | El mapa 2D no puede capturar la estructura topológica real |

**Dos usos distintos de la reducción de dimensionalidad:**

```
1. PREPROCESADO → reducir dimensiones ANTES de clustering
   Objetivo: mejorar la calidad intrínseca del clustering
   Método recomendado: PCA (lineal, reversible, rápido)

2. VISUALIZACIÓN → proyectar resultados YA obtenidos a 2D/3D
   Objetivo: inspeccionar y comunicar el clustering
   Métodos recomendados: t-SNE, UMAP (no lineales, no reversibles)
```

> "Estas son dos tareas completamente distintas. No usamos t-SNE para preprocesar clustering porque distorsiona las distancias globales. No usamos PCA para visualización final porque puede ocultar estructuras no lineales. El error de confundirlos es frecuente."

---

## Teoría 2 — PCA como preprocesado (7 min)

### Guión del instructor

**¿Qué hace PCA?**

> "PCA busca las direcciones de máxima varianza en los datos. La primera componente principal captura la mayor varianza posible; la segunda, la mayor varianza ortogonal a la primera; y así sucesivamente."

**Matemáticamente:**
- Descomposición en valores propios de la matriz de covarianza: `Σ = VΛVᵀ`
- Las columnas de V son los vectores propios (direcciones principales)
- Los valores de Λ son las varianzas explicadas por cada componente
- Proyección: `Z = X_centrado · V[:, :k]`

**Criterios para elegir k (número de componentes):**

1. **Varianza explicada acumulada**: retener suficientes componentes para explicar 80-95% de la varianza total
2. **Codo en la curva de varianza**: punto donde el beneficio marginal cae bruscamente
3. **Regla de Kaiser**: retener componentes con valor propio > 1 (solo para datos estandarizados)

**Ventajas para clustering:**
- Elimina ruido: las últimas componentes suelen capturar ruido, no estructura
- Elimina multicolinealidad: las componentes son ortogonales por construcción
- Reduce coste computacional: menos dimensiones = clustering más rápido
- Mejora la geometría de distancias: varianza concentrada en pocas dimensiones

**Limitación importante:**
> "PCA es lineal. Si la estructura del dato es intrínsecamente no lineal (como un toroide o una espiral), PCA puede perder información estructural relevante. Para esos casos, existen métodos como Kernel PCA, pero quedan fuera de este módulo."

---

## Práctica 1 — PCA + K-Means pipeline (8 min)

### Celda 1 — Imports y datos de alta dimensionalidad

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Generamos datos con estructura real en pocas dimensiones, embebidos en alta dim.
n_samples = 400
n_features_real = 3   # estructura real en 3 dimensiones
n_features_total = 50  # datos "observados" en 50 dimensiones

# Datos base: 4 clusters en 3D
X_real, y_true = make_blobs(n_samples=n_samples, n_features=n_features_real,
                             centers=4, cluster_std=0.8, random_state=42)

# Embebemos en 50D: proyección aleatoria + ruido
proyeccion = np.random.randn(n_features_real, n_features_total)
X_alto = X_real @ proyeccion + np.random.randn(n_samples, n_features_total) * 2.0

scaler = StandardScaler()
X_alto_scaled = scaler.fit_transform(X_alto)

print(f"Shape datos originales (alta dim.): {X_alto_scaled.shape}")
print(f"Dimensiones reales con estructura: {n_features_real}")
```

### Celda 2 — Comparación: clustering directo vs. con PCA

```python
from sklearn.metrics import adjusted_rand_score

resultados = []

# 1. K-Means directo en 50D
km_alto = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_alto = km_alto.fit_predict(X_alto_scaled)
sil_alto = silhouette_score(X_alto_scaled, labels_alto)
ari_alto = adjusted_rand_score(y_true, labels_alto)
resultados.append({'Método': 'K-Means 50D (sin PCA)', 'Silhouette': sil_alto, 'ARI': ari_alto})

# 2. PCA + K-Means con distintos k de componentes
for n_comp in [2, 3, 5, 10, 20]:
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_alto_scaled)
    var_exp = pca.explained_variance_ratio_.sum()

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    ari = adjusted_rand_score(y_true, labels)
    resultados.append({
        'Método': f'PCA({n_comp}) + K-Means',
        'Silhouette': sil,
        'ARI': ari,
        'Var. explicada': f'{var_exp:.1%}'
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
```

### Celda 3 — Curva de varianza explicada y elección de k

```python
pca_full = PCA(random_state=42)
pca_full.fit(X_alto_scaled)

var_acum = np.cumsum(pca_full.explained_variance_ratio_)
var_ind = pca_full.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Varianza individual
ax1 = axes[0]
ax1.bar(range(1, 21), var_ind[:20] * 100, color='steelblue', alpha=0.7, label='Varianza individual')
ax1.set_xlabel('Componente principal')
ax1.set_ylabel('% Varianza explicada')
ax1.set_title('Scree Plot (primeras 20 componentes)')
ax1.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k real = 3')
ax1.legend()

# Varianza acumulada
ax2 = axes[1]
ax2.plot(range(1, len(var_acum) + 1), var_acum * 100, 'o-', color='steelblue',
         markersize=4, linewidth=2)
ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% varianza')
ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% varianza')

# Marcar los umbrales
k_80 = np.argmax(var_acum >= 0.80) + 1
k_95 = np.argmax(var_acum >= 0.95) + 1
ax2.axvline(x=k_80, color='orange', linestyle=':', alpha=0.7)
ax2.axvline(x=k_95, color='red', linestyle=':', alpha=0.7)
ax2.annotate(f'k={k_80}', xy=(k_80, 80), xytext=(k_80+1, 75),
             fontsize=9, color='orange')
ax2.annotate(f'k={k_95}', xy=(k_95, 95), xytext=(k_95+1, 90),
             fontsize=9, color='red')

ax2.set_xlabel('Número de componentes')
ax2.set_ylabel('% Varianza acumulada')
ax2.set_title('Varianza acumulada explicada')
ax2.legend()
ax2.set_xlim(1, 30)

plt.tight_layout()
plt.suptitle('Análisis de componentes principales — Selección de k óptimo',
             y=1.02, fontsize=13, fontweight='bold')
plt.show()

print(f"\nComponentes para 80% de varianza: {k_80}")
print(f"Componentes para 95% de varianza: {k_95}")
```

---

## Teoría 3 — t-SNE y UMAP para visualización (5 min)

### Guión del instructor

> "Una vez hemos ejecutado el clustering, queremos *ver* los resultados. Proyectar a 2D con PCA a veces es suficiente, pero si la estructura es no lineal, podemos usar t-SNE o UMAP."

**t-SNE (t-distributed Stochastic Neighbor Embedding)**

- Preserva similitudes locales: puntos cercanos en el espacio original → cercanos en 2D
- No preserva distancias globales (la distancia entre clusters en 2D no es interpretable)
- Parámetro clave: **perplexity** (rango típico: 5–50)
  - Bajo: estructura muy local, clusters fragmentados
  - Alto: estructura más global, clusters pueden comprimirse
- **Computacionalmente costoso**: O(n² log n). Para n > 10.000, usar PCA primero
- No determinista por defecto (usar `random_state`)
- **No sirve para preprocesar clustering**: las distorsiones no lineales lo hacen inadecuado

**UMAP (Uniform Manifold Approximation and Projection)**

- Más rápido que t-SNE para datasets grandes
- Mejor preservación de la estructura global (grupos de clusters y sus relaciones)
- Parámetros clave:
  - `n_neighbors` (5–50): balance local/global, análogo a perplexity
  - `min_dist` (0.0–0.99): compactación de los clusters en la proyección
- También puede usarse como reducción de dimensionalidad (no solo a 2D)
- Más reproducible que t-SNE con `random_state`

**Regla práctica:**

```
¿Para qué?          → Usar
Preprocesar clustering  → PCA
Visualizar resultados   → t-SNE o UMAP
Explorar estructura     → UMAP (más rápido, más fiel a escala global)
```

---

## Práctica 2 — Visualización comparativa PCA / t-SNE / UMAP (7 min)

### Celda 4 — Proyecciones comparativas con labels de clustering

```python
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_DISPONIBLE = True
except ImportError:
    UMAP_DISPONIBLE = False
    print("UMAP no disponible. Instalar con: pip install umap-learn")

# Usamos PCA a 3 componentes para clustering
pca3 = PCA(n_components=3, random_state=42)
X_pca3 = pca3.fit_transform(X_alto_scaled)

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_final = km_final.fit_predict(X_pca3)

colores = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
color_map = [colores[l] for l in labels_final]

# Número de proyecciones disponibles
n_plots = 3 if UMAP_DISPONIBLE else 2
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

# --- PCA 2D ---
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_alto_scaled)

axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1], c=color_map, alpha=0.6, s=30)
axes[0].set_title(f'PCA (var. explicada: {pca2.explained_variance_ratio_.sum():.1%})',
                  fontsize=11)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# --- t-SNE ---
print("Calculando t-SNE... (puede tardar ~10s)")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_alto_scaled)

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_map, alpha=0.6, s=30)
axes[1].set_title('t-SNE (perplexity=30)', fontsize=11)
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')

# --- UMAP (si disponible) ---
if UMAP_DISPONIBLE:
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=42)
    X_umap = reducer.fit_transform(X_alto_scaled)

    axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=color_map, alpha=0.6, s=30)
    axes[2].set_title('UMAP (n_neighbors=15, min_dist=0.1)', fontsize=11)
    axes[2].set_xlabel('Dim 1')
    axes[2].set_ylabel('Dim 2')

# Leyenda de clusters
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colores[k], markersize=10,
                          label=f'Cluster {k}')
                  for k in range(4)]
axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8)

plt.suptitle('Comparación de proyecciones 2D con labels de K-Means\n(datos 50D → 2D)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Celda 5 — Efecto de perplexity en t-SNE (exploración pedagógica)

```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
perplexities = [5, 15, 30, 50]

for ax, perp in zip(axes, perplexities):
    tsne_p = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=500)
    X_p = tsne_p.fit_transform(X_pca3)  # usamos PCA3 para acelerar

    ax.scatter(X_p[:, 0], X_p[:, 1], c=color_map, alpha=0.5, s=20)
    ax.set_title(f'perplexity = {perp}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Efecto del parámetro perplexity en t-SNE\n'
             '(los clusters son reales — la distorsión es del método)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n⚠️  ADVERTENCIA PEDAGÓGICA:")
print("Las distancias ENTRE clusters en t-SNE NO son interpretables.")
print("La distancia DENTRO de un cluster tampoco es directamente comparable entre clusters.")
print("t-SNE solo preserva relaciones de vecindad LOCAL.")
```

### Celda 6 — Pipeline completo recomendado

```python
# ============================================================
# PIPELINE RECOMENDADO: PCA → Clustering → t-SNE/UMAP para viz
# ============================================================

from sklearn.pipeline import Pipeline

# Paso 1: Estandarizar
scaler_pipe = StandardScaler()
X_pipe = scaler_pipe.fit_transform(X_alto)

# Paso 2: PCA para preprocesar clustering
pca_pipe = PCA(n_components=0.90, random_state=42)  # retener 90% varianza
X_reduced = pca_pipe.fit_transform(X_pipe)
n_comp_seleccionados = pca_pipe.n_components_
var_ret = pca_pipe.explained_variance_ratio_.sum()
print(f"PCA: {X_pipe.shape[1]}D → {n_comp_seleccionados}D "
      f"(varianza retenida: {var_ret:.1%})")

# Paso 3: Clustering en espacio reducido
km_pipe = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_pipe = km_pipe.fit_predict(X_reduced)
sil_pipe = silhouette_score(X_reduced, labels_pipe)
ari_pipe = adjusted_rand_score(y_true, labels_pipe)
print(f"K-Means: Silhouette = {sil_pipe:.3f} | ARI = {ari_pipe:.3f}")

# Paso 4: t-SNE solo para visualización del resultado
tsne_viz = TSNE(n_components=2, perplexity=30, random_state=42)
X_viz = tsne_viz.fit_transform(X_pipe)  # proyectamos los datos originales

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=labels_pipe,
                      cmap='Set1', alpha=0.7, s=40)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Pipeline completo: PCA({n_comp_seleccionados}D) → K-Means → t-SNE viz\n'
          f'ARI={ari_pipe:.3f} | Silhouette={sil_pipe:.3f}',
          fontsize=11)
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')
plt.tight_layout()
plt.show()

print("\n📌 RESUMEN DEL PIPELINE:")
print(f"   Datos originales:  {X_alto.shape}")
print(f"   Tras PCA (90% var): {X_reduced.shape}")
print(f"   Clustering: K-Means k=4")
print(f"   Visualización: t-SNE 2D")
print(f"   ⚠️  t-SNE se usa SOLO para visualizar, no afecta al clustering")
```

---

## Notas de producción

### Para la presentación (slides)
- **Slide 1**: "Dos usos ≠ de la reducción de dim." — tabla con Preprocesado vs. Visualización
- **Slide 2**: Scree plot animado — mostrar codo y umbral 80%/95%
- **Slide 3**: Las 3 proyecciones lado a lado (PCA / t-SNE / UMAP) — hacer énfasis visual en la diferencia de separación
- **Slide 4**: Advertencia de t-SNE — mostrar cómo perplexity cambia la apariencia sin cambiar los datos
- **Slide 5**: Diagrama del pipeline completo (cajas: Raw → Scale → PCA → Cluster → t-SNE → Viz)

### Para el handout
- Tabla comparativa PCA / t-SNE / UMAP (objetivo, preserva, coste, parámetro clave, cuándo usar)
- Fórmula PCA: `Z = X_centrado · V[:, :k]`, diagrama de eigenvectores
- Criterios de selección de k: varianza acumulada, Scree plot, Kaiser
- Advertencia t-SNE: caja negra con ⚠️ "Las distancias entre clusters en la proyección NO son interpretables"
- Pipeline recomendado como diagrama de flujo

### Ejercicios propuestos (notebook)
1. En el dataset de clientes (mall/e-commerce de sesiones anteriores), aplicar PCA antes de K-Means. ¿Cuántas componentes retener? ¿Mejora el Silhouette score?
2. Proyectar los resultados del SOM del Bloque 2.2 con t-SNE. ¿Son más visibles los perfiles?
3. *(Avanzado)* Comparar `n_neighbors=5` vs. `n_neighbors=30` en UMAP con el dataset actual. ¿Qué información pierde cada uno?

---

## Tabla de tiempos (verificación)

| Segmento | Duración real |
|---|---|
| Teoría 1: alta dimensionalidad | 8 min |
| Teoría 2: PCA preprocesado | 7 min |
| Práctica 1: PCA pipeline (3 celdas) | 8 min |
| Teoría 3: t-SNE y UMAP | 5 min |
| Práctica 2: comparativa viz (3 celdas) | 7 min |
| **Total** | **35 min** ✓ |
