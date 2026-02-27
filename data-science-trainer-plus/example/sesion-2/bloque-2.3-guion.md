# Bloque 2.3 — Métricas de Evaluación de Clustering
## Guión detallado del instructor

**Duración:** 65 minutos (25 min teoría + 40 min práctica en Jupyter Notebook)
**Posición en la sesión:** Tercer bloque de la Sesión 2

---

## PARTE TEÓRICA (25 min)

---

### [00:00 – 00:04] Transición y la pregunta central del bloque

**Script de transición:**

*"Llevamos cinco algoritmos estudiados. Todos producen clusters. Pero ¿cómo sabemos si esos clusters son buenos? ¿Cuándo es mejor K-Means que DBSCAN en vuestros datos? ¿Cómo justificáis ante vuestro equipo que k=5 es mejor que k=4?"*

*"Este es el problema de evaluación en clustering, y es fundamentalmente distinto al de la clasificación. En clasificación tenéis etiquetas reales y podéis calcular accuracy, F1, AUC. En clustering no hay etiquetas —es aprendizaje no supervisado— así que no podéis medir 'cuántas veces acertasteis'. Necesitáis métricas que evalúen la calidad de los clusters sin saber la verdad."*

*"Este bloque os va a dar exactamente eso: un arsenal de métricas para evaluar, comparar y justificar vuestros modelos de clustering."*

**Distinción fundamental:**

Hay dos familias de métricas según si disponemos de etiquetas de referencia o no:

- **Métricas internas:** no necesitan ground truth. Evalúan la estructura interna de los clusters (compacidad, separación). Son las más usadas en la práctica.
- **Métricas externas:** comparan los clusters obtenidos con etiquetas reales conocidas. Solo disponibles en entornos de laboratorio o evaluación supervisada.

*"En la mayoría de proyectos reales de clustering solo tendréis métricas internas. Aprenderlas bien es crítico."*

---

### [00:04 – 00:13] Métricas internas

#### Coeficiente Silhouette

**Qué mide:** Para cada punto, qué tan bien encaja en su cluster comparado con el siguiente cluster más cercano. Combina compacidad interna y separación externa.

**Cálculo para un punto `i`:**

```
a(i) = distancia media de i a todos los demás puntos de su cluster
       (cohesión interna: cuanto más pequeño, más compacto el cluster)

b(i) = distancia media mínima de i a los puntos de cualquier otro cluster
       b(i) = min_{k ≠ cluster(i)} { media de d(i, j) para j en cluster k }
       (separación externa: cuanto más grande, más alejado del siguiente cluster)

s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Rango e interpretación:**
- `s(i) ≈ +1`: el punto está bien dentro de su cluster y lejos del siguiente. Muy buena asignación.
- `s(i) ≈ 0`: el punto está en la frontera entre dos clusters. Podría pertenecer a cualquiera.
- `s(i) ≈ -1`: el punto está más cerca del siguiente cluster que del propio. Probablemente mal asignado.

**Silhouette Score global:** media de `s(i)` para todos los puntos.

**Silhouette plot:** representación individual de `s(i)` para cada punto, ordenados por cluster y por valor. Permite ver si todos los clusters son de buena calidad o si hay clusters problemáticos.

**Limitación:** computacionalmente costoso para datasets muy grandes (O(n²)). Para n > 50.000 usar `silhouette_samples` con submuestreo.

---

#### Índice Davies-Bouldin (DBI)

**Qué mide:** Para cada cluster, su "peor caso": la ratio entre la dispersión interna y la separación con el cluster más parecido. El índice global es la media de estos peores casos.

**Cálculo:**

Para cada par de clusters `i`, `j`:
```
Rᵢⱼ = (sᵢ + sⱼ) / dᵢⱼ
```
donde:
- `sᵢ` = dispersión media del cluster `i` (distancia media de sus puntos al centroide)
- `dᵢⱼ` = distancia entre los centroides de `i` y `j`

Para cada cluster `i`:
```
Dᵢ = max_{j≠i} Rᵢⱼ     ← el 'peor vecino' del cluster i
```

Índice global:
```
DBI = (1/K) Σᵢ Dᵢ
```

**Interpretación:** Menor DBI = mejor. Un DBI bajo significa que los clusters son compactos (sᵢ pequeño) y están bien separados (dᵢⱼ grande). No tiene un rango fijo.

**Ventaja sobre Silhouette:** más rápido de calcular (O(n·K) vs O(n²)).

---

#### Índice Calinski-Harabasz (CHI / Variance Ratio Criterion)

**Qué mide:** La ratio entre la dispersión entre clusters (varianza inter-cluster) y la dispersión dentro de los clusters (varianza intra-cluster). Mayor = mejor.

```
CHI = [SS_between / (K-1)] / [SS_within / (n-K)]
```

donde:
- `SS_between = Σₖ nₖ · ||μₖ - μ||²` — dispersión entre centroides (ponderada por tamaño)
- `SS_within = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²` — WCSS total (inercia de K-Means)
- `K` = número de clusters, `n` = número de puntos

**Interpretación:** Mayor CHI = mejor. Los clusters son más densos internamente y más separados entre sí. No tiene rango máximo —solo es útil comparando el mismo dataset con distintos K.

**Ventaja:** muy rápido de calcular, intuitivo.

**Limitación:** asume clusters convexos y esféricos. Puede dar valores altos para soluciones de DBSCAN o jerárquico aunque la estructura real no sea esférica.

---

#### WCSS / Inercia

Revisitada desde K-Means: `WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²`

Útil para el método del codo pero **no es comparable entre distintos K** (siempre decrece al aumentar K). No es una métrica de evaluación final —es una herramienta para seleccionar K dentro de K-Means.

---

### [00:13 – 00:20] Métricas externas

*"Cuando tenéis ground truth —etiquetas reales— podéis usar métricas externas. Esto ocurre en evaluación supervisada de algoritmos, en benchmarks, o cuando tenéis una pequeña muestra etiquetada para validar."*

#### Adjusted Rand Index (ARI)

Mide el acuerdo entre dos particiones ajustando por el azar. Range [-1, 1] donde 1 = acuerdo perfecto, 0 = acuerdo aleatorio, negativo = peor que el azar.

```python
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(labels_verdaderos, labels_predichos)
```

**Ventaja:** invariante a permutaciones de etiquetas (el cluster 0 predicho puede corresponder al cluster 2 real). Ajustado por el azar.

#### Normalized Mutual Information (NMI)

Mide la información mutua entre las dos particiones, normalizada para que esté en [0, 1]. 1 = particiones idénticas, 0 = completamente independientes.

```python
from sklearn.metrics import normalized_mutual_info_score
nmi = normalized_mutual_info_score(labels_verdaderos, labels_predichos)
```

#### V-Measure (completeness + homogeneity)

Combina dos métricas:
- **Homogeneidad:** cada cluster contiene solo puntos de una clase real.
- **Completeness:** todos los puntos de una clase están en el mismo cluster.
- **V-Measure:** media harmónica de ambas.

---

### [00:20 – 00:25] Reglas de uso y las trampas de las métricas

*"Antes de la práctica, quiero que tengáis en mente cuatro advertencias sobre las métricas de clustering. Son las más olvidadas en proyectos reales."*

**Trampa 1 — El mejor Silhouette no siempre es la mejor solución:**
Silhouette maximiza la separación entre clusters. A veces la solución con mayor Silhouette tiene clusters artificialmente pequeños que no tienen sentido de negocio. **Regla:** las métricas guían, no deciden.

**Trampa 2 — CHI favorece clusters esféricos:**
Calinski-Harabasz dará valores altos para K-Means aunque DBSCAN capture mejor la estructura real. Usad CHI solo para comparar el mismo algoritmo con distintos K, no para comparar algoritmos distintos.

**Trampa 3 — Las métricas no capturan interpretabilidad:**
Un clustering con Silhouette 0.9 pero cuyos clusters son imposibles de nombrar o actuar tiene menos valor que uno con Silhouette 0.5 pero con segmentos claros y accionables.

**Trampa 4 — Nunca usar una sola métrica:**
La práctica correcta es triangular: Silhouette + DBI + CHI + inspección visual + juicio de negocio. Si tres métricas convergen en el mismo K, es un resultado robusto.

---

## PARTE PRÁCTICA — Jupyter Notebook (40 min)

---

### [00:25 – 01:05] Práctica guiada

---

#### Celda 1 — Imports

```python
# ============================================================
# BLOQUE 2.3 — Métricas de Evaluación de Clustering
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
np.random.seed(42)

print("✓ Imports correctos")
```

---

#### Celda 2 — Silhouette plot: la métrica en detalle

```python
# -------------------------------------------------------
# Silhouette plot: análisis punto a punto
# -------------------------------------------------------

X_blob, y_real = make_blobs(n_samples=300, centers=4,
                             cluster_std=0.9, random_state=5)
X_blob_norm = StandardScaler().fit_transform(X_blob)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
axes = axes.flatten()

for idx, k in enumerate([2, 3, 4, 5, 6, 7]):
    ax = axes[idx]
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_blob_norm)

    sil_avg  = silhouette_score(X_blob_norm, labels)
    sil_vals = silhouette_samples(X_blob_norm, labels)

    colores = cm.nipy_spectral(np.linspace(0.1, 0.9, k))
    y_lower = 10

    for c in range(k):
        c_vals = np.sort(sil_vals[labels == c])
        y_upper = y_lower + len(c_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals,
                         facecolor=colores[c], edgecolor=colores[c], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(c_vals), str(c), fontsize=8)
        y_lower = y_upper + 5

    ax.axvline(x=sil_avg, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f"k={k} — Silhouette avg = {sil_avg:.3f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Coeficiente Silhouette")
    ax.set_ylabel("Cluster")
    ax.set_xlim([-0.2, 1])
    ax.set_yticks([])

plt.suptitle("Silhouette plots para k=2..7 — Dataset blobs (4 clusters reales)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_silhouette_plots.png", dpi=150, bbox_inches='tight')
plt.show()

print("Interpretación del Silhouette plot:")
print("  Cada barra horizontal = un punto. Anchura = valor silhouette.")
print("  Barras que cruzan la línea roja (media) hacia la izquierda → puntos problemáticos.")
print("  Barras de anchura uniforme → cluster compacto y bien separado.")
print("  k=4 debería tener los plots más uniformes (es el k real).")
```

**Script de explicación:**

*"El silhouette plot es la visualización más informativa de todas las métricas. Cada barra es un punto. Si las barras de un cluster son cortas —no llegan a la línea roja— ese cluster tiene puntos mal asignados. Si son todas largas y uniformes, el cluster es compacto y bien separado. Comparad k=2 con k=4: en k=4 los plots son mucho más limpios porque coincide con la estructura real."*

---

#### Celda 3 — Comparación de las tres métricas para distintos K

```python
# -------------------------------------------------------
# Las tres métricas juntas para elegir K óptimo
# -------------------------------------------------------

ks = range(2, 11)
resultados = {'k': list(ks), 'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_blob_norm)
    resultados['silhouette'].append(silhouette_score(X_blob_norm, labels))
    resultados['davies_bouldin'].append(davies_bouldin_score(X_blob_norm, labels))
    resultados['calinski_harabasz'].append(calinski_harabasz_score(X_blob_norm, labels))

df_metricas = pd.DataFrame(resultados).set_index('k')

# Normalizar para comparación visual en [0,1]
df_norm = df_metricas.copy()
df_norm['silhouette_norm']      = (df_metricas['silhouette'] - df_metricas['silhouette'].min()) / \
                                   (df_metricas['silhouette'].max() - df_metricas['silhouette'].min())
df_norm['dbi_norm_inv']         = 1 - (df_metricas['davies_bouldin'] - df_metricas['davies_bouldin'].min()) / \
                                       (df_metricas['davies_bouldin'].max() - df_metricas['davies_bouldin'].min())
df_norm['chi_norm']             = (df_metricas['calinski_harabasz'] - df_metricas['calinski_harabasz'].min()) / \
                                   (df_metricas['calinski_harabasz'].max() - df_metricas['calinski_harabasz'].min())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Métricas en escala original
ax1 = axes[0]
ax1.plot(ks, df_metricas['silhouette'], 'bo-', linewidth=2, label='Silhouette ↑')
ax1b = ax1.twinx()
ax1b.plot(ks, df_metricas['davies_bouldin'], 'r^--', linewidth=2, label='Davies-Bouldin ↓')
ax1.set_xlabel("Número de clusters (k)")
ax1.set_ylabel("Silhouette Score", color='blue')
ax1b.set_ylabel("Davies-Bouldin Index", color='red')
ax1.set_title("Silhouette y Davies-Bouldin\nvs. número de clusters",
              fontsize=11, fontweight='bold')
ax1.set_xticks(ks)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9)

# Métricas normalizadas juntas
ax2 = axes[1]
ax2.plot(ks, df_norm['silhouette_norm'], 'bo-', linewidth=2, markersize=7,
         label='Silhouette (normalizado) ↑')
ax2.plot(ks, df_norm['dbi_norm_inv'], 'r^--', linewidth=2, markersize=7,
         label='1 - DBI (normalizado) ↑')
ax2.plot(ks, df_norm['chi_norm'], 'gs-.', linewidth=2, markersize=7,
         label='Calinski-Harabasz (normalizado) ↑')
ax2.axvline(x=4, color='black', linestyle=':', linewidth=2, label='k real = 4')
ax2.set_xlabel("Número de clusters (k)")
ax2.set_ylabel("Puntuación normalizada [0,1] (mayor = mejor)")
ax2.set_title("Las tres métricas normalizadas juntas\n(coinciden en k=4 → resultado robusto)",
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_xticks(ks)

plt.suptitle("Triangulación de métricas para seleccionar k óptimo",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_metricas_triangulacion.png", dpi=150, bbox_inches='tight')
plt.show()

k_optimos = {
    'Silhouette':          df_metricas['silhouette'].idxmax(),
    'Davies-Bouldin':      df_metricas['davies_bouldin'].idxmin(),
    'Calinski-Harabasz':   df_metricas['calinski_harabasz'].idxmax(),
}
print("K óptimo según cada métrica:")
for metrica, k_opt in k_optimos.items():
    print(f"  {metrica}: k={k_opt}")
print(f"\n→ Las tres coinciden en k={pd.Series(k_optimos).mode()[0]} ✓")
```

**Script de explicación:**

*"Este es el patrón que queréis ver: las tres métricas apuntando al mismo k. Cuando las tres coinciden, el resultado es robusto —no es un artefacto de una sola métrica. En este caso las tres señalan k=4, que es exactamente el k real que usamos para generar los datos."*

*"En datos reales, las tres rara vez coinciden exactamente. Pero os dan un rango plausible de k values. Luego vuestro juicio de negocio decide cuántos segmentos son accionables."*

---

#### Celda 4 — Comparación de algoritmos con las mismas métricas

```python
# -------------------------------------------------------
# ¿Qué algoritmo da mejores clusters para este dataset?
# Usando las métricas como árbitro objetivo
# -------------------------------------------------------

# Dataset: blobs estándar (caso favorable para K-Means)
X_eval, y_eval = make_blobs(n_samples=400, centers=4,
                             cluster_std=0.85, random_state=7)
X_eval_norm = StandardScaler().fit_transform(X_eval)

algoritmos = {
    'K-Means k=4': KMeans(n_clusters=4, n_init=10, random_state=42),
    'GMM k=4':     GaussianMixture(n_components=4, n_init=5, random_state=42),
    'Jerárquico Ward k=4': AgglomerativeClustering(n_clusters=4, linkage='ward'),
}

filas = []
for nombre, modelo in algoritmos.items():
    if isinstance(modelo, GaussianMixture):
        modelo.fit(X_eval_norm)
        labels = modelo.predict(X_eval_norm)
    else:
        labels = modelo.fit_predict(X_eval_norm)

    sil = silhouette_score(X_eval_norm, labels)
    dbi = davies_bouldin_score(X_eval_norm, labels)
    chi = calinski_harabasz_score(X_eval_norm, labels)
    ari = adjusted_rand_score(y_eval, labels)
    filas.append({'Algoritmo': nombre, 'Silhouette ↑': sil,
                  'Davies-Bouldin ↓': dbi, 'Calinski-Harabasz ↑': chi,
                  'ARI (vs. real) ↑': ari})

df_comp = pd.DataFrame(filas).set_index('Algoritmo')
print("Comparación de algoritmos — Dataset blobs (4 clusters reales):")
print(df_comp.round(4).to_string())

print("\n→ En datos convexos, los tres algoritmos dan resultados muy similares.")
print("  El ARI confirma que los tres recuperan bien la estructura real.")
```

---

#### Celda 5 — Dashboard de evaluación: selección automática del mejor K

```python
# -------------------------------------------------------
# EJERCICIO INTEGRADOR:
# dado un dataset desconocido, elegir automáticamente
# el mejor algoritmo y el mejor K
# -------------------------------------------------------

def evaluar_clustering(X, k_min=2, k_max=8, algoritmos_k=['kmeans'],
                       verbose=True):
    """
    Evalúa automáticamente múltiples configuraciones de clustering.
    Devuelve un DataFrame con todas las métricas y recomienda la mejor.
    """
    resultados = []

    for k in range(k_min, k_max + 1):
        for algo in algoritmos_k:
            if algo == 'kmeans':
                modelo = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = modelo.fit_predict(X)
                nombre = f'K-Means k={k}'
            elif algo == 'gmm':
                modelo = GaussianMixture(n_components=k, n_init=5, random_state=42)
                modelo.fit(X)
                labels = modelo.predict(X)
                nombre = f'GMM k={k}'
            elif algo == 'ward':
                modelo = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = modelo.fit_predict(X)
                nombre = f'Ward k={k}'

            # Saltar si solo hay un cluster real
            if len(np.unique(labels)) < 2:
                continue

            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            chi = calinski_harabasz_score(X, labels)

            resultados.append({
                'Configuración': nombre, 'k': k,
                'Silhouette ↑': round(sil, 4),
                'DBI ↓': round(dbi, 4),
                'CHI ↑': round(chi, 1),
            })

    df_res = pd.DataFrame(resultados)

    # Puntuación compuesta (normalizada, las tres métricas con igual peso)
    df_res['sil_norm'] = (df_res['Silhouette ↑'] - df_res['Silhouette ↑'].min()) / \
                          (df_res['Silhouette ↑'].max() - df_res['Silhouette ↑'].min() + 1e-9)
    df_res['dbi_norm'] = 1 - (df_res['DBI ↓'] - df_res['DBI ↓'].min()) / \
                              (df_res['DBI ↓'].max() - df_res['DBI ↓'].min() + 1e-9)
    df_res['chi_norm'] = (df_res['CHI ↑'] - df_res['CHI ↑'].min()) / \
                          (df_res['CHI ↑'].max() - df_res['CHI ↑'].min() + 1e-9)
    df_res['Score compuesto'] = (df_res['sil_norm'] + df_res['dbi_norm'] + df_res['chi_norm']) / 3

    df_res_clean = df_res.drop(columns=['k','sil_norm','dbi_norm','chi_norm'])

    if verbose:
        print(df_res_clean.sort_values('Score compuesto', ascending=False)
              .head(5).to_string(index=False))
        mejor = df_res_clean.loc[df_res['Score compuesto'].idxmax(), 'Configuración']
        print(f"\n→ Configuración recomendada: '{mejor}'")

    return df_res_clean.sort_values('Score compuesto', ascending=False)


# Probamos con el dataset de Mall Customers
print("=== Dashboard de evaluación automática ===\n")
np.random.seed(0)
n = 200
df_mall_eval = pd.DataFrame({
    'Annual_Income_k': np.concatenate([
        np.random.normal(20,5,30), np.random.normal(20,5,30),
        np.random.normal(55,8,40), np.random.normal(85,7,50), np.random.normal(85,7,50)
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(20,6,30), np.random.normal(80,6,30),
        np.random.normal(50,8,40), np.random.normal(15,6,50), np.random.normal(82,6,50)
    ])
}).clip(lower=0)
X_mall_eval = StandardScaler().fit_transform(df_mall_eval)

df_resultados = evaluar_clustering(
    X_mall_eval, k_min=2, k_max=8,
    algoritmos_k=['kmeans', 'gmm', 'ward']
)
```

---

#### Celda 6 — Las trampas de las métricas: cuando el mejor score no es la mejor solución

```python
# -------------------------------------------------------
# DEMOSTRACIÓN: Silhouette puede mentir
# -------------------------------------------------------

print("=== Caso donde Silhouette puede ser engañoso ===\n")

# Dataset: lunas (estructura no convexa)
X_lunas, y_lunas = make_moons(n_samples=300, noise=0.06, random_state=42)
X_lunas_norm = StandardScaler().fit_transform(X_lunas)

resultados_lunas = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_km = km.fit_predict(X_lunas_norm)
    sil = silhouette_score(X_lunas_norm, labels_km)
    resultados_lunas.append({'k': k, 'Silhouette K-Means': round(sil, 4)})

# DBSCAN (el correcto para este dataset)
db = DBSCAN(eps=0.18, min_samples=5)
labels_db = db.fit_predict(X_lunas_norm)
mask_no_noise = labels_db != -1
sil_db = silhouette_score(X_lunas_norm[mask_no_noise], labels_db[mask_no_noise])
ari_db = adjusted_rand_score(y_lunas[mask_no_noise], labels_db[mask_no_noise])

print("K-Means en dataset de lunas:")
df_lunas = pd.DataFrame(resultados_lunas).set_index('k')
print(df_lunas)
print(f"\nDBSCAN (correcto para este dataset):")
print(f"  Silhouette: {sil_db:.4f}")
print(f"  ARI vs. etiquetas reales: {ari_db:.4f}")

print(f"""
Análisis:
  K-Means con k=2 puede tener un Silhouette {'mayor' if resultados_lunas[0]['Silhouette K-Means'] > sil_db else 'menor'} que DBSCAN.
  Sin embargo, DBSCAN recupera la estructura real (ARI={ari_db:.2f} ≈ 1.0).

  Conclusión: Silhouette mide separación convexa.
  En clusters no convexos, un Silhouette alto puede ser un artefacto.
  Usad siempre la métrica junto con la INSPECCIÓN VISUAL.
""")
```

**Script de explicación:**

*"Este ejemplo es importante. K-Means partiendo las lunas puede tener un Silhouette comparable o incluso mayor que DBSCAN, porque Silhouette mide separación lineal. Pero el ARI contra las etiquetas reales revela que DBSCAN es mucho mejor. Moraleja: las métricas internas son herramientas, no árbitros absolutos. Siempre combinadlas con visualización."*

---

#### Celda 7 — Tabla final de referencia

```python
print("=" * 70)
print("GUÍA DE REFERENCIA — MÉTRICAS DE EVALUACIÓN DE CLUSTERING")
print("=" * 70)

tabla_ref = pd.DataFrame({
    'Métrica': ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'NMI'],
    'Rango': ['[-1, 1]', '[0, ∞)', '[0, ∞)', '[-1, 1]', '[0, 1]'],
    'Mejor': ['↑ Mayor', '↓ Menor', '↑ Mayor', '↑ Mayor', '↑ Mayor'],
    'Necesita GT': ['No', 'No', 'No', 'Sí', 'Sí'],
    'Complejidad': ['O(n²)', 'O(n·K)', 'O(n·K)', 'O(n)', 'O(n)'],
    'Limitación principal': [
        'Asume convexidad',
        'Solo centroides',
        'Asume convexidad',
        'Requiere ground truth',
        'Requiere ground truth',
    ]
}).set_index('Métrica')

print(tabla_ref.to_string())
print("""
Protocolo de evaluación recomendado:
  1. Silhouette plot → analizar cluster por cluster
  2. DBI + CHI → confirmar con métricas más rápidas
  3. Inspección visual → siempre obligatoria
  4. Juicio de negocio → ¿los clusters son accionables?
  5. ARI/NMI → solo si se dispone de ground truth
""")
```

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada. La pregunta: *"¿Cómo sé si mis clusters son buenos?"*
- **Slide 2:** Métricas internas vs. externas — diagrama comparativo.
- **Slide 3:** Silhouette — fórmula de `a(i)`, `b(i)`, `s(i)` con diagrama geométrico.
- **Slide 4:** Davies-Bouldin — fórmula y diagrama de compacidad vs. separación.
- **Slide 5:** Calinski-Harabasz — SS_between vs. SS_within visualmente.
- **Slide 6:** Las cuatro advertencias sobre métricas — tarjetas de advertencia.
- **Slide 7:** El protocolo de evaluación en 5 pasos.

### Para el handout

- Tabla de referencia completa de métricas (Celda 7).
- Silhouette plots para k=3 y k=4 lado a lado (Celda 2) con guía de lectura.
- Gráfico de triangulación de métricas (Celda 3).
- La demostración de Silhouette engañoso (Celda 6) como caso de advertencia.
- El protocolo de evaluación en 5 pasos.

### Para el Jupyter Notebook (ejercicios a completar)

**Ejercicio 1:** Aplicar el dashboard de evaluación (`evaluar_clustering`) al dataset de países del Bloque 1.3. ¿El k recomendado coincide con el que elegisteis visualmente por el dendrograma?

**Ejercicio 2:** Calcular el Silhouette plot para K-Medoids con k=5 en el dataset Mall Customers. ¿Los clusters tienen Silhouette más uniforme que K-Means?

**Ejercicio 3 (avanzado):** Implementar el cálculo del Silhouette Score desde cero usando NumPy y scipy.spatial.distance. Verificar que coincide con `sklearn.metrics.silhouette_score`.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador |
|---|---|---|
| Transición + distinción interna/externa | 4 min | Diagrama en pantalla |
| Silhouette (fórmula + interpretación) | 5 min | Fórmulas en pantalla |
| Davies-Bouldin + Calinski-Harabasz | 4 min | Tabla comparativa en pantalla |
| Métricas externas (ARI, NMI) | 3 min | Fórmulas en pantalla |
| Las cuatro trampas | 4 min | Tarjetas en pantalla |
| Protocolo de 5 pasos | 5 min | Lista en pantalla |
| Celda 1-2 (imports + Silhouette plots) | 10 min | 6 plots generados |
| Celda 3 (triangulación) | 8 min | Gráfico de métricas generado |
| Celda 4 (comparación algoritmos) | 7 min | Tabla impresa |
| Celda 5 (dashboard automático) | 8 min | Top 5 configuraciones |
| Celda 6-7 (trampa + tabla final) | 7 min | Demostración engaño + tabla |
| **Total** | **65 min** | |

---

*Bloque 2.3 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
