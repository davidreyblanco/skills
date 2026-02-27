# Bloque 1.2 — K-Means y K-Medoids
## Guión detallado del instructor

**Duración:** 110 minutos
- Teoría K-Means: 30 min
- Práctica K-Means: 35 min
- Teoría K-Medoids: 20 min
- Práctica K-Medoids: 25 min

**Posición en la sesión:** Segundo bloque de la Sesión 1, tras el Bloque 1.1

---

## PARTE A — K-MEANS

---

### TEORÍA K-MEANS (30 min)

---

### [00:00 – 00:05] Transición desde el Bloque 1.1 y motivación

> *Nota para el instructor: Recapitula brevemente el bloque anterior antes de arrancar. Un minuto de recapitulación entre bloques reduce la carga cognitiva acumulada.*

**Script de transición:**

*"En el bloque anterior vimos que el clustering consiste en encontrar estructura en datos sin etiquetar, y que la elección de la métrica de distancia es crítica. Ahora vamos a ver el primer y más utilizado algoritmo de clustering: K-Means. Si solo pudiérais aprender un algoritmo de clustering hoy, sería este. No porque sea el mejor en todos los casos —veremos que tiene limitaciones importantes— sino porque es el punto de referencia contra el que se compara todo lo demás y porque en la práctica, con datos bien preparados, funciona sorprendentemente bien."*

---

### [00:05 – 00:15] El algoritmo de Lloyd: mecánica paso a paso

**Concepto central:**

K-Means busca dividir `n` puntos en `k` grupos minimizando la **inercia**, también llamada WCSS (Within-Cluster Sum of Squares): la suma de las distancias al cuadrado de cada punto a su centroide más cercano.

```
WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ − μₖ||²
```

donde `μₖ` es el centroide (media aritmética) del cluster `k`.

**Los cuatro pasos del algoritmo de Lloyd:**

**Paso 0 — Inicialización:** Elige `k` centroides iniciales. (¿Cómo? Lo veremos a continuación.) En este paso se decide aleatoriamente de dónde parte la búsqueda.

**Paso 1 — Asignación:** Asigna cada punto al centroide más cercano según la distancia euclidiana.
```
Cₖ = { xᵢ : ||xᵢ − μₖ||² ≤ ||xᵢ − μⱼ||² para todo j ≠ k }
```
Cada punto pertenece exactamente a un cluster. Este es el **clustering duro** (hard assignment).

**Paso 2 — Actualización:** Recalcula cada centroide como la media de los puntos asignados a ese cluster.
```
μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
```
El centroide "se mueve" hacia donde están sus puntos.

**Paso 3 — Convergencia:** Repite Pasos 1 y 2 hasta que los centroides no se muevan (o se muevan menos que un umbral ε) o se alcance el número máximo de iteraciones.

**¿Por qué converge?** Porque cada iteración garantiza que la WCSS no aumenta: la asignación nunca empeora el coste para los puntos dados los centroides, y la actualización nunca empeora el coste para los centroides dada la asignación. La función de coste está acotada inferiormente por cero, y es un entero no negativo en su versión discreta. Por tanto, debe converger en un número finito de pasos.

**Complejidad computacional:** `O(n · k · d · i)` donde `n` = número de puntos, `k` = número de clusters, `d` = dimensiones, `i` = iteraciones. Escala linealmente con el número de puntos, lo que lo hace apto para datasets grandes.

**Analogía visual para explicar en clase:**

*"Imaginad que tenéis que colocar k banderas en un mapa de puntos. Primero ponéis las banderas al azar. Luego, cada punto 'decide' cuál es su bandera más cercana (Paso 1). Después, cada bandera se mueve al centro geográfico de sus puntos (Paso 2). Repetís hasta que las banderas se quedan quietas. Eso es exactamente K-Means."*

**Slide sugerida:** Animación o secuencia de 4-5 imágenes mostrando la evolución de los centroides en cada iteración sobre un dataset de blobs 2D.

---

### [00:15 – 00:20] El problema de la inicialización y K-Means++

**El problema:**

K-Means no garantiza encontrar el óptimo global de la WCSS. El resultado depende de dónde se colocan los centroides iniciales. Con inicialización aleatoria pura (algoritmo de Forgy: elegir k puntos al azar del dataset), es posible caer en mínimos locales malos.

**Demostración mental:**

*"Imaginad 4 clusters bien separados. Si por casualidad los 4 centroides iniciales caen todos dentro del mismo cluster real, el algoritmo va a hacer un trabajo terrible. Con datos de alta dimensión, esto ocurre más a menudo de lo que parece."*

**La solución: K-Means++** (Arthur & Vassilvitskii, 2007)

El algoritmo K-Means++ cambia solo la inicialización, manteniendo el resto igual. La idea es sembrar los centroides iniciales lo más separados posible entre sí:

1. Elige el primer centroide `μ₁` aleatoriamente de forma uniforme entre todos los puntos.
2. Para cada punto `xᵢ`, calcula `D(xᵢ)²`: su distancia al cuadrado al centroide más cercano ya elegido.
3. Elige el siguiente centroide con probabilidad proporcional a `D(xᵢ)²`. Los puntos más alejados de cualquier centroide existente tienen mayor probabilidad de ser elegidos.
4. Repite los pasos 2 y 3 hasta tener `k` centroides.
5. Continúa con el algoritmo de Lloyd estándar.

**Garantía teórica:** K-Means++ tiene una cota de aproximación esperada de `O(log k)` veces el óptimo global. En la práctica, da resultados significativamente mejores que la inicialización aleatoria pura y es el **default de scikit-learn** desde hace años (`init='k-means++'`).

**Script de cierre del subtema:**

*"En scikit-learn, K-Means++ está activado por defecto. No tenéis que hacer nada especial. Pero sí debéis saber que incluso con K-Means++, scikit-learn ejecuta el algoritmo `n_init=10` veces con distintas semillas y se queda con la que tiene la menor WCSS. Esto reduce enormemente el riesgo de mínimo local. En producción, con datasets grandes, a veces se reduce `n_init` a 3 por tiempo de cómputo."*

---

### [00:20 – 00:27] Cómo elegir K: el método del codo y sus límites

**El problema fundamental del clustering particional:**

K-Means requiere especificar `k` de antemano. En la práctica, rara vez sabemos cuántos grupos hay realmente. Necesitamos métodos para estimarlo.

**Método del codo (Elbow Method):**

Entrena K-Means para `k = 1, 2, 3, ..., 10` (o más) y registra la WCSS para cada valor de `k`. Al graficar WCSS vs. k se obtiene una curva decreciente. El "codo" —el punto donde la reducción marginal de WCSS empieza a ser pequeña— sugiere el `k` óptimo.

**Intuición matemática:** Con `k = n` (un cluster por punto), la WCSS es 0. Con `k = 1`, la WCSS es máxima. La curva siempre decrece, pero los primeros clusters añadidos reducen mucho la WCSS (capturan estructura real). A partir de cierto `k`, los clusters adicionales solo "parten" clusters ya compactos y la reducción es marginal.

**Limitación honesta que hay que comunicar a los alumnos:**

*"El método del codo tiene un problema: en datos reales, el codo muchas veces no es obvio. La curva puede ser suave sin un codo claro. En ese caso necesitáis combinar el codo con la métrica Silhouette (Bloque 2.3) y, sobre todo, con la lógica de negocio. Recordad que el objetivo no es encontrar el k 'matemáticamente correcto', sino el k que produzca segmentos accionables y comprensibles."*

**Regla empírica:** `k ≈ √(n/2)` es un punto de partida razonable para datasets sin información previa. Es solo un punto de partida, no un resultado.

---

### [00:27 – 00:30] Limitaciones de K-Means — Saber cuándo NO usarlo

*"K-Means es el algoritmo de referencia pero tiene limitaciones que todo profesional debe conocer. Si las ignoráis, obtendréis resultados que parecen correctos pero que son completamente erróneos."*

**Limitación 1 — Solo clusters convexos y esféricos:** K-Means asume implícitamente que los clusters son esféricos e isótropos (iguales en todas las direcciones). No puede descubrir clusters de forma arbitraria (lunas, anillos, formas irregulares). *Alternativa: DBSCAN, espectral, SOM.*

**Limitación 2 — Sensibilidad a outliers:** El centroide es la media aritmética. Un outlier extremo puede "tirar" del centroide hacia él, distorsionando el cluster. *Alternativa: K-Medoids (siguiente sección), que usa la mediana implícita.*

**Limitación 3 — Sensibilidad a la escala:** Ya lo vimos en el Bloque 1.1. Las variables con mayor rango dominan la distancia. *Solución: normalizar siempre.*

**Limitación 4 — Clusters de tamaño muy desigual:** K-Means tiende a crear clusters de tamaño similar. Si la realidad tiene un cluster con el 80% de los datos y otro con el 2%, K-Means lo hará mal. *Alternativa: GMM, DBSCAN.*

**Limitación 5 — Mínimos locales:** A pesar de K-Means++, no hay garantía del óptimo global. *Solución: múltiples re-inicializaciones (`n_init`).*

**Slide sugerida:** Cinco tarjetas visuales, una por limitación, con un icono de "advertencia" y la alternativa recomendada para cada caso.

---

## PRÁCTICA K-MEANS — Jupyter Notebook (35 min)

---

### [00:30 – 01:05] Práctica guiada

> *Nota para el instructor: Abre el notebook `sesion1_bloque2_kmeans_kmedoids.ipynb`. Este notebook es continuación del Bloque 1.1 en la misma sesión — los alumnos deben tenerlo abierto. Comenta en voz alta cada celda antes de ejecutarla.*

---

#### Celda 1 — Imports

```python
# ============================================================
# BLOQUE 1.2 — K-Means y K-Medoids
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
np.random.seed(42)

print("✓ Imports correctos")
```

---

#### Celda 2 — Implementación manual del algoritmo de Lloyd (paso a paso)

> *Nota: Esta celda tiene un propósito pedagógico: mostrar el algoritmo desde cero antes de usar scikit-learn. No es la implementación que usarían en producción.*

```python
def kmeans_manual(X, k, n_iter=10, seed=42):
    """
    Implementación didáctica del algoritmo de Lloyd (K-Means básico).
    NO usar en producción — usar sklearn.cluster.KMeans.
    """
    rng = np.random.RandomState(seed)

    # Paso 0: Inicialización aleatoria (Forgy)
    idx_init = rng.choice(len(X), size=k, replace=False)
    centroides = X[idx_init].copy()

    historial = [centroides.copy()]  # guardamos la evolución

    for iteracion in range(n_iter):
        # Paso 1: Asignación — cada punto al centroide más cercano
        distancias = np.linalg.norm(
            X[:, np.newaxis, :] - centroides[np.newaxis, :, :], axis=2
        )
        asignaciones = np.argmin(distancias, axis=1)

        # Paso 2: Actualización — recalcular centroides como media de sus puntos
        nuevos_centroides = np.array([
            X[asignaciones == j].mean(axis=0) if (asignaciones == j).any()
            else centroides[j]  # cluster vacío: mantener centroide
            for j in range(k)
        ])

        historial.append(nuevos_centroides.copy())

        # Paso 3: Convergencia
        if np.allclose(centroides, nuevos_centroides, atol=1e-6):
            print(f"  Convergencia alcanzada en iteración {iteracion + 1}")
            break

        centroides = nuevos_centroides

    wcss = sum(
        np.sum((X[asignaciones == j] - centroides[j]) ** 2)
        for j in range(k)
    )

    return asignaciones, centroides, wcss, historial


# --- Generamos datos y ejecutamos ---
X_demo, y_real = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)
X_demo_norm = StandardScaler().fit_transform(X_demo)

print("Ejecutando K-Means manual con k=3:")
labels, centroides_finales, wcss_final, historial = kmeans_manual(X_demo_norm, k=3)
print(f"  WCSS final: {wcss_final:.4f}")
print(f"  Puntos por cluster: {[np.sum(labels == j) for j in range(3)]}")
```

**Script de explicación:**

*"Fijaos en la función: son literalmente tres pasos dentro de un bucle. Paso 1 calcula qué centroide está más cerca de cada punto. Paso 2 mueve los centroides. El bucle para cuando los centroides ya no se mueven. Eso es todo K-Means. La magia y la limitación están en que el centroide es la media aritmética —eso es lo que vamos a cuestionar con K-Medoids."*

---

#### Celda 3 — Visualización de la evolución iterativa

```python
def plot_evolucion_kmeans(X, historial, labels_finales, k, max_iter_mostrar=5):
    """Muestra cómo evolucionan los centroides a lo largo de las iteraciones."""
    n_iter = min(len(historial), max_iter_mostrar)
    fig, axes = plt.subplots(1, n_iter, figsize=(4 * n_iter, 4))
    if n_iter == 1:
        axes = [axes]

    colores = plt.cm.tab10(np.linspace(0, 0.5, k))

    for idx, ax in enumerate(axes):
        if idx < len(historial) - 1:
            # Asignaciones provisionales para esta iteración
            dists = np.linalg.norm(
                X[:, np.newaxis, :] - historial[idx][np.newaxis, :, :], axis=2
            )
            labels_iter = np.argmin(dists, axis=1)
            titulo = f"Iteración {idx}" if idx > 0 else "Inicialización"
        else:
            labels_iter = labels_finales
            titulo = "Convergencia"

        ax.scatter(X[:, 0], X[:, 1], c=labels_iter, cmap='tab10',
                   alpha=0.5, s=20)
        centroides_iter = historial[idx]
        ax.scatter(centroides_iter[:, 0], centroides_iter[:, 1],
                   c='red', marker='X', s=200, zorder=5,
                   edgecolors='black', linewidths=1.5, label='Centroides')
        ax.set_title(titulo, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Evolución de K-Means: de inicialización a convergencia",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("img_evolucion_kmeans.png", dpi=150, bbox_inches='tight')
    plt.show()


plot_evolucion_kmeans(X_demo_norm, historial, labels, k=3)
```

**Script de explicación:**

*"Este es el gráfico más importante para entender K-Means intuitivamente. Las X rojas son los centroides. En la inicialización están en posiciones aleatorias. En la primera iteración, los puntos se asignan a su X más cercana y las X se mueven al centro de sus grupos. En pocas iteraciones el algoritmo converge. Guardad este gráfico mentalmente: cuando algo falla en K-Means, normalmente es porque los centroides iniciales estaban en una posición muy mala."*

---

#### Celda 4 — K-Means con scikit-learn + método del codo

```python
# ---- CASO PRÁCTICO: Dataset Mall Customers ----
# Segmentación de clientes por Annual Income y Spending Score
# Fuente: Kaggle (incluido en la carpeta datasets/)

# Para la demo en clase usamos datos sintéticos que replican la estructura
# del dataset original. En producción, cargar con pd.read_csv('mall_customers.csv')

np.random.seed(0)
n = 200
ingresos   = np.concatenate([
    np.random.normal(20,  5,  30),   # bajo ingreso, bajo gasto
    np.random.normal(20,  5,  30),   # bajo ingreso, alto gasto
    np.random.normal(55,  8,  40),   # ingreso medio
    np.random.normal(85,  7,  50),   # alto ingreso, bajo gasto
    np.random.normal(85,  7,  50),   # alto ingreso, alto gasto
])
gasto = np.concatenate([
    np.random.normal(20,  6,  30),
    np.random.normal(80,  6,  30),
    np.random.normal(50,  8,  40),
    np.random.normal(15,  6,  50),
    np.random.normal(82,  6,  50),
])
df_mall = pd.DataFrame({'Annual_Income_k': ingresos, 'Spending_Score': gasto})
df_mall = df_mall.clip(lower=0)  # sin negativos

print(f"Dataset: {df_mall.shape[0]} clientes, {df_mall.shape[1]} variables")
print(df_mall.describe().round(1))
```

---

#### Celda 5 — Método del codo

```python
# Normalización
scaler = StandardScaler()
X_mall = scaler.fit_transform(df_mall)

# Método del codo: WCSS para k = 1..10
wcss_lista = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_mall)
    wcss_lista.append(km.inertia_)

# Visualización
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_range, wcss_lista, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel("Número de clusters (k)", fontsize=12)
ax.set_ylabel("WCSS (Inercia)", fontsize=12)
ax.set_title("Método del Codo — Dataset Mall Customers", fontsize=13, fontweight='bold')
ax.set_xticks(k_range)

# Anotación manual del codo
ax.annotate('Codo ≈ k=5',
            xy=(5, wcss_lista[4]),
            xytext=(6.5, wcss_lista[4] + 0.3 * (wcss_lista[0] - wcss_lista[-1])),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig("img_codo_mall.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nReducción de WCSS al pasar de k=4 a k=5: "
      f"{wcss_lista[3]-wcss_lista[4]:.3f}")
print(f"Reducción de WCSS al pasar de k=5 a k=6: "
      f"{wcss_lista[4]-wcss_lista[5]:.3f}")
print("→ El salto es mayor en k=4→5, confirmando k=5 como codo.")
```

**Script de explicación del codo:**

*"La curva cae bruscamente de k=1 a k=5 y luego se aplana. Eso es el codo. Añadir un sexto cluster apenas reduce la WCSS porque ya no estamos capturando estructura real, solo partiendo clusters que ya eran buenos. Fijémonos en los valores numéricos: el salto de k=4 a k=5 es mayor que el de k=5 a k=6."*

---

#### Celda 6 — Resultado final de K-Means con k=5

```python
# Entrenamiento final
km_final = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
df_mall['Cluster'] = km_final.fit_predict(X_mall)

# Centroides en escala original
centroides_orig = scaler.inverse_transform(km_final.cluster_centers_)
df_centroides = pd.DataFrame(
    centroides_orig,
    columns=['Annual_Income_k', 'Spending_Score']
)
df_centroides.index.name = 'Cluster'

# Visualización
colores = ['#e41a1c','#377eb8','#4daf4a','#ff7f00','#984ea3']
fig, ax = plt.subplots(figsize=(10, 7))

for c in range(5):
    mask = df_mall['Cluster'] == c
    ax.scatter(
        df_mall.loc[mask, 'Annual_Income_k'],
        df_mall.loc[mask, 'Spending_Score'],
        color=colores[c], alpha=0.7, s=60, label=f'Cluster {c}'
    )

# Centroides
ax.scatter(
    df_centroides['Annual_Income_k'],
    df_centroides['Spending_Score'],
    c='black', marker='X', s=250, zorder=5, label='Centroides'
)

ax.set_xlabel("Ingresos anuales (k€)", fontsize=12)
ax.set_ylabel("Spending Score (0–100)", fontsize=12)
ax.set_title("K-Means k=5 — Segmentación de clientes Mall", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("img_kmeans_mall.png", dpi=150, bbox_inches='tight')
plt.show()

# Interpretación de negocio
print("\nPerfil de cada cluster (medias en escala original):")
print(df_mall.groupby('Cluster')[['Annual_Income_k','Spending_Score']].mean().round(1))
```

**Script de interpretación:**

*"Ahora viene la parte más importante: dar nombre a los clusters. El algoritmo no sabe nada de negocio —eso lo ponemos nosotros. Mirando los centroides podemos identificar cinco perfiles:"*
- *"Cluster con ingresos bajos y gasto bajo → 'Ahorradores con presupuesto ajustado'"*
- *"Cluster con ingresos bajos y gasto alto → 'Compradores impulsivos (alto riesgo de deuda)'"*
- *"Cluster con ingresos medios → 'Clientes estándar'"*
- *"Cluster con ingresos altos y gasto bajo → 'Ahorradores premium'"*
- *"Cluster con ingresos altos y gasto alto → 'VIPs — máxima prioridad de retención'"*

*"Esta interpretación es la entrega real. No un número, sino una narrativa de negocio."*

---

#### Celda 7 — Comparación K-Means vs K-Means++ (demo de inicialización)

```python
# ¿Cuánto importa la inicialización?

resultados = []

for metodo in ['random', 'k-means++']:
    wcss_runs = []
    for seed in range(20):
        km = KMeans(n_clusters=5, init=metodo, n_init=1, random_state=seed)
        km.fit(X_mall)
        wcss_runs.append(km.inertia_)
    resultados.append({
        'Método': metodo,
        'WCSS media': np.mean(wcss_runs),
        'WCSS std':   np.std(wcss_runs),
        'WCSS min':   np.min(wcss_runs),
        'WCSS max':   np.max(wcss_runs),
    })

df_res = pd.DataFrame(resultados).set_index('Método')
print("Comparación de inicialización (20 runs, n_init=1 cada una):")
print(df_res.round(4))

# Visualización como boxplot
fig, ax = plt.subplots(figsize=(8, 5))
data_random   = [KMeans(n_clusters=5, init='random',    n_init=1, random_state=s).fit(X_mall).inertia_ for s in range(30)]
data_kpp      = [KMeans(n_clusters=5, init='k-means++', n_init=1, random_state=s).fit(X_mall).inertia_ for s in range(30)]
ax.boxplot([data_random, data_kpp],
           labels=['Inicialización\naleatoria', 'K-Means++'],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
ax.set_ylabel("WCSS (Inercia)", fontsize=12)
ax.set_title("Variabilidad de WCSS según método de inicialización\n(30 runs, n_init=1)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("img_kmeans_vs_kpp.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación:**

*"Con inicialización aleatoria, el WCSS varía mucho entre runs: a veces encontramos una buena solución, a veces una mala. Con K-Means++ la varianza es mucho menor y el mínimo es mejor. En scikit-learn, el parámetro `n_init=10` ya ejecuta esto automáticamente y se queda con el mejor resultado — por eso es el default."*

---

## PARTE B — K-MEDOIDS

---

### TEORÍA K-MEDOIDS (20 min)

---

### [01:05 – 01:10] Motivación: ¿Por qué K-Medoids?

**Script de transición:**

*"K-Means tiene una vulnerabilidad fundamental: el centroide es la media aritmética de los puntos del cluster. Esto tiene un problema grave: la media puede ser un punto que no existe en el dataset. Y si hay outliers, la media se 'contamina'."*

**Ejemplo numérico inmediato:**

*"Imaginad un cluster con cuatro clientes con ingresos de 20k, 22k, 21k y 85k euros. La media es (20+22+21+85)/4 = 37k. Ese centroide de 37k no representa bien a nadie del cluster: los tres primeros tienen ~21k y el cuarto es un outlier en 85k. El 'representante' del cluster no es un cliente real."*

*"K-Medoids soluciona esto de forma elegante: en lugar de la media, usa el **medoide** — el punto real del dataset que minimiza la distancia media a todos los demás puntos de su cluster. El representante siempre es un cliente que existe."*

**Ventajas inmediatas:**
1. **Robustez ante outliers:** el medoide no puede ser 'tirado' hacia un outlier porque es un punto real del dataset.
2. **Interpretabilidad:** cada cluster está representado por un caso real. Puedes decir: *"Este segmento se parece al cliente #1234"*.
3. **Funciona con cualquier métrica de distancia:** no requiere que la media tenga sentido. Funciona con distancias no-euclidianas, datos mixtos o incluso distancias entre strings (edit distance).

---

### [01:10 – 01:20] El algoritmo PAM (Partitioning Around Medoids)

**Desarrollado por Kaufman & Rousseeuw (1990). Es el algoritmo de K-Medoids más conocido y sigue siendo la referencia.**

**Notación:** Dado un conjunto `S` de `n` puntos y una función de distancia `d(i,j)`, PAM busca un conjunto `M` de `k` medoides tal que el coste total sea mínimo:

```
COSTE = Σᵢ∉M  min_{m∈M} d(i, m)
```

*(La suma de distancias de cada punto no-medoide al medoide más cercano.)*

**Fase BUILD — Inicialización inteligente:**

A diferencia de K-Means que inicializa aleatoriamente, PAM tiene una fase de inicialización determinista:

1. Elige el primer medoide `m₁`: el punto que minimiza la suma de distancias a todos los demás (el punto más "central" del dataset completo).
2. Para cada punto candidato `x` a ser el segundo medoide, calcula cuánto reduciría el coste total añadirlo. Elige el que más reduce el coste.
3. Repite hasta tener `k` medoides.

**Fase SWAP — Optimización iterativa:**

Una vez inicializados los `k` medoides, PAM intenta mejoras sistemáticas:

1. Para cada par `(mᵢ, xⱼ)` donde `mᵢ` es un medoide actual y `xⱼ` es un punto no-medoide:
   - Calcula el coste del swap: ¿cuánto cambiaría el coste total si `xⱼ` reemplazara a `mᵢ`?
2. Si existe algún swap que reduce el coste, realiza el que más lo reduce.
3. Repite hasta que no haya swaps beneficiosos.

**Complejidad computacional:**

- Fase BUILD: `O(k · n²)`
- Fase SWAP por iteración: `O(k · (n-k)²)`. Cada iteración evalúa `k × (n-k)` swaps posibles, y cada evaluación cuesta `O(n-k)`.
- Para `n` grande, PAM es significativamente más lento que K-Means. Para `n < 5.000` es perfectamente viable.

**Variantes para datasets grandes:**

| Variante | Idea | Complejidad | Cuándo usar |
|---|---|---|---|
| PAM | Exacto, todos los swaps | O(k·(n-k)²) | n < 5.000 |
| CLARA | Muestrea subconjuntos, aplica PAM a cada uno | O(k·s²) | n ~ 10⁴–10⁵ |
| CLARANS | Búsqueda aleatoria de vecinos en el espacio de soluciones | O(n²) | n ~ 10⁵ |

**scikit-learn-extra implementa los tres.** El parámetro `method` de `KMedoids` acepta `'pam'`, `'alternate'` (variante más rápida) y `'fastpam1'`.

---

### [01:20 – 01:25] K-Means vs. K-Medoids: cuándo elegir cada uno

**Tabla comparativa definitiva:**

| Criterio | K-Means | K-Medoids |
|---|---|---|
| Representante del cluster | Media (puede no existir) | Punto real del dataset |
| Robustez ante outliers | Baja | **Alta** |
| Métrica de distancia | Solo euclidiana (nativa) | **Cualquier métrica** |
| Velocidad (n grande) | **Muy rápido** O(n·k·d·i) | Más lento O(k·(n-k)²) |
| Interpretabilidad | Media | **Alta** — caso real |
| Datos mixtos / categóricos | No nativo | **Sí**, con la métrica adecuada |
| Default scikit | `sklearn.cluster.KMeans` | `sklearn_extra.cluster.KMedoids` |

**Regla práctica para elegir:**

*"Usad K-Means cuando tengáis muchos datos, las variables sean numéricas y continuas, y no haya muchos outliers. Usad K-Medoids cuando los outliers sean un problema, cuando necesitéis que cada cluster esté representado por un caso real (útil para presentaciones a negocio), o cuando estéis trabajando con distancias que no son euclidianas —por ejemplo, distancias entre perfiles de comportamiento discreto, o datos que incluyen variables categóricas."*

---

## PRÁCTICA K-MEDOIDS — Jupyter Notebook (25 min)

---

### [01:25 – 01:50] Práctica guiada

---

#### Celda 8 — Instalación y verificación de scikit-learn-extra

```python
# scikit-learn-extra no viene con scikit-learn estándar
# Instalar con: pip install scikit-learn-extra

try:
    from sklearn_extra.cluster import KMedoids
    print("✓ scikit-learn-extra disponible")
except ImportError:
    print("✗ Instalando scikit-learn-extra...")
    import subprocess
    subprocess.run(["pip", "install", "scikit-learn-extra", "-q"])
    from sklearn_extra.cluster import KMedoids
    print("✓ scikit-learn-extra instalado y cargado")
```

---

#### Celda 9 — Demostración del impacto de outliers: K-Means vs. K-Medoids

```python
# -------------------------------------------------------
# EXPERIMENTO: ¿Cómo afectan los outliers a K-Means
# pero no a K-Medoids?
# -------------------------------------------------------

from sklearn_extra.cluster import KMedoids

# Dataset base: 3 clusters bien separados
np.random.seed(42)
X_base, _ = make_blobs(n_samples=120, centers=[[-3, 0], [0, 0], [3, 0]],
                        cluster_std=0.6, random_state=42)

# Añadimos 5 outliers extremos artificiales
outliers = np.array([
    [-3, 8], [-3, 9],   # outliers sobre el cluster izquierdo
    [3,  -8], [3, -9],  # outliers sobre el cluster derecho
    [0,  10]            # outlier sobre el cluster central
])

X_con_outliers = np.vstack([X_base, outliers])

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_con_outliers)
n_outliers = len(outliers)

# Entrenamos ambos algoritmos con k=3
km  = KMeans(n_clusters=3, n_init=10, random_state=42)
kmd = KMedoids(n_clusters=3, method='pam', random_state=42)

labels_km  = km.fit_predict(X_scaled)
labels_kmd = kmd.fit_predict(X_scaled)

# Obtenemos centroides/medoides en escala original
centroides_km  = scaler.inverse_transform(km.cluster_centers_)
medoides_kmd   = scaler.inverse_transform(kmd.cluster_centers_)

# ---- Visualización comparativa ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, labels, representantes, titulo, color_rep in zip(
    axes,
    [labels_km, labels_kmd],
    [centroides_km, medoides_kmd],
    ["K-Means (sensible a outliers)", "K-Medoids (robusto a outliers)"],
    ['red', 'green']
):
    # Puntos normales
    scatter = ax.scatter(
        X_con_outliers[:-n_outliers, 0],
        X_con_outliers[:-n_outliers, 1],
        c=labels[:-n_outliers], cmap='tab10', alpha=0.7, s=40
    )
    # Outliers marcados con estrella
    ax.scatter(
        outliers[:, 0], outliers[:, 1],
        c='black', marker='*', s=250, zorder=5, label='Outliers'
    )
    # Representantes
    ax.scatter(
        representantes[:, 0], representantes[:, 1],
        c=color_rep, marker='X', s=300, zorder=6,
        edgecolors='black', linewidths=1.5,
        label='Centroides' if color_rep=='red' else 'Medoides'
    )
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")

plt.suptitle("Impacto de outliers en K-Means vs. K-Medoids",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_kmeans_vs_kmedoids_outliers.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación — este es el momento clave del bloque:**

*"Fijaos en las X rojas (K-Means) y en las X verdes (K-Medoids). Los clusters reales son tres grupos horizontales. Los outliers son las estrellas negras arriba y abajo."*

*"En K-Means, los centroides rojos están desplazados hacia los outliers porque la media se contamina. El cluster izquierdo tiene su centroide 'subido' hacia los outliers de arriba. En K-Medoids, los medoides verdes están en el centro real de cada cluster porque son puntos reales del dataset — los outliers no pueden moverlos."*

*"En un proyecto real, esto significa que con K-Means vuestro segmento 'cliente típico' podría estar representado por un perfil que no existe, distorsionado por cuatro transacciones fraudulentas o por cuatro clientes VIP extremos."*

---

#### Celda 10 — Cuantificación del desplazamiento de representantes

```python
# Medimos cuánto se desplazan los representantes respecto al centro real

# Calculamos los centros "reales" (sin outliers) para comparar
km_sin_outliers  = KMeans(n_clusters=3, n_init=10, random_state=42)
km_sin_outliers.fit(scaler.transform(X_base))
centros_reales = scaler.inverse_transform(km_sin_outliers.cluster_centers_)

# Ordenamos clusters por coordenada X para comparar correctamente
def ordenar_clusters(centers):
    return centers[np.argsort(centers[:, 0])]

reales = ordenar_clusters(centros_reales)
km_c   = ordenar_clusters(centroides_km)
kmd_c  = ordenar_clusters(medoides_kmd)

print("Distancia de cada representante al centro real del cluster:")
print("-" * 55)
for i, (r, km_ci, kmd_ci) in enumerate(zip(reales, km_c, kmd_c)):
    d_km  = np.linalg.norm(km_ci - r)
    d_kmd = np.linalg.norm(kmd_ci - r)
    print(f"Cluster {i+1}:  K-Means desplazado {d_km:.3f} unidades  |"
          f"  K-Medoids desplazado {d_kmd:.3f} unidades")

print("\n→ K-Medoids mantiene sus representantes mucho más cerca del centro real.")
```

---

#### Celda 11 — K-Medoids aplicado al dataset Mall Customers

```python
# Comparación directa sobre el mismo dataset de negocio

kmd_mall = KMedoids(n_clusters=5, method='pam', random_state=42)
df_mall['Cluster_KMedoids'] = kmd_mall.fit_predict(X_mall)

medoides_orig = scaler.inverse_transform(kmd_mall.cluster_centers_)

# Visualización lado a lado
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, col_cluster, representantes, titulo, marker_color in zip(
    axes,
    ['Cluster', 'Cluster_KMedoids'],
    [centroides_orig, medoides_orig],
    ['K-Means k=5', 'K-Medoids k=5'],
    ['red', 'green']
):
    for c in range(5):
        mask = df_mall[col_cluster] == c
        ax.scatter(
            df_mall.loc[mask, 'Annual_Income_k'],
            df_mall.loc[mask, 'Spending_Score'],
            alpha=0.6, s=50
        )
    ax.scatter(
        representantes[:, 0], representantes[:, 1],
        c=marker_color, marker='X', s=250, zorder=5,
        edgecolors='black', linewidths=1.5
    )
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel("Ingresos anuales (k€)")
    ax.set_ylabel("Spending Score")

plt.suptitle("Mall Customers — Comparación K-Means vs K-Medoids",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_mall_kmeans_vs_kmedoids.png", dpi=150, bbox_inches='tight')
plt.show()

# Mostrar los medoides como filas reales del dataset
print("\nMediantas: los 5 clientes 'representativos' según K-Medoids:")
df_medoides = df_mall.iloc[kmd_mall.medoid_indices_][
    ['Annual_Income_k', 'Spending_Score']
].copy()
df_medoides.index = [f'Medoide Cluster {i}' for i in range(5)]
print(df_medoides.round(1))
print("\n→ Estos son clientes reales del dataset. Existen.")
```

**Script de explicación:**

*"Aquí está la gran diferencia práctica: los medoides son filas reales de vuestro dataset. Si vais a presentar los resultados al equipo de marketing, podéis decir: 'Este segmento se parece al cliente 47, que compra así y gasta así'. Con K-Means, el centroide es un cliente imaginario que quizás no existe en vuestros sistemas."*

---

#### Celda 12 — Mini-ejercicio: ¿Cuándo escalar importa para K-Medoids?

```python
# Ejercicio guiado: K-Medoids SIN normalizar vs. CON normalizar
# (mismo punto que con K-Means pero importante repetirlo)

kmd_sin_norm = KMedoids(n_clusters=5, method='pam', random_state=42)
kmd_con_norm = KMedoids(n_clusters=5, method='pam', random_state=42)

labels_sin = kmd_sin_norm.fit_predict(df_mall[['Annual_Income_k','Spending_Score']].values)
labels_con = kmd_con_norm.fit_predict(X_mall)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, labels, titulo in zip(
    axes,
    [labels_sin, labels_con],
    ['K-Medoids SIN normalizar', 'K-Medoids CON normalizar']
):
    ax.scatter(df_mall['Annual_Income_k'], df_mall['Spending_Score'],
               c=labels, cmap='tab10', alpha=0.7, s=50)
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xlabel("Ingresos anuales (k€)")
    ax.set_ylabel("Spending Score")

plt.suptitle("Impacto de la normalización en K-Medoids",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("Conclusión: K-Medoids también requiere normalización.")
print("La escala afecta a las distancias, independientemente del algoritmo.")
```

---

#### Celda 13 — Resumen comparativo del bloque

```python
print("=" * 60)
print("RESUMEN BLOQUE 1.2 — K-Means y K-Medoids")
print("=" * 60)

resumen = {
    "K-Means":   {"Velocidad": "★★★★★", "Robustez outliers": "★★☆☆☆",
                  "Interpretabilidad": "★★★☆☆", "Métricas flexibles": "★★☆☆☆"},
    "K-Medoids": {"Velocidad": "★★★☆☆", "Robustez outliers": "★★★★★",
                  "Interpretabilidad": "★★★★★", "Métricas flexibles": "★★★★★"},
}

df_resumen = pd.DataFrame(resumen).T
print(df_resumen.to_string())

print("""
Cuándo usar K-Means:
  ✓ Dataset grande (n > 50.000)
  ✓ Variables numéricas continuas bien escaladas
  ✓ No hay outliers extremos
  ✓ Velocidad es prioritaria

Cuándo usar K-Medoids:
  ✓ Outliers presentes o sospechados
  ✓ Necesitas representantes reales (presentaciones, CRM)
  ✓ Distancias no-euclidianas (datos mixtos, texto, etc.)
  ✓ Dataset pequeño-mediano (n < 10.000 con PAM)
""")
```

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada K-Means. Animación de los 4 pasos de Lloyd.
- **Slide 2:** Los 4 pasos del algoritmo con pseudocódigo y fórmulas.
- **Slide 3:** K-Means++ — diagrama mostrando la probabilidad proporcional a D(x)².
- **Slide 4:** Método del codo — gráfico de WCSS con anotación del codo.
- **Slide 5:** Las 5 limitaciones de K-Means — tarjetas de advertencia.
- **Slide 6:** Portada K-Medoids. El ejemplo de ingresos con la media contaminada.
- **Slide 7:** Algoritmo PAM — fases BUILD y SWAP con diagrama de flujo.
- **Slide 8:** Tabla comparativa K-Means vs. K-Medoids.
- **Slide 9:** Resultado visual del experimento de outliers (los dos scatter plots lado a lado).

### Para el handout

- Tabla comparativa K-Means vs. K-Medoids (criterios de selección).
- Pseudocódigo de Lloyd (4 pasos) y pseudocódigo de PAM (BUILD + SWAP).
- Tabla variantes de K-Medoids (PAM, CLARA, CLARANS).
- Los gráficos: evolución de centroides, método del codo, comparación con outliers.
- Checklist de decisión: *¿Hay outliers? → K-Medoids. ¿n > 50k? → K-Means. ¿Necesito representantes reales? → K-Medoids.*

### Para el Jupyter Notebook (ejercicios a completar por los alumnos)

**Ejercicio 1 (Celda 9 ampliada):** Repetir el experimento de outliers variando el número de outliers (0, 2, 5, 10). ¿A partir de cuántos outliers empieza K-Means a dar resultados claramente peores?

**Ejercicio 2 (Celda 5 ampliada):** Añadir la curva de Silhouette Score al gráfico del codo. ¿El k óptimo según Silhouette coincide con el del codo? (Anticipación al Bloque 2.3.)

**Ejercicio 3 (Celda 11 ampliada):** Usar `method='alternate'` en lugar de `'pam'` para K-Medoids. Comparar los medoides resultantes y el tiempo de ejecución con `%%time`.

**Ejercicio 4 (avanzado):** Implementar CLARA manualmente: (1) tomar 5 muestras aleatorias del 20% del dataset, (2) aplicar PAM a cada muestra, (3) asignar todos los puntos al medoide más cercano de la mejor solución, (4) comparar con PAM sobre el dataset completo.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador de progreso |
|---|---|---|
| Transición desde Bloque 1.1 | 5 min | Pregunta de conexión respondida |
| Algoritmo de Lloyd (4 pasos) | 10 min | Diagrama en pantalla |
| Inicialización y K-Means++ | 5 min | Gráfico de variabilidad explicado |
| Método del codo + limitaciones | 10 min | Tabla de limitaciones en pantalla |
| Práctica Celdas 1-3 (manual + evolución) | 10 min | Gráfico de evolución generado |
| Práctica Celdas 4-7 (Mall + codo + comparativa) | 25 min | Segmentación final interpretada |
| **Pausa de 5 min** (si el ritmo lo permite) | 5 min | — |
| Motivación K-Medoids + ejemplo numérico | 5 min | Pregunta retórica planteada |
| Algoritmo PAM (BUILD + SWAP) | 10 min | Tabla PAM/CLARA/CLARANS en pantalla |
| Tabla comparativa K-Means vs. K-Medoids | 5 min | Tabla en pantalla |
| Práctica Celdas 8-13 (outliers + Mall + resumen) | 25 min | Gráfico comparativo generado |
| **Total** | **115 min** *(~5 min de margen sobre los 110)* | |

> *Nota: Si el grupo va lento en la práctica de K-Medoids, omitir la Celda 12 (impacto de normalización) y remitir al ejercicio como tarea.*

---

*Bloque 1.2 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
