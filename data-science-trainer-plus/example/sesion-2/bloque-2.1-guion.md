# Bloque 2.1 — Gaussian Mixture Models (GMM) y el Algoritmo EM
## Guión detallado del instructor

**Duración:** 70 minutos (35 min teoría + 35 min práctica en Jupyter Notebook)
**Posición en la sesión:** Primer bloque de la Sesión 2

---

## PARTE TEÓRICA (35 min)

---

### [00:00 – 00:06] Apertura de la Sesión 2 y transición

**Script de apertura:**

*"Bienvenidos a la Sesión 2. En la sesión anterior construimos la base: K-Means, K-Medoids, clustering jerárquico y DBSCAN. Hoy vamos a subir un nivel. Los algoritmos de ayer producen lo que se llama clustering duro: cada punto pertenece exactamente a un cluster con certeza absoluta. El mundo real rara vez funciona así."*

*"Pensad en un cliente de vuestra empresa. ¿Pertenece al segmento 'cazador de ofertas' o al segmento 'comprador de conveniencia'? Probablemente a los dos, en proporciones distintas según el momento. O pensad en una transacción financiera: ¿es normal o fraudulenta? A veces no hay una respuesta binaria —hay un grado de sospecha. Los Gaussian Mixture Models son la respuesta matemática a esta necesidad: asignan probabilidades de pertenencia, no etiquetas."*

**Recapitulación rápida de la Sesión 1 (2 min):**

*"Antes de arrancar: ¿alguna pregunta de la sesión anterior? ¿Algo que no quedó claro con K-Means, K-Medoids, el dendrograma o DBSCAN?"*

---

### [00:06 – 00:14] Limitaciones del clustering duro y la necesidad de probabilidades

**El problema del clustering duro:**

K-Means asigna cada punto `xᵢ` al cluster `k` según:
```
zᵢ = argmin_k ||xᵢ - μₖ||²
```
Es una asignación binaria: `zᵢ ∈ {1, 2, ..., K}`. No hay matices.

**Situaciones donde esto falla:**

1. **Puntos en la frontera entre clusters:** Un punto equidistante de dos centroides recibe la misma etiqueta definitiva que un punto en el centro de un cluster. No hay forma de saber que es un caso ambiguo.

2. **Clusters elípticos o con correlación:** K-Means asume que todos los clusters tienen la misma forma esférica (misma covarianza). Si los datos tienen clusters elongados en distintas direcciones, K-Means los parte incorrectamente.

3. **Clusters con tamaños muy distintos:** K-Means tiende a producir clusters del mismo tamaño aunque la realidad sea que el 80% de los puntos pertenece a un solo grupo.

**Ejemplo visual que funciona muy bien en clase:**

*"Imaginad que medís la altura y el peso de una población. Hay claramente dos grupos: hombres y mujeres. Pero los grupos no son esféricos —hay correlación positiva entre altura y peso, y los dos grupos tienen formas elípticas distintas. K-Means intentará dividir esto con círculos y fallará en la zona de superposición. Un GMM pondrá una elipse diferente sobre cada grupo y asignará probabilidades en la zona de solapamiento."*

---

### [00:14 – 00:22] El modelo: mezcla de distribuciones gaussianas

**Definición formal:**

Un Gaussian Mixture Model asume que los datos han sido generados por `K` distribuciones gaussianas multivariantes. La densidad de probabilidad del modelo completo es:

```
p(x) = Σₖ₌₁ᴷ  πₖ · 𝒩(x | μₖ, Σₖ)
```

donde:
- `πₖ` es el **peso** del componente `k`: la probabilidad a priori de que un punto aleatorio pertenezca al componente `k`. Cumple `Σπₖ = 1`, `πₖ ≥ 0`.
- `𝒩(x | μₖ, Σₖ)` es la densidad de una gaussiana multivariante con **media** `μₖ` y **covarianza** `Σₖ`.
- La densidad gaussiana multivariante en `d` dimensiones es:

```
𝒩(x | μ, Σ) = (1 / ((2π)^(d/2) |Σ|^(1/2))) · exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))
```

**Los tres parámetros que describe cada gaussiana:**

- `μₖ` — el centro del cluster (vector de medias): equivalente al centroide de K-Means.
- `Σₖ` — la matriz de covarianza: describe la forma y orientación del cluster. Una diagonal grande → cluster alargado. Un término off-diagonal → cluster rotado.
- `πₖ` — el peso: qué fracción del dataset pertenece a este componente.

**Tipos de covarianza en scikit-learn (parámetro `covariance_type`):**

| Tipo | Descripción | Parámetros | Cuándo |
|---|---|---|---|
| `'full'` | Cada cluster tiene su propia matriz de covarianza completa | K·d²/2 | Clusters con forma y orientación distintas |
| `'tied'` | Todos los clusters comparten la misma covarianza | d²/2 | Clusters con misma forma, distintos centros |
| `'diag'` | Covarianzas diagonales (sin correlación entre features) | K·d | Clusters elípticos alineados con los ejes |
| `'spherical'` | Una varianza por cluster, clusters esféricos | K | Equivalente suave a K-Means |

**Script de explicación:**

*"Hay un parámetro que os cambia por completo el comportamiento del GMM: `covariance_type`. Con `full` cada cluster puede tener la elipse que quiera —rotada, alargada, aplastada—. Con `spherical` cada cluster es un círculo de tamaño distinto, que es básicamente K-Means con asignación suave. La elección depende de cuántos datos tenéis y de lo compleja que sea la forma de vuestros clusters."*

---

### [00:22 – 00:30] El algoritmo EM: Expectation-Maximization

**EM es el algoritmo de optimización que entrena el GMM. La analogía con K-Means es directa.**

**El problema de optimización:**

Queremos encontrar los parámetros `θ = {πₖ, μₖ, Σₖ}` que maximizan la log-verosimilitud de los datos:

```
log L(θ) = Σᵢ log p(xᵢ | θ) = Σᵢ log (Σₖ πₖ · 𝒩(xᵢ | μₖ, Σₖ))
```

El problema es que la suma dentro del logaritmo hace la optimización directa intratable.

**La solución EM — dos pasos que se alternan:**

**Paso E (Expectation):** Dada la estimación actual de los parámetros `θ`, calcula la **responsabilidad** `rᵢₖ`: la probabilidad posterior de que el punto `xᵢ` haya sido generado por el componente `k`.

```
rᵢₖ = P(k | xᵢ, θ) = (πₖ · 𝒩(xᵢ | μₖ, Σₖ)) / Σⱼ (πⱼ · 𝒩(xᵢ | μⱼ, Σⱼ))
```

Cada punto `xᵢ` recibe un vector de K responsabilidades que suman 1: `[rᵢ₁, rᵢ₂, ..., rᵢₖ]`. Esta es la **asignación suave** (soft assignment). Comparad con K-Means donde el punto se asignaba a un solo cluster con responsabilidad 1.

**Paso M (Maximization):** Dadas las responsabilidades, actualiza los parámetros maximizando la verosimilitud esperada:

```
πₖ_nuevo  = (1/n) Σᵢ rᵢₖ                                  ← peso = fracción efectiva del cluster
μₖ_nuevo  = Σᵢ rᵢₖ · xᵢ / Σᵢ rᵢₖ                         ← media ponderada
Σₖ_nuevo  = Σᵢ rᵢₖ · (xᵢ - μₖ)(xᵢ - μₖ)ᵀ / Σᵢ rᵢₖ       ← covarianza ponderada
```

**Convergencia:** Se repite E-M hasta que la log-verosimilitud deja de crecer (o el cambio es menor que un umbral `tol`). Garantiza convergencia a un máximo local, no global.

**Analogía con K-Means:**

| K-Means | GMM + EM |
|---|---|
| Inicialización con K-Means++ | Inicialización por defecto con K-Means |
| Paso Asignación: distancia al centroide más cercano (duro) | Paso E: responsabilidades probabilísticas (suave) |
| Paso Actualización: media aritmética | Paso M: media ponderada, covarianza ponderada |
| Objetivo: minimizar WCSS | Objetivo: maximizar log-verosimilitud |
| Resultado: etiquetas duras | Resultado: probabilidades de pertenencia |

*"EM es K-Means con asignaciones suaves y forma elíptica. Si hacéis `covariance_type='spherical'` y aplicáis argmax a las responsabilidades, obtenéis exactamente K-Means."*

---

### [00:30 – 00:35] Selección del número de componentes: BIC y AIC

**El problema:**

Al igual que K-Means necesita elegir `k`, GMM necesita elegir el número de componentes `K`. Pero en GMM tenemos una ventaja: la verosimilitud es una función de coste natural que podemos comparar.

**El riesgo del sobreajuste:**

Con K = n (un componente por punto), la log-verosimilitud es máxima pero el modelo no generaliza —ha memorizando los datos. Necesitamos penalizar la complejidad.

**AIC (Akaike Information Criterion):**
```
AIC = 2·p − 2·log L
```
donde `p` es el número de parámetros libres del modelo. Menor AIC = mejor.

**BIC (Bayesian Information Criterion):**
```
BIC = p·log(n) − 2·log L
```
BIC penaliza más fuerte que AIC porque el factor `log(n)` crece con el tamaño del dataset. Para datasets grandes, BIC tiende a seleccionar modelos más simples (menos componentes).

**Regla práctica:**
- Usar BIC como criterio principal cuando el dataset es grande (n > 500).
- Usar AIC cuando se prefiere capturar más estructura aunque el modelo sea más complejo.
- Cuando BIC y AIC coinciden, hay consenso claro.
- Cuando difieren, usar el criterio de negocio: ¿cuántos segmentos accionables puede gestionar el equipo?

---

## PARTE PRÁCTICA — Jupyter Notebook (35 min)

---

### [00:35 – 01:10] Práctica guiada

---

#### Celda 1 — Imports

```python
# ============================================================
# BLOQUE 2.1 — Gaussian Mixture Models y el Algoritmo EM
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize'] = (11, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
np.random.seed(42)

print("✓ Imports correctos")
```

---

#### Celda 2 — Visualización intuitiva: asignación dura vs. suave

```python
# -------------------------------------------------------
# K-Means (duro) vs. GMM (suave) en el mismo dataset
# -------------------------------------------------------

from sklearn.cluster import KMeans

# Dataset con zona de solapamiento entre dos clusters
np.random.seed(3)
X_overlap = np.vstack([
    np.random.multivariate_normal([0, 0], [[1.5, 0.8],[0.8, 0.6]], 200),
    np.random.multivariate_normal([3, 2], [[1.0, -0.5],[-0.5, 0.8]], 200),
])

X_norm = StandardScaler().fit_transform(X_overlap)

# K-Means
km = KMeans(n_clusters=2, n_init=10, random_state=0)
labels_km = km.fit_predict(X_norm)

# GMM
gmm = GaussianMixture(n_components=2, covariance_type='full',
                      n_init=5, random_state=0)
gmm.fit(X_norm)
labels_gmm   = gmm.predict(X_norm)
proba_gmm    = gmm.predict_proba(X_norm)  # probabilidades de pertenencia
incertidumbre = 1 - proba_gmm.max(axis=1)  # 0 = seguro, 0.5 = máxima incertidumbre

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# K-Means: asignación dura
ax = axes[0]
ax.scatter(X_norm[:, 0], X_norm[:, 1], c=labels_km,
           cmap='bwr', alpha=0.6, s=25)
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='black', marker='X', s=200, zorder=5, label='Centroides')
ax.set_title("K-Means — Asignación dura\n(cada punto = un color, sin matices)",
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# GMM: probabilidad de pertenencia al componente 0
ax = axes[1]
sc = ax.scatter(X_norm[:, 0], X_norm[:, 1],
                c=proba_gmm[:, 0], cmap='RdBu', alpha=0.8, s=25,
                vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='P(componente 0 | x)')
ax.set_title("GMM — Probabilidad de pertenencia\n(gradiente = incertidumbre)",
             fontsize=10, fontweight='bold')

# GMM: incertidumbre (zona de frontera)
ax = axes[2]
sc2 = ax.scatter(X_norm[:, 0], X_norm[:, 1],
                 c=incertidumbre, cmap='hot_r', alpha=0.8, s=25,
                 vmin=0, vmax=0.5)
plt.colorbar(sc2, ax=ax, label='Incertidumbre (0=seguro, 0.5=máx)')
ax.set_title("GMM — Mapa de incertidumbre\n(rojo = zona de frontera ambigua)",
             fontsize=10, fontweight='bold')

plt.suptitle("K-Means vs. GMM: asignación dura vs. asignación probabilística",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_gmm_vs_kmeans_soft.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Puntos con incertidumbre > 0.3: {(incertidumbre > 0.3).sum()} "
      f"({(incertidumbre > 0.3).mean()*100:.1f}%)")
print("→ Estos son los puntos 'frontera' que K-Means clasifica con falsa certeza.")
```

**Script de explicación:**

*"La imagen central es la clave: el gradiente de color muestra la probabilidad de pertenecer al componente azul. Los puntos totalmente rojos son con certeza del componente rojo; los totalmente azules, del azul. Pero hay una zona intermedia donde los puntos son violetas —pertenecen a ambos en distintas proporciones. Ese gradiente es información que K-Means descarta completamente."*

*"El tercer gráfico muestra la incertidumbre: los puntos más calientes son los más ambiguos. En un proyecto real, esos son los clientes 'en la frontera' entre dos segmentos —los más interesantes para estrategias de cross-selling o para campañas de reactivación."*

---

#### Celda 3 — Visualización de las elipses de covarianza

```python
def plot_elipses_gmm(gmm, ax, n_std=2.0, alpha=0.25, colores=None):
    """
    Dibuja las elipses de covarianza de un GMM entrenado.
    n_std: número de desviaciones estándar para el radio de la elipse.
    """
    if colores is None:
        colores = plt.cm.tab10(np.linspace(0, 0.5, gmm.n_components))

    for k, (mean, cov, color) in enumerate(
        zip(gmm.means_, gmm.covariances_, colores)
    ):
        # Descomposición propia para obtener ejes y ángulo
        if gmm.covariance_type == 'full':
            cov_2d = cov
        elif gmm.covariance_type == 'diag':
            cov_2d = np.diag(cov)
        elif gmm.covariance_type in ('spherical', 'tied'):
            cov_2d = np.eye(2) * (cov if gmm.covariance_type == 'spherical'
                                   else cov[0, 0])
        else:
            cov_2d = cov

        vals, vecs = np.linalg.eigh(cov_2d[:2, :2])
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(np.abs(vals))

        elipse = Ellipse(
            xy=mean[:2], width=width, height=height, angle=angle,
            edgecolor=color, facecolor=color, alpha=alpha, linewidth=2
        )
        ax.add_patch(elipse)
        ax.scatter(*mean[:2], c=[color], s=150, marker='X',
                   zorder=5, edgecolors='black', linewidths=1)


# Comparación de tipos de covarianza
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

cov_types = ['full', 'tied', 'diag', 'spherical']
titulos   = [
    "full\n(elipses libres por cluster)",
    "tied\n(misma forma, distintos centros)",
    "diag\n(ejes alineados, sin rotación)",
    "spherical\n(círculos, similar a K-Means)"
]

# Dataset con clusters de distinta forma
np.random.seed(7)
X_elip = np.vstack([
    np.random.multivariate_normal([-2, 0], [[2.0, 1.2],[1.2, 0.4]], 150),
    np.random.multivariate_normal([2,  1], [[0.5, -0.3],[-0.3, 1.5]], 150),
    np.random.multivariate_normal([0, -3], [[0.3, 0],[0, 0.3]], 100),
])
X_elip_norm = StandardScaler().fit_transform(X_elip)

colores_elip = ['#e41a1c','#377eb8','#4daf4a']

for ax, ctype, titulo in zip(axes, cov_types, titulos):
    gmm_c = GaussianMixture(n_components=3, covariance_type=ctype,
                             n_init=5, random_state=0)
    gmm_c.fit(X_elip_norm)
    labels_c = gmm_c.predict(X_elip_norm)

    ax.scatter(X_elip_norm[:, 0], X_elip_norm[:, 1],
               c=labels_c, cmap='tab10', alpha=0.5, s=20)
    plot_elipses_gmm(gmm_c, ax, colores=colores_elip)
    ax.set_title(f"covariance_type='{ctype}'\n{titulo}", fontsize=9, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

plt.suptitle("Impacto de covariance_type en las formas de los clusters",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_gmm_covariance_types.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Script de explicación:**

*"Cada panel muestra el mismo dataset con un tipo de covarianza distinto. Con `full` cada cluster puede ser la elipse que le corresponde —libre en forma, tamaño y orientación—. Con `spherical` los clusters son círculos: si lo recordáis, eso es básicamente K-Means suave. La elección de `covariance_type` afecta profundamente la solución, y también el número de parámetros que hay que estimar —más parámetros requieren más datos."*

---

#### Celda 4 — Selección de K con BIC y AIC

```python
# -------------------------------------------------------
# Curvas BIC y AIC para elegir el número de componentes
# -------------------------------------------------------

# Dataset de churn de telecomunicaciones (sintético)
np.random.seed(0)
n = 500

# Simulamos 4 perfiles distintos de cliente de telecom
antiguedad = np.concatenate([
    np.random.normal(24,  6, 120),   # clientes nuevos
    np.random.normal(48, 10, 150),   # clientes medios
    np.random.normal(72, 12, 130),   # clientes veteranos no fieles
    np.random.normal(60,  8, 100),   # clientes veteranos fieles
])
llamadas = np.concatenate([
    np.random.normal(150, 30, 120),
    np.random.normal(200, 40, 150),
    np.random.normal(80,  20, 130),
    np.random.normal(300, 35, 100),
])
factura = np.concatenate([
    np.random.normal(30, 8,  120),
    np.random.normal(55, 12, 150),
    np.random.normal(40, 10, 130),
    np.random.normal(90, 15, 100),
])
churn_prob = np.concatenate([
    np.random.beta(3, 2, 120),
    np.random.beta(2, 4, 150),
    np.random.beta(5, 2, 130),
    np.random.beta(1, 6, 100),
])

df_telecom = pd.DataFrame({
    'antiguedad_meses': antiguedad,
    'llamadas_mes':     llamadas,
    'factura_media':    factura,
    'prob_churn':       churn_prob,
})
df_telecom = df_telecom.clip(lower=0)

X_tel = StandardScaler().fit_transform(df_telecom)

# Calculamos BIC y AIC para K = 1..10
ks_range = range(1, 11)
bic_vals, aic_vals, ll_vals = [], [], []

for k in ks_range:
    gmm_k = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=5, random_state=42)
    gmm_k.fit(X_tel)
    bic_vals.append(gmm_k.bic(X_tel))
    aic_vals.append(gmm_k.aic(X_tel))
    ll_vals.append(gmm_k.score(X_tel))  # log-verosimilitud media

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BIC y AIC
ax1 = axes[0]
ax1.plot(ks_range, bic_vals, 'bs-', linewidth=2, markersize=8, label='BIC')
ax1.plot(ks_range, aic_vals, 'r^--', linewidth=2, markersize=8, label='AIC')
k_bic = np.argmin(bic_vals) + 1
k_aic = np.argmin(aic_vals) + 1
ax1.axvline(x=k_bic, color='blue', linestyle=':', alpha=0.7,
            label=f'Mín. BIC → k={k_bic}')
ax1.axvline(x=k_aic, color='red', linestyle=':', alpha=0.7,
            label=f'Mín. AIC → k={k_aic}')
ax1.set_xlabel("Número de componentes (K)", fontsize=11)
ax1.set_ylabel("Criterio de información (menor = mejor)", fontsize=11)
ax1.set_title("BIC y AIC para seleccionar K\n(Dataset Telecom Churn)",
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xticks(ks_range)

# Log-verosimilitud
ax2 = axes[1]
ax2.plot(ks_range, ll_vals, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel("Número de componentes (K)", fontsize=11)
ax2.set_ylabel("Log-verosimilitud media", fontsize=11)
ax2.set_title("Log-verosimilitud vs. K\n(siempre crece — no sirve sola)",
              fontsize=11, fontweight='bold')
ax2.set_xticks(ks_range)

plt.suptitle("Selección del número de componentes GMM",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_gmm_bic_aic.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"K óptimo según BIC: {k_bic}")
print(f"K óptimo según AIC: {k_aic}")
```

**Script de explicación:**

*"El gráfico de la derecha ilustra el problema central: la log-verosimilitud siempre crece al añadir componentes. Si la usásemos sola, siempre elegiríamos K = n. Por eso necesitamos BIC y AIC: penalizan el número de parámetros. La curva del BIC tiene un mínimo claro —ahí está el K óptimo según BIC. Si BIC y AIC coinciden, es un resultado robusto."*

---

#### Celda 5 — GMM entrenado y perfilado de segmentos

```python
# Entrenamos el GMM final con el K elegido por BIC
k_final = k_bic
gmm_final = GaussianMixture(n_components=k_final, covariance_type='full',
                             n_init=10, random_state=42)
gmm_final.fit(X_tel)

df_telecom['cluster_gmm'] = gmm_final.predict(X_tel)
proba_final = gmm_final.predict_proba(X_tel)

# Perfil de cada componente
print("Perfil medio de cada componente GMM:")
perfil_gmm = df_telecom.groupby('cluster_gmm')[df_telecom.columns[:-1]].mean().round(1)
perfil_gmm['peso (%)'] = (
    df_telecom['cluster_gmm'].value_counts(normalize=True) * 100
).sort_index().round(1)
print(perfil_gmm)

# Visualización: probabilidades de los 5 puntos más ambiguos
incert = 1 - proba_final.max(axis=1)
top_ambiguos = np.argsort(incert)[-5:][::-1]
print("\nLos 5 clientes más ambiguos (mayor incertidumbre):")
df_ambiguos = pd.DataFrame(
    proba_final[top_ambiguos],
    columns=[f'P(cluster {k})' for k in range(k_final)],
    index=[f'Cliente {i}' for i in top_ambiguos]
).round(3)
print(df_ambiguos)
print("\n→ Estos clientes no pertenecen claramente a ningún segmento.")
print("  Son candidatos a campañas de 'definición de perfil' (encuestas, A/B tests).")
```

---

#### Celda 6 — Visualización del resultado con elipses

```python
# Proyección 2D para visualizar (usamos las dos primeras features)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colores_gmm = ['#e41a1c','#377eb8','#4daf4a','#ff7f00']

# Panel izquierdo: scatter con asignación hard
ax1 = axes[0]
for c in range(k_final):
    mask = df_telecom['cluster_gmm'] == c
    ax1.scatter(
        df_telecom.loc[mask, 'antiguedad_meses'],
        df_telecom.loc[mask, 'factura_media'],
        c=colores_gmm[c % len(colores_gmm)], alpha=0.5, s=30,
        label=f'Componente {c} (n={mask.sum()})'
    )
ax1.set_xlabel("Antigüedad (meses)")
ax1.set_ylabel("Factura media (€)")
ax1.set_title(f"GMM k={k_final} — Asignación hard\n(argmax de probabilidades)",
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=9)

# Panel derecho: incertidumbre
ax2 = axes[1]
sc = ax2.scatter(
    df_telecom['antiguedad_meses'],
    df_telecom['factura_media'],
    c=incert, cmap='YlOrRd', s=30, alpha=0.8
)
plt.colorbar(sc, ax=ax2, label='Incertidumbre de asignación')
ax2.set_xlabel("Antigüedad (meses)")
ax2.set_ylabel("Factura media (€)")
ax2.set_title("Mapa de incertidumbre\n(amarillo = seguro, rojo = ambiguo)",
              fontsize=10, fontweight='bold')

plt.suptitle("GMM aplicado a segmentación de clientes de telecom",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("img_gmm_telecom_resultado.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

#### Celda 7 — Interpretación de negocio de los segmentos

```python
# Nomenclatura de los segmentos basada en el perfil medio
nombres_segmento = {
    0: "Clientes nuevos de bajo valor",
    1: "Clientes consolidados activos",
    2: "Clientes veteranos en riesgo",
    3: "Clientes VIP fieles",
}

acciones = {
    0: "Onboarding mejorado, ofertas de bienvenida",
    1: "Cross-selling de productos premium",
    2: "Programa de retención urgente, llamada proactiva",
    3: "Programa de fidelidad exclusivo, upselling",
}

print("=" * 60)
print("INTERPRETACIÓN DE NEGOCIO — GMM Segmentación Telecom")
print("=" * 60)
for c in range(k_final):
    n_seg = (df_telecom['cluster_gmm'] == c).sum()
    pct   = n_seg / len(df_telecom) * 100
    print(f"\nComponente {c}: '{nombres_segmento.get(c, 'Por definir')}'")
    print(f"  Tamaño: {n_seg} clientes ({pct:.1f}%)")
    print(f"  Acción: {acciones.get(c, 'Pendiente de definir')}")

print("\nVentaja del GMM sobre K-Means:")
print("  Los clientes ambiguos no reciben una etiqueta forzada.")
print("  Se pueden tratar con estrategias mixtas o como prioridad de análisis.")
```

---

## NOTAS DE PRODUCCIÓN

### Para las slides

- **Slide 1:** Portada. Pregunta: *"¿Un cliente pertenece al 100% a un único segmento?"*
- **Slide 2:** Clustering duro vs. suave — diagrama con la misma frontera vista con K-Means (línea dura) y GMM (gradiente de probabilidad).
- **Slide 3:** Fórmula `p(x) = Σ πₖ 𝒩(x|μₖ,Σₖ)` descompuesta visualmente: tres gaussianas coloreadas que se suman.
- **Slide 4:** El algoritmo EM — tabla comparativa con K-Means, paso a paso.
- **Slide 5:** Los cuatro tipos de covarianza — los cuatro paneles de la Celda 3.
- **Slide 6:** BIC y AIC — gráfica con los mínimos señalados y la explicación de la penalización.
- **Slide 7:** Tabla comparativa K-Means vs. GMM (cuándo usar cada uno).

### Para el handout

- Tabla comparativa K-Means vs. GMM con fórmulas del paso E y paso M.
- Tabla de `covariance_type`: descripción, parámetros, cuándo usar.
- Los gráficos de elipses de covarianza (Celda 3).
- El mapa de incertidumbre (Celda 2 y Celda 6) con guía de interpretación.
- Guía de decisión BIC vs. AIC.

### Para el Jupyter Notebook (ejercicios a completar)

**Ejercicio 1:** Aplicar GMM con los cuatro tipos de covarianza al dataset de países del Bloque 1.3. ¿Cuál produce clusters más interpretables? ¿Cuál minimiza el BIC?

**Ejercicio 2:** Para el dataset de telecom, añadir la columna `probabilidad_maxima` al DataFrame y filtrar los clientes con `max_prob < 0.6`. ¿Cuántos son? ¿A qué cluster pertenecen mayoritariamente?

**Ejercicio 3 (avanzado):** Implementar una iteración del algoritmo EM manualmente: dado un GMM ya inicializado con `gmm.fit()`, programar el paso E (responsabilidades) usando NumPy y verificar que coincide con `gmm.predict_proba()`.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador |
|---|---|---|
| Apertura Sesión 2 + recapitulación | 6 min | Preguntas respondidas |
| Limitaciones del clustering duro | 8 min | Ejemplo altura/peso en pantalla |
| El modelo GMM (fórmula + parámetros) | 8 min | Fórmula descompuesta en pantalla |
| El algoritmo EM (pasos E y M) | 9 min | Tabla comparativa con K-Means |
| BIC y AIC | 4 min | Fórmulas en pantalla |
| Celda 1-2 (imports + soft vs. hard) | 8 min | Mapa de incertidumbre generado |
| Celda 3 (elipses de covarianza) | 7 min | Los 4 paneles generados |
| Celda 4 (BIC y AIC) | 7 min | K óptimo identificado |
| Celda 5-7 (telecom + interpretación) | 13 min | Tabla de negocio impresa |
| **Total** | **70 min** | |

---

*Bloque 2.1 desarrollado para el módulo "Algoritmos de Clustering" — Máster en Ciencia de Datos*
