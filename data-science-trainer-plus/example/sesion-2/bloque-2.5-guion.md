# Bloque 2.5 — Proyecto Integrador: Segmentación de Clientes E-Commerce
**Sesión 2 | Duración: 55 minutos**

---

## Tabla de tiempos

| Segmento | Contenido | Tiempo |
|---|---|---|
| Introducción | Contexto del problema y datos | 5 min |
| Fase 1 | Carga, EDA y preparación | 10 min |
| Fase 2 | Ingeniería de features: RFM+ | 10 min |
| Fase 3 | Selección de algoritmo | 10 min |
| Fase 4 | Perfilado e interpretación | 10 min |
| Fase 5 | Presentación de resultados | 10 min |
| **Total** | | **55 min** |

---

## Introducción — Contexto del proyecto (5 min)

### Guión del instructor

> "Con todo lo que hemos visto en estas 10 horas, vamos a cerrar el módulo con un proyecto integrador completo. La idea es simular lo que haría un Data Scientist en una empresa real: desde los datos en bruto hasta una recomendación de negocio accionable."

**El problema:**

Una plataforma de e-commerce quiere lanzar una campaña de retención personalizada. Tienen datos de todas las transacciones del último año. El equipo de marketing necesita saber:

1. ¿Cuántos segmentos de clientes existen realmente?
2. ¿Qué caracteriza a cada segmento?
3. ¿Qué acción de marketing le corresponde a cada uno?

**Dataset: Online Retail II (UCI)**

- Dataset público ampliamente utilizado en marketing analytics
- ~500.000 transacciones de una tienda mayorista UK (2009-2011)
- Variables: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

> "Para este ejercicio usaremos una versión sintética generada a partir del mismo esquema para garantizar reproducibilidad. Los patrones reflejan los del dataset real."

**Metodología del proyecto:**

```
RAW DATA
    ↓
EDA (distribuciones, outliers, nulos)
    ↓
Feature Engineering (RFM+)
    ↓
Preprocesado (escala + PCA opcional)
    ↓
Selección de algoritmo + validación con métricas
    ↓
Perfilado de segmentos
    ↓
Acción de negocio
```

---

## Fase 1 — Carga, EDA y preparación (10 min)

### Guión del instructor

> "Lo primero siempre es explorar los datos. El clustering no es una excepción — basura entra, basura sale."

### Celda 1 — Generación del dataset sintético y EDA

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -----------------------------------------------
# Generación de dataset sintético de transacciones
# -----------------------------------------------
n_clientes = 1500
fecha_referencia = datetime(2024, 12, 31)

# Perfiles latentes (5 segmentos reales que intentaremos descubrir)
perfiles = {
    'Champions':     {'n': 250, 'recencia': (1, 30),   'freq': (15, 30),  'valor': (500, 2000)},
    'Leales':        {'n': 350, 'recencia': (15, 60),  'freq': (8, 20),   'valor': (200, 800)},
    'En riesgo':     {'n': 300, 'recencia': (90, 180), 'freq': (5, 15),   'valor': (150, 600)},
    'Hibernando':    {'n': 350, 'recencia': (180, 365),'freq': (1, 5),    'valor': (50, 250)},
    'Perdidos':      {'n': 250, 'recencia': (300, 365),'freq': (1, 3),    'valor': (20, 100)},
}

filas = []
for perfil, params in perfiles.items():
    n = params['n']
    recencias = np.random.randint(*params['recencia'], n)
    frecuencias = np.random.randint(*params['freq'], n)
    valores_medios = np.random.uniform(*params['valor'], n)
    n_tickets = np.random.randint(1, 20, n)
    devoluciones = np.random.uniform(0, 0.15 if perfil != 'Perdidos' else 0.3, n)
    categorias_unicas = np.random.randint(1, 8, n)

    for i in range(n):
        filas.append({
            'CustomerID': f'C{len(filas)+1:05d}',
            'Recencia': recencias[i],
            'Frecuencia': frecuencias[i],
            'Valor_total': round(valores_medios[i] * n_tickets[i] / 5, 2),
            'Valor_medio_ticket': round(valores_medios[i], 2),
            'Tasa_devolucion': round(devoluciones[i], 3),
            'Categorias_distintas': categorias_unicas[i],
            'Perfil_real': perfil
        })

df = pd.DataFrame(filas).sample(frac=1, random_state=42).reset_index(drop=True)
print("Shape del dataset:", df.shape)
print("\nDistribución de perfiles reales (solo para validación al final):")
print(df['Perfil_real'].value_counts())
print("\nPrimeras filas:")
df.drop(columns='Perfil_real').head()
```

### Celda 2 — EDA visual

```python
features_numericas = ['Recencia', 'Frecuencia', 'Valor_total',
                      'Valor_medio_ticket', 'Tasa_devolucion', 'Categorias_distintas']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for i, col in enumerate(features_numericas):
    axes[i].hist(df[col], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[i].set_title(f'{col}', fontsize=11)
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')

    # Añadir estadísticos
    median_val = df[col].median()
    mean_val = df[col].mean()
    axes[i].axvline(median_val, color='orange', linestyle='--', linewidth=1.5,
                    label=f'Mediana: {median_val:.1f}')
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                    label=f'Media: {mean_val:.1f}')
    axes[i].legend(fontsize=8)

plt.suptitle('EDA — Distribuciones de variables del cliente', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Detección de outliers
print("\n📊 Estadísticos descriptivos:")
print(df[features_numericas].describe().round(2).to_string())

# Skewness
print("\n📐 Asimetría (skewness):")
for col in features_numericas:
    skew = df[col].skew()
    flag = " ⚠️ Alta asimetría" if abs(skew) > 1 else ""
    print(f"   {col:30s}: {skew:+.3f}{flag}")
```

---

## Fase 2 — Ingeniería de features: RFM+ (10 min)

### Guión del instructor

> "La transformación de variables crudas en features significativas es, en muchos casos, la parte más importante del proyecto. Aquí aplicamos la lógica RFM (Recency, Frequency, Monetary), un framework clásico de marketing adaptado."

**¿Qué es RFM?**

| Dimensión | Pregunta | Variable en nuestro dataset |
|---|---|---|
| **R**ecency | ¿Cuándo compró por última vez? | `Recencia` (días) |
| **F**requency | ¿Cuántas veces ha comprado? | `Frecuencia` (transacciones) |
| **M**onetary | ¿Cuánto dinero ha gastado? | `Valor_total` (€) |

**Extensiones (+):**
- `Valor_medio_ticket`: distingue compradores frecuentes-baratos de infrecuentes-premium
- `Tasa_devolucion`: indicador de satisfacción/fraude
- `Categorias_distintas`: breadth de interés del cliente

### Celda 3 — Transformación logarítmica y estandarización

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Las variables monetarias suelen tener distribución log-normal
features_log = ['Recencia', 'Valor_total', 'Valor_medio_ticket']
features_directas = ['Frecuencia', 'Tasa_devolucion', 'Categorias_distintas']

df_proc = df.copy()

# Transformación log (suavizar colas largas)
for col in features_log:
    df_proc[f'{col}_log'] = np.log1p(df_proc[col])

# Dataset final de features
features_modelo = [f'{c}_log' for c in features_log] + features_directas
X_raw = df_proc[features_modelo].values

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Features del modelo:")
for i, f in enumerate(features_modelo):
    print(f"   [{i+1}] {f}")
print(f"\nShape: {X_scaled.shape}")

# Correlación entre features
fig, ax = plt.subplots(figsize=(8, 6))
corr = pd.DataFrame(X_scaled, columns=features_modelo).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, vmin=-1, vmax=1)
ax.set_title('Correlación entre features (post-escala)', fontsize=12)
plt.tight_layout()
plt.show()
```

### Celda 4 — Decisión: ¿usar PCA?

```python
# Análisis de varianza explicada
pca_check = PCA(random_state=42)
pca_check.fit(X_scaled)

var_acum = np.cumsum(pca_check.explained_variance_ratio_)
k_90 = np.argmax(var_acum >= 0.90) + 1

print(f"Dimensiones originales: {X_scaled.shape[1]}")
print(f"Componentes para retener 90% varianza: {k_90}")
print(f"Varianza retenida con todas las componentes: 100%")

# Con 6 features no hay maldición severa, pero PCA puede ayudar
# si hay alta correlación entre variables
if k_90 <= 4:
    print("\n→ Usaremos PCA para reducir a", k_90, "componentes")
    pca_final = PCA(n_components=k_90, random_state=42)
    X_modelo = pca_final.fit_transform(X_scaled)
else:
    print(f"\n→ Con solo {X_scaled.shape[1]} features y {k_90} comp. para 90%, "
          "trabajamos directamente con las features originales estandarizadas")
    X_modelo = X_scaled
    pca_final = None

print(f"Shape datos para clustering: {X_modelo.shape}")
```

---

## Fase 3 — Selección del algoritmo (10 min)

### Guión del instructor

> "Ahora aplicamos lo aprendido en el Bloque 2.3: no elegimos un algoritmo arbitrariamente, sino que comparamos varios con métricas objetivas."

### Celda 5 — Comparación multialgoritmo

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn_extra.cluster import KMedoids

resultados_comparacion = []

k_candidatos = [3, 4, 5, 6]

print("Evaluando algoritmos...\n")

for k in k_candidatos:
    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    lbl = km.fit_predict(X_modelo)
    resultados_comparacion.append({
        'Algoritmo': f'K-Means k={k}',
        'k': k,
        'Silhouette': silhouette_score(X_modelo, lbl),
        'DBI': davies_bouldin_score(X_modelo, lbl),
        'CHI': calinski_harabasz_score(X_modelo, lbl),
        'labels': lbl
    })

    # GMM
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42, n_init=5)
    lbl = gmm.fit_predict(X_modelo)
    resultados_comparacion.append({
        'Algoritmo': f'GMM k={k}',
        'k': k,
        'Silhouette': silhouette_score(X_modelo, lbl),
        'DBI': davies_bouldin_score(X_modelo, lbl),
        'CHI': calinski_harabasz_score(X_modelo, lbl),
        'labels': lbl
    })

df_comp = pd.DataFrame(resultados_comparacion)

# Normalizar para score compuesto
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
df_comp['sil_norm'] = mms.fit_transform(df_comp[['Silhouette']])
df_comp['dbi_norm_inv'] = 1 - mms.fit_transform(df_comp[['DBI']])
df_comp['chi_norm'] = mms.fit_transform(df_comp[['CHI']])
df_comp['Score_compuesto'] = (df_comp['sil_norm'] +
                               df_comp['dbi_norm_inv'] +
                               df_comp['chi_norm']) / 3

# Mostrar ranking
cols_display = ['Algoritmo', 'Silhouette', 'DBI', 'CHI', 'Score_compuesto']
print(df_comp[cols_display].sort_values('Score_compuesto', ascending=False)
      .round(4).to_string(index=False))

# Selección del mejor
mejor = df_comp.sort_values('Score_compuesto', ascending=False).iloc[0]
print(f"\n✅ Mejor configuración: {mejor['Algoritmo']} "
      f"(Score compuesto: {mejor['Score_compuesto']:.4f})")
```

### Celda 6 — Ajuste fino del modelo seleccionado

```python
# Ajustamos K-Means con el k óptimo encontrado
k_optimo = int(df_comp.sort_values('Score_compuesto', ascending=False).iloc[0]['k'])
algoritmo_optimo = df_comp.sort_values('Score_compuesto', ascending=False).iloc[0]['Algoritmo']

print(f"Modelo final: {algoritmo_optimo}")
print(f"k óptimo: {k_optimo}")

# Re-entrenar con más inits para mayor estabilidad
if 'K-Means' in algoritmo_optimo:
    modelo_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=30)
else:
    modelo_final = GaussianMixture(n_components=k_optimo, covariance_type='full',
                                   random_state=42, n_init=15)

labels_finales = modelo_final.fit_predict(X_modelo)
df_proc['Segmento'] = labels_finales

print(f"\nDistribución de segmentos:")
print(df_proc['Segmento'].value_counts().sort_index())

sil = silhouette_score(X_modelo, labels_finales)
dbi = davies_bouldin_score(X_modelo, labels_finales)
chi = calinski_harabasz_score(X_modelo, labels_finales)
print(f"\nMétricas modelo final:")
print(f"   Silhouette:  {sil:.4f}")
print(f"   Davies-Bouldin: {dbi:.4f}")
print(f"   Calinski-Harabasz: {chi:.1f}")
```

---

## Fase 4 — Perfilado e interpretación (10 min)

### Guión del instructor

> "Tenemos los segmentos. Ahora viene la parte que más le importa al negocio: ¿qué significa cada número? Un clustering sin interpretación es inútil."

### Celda 7 — Perfiles estadísticos

```python
# Perfil estadístico por segmento
perfil = df_proc.groupby('Segmento')[features_numericas].agg(['mean', 'median']).round(2)

# Tabla de medias para interpretación
medias = df_proc.groupby('Segmento')[features_numericas].mean().round(2)
medias['N_clientes'] = df_proc.groupby('Segmento').size()
medias['%_clientes'] = (medias['N_clientes'] / len(df_proc) * 100).round(1)

print("📊 Perfil medio por segmento:\n")
print(medias.to_string())
```

### Celda 8 — Radar chart y etiquetado de segmentos

```python
from matplotlib.patches import FancyArrowPatch

# Normalizar features para radar chart (0-1)
medias_norm = medias[features_numericas].copy()

# Recencia: invertir (menor recencia = mejor → más activo)
medias_norm['Recencia'] = 1 - (medias_norm['Recencia'] - medias_norm['Recencia'].min()) / \
                              (medias_norm['Recencia'].max() - medias_norm['Recencia'].min())
# Tasa devolución: invertir
medias_norm['Tasa_devolucion'] = 1 - (medias_norm['Tasa_devolucion'] -
                                       medias_norm['Tasa_devolucion'].min()) / \
                                      (medias_norm['Tasa_devolucion'].max() -
                                       medias_norm['Tasa_devolucion'].min() + 1e-9)

for col in ['Frecuencia', 'Valor_total', 'Valor_medio_ticket', 'Categorias_distintas']:
    min_v = medias_norm[col].min()
    max_v = medias_norm[col].max()
    medias_norm[col] = (medias_norm[col] - min_v) / (max_v - min_v + 1e-9)

# Renombrar para el gráfico
labels_radar = ['Actividad\nreciente', 'Frecuencia', 'Valor\ntotal',
                'Ticket\nmedio', 'Fidelidad\n(dev. inv.)', 'Amplitud\ncategorías']

n_seg = k_optimo
colores_seg = plt.cm.Set2(np.linspace(0, 1, n_seg))

fig = plt.figure(figsize=(14, 6))

# Radar chart
ax_radar = fig.add_subplot(121, polar=True)
angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
angles += angles[:1]

for seg in range(n_seg):
    vals = medias_norm.iloc[seg][features_numericas].tolist()
    # Reordenar para el radar
    vals_radar = [medias_norm.iloc[seg]['Recencia'],
                  medias_norm.iloc[seg]['Frecuencia'],
                  medias_norm.iloc[seg]['Valor_total'],
                  medias_norm.iloc[seg]['Valor_medio_ticket'],
                  medias_norm.iloc[seg]['Tasa_devolucion'],
                  medias_norm.iloc[seg]['Categorias_distintas']]
    vals_radar += vals_radar[:1]
    ax_radar.plot(angles, vals_radar, 'o-', linewidth=2,
                  color=colores_seg[seg], label=f'Segmento {seg}')
    ax_radar.fill(angles, vals_radar, alpha=0.15, color=colores_seg[seg])

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(labels_radar, size=9)
ax_radar.set_title('Radar: perfil por segmento', fontsize=11, pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

# Barras de tamaño
ax_bar = fig.add_subplot(122)
sizes = [medias.loc[s, 'N_clientes'] for s in range(n_seg)]
bars = ax_bar.barh(range(n_seg),
                   sizes,
                   color=colores_seg,
                   edgecolor='white', linewidth=1.5)
ax_bar.set_yticks(range(n_seg))
ax_bar.set_yticklabels([f'Segmento {s}' for s in range(n_seg)], fontsize=10)
ax_bar.set_xlabel('Número de clientes')
ax_bar.set_title('Tamaño de cada segmento', fontsize=11)

for bar, size in zip(bars, sizes):
    pct = size / len(df_proc) * 100
    ax_bar.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{size} ({pct:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.suptitle('Caracterización de segmentos de clientes',
             fontsize=13, fontweight='bold', y=1.02)
plt.show()
```

### Celda 9 — Asignación de nombres y validación

```python
# El instructor facilita la interpretación con los estudiantes
# basándose en el radar chart anterior

# Ejemplo de asignación de nombres (ajustar según los resultados reales)
# Los estudiantes deben proponer sus propios nombres basándose en los datos

# Tabla de interpretación guiada
print("=" * 65)
print("GUÍA DE INTERPRETACIÓN DE SEGMENTOS")
print("=" * 65)
print()
print("Para asignar nombre a cada segmento, considera:")
print()
print("  Alta Actividad Reciente + Alta Frecuencia + Alto Valor")
print("  → Champions / Clientes VIP activos")
print()
print("  Media Actividad + Media Frecuencia + Medio Valor")
print("  → Clientes Leales / Base estable")
print()
print("  Baja Actividad Reciente + Alta Frecuencia histórica")
print("  → En riesgo de abandono / Despertar necesario")
print()
print("  Muy Baja Actividad + Baja Frecuencia + Bajo Valor")
print("  → Hibernando / Prácticamente perdidos")
print()
print("  Alta Tasa Devolución")
print("  → Perfil de riesgo / Insatisfacción o fraude")
print()

# Mapeo propuesto (adaptable)
nombres_segmentos_base = {
    0: 'Segmento A', 1: 'Segmento B', 2: 'Segmento C',
    3: 'Segmento D', 4: 'Segmento E'
}

# Validación con ARI (usando los perfiles reales que solo el instructor conoce)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(df['Perfil_real'], labels_finales)
print(f"ARI vs. perfiles reales (solo para validación interna): {ari:.4f}")
print("(Un ARI > 0.6 indica que el clustering captura bien los segmentos reales)")
```

---

## Fase 5 — Presentación de resultados (10 min)

### Guión del instructor

> "El último paso, y uno que se subestima en la formación técnica: comunicar los resultados de forma que el equipo de marketing pueda actuar. Un cluster sin nombre y sin acción es ruido."

### Celda 10 — Dashboard final y recomendaciones

```python
# Visualización TSNE final de los segmentos
from sklearn.manifold import TSNE

# Asignación de nombres ilustrativos (el instructor ajusta tras ver los perfiles)
# En el aula: pedir a los estudiantes que propongan nombres
nombres_ejemplo = {
    0: '⭐ Champions',
    1: '💙 Leales',
    2: '⚠️ En riesgo',
    3: '💤 Hibernando',
    4: '🚫 Perdidos'
}
# Solo aplicamos si k_optimo == 5
if k_optimo == 5:
    df_proc['Nombre_segmento'] = df_proc['Segmento'].map(nombres_ejemplo)
else:
    df_proc['Nombre_segmento'] = 'Segmento ' + df_proc['Segmento'].astype(str)

# t-SNE para visualización
print("Calculando proyección t-SNE para visualización final...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatter con segmentos
colores_finales = plt.cm.Set2(np.linspace(0, 1, k_optimo))
for seg in range(k_optimo):
    mask = df_proc['Segmento'] == seg
    nombre = df_proc.loc[mask, 'Nombre_segmento'].iloc[0]
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=[colores_finales[seg]], alpha=0.6, s=20, label=nombre)

axes[0].set_title('Segmentación de clientes (proyección t-SNE)', fontsize=11)
axes[0].set_xlabel('Dimensión 1')
axes[0].set_ylabel('Dimensión 2')
axes[0].legend(fontsize=9, markerscale=2)

# Heatmap de perfiles
perfil_heatmap = medias[features_numericas].copy()
# Normalizar por columna para el heatmap
perfil_heatmap_norm = (perfil_heatmap - perfil_heatmap.min()) / \
                      (perfil_heatmap.max() - perfil_heatmap.min() + 1e-9)

sns.heatmap(perfil_heatmap_norm, annot=perfil_heatmap.values.round(1),
            fmt='.1f', cmap='YlOrRd', ax=axes[1],
            linewidths=0.5, linecolor='white',
            xticklabels=[c.replace('_', '\n') for c in features_numericas],
            yticklabels=[df_proc.loc[df_proc['Segmento']==s, 'Nombre_segmento'].iloc[0]
                         for s in range(k_optimo)])

axes[1].set_title('Perfil de segmentos (valores reales, escala por color)', fontsize=11)
axes[1].set_xlabel('')

plt.suptitle('Dashboard final — Segmentación de clientes e-commerce',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Celda 11 — Tabla de acciones de marketing

```python
# Tabla de recomendaciones (para generar PDF/presentación)
acciones = pd.DataFrame({
    'Segmento': ['Champions', 'Leales', 'En riesgo', 'Hibernando', 'Perdidos'],
    'Descripción': [
        'Alta recencia, frecuencia y valor',
        'Compran regularmente, valor medio',
        'Activos antes, inactivos últimos meses',
        'Sin compras desde 6-12 meses',
        'Sin compras desde >12 meses, bajo valor'
    ],
    'Acción recomendada': [
        'Programa de fidelización premium, early access',
        'Descuentos por volumen, newsletter VIP',
        'Campaña de reactivación, encuesta de satisfacción',
        'Oferta win-back agresiva, recordatorio personalizado',
        'Evaluar coste-beneficio de campaña vs. baja de BBDD'
    ],
    'Canal sugerido': [
        'Email + App push + Gestor personal',
        'Email + SMS',
        'Email personalizado + Retargeting',
        'Email + SMS oferta limitada',
        'Email (coste mínimo)'
    ],
    'KPI objetivo': [
        'Mantener ticket medio, NPS > 8',
        'Aumentar frecuencia +20%',
        'Reactivar >30% en 90 días',
        'Reactivar >15% en 60 días',
        'ROI positivo o depurar'
    ]
})

print("=" * 90)
print("PLAN DE ACCIÓN DE MARKETING POR SEGMENTO")
print("=" * 90)
print(acciones.to_string(index=False))
print()
print(f"📈 Inversión recomendada:")
print(f"   Champions ({medias.loc[0,'N_clientes'] if k_optimo>0 else 'N/A'} clientes): Alta")
print(f"   Leales: Media-Alta")
print(f"   En riesgo: Media (ROI potencial alto)")
print(f"   Hibernando: Baja-Media")
print(f"   Perdidos: Muy baja (evaluar individualmente)")
```

---

## Notas de producción

### Para la presentación (slides)
- **Slide 1**: El problema de negocio — foto de e-commerce + 3 preguntas clave
- **Slide 2**: Metodología como diagrama de flujo (6 pasos con iconos)
- **Slide 3**: EDA — 2-3 histogramas más relevantes + outliers
- **Slide 4**: Explicación de RFM con iconos visuales
- **Slide 5**: Tabla de comparación de algoritmos con score compuesto
- **Slide 6**: Radar chart + tamaño de segmentos
- **Slide 7**: Dashboard final (t-SNE + heatmap)
- **Slide 8**: Tabla de acciones de marketing — la más importante para el negocio

### Para el handout
- Descripción del dataset y variables
- Justificación de la transformación log (con histograma antes/después)
- Tabla de métricas de evaluación comparativa
- Clave de interpretación RFM
- Tabla de acciones de marketing (reproducible)

### Ejercicio de entrega propuesto
> "Aplicar el mismo pipeline a un dataset propio o alternativo (Superstore, Instacart, o datos propios de la empresa). Entregar un Jupyter Notebook con: EDA, feature engineering justificado, selección de algoritmo con métricas, perfilado de segmentos con nombres, y tabla de acciones."

### Variantes para reutilizar en clase
- Dataset bancario: segmentación de cuentas por comportamiento transaccional
- Dataset de salud: segmentación de pacientes por adherencia/riesgo
- Dataset de HR: segmentación de empleados por engagement

---

## Tabla de tiempos (verificación)

| Segmento | Duración real |
|---|---|
| Introducción | 5 min |
| Fase 1: EDA (2 celdas) | 10 min |
| Fase 2: RFM+ (2 celdas) | 10 min |
| Fase 3: Selección algoritmo (2 celdas) | 10 min |
| Fase 4: Perfilado (3 celdas) | 10 min |
| Fase 5: Resultados (2 celdas) | 10 min |
| **Total** | **55 min** ✓ |
