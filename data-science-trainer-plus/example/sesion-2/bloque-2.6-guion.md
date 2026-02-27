# Bloque 2.6 — Buenas Prácticas, Escalabilidad y Cierre del Módulo
**Sesión 2 | Duración: 15 minutos**

---

## Tabla de tiempos

| Segmento | Contenido | Tiempo |
|---|---|---|
| Parte 1 | Errores frecuentes y cómo evitarlos | 5 min |
| Parte 2 | Escalabilidad: ¿qué usar con datos grandes? | 3 min |
| Parte 3 | Mapa de decisión final | 3 min |
| Parte 4 | Cierre, recursos y próximos pasos | 4 min |
| **Total** | | **15 min** |

---

## Parte 1 — Errores frecuentes y cómo evitarlos (5 min)

### Guión del instructor

> "Antes de cerrar, quiero compartir un inventario de los errores más comunes que se cometen en proyectos reales de clustering. Los he visto en proyectos de consultoría y en trabajos de fin de máster. Saber cuáles son ya es evitar la mitad."

### Celda 1 — Checklist de errores frecuentes (ejecutable como recordatorio)

```python
# ============================================================
# CHECKLIST DE BUENAS PRÁCTICAS EN CLUSTERING
# Guarda este notebook como referencia para proyectos reales
# ============================================================

checklist = {
    "PREPROCESADO": [
        ("✅", "Estandarizar siempre antes de clustering (media 0, std 1)"),
        ("✅", "Revisar outliers ANTES de clustering, no después"),
        ("✅", "Aplicar log-transform a variables monetarias/de conteo con cola larga"),
        ("✅", "Comprobar multicolinealidad — features correladas distorsionan las distancias"),
        ("❌", "NUNCA incluir el target (si existe) en las features de clustering"),
        ("❌", "NUNCA imputar nulos con la media si hay muchos — pueden crear un cluster espurio"),
    ],
    "SELECCIÓN DE ALGORITMO": [
        ("✅", "Probar al menos 2-3 algoritmos antes de decidir"),
        ("✅", "Usar siempre múltiples métricas (no solo Silhouette)"),
        ("✅", "Considerar la interpretabilidad del negocio al elegir k"),
        ("❌", "NUNCA elegir k basándose en una sola métrica ni por defecto (k=3 o k=5)"),
        ("❌", "NUNCA usar K-Means si los clusters no son convexos o hay ruido significativo"),
        ("❌", "NUNCA usar DBSCAN sin explorar el k-distance graph para calibrar ε"),
    ],
    "INTERPRETACIÓN": [
        ("✅", "Asignar nombres de negocio a cada cluster — los números no comunican"),
        ("✅", "Validar con expertos de dominio que los segmentos tienen sentido"),
        ("✅", "Comprobar estabilidad: re-ejecutar con distintas semillas"),
        ("✅", "Verificar que cada cluster es accionable (se puede hacer algo diferente con él)"),
        ("❌", "NUNCA presentar el clustering como 'la verdad' — es una aproximación"),
        ("❌", "NUNCA confundir el score de Silhouette alto con 'el clustering es correcto'"),
    ],
    "VISUALIZACIÓN": [
        ("✅", "Usar PCA para preprocesar, t-SNE/UMAP solo para visualizar"),
        ("✅", "Aclarar en cualquier gráfico t-SNE que las distancias entre clusters NO son interpretables"),
        ("✅", "Incluir siempre el tamaño de cada cluster en los gráficos"),
        ("❌", "NUNCA sacar conclusiones sobre relaciones entre clusters a partir de t-SNE"),
    ],
    "PRODUCCIÓN": [
        ("✅", "Guardar el scaler y el modelo entrenado (pickle/joblib) para predecir nuevos clientes"),
        ("✅", "Documentar el pipeline completo (parámetros, versiones de librerías)"),
        ("✅", "Establecer un proceso de re-entrenamiento periódico"),
        ("❌", "NUNCA asumir que el clustering de hace 6 meses es válido hoy sin validarlo"),
    ]
}

for categoria, items in checklist.items():
    print(f"\n{'='*55}")
    print(f"  {categoria}")
    print(f"{'='*55}")
    for simbolo, texto in items:
        print(f"  {simbolo}  {texto}")

print("\n\n💾 Guarda este notebook — es tu referencia para proyectos reales.")
```

---

## Parte 2 — Escalabilidad: ¿qué usar con datos grandes? (3 min)

### Guión del instructor

> "Todo lo que hemos visto funciona bien hasta ~50.000-100.000 registros. En proyectos con millones de clientes hay que hacer ajustes."

### Tabla de escalabilidad (para la presentación)

| Algoritmo | Complejidad | Escala a 1M+ | Alternativa escalable |
|---|---|---|---|
| K-Means | O(n·k·iter) | ✅ Con `MiniBatchKMeans` | `MiniBatchKMeans` (sklearn) |
| K-Medoids (PAM) | O(n²) | ❌ | CLARA, CLARANS |
| Jerárquico | O(n² log n) – O(n³) | ❌ | Bisecting K-Means |
| DBSCAN | O(n log n) con índice | ⚠️ Con HDBSCAN | HDBSCAN (hdbscan library) |
| GMM (EM) | O(n·k·d²·iter) | ⚠️ Con muchos features | Variational Bayes GMM |
| SOM | O(n·m·iter) | ✅ Con m moderado | `minisom` + submuestras |

```python
# Demo rápida: MiniBatchKMeans vs KMeans en tiempo
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
import numpy as np

# Dataset grande simulado
X_grande, _ = make_blobs(n_samples=200_000, n_features=10,
                         centers=6, random_state=42)

# KMeans estándar
t0 = time.time()
km = KMeans(n_clusters=6, random_state=42, n_init=5, max_iter=100)
km.fit(X_grande)
t_km = time.time() - t0

# MiniBatchKMeans
t0 = time.time()
mbkm = MiniBatchKMeans(n_clusters=6, random_state=42, n_init=5,
                        batch_size=2048, max_iter=100)
mbkm.fit(X_grande)
t_mbkm = time.time() - t0

print(f"KMeans estándar (200k muestras):  {t_km:.2f}s")
print(f"MiniBatchKMeans (200k muestras):  {t_mbkm:.2f}s")
print(f"Aceleración: {t_km/t_mbkm:.1f}x")

# Comparar inercia (aproximación de calidad)
print(f"\nKMeans inertia:          {km.inertia_:,.0f}")
print(f"MiniBatchKMeans inertia: {mbkm.inertia_:,.0f}")
print(f"Diferencia relativa: {abs(km.inertia_-mbkm.inertia_)/km.inertia_:.2%}")
```

---

## Parte 3 — Mapa de decisión final (3 min)

### Guión del instructor

> "Cierro con el árbol de decisión que podéis usar en cualquier proyecto. No es un oráculo, pero es un punto de partida sólido."

### Celda 3 — Árbol de decisión como texto ejecutable

```python
# ============================================================
# ÁRBOL DE DECISIÓN PARA SELECCIÓN DE ALGORITMO DE CLUSTERING
# ============================================================

arbol = """
¿Cuántas muestras tengo?
│
├── < 10.000 → Puedo usar cualquier algoritmo
│
└── > 100.000 → Evitar: K-Medoids PAM, Jerárquico completo
               Preferir: MiniBatchKMeans, HDBSCAN, SOM con submuestra

¿Conozco el número de clusters k de antemano?
│
├── Sí → K-Means / K-Medoids / GMM
│
└── No → Explorar k con: Elbow + Silhouette + DBI
         O usar: DBSCAN / HDBSCAN (determinan k automáticamente)

¿Esperan clusters esféricos y bien separados?
│
├── Sí → K-Means (rápido, interpretable)
│
└── No → ¿Hay ruido/outliers significativos?
          │
          ├── Sí → DBSCAN / HDBSCAN
          │
          └── No → ¿Clusters con forma irregular pero sin ruido?
                    │
                    ├── Sí → Jerárquico (Ward) o GMM
                    └── No → SOM (exploración topológica)

¿Necesito probabilidades de pertenencia (soft assignment)?
│
├── Sí → GMM
└── No → K-Means, K-Medoids, Jerárquico, DBSCAN

¿Los outliers son datos importantes (no errores)?
│
├── Sí (ej. detección de anomalías) → DBSCAN / HDBSCAN
└── No → K-Means o K-Medoids (más robusto que K-Means)

¿Necesito visualizar relaciones topológicas entre clusters?
│
└── Sí → SOM (U-Matrix, component planes)
"""

print(arbol)
```

---

## Parte 4 — Cierre, recursos y próximos pasos (4 min)

### Guión del instructor

> "Hemos cubierto en 10 horas lo que muchos proyectos de ciencia de datos requieren: desde los fundamentos matemáticos hasta la puesta en producción. Pero este es el punto de partida, no el final."

### Celda 4 — Resumen del módulo y recursos

```python
resumen = """
╔══════════════════════════════════════════════════════════════════╗
║          RESUMEN DEL MÓDULO — ALGORITMOS DE CLUSTERING          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  SESIÓN 1 (5h)                                                   ║
║  ├── Bloque 1.1: Fundamentos y métricas de distancia            ║
║  ├── Bloque 1.2: K-Means y K-Medoids                            ║
║  ├── Bloque 1.3: Clustering jerárquico                          ║
║  └── Bloque 1.4: DBSCAN                                         ║
║                                                                  ║
║  SESIÓN 2 (5h)                                                   ║
║  ├── Bloque 2.1: Modelos de mezcla gaussiana (GMM + EM)         ║
║  ├── Bloque 2.2: Mapas auto-organizados de Kohonen (SOM)        ║
║  ├── Bloque 2.3: Métricas de evaluación                         ║
║  ├── Bloque 2.4: Reducción de dimensionalidad                   ║
║  ├── Bloque 2.5: Proyecto integrador (RFM e-commerce)           ║
║  └── Bloque 2.6: Buenas prácticas y cierre ← aquí estamos      ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ALGORITMOS CUBIERTOS:                                           ║
║   K-Means  ·  K-Medoids  ·  Hierarchical  ·  DBSCAN             ║
║   GMM (EM)  ·  Kohonen SOM                                      ║
║                                                                  ║
║  MÉTRICAS: Silhouette · DBI · CHI · WCSS · ARI · NMI            ║
║                                                                  ║
║  PREPROCESADO: Escala · Log-transform · PCA                     ║
║  VISUALIZACIÓN: t-SNE · UMAP · Radar · Heatmap · U-Matrix       ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(resumen)

recursos = """
📚 RECURSOS RECOMENDADOS

LIBROS
  • "The Elements of Statistical Learning" — Hastie, Tibshirani, Friedman
    Cap. 14: Unsupervised Learning (disponible free en web oficial)
  • "Pattern Recognition and Machine Learning" — Bishop
    Cap. 9: Mixture Models and EM
  • "Introduction to Machine Learning with Python" — Müller & Guido
    (enfoque práctico, scikit-learn)

PAPERS CLAVE
  • Lloyd (1982) — "Least squares quantization in PCM" (K-Means original)
  • Arthur & Vassilvitskii (2007) — "k-means++: The advantages of careful seeding"
  • Ester et al. (1996) — "A density-based algorithm for discovering clusters" (DBSCAN)
  • Kohonen (1982) — "Self-organized formation of topologically correct feature maps"
  • McInnes et al. (2018) — "UMAP: Uniform Manifold Approximation and Projection" (arXiv)

DOCUMENTACIÓN
  • scikit-learn clustering: https://scikit-learn.org/stable/modules/clustering.html
  • minisom: https://github.com/JustGlowing/minisom
  • UMAP: https://umap-learn.readthedocs.io
  • HDBSCAN: https://hdbscan.readthedocs.io

DATASETS PARA PRACTICAR
  • UCI ML Repository — Online Retail, Customer Segmentation
  • Kaggle — Mall Customers, Credit Card Dataset, E-Commerce Data
  • scikit-learn — make_blobs, make_moons, make_circles (sintéticos)
"""
print(recursos)
```

### Celda 5 — Próximos pasos (continuación del máster)

```python
proximos_pasos = """
🎯 PRÓXIMOS PASOS EN EL MÁSTER

Este módulo conecta con:

  → MÓDULO SIGUIENTE: Detección de anomalías
     Isolation Forest, LOF, Autoencoders
     (Clustering como baseline para anomaly detection)

  → MÓDULO: Aprendizaje semi-supervisado
     Usar clustering para generar pseudo-labels
     Label propagation, self-training

  → MÓDULO: Series temporales
     Clustering de series: DTW, k-Shape, temporal SOM
     Segmentación de comportamiento a lo largo del tiempo

  → PROYECTO FINAL
     Aplicar el pipeline completo a datos reales de la empresa
     Deliverable: notebook documentado + presentación ejecutiva

💡 CONSEJO FINAL

El clustering es tanto ciencia como arte.
Las métricas guían, pero el juicio del dominio decide.
El mejor clustering no es el que tiene el Silhouette más alto,
sino el que lleva a las mejores decisiones de negocio.
"""
print(proximos_pasos)
```

---

## Notas de producción

### Para la presentación (slides)
- **Slide 1**: Checklist en 4 categorías — presentar como "errores que cuestan dinero"
- **Slide 2**: Tabla de escalabilidad — resaltar MiniBatchKMeans y HDBSCAN
- **Slide 3**: Árbol de decisión como diagrama visual (no texto)
- **Slide 4**: Mapa visual de los 6 algoritmos con sus dominios de aplicación
- **Slide 5**: Lista de recursos + QR code al repositorio del curso
- **Slide 6** (última): Frase de cierre — "El mejor clustering lleva a la mejor decisión"

### Para el handout
- Checklist en una página (A4 doble cara) para uso en proyectos
- Árbol de decisión como diagrama (printable)
- Tabla comparativa final de los 6 algoritmos cubiertos
- Lista de recursos con URLs

### Encuesta de cierre (sugerida)
Pedir a los estudiantes que respondan en papel o digital:
1. ¿Qué algoritmo crees que usarás más en tu trabajo?
2. ¿Qué concepto te ha costado más entender?
3. ¿Qué echas en falta en el módulo?

---

## Tabla de tiempos (verificación)

| Segmento | Duración real |
|---|---|
| Parte 1: Errores frecuentes | 5 min |
| Parte 2: Escalabilidad | 3 min |
| Parte 3: Árbol de decisión | 3 min |
| Parte 4: Cierre y recursos | 4 min |
| **Total** | **15 min** ✓ |
