---
name: data-science-trainer-plus
description: Design data science and machine learning training modules with detailed instructor guides, notebooks and slides. Produces ready-to-deliver course materials.
license: Open Source. LICENSE.txt has complete terms
compatibility: Designed for Claude Code, Codex (or similar products)
---
# Agent Skill: Diseño y Desarrollo de Formaciones Técnicas Especializadas

## Propósito

Esta Skill capacita al agente para diseñar y desarrollar formaciones técnicas adaptadas a:
* Una temática específica
* Un perfil concreto de alumnos
* Un tiempo determinado
* Un formato de impartición definido

La habilidad cubre el proceso completo desde el análisis inicial hasta la generación de materiales listos para impartir, incluyendo guiones detallados del instructor, notebooks de alumno y especificaciones de slides y handouts.

---

## Referencia de Calidad

El directorio `example/` contiene un módulo completo de referencia (Algoritmos de Clustering, 10 horas, 2 sesiones). El agente **debe leer estos archivos antes de generar cualquier material** y usarlos como estándar de profundidad, formato y usabilidad:

| Archivo | Qué ilustra |
|---|---|
| `example/SYLLABUS.md` | Estructura completa de syllabus con metadatos, bloques, recursos y notas para instructor |
| `example/sesion-1/bloque-1.1-guion.md` | Guión de bloque introductorio (55 min): timestamps, scripts, código, notas de producción |
| `example/sesion-1/bloque-1.2-guion.md` | Guión de bloque extenso con dos subpartes (110 min) |
| `example/sesion-1/notebooks/bloque-1.1-distancias.ipynb` | Notebook de alumno: estructura, celdas de ejercicio, setup de verificación |

⚠️ El material generado debe ser comparable en profundidad y usabilidad con estos archivos. Un guión superficial o un notebook incompleto no cumplen el estándar.

---

## Alcance

Cuando esta Skill se activa, el agente debe ser capaz de:

1. **Buscar y analizar formaciones existentes**
   * Revisar MOOCs, programas universitarios, bootcamps y certificaciones relevantes.
   * Identificar: objetivos formativos, estructura de contenidos, nivel de profundidad, metodologías.
   * Detectar buenas prácticas y vacíos formativos.

2. **Diseñar la estructura formativa**

   Definir:
   * Objetivos generales y específicos
   * Competencias a desarrollar
   * Resultados de aprendizaje (verbos de Bloom)

   Proponer:
   * Esquema completo del syllabus
   * Layout de contenidos
   * Metodología didáctica y estrategia pedagógica
   * Evaluación (si aplica)

3. **Planificar la formación**
   * Distribución temporal por sesiones
   * Asignación de tiempos por bloque con tabla detallada
   * Integración de pausas, ejercicios, dinámicas y espacios de discusión

4. **Desarrollar los contenidos en profundidad**

   Para cada bloque temático:
   * Explicación conceptual rigurosa con fórmulas o pseudocódigo cuando aplique
   * Analogías y ejemplos aplicados a casos reales
   * Código funcional comentado
   * Ejercicios prácticos con solución
   * Errores comunes y cómo evitarlos
   * Posibles preguntas del alumnado y respuestas sugeridas

---

## Comportamiento Obligatorio del Agente

Antes de diseñar la formación, el agente debe realizar preguntas estratégicas para adaptar correctamente la propuesta.

🔎 Preguntas mínimas obligatorias
1. ¿Cuál es la temática exacta?
2. ¿Cuánto tiempo total disponible hay?
3. ¿Cómo se distribuye el tiempo (número de sesiones)?
4. ¿Cuál es el perfil de los alumnos?
   * Nivel técnico
   * Experiencia previa
   * Conocimientos previos específicos
5. ¿Idioma de la formación?
6. ¿Herramientas o lenguaje de desarrollo requerido?
7. ¿Formato? (Presencial / Remoto / Híbrido)
8. ¿Se prioriza enfoque práctico, teórico o equilibrado?
9. ¿Existen restricciones institucionales?
10. ¿Se requiere evaluación formal?

⚠️ El agente no debe asumir información no proporcionada.

---

## Flujo de Ejecución

### Fase 1 — Análisis
* Recoger contexto. Confirmar objetivos. Detectar limitaciones.

### Fase 2 — Benchmark
* Buscar programas similares (MOOCs, universidades, plataformas técnicas).
* Extraer estructura y objetivos comparables. Identificar enfoques dominantes.

### Fase 3 — Diseño Macro
* Definir objetivos generales, competencias y resultados esperados.
* Crear estructura del syllabus y mapa conceptual.

### Fase 4 — Planificación Temporal
* Dividir por sesiones. Asignar tiempos. Diseñar progresión pedagógica.

### Fase 5 — Desarrollo Profundo

Para cada bloque:
* Conceptos clave y explicación técnica detallada
* Ejemplo aplicado y código demostrativo (si aplica)
* Ejercicio práctico con solución guiada
* Discusión y posibles preguntas del alumnado

### Fase 6 — Generación de Materiales por Bloque

Generar los siguientes archivos respetando la estructura de carpetas y naming convention:

```
nombre-del-modulo/
├── SYLLABUS.md
├── sesion-N/
│   ├── bloque-N.X-guion.md            ← guión detallado del instructor
│   └── notebooks/
│       └── bloque-N.X-[tema].ipynb    ← notebook de alumno
```

---

#### 6a — SYLLABUS.md

Ver `example/SYLLABUS.md` como referencia. Debe incluir:

* **Tabla de metadatos** del módulo: duración, nivel, lenguaje, entorno, modalidad
* **Descripción del módulo** (2-4 párrafos contextualizando el módulo y el perfil del alumno)
* **Objetivos de aprendizaje** numerados con verbos de Bloom (comprender, implementar, comparar, evaluar...)
* **Prerrequisitos** concretos y verificables
* **Estructura completa** por sesiones con tabla de bloques y tiempos
* **Descripción detallada de cada bloque** (teoría + práctica, con contenidos específicos)
* **Evaluación** con tabla de componentes, pesos y descripción (si aplica)
* **Recursos**: libros con edición, MOOCs con plataforma, documentación oficial, datasets con fuente
* **Herramientas y setup** con comandos de instalación completos (pip, conda, etc.)
* **Notas para el instructor** con adaptaciones según ritmo del grupo y consejos pedagógicos

---

#### 6b — Guión del Instructor (`bloque-N.X-guion.md`)

El guión es el **documento central** de cada bloque. Es el script completo que el instructor sigue durante la sesión — no un esquema de puntos, sino un documento desde el que se puede impartir directamente.

Ver `example/sesion-1/bloque-1.1-guion.md` y `example/sesion-1/bloque-1.2-guion.md`.

**Estructura obligatoria del guión:**

```
# Bloque N.X — [Título]
## Guión detallado del instructor

**Duración:** X minutos ([desglose: X min teoría + X min práctica])
**Posición en la sesión:** [descripción de dónde cae en la sesión]

---

## PARTE TEÓRICA (X min)

---

### [HH:MM – HH:MM] Título de la subsección

> *Nota para el instructor: contexto o instrucción de preparación.*

**Script de [apertura / transición / explicación]:**

*"Texto exacto que dice el instructor entre comillas, en cursiva."*

**Concepto central:**
Explicación técnica rigurosa...

**Analogía para explicarlo:**
*"Analogía en cursiva..."*

[Fórmulas, tablas, pseudocódigo según corresponda]

**Punto de discusión rápido (X min):**
*"Pregunta que lanza el instructor al grupo..."*

**Slide sugerida:** Descripción del contenido de la slide (qué mostrar visualmente).

---

## PARTE PRÁCTICA — Jupyter Notebook (X min)

---

### [HH:MM – HH:MM] Práctica guiada

> *Nota para el instructor: instrucciones de apertura del notebook, qué deben tener los alumnos abierto.*

---

#### Celda N — [Nombre descriptivo]

```python
# código completo, funcional y comentado
```

**Script de explicación / Nota al instructor:**
*"Texto que dice el instructor mientras ejecuta la celda..."*

---

## NOTAS DE PRODUCCIÓN

### Para las slides
- **Slide N:** Descripción del contenido (qué texto, qué visual, qué tabla).

### Para el handout (papel o PDF)
Lista de contenidos que debe incluir el handout del bloque.

### Para el Jupyter Notebook (entrega a alumnos)
**Ejercicio N:** Descripción del ejercicio con marcadores `# TODO:` para el alumno.

---

## GESTIÓN DEL TIEMPO

| Segmento | Duración | Indicador de progreso |
|---|---|---|
| ... | ... | ... |
| **Total** | **X min** | |

---
*[Pie de página con crédito del módulo]*
```

**Reglas de formato del guión:**
* Las notas de instructor van en blockquote con cursiva: `> *Nota: ...*`
* Los scripts del instructor (lo que dice) van en cursiva entre comillas: `*"..."*`
* Las sugerencias de slides van precedidas de `**Slide sugerida:**`
* El código va en bloques de código con el lenguaje especificado
* Los marcadores de tiempo son obligatorios en todas las subsecciones: `[HH:MM – HH:MM]`
* El código en la parte práctica debe ser completo y ejecutable (no fragmentos incompletos)
* Cada celda de código va seguida de su script de explicación

---

#### 6c — Jupyter Notebook (`bloque-N.X-[tema].ipynb`)

El notebook es la versión para el alumno. Ver `example/sesion-1/notebooks/bloque-1.1-distancias.ipynb`.

**Estructura obligatoria del notebook:**

1. **Celda Markdown de título**: nombre del bloque, módulo, duración y cómo usar el notebook
2. **Celda de setup y verificación de entorno**: imports + verificación de librerías requeridas y opcionales con ✅/❌
3. **Celdas de código** con el código del guión, limpio y comentado (sin los scripts del instructor)
4. **Celdas Markdown intercaladas** con:
   * Explicaciones conceptuales en versión alumno (sin el script del instructor)
   * Separadores `---` entre secciones
   * Cabeceras `####` para cada celda numerada
5. **Celdas de ejercicio** marcadas con `# EJERCICIO` o `# TODO:` con espacio vacío para que el alumno trabaje
6. **Celda de cierre** con sección "Para explorar más" y ejercicios propuestos de mayor dificultad

**Reglas del notebook:**
* Las celdas deben poder ejecutarse en orden sin errores
* Los imports van todos en las primeras celdas (nunca a mitad del notebook)
* Los mensajes de confirmación (`print("✓ ...")`) ayudan al alumno a verificar su entorno
* Las celdas de ejercicio incluyen el enunciado como comentario o en Markdown y código vacío o con `# TODO:`
* El notebook NO incluye los scripts del instructor ni las notas de producción — esos van solo en el guión

---

## Estándares de Calidad

La formación debe:
* Estar alineada con el nivel real del alumnado.
* Fomentar aplicabilidad práctica con casos reales.
* Mantener coherencia progresiva entre bloques y sesiones.
* Equilibrar teoría y práctica (aproximadamente 40% / 60% salvo indicación contraria).
* Optimizar el tiempo disponible (los tiempos del guión deben sumar exactamente la duración del bloque).
* Incluir ejemplos actuales y realistas.
* Estar lista para impartirse sin rediseño adicional.
* Incluir referencias bibliográficas y recursos para profundizar.
* Seguir el planteamiento iterativo: **plan general → plan de módulos → desarrollo de contenidos → generación de materiales**.

**Profundidad mínima por bloque:**
* Guión: scripte completo del instructor con marcas de tiempo, código funcional y notas de producción accionables
* Notebook: ejecutable de inicio a fin sin errores, con al menos un ejercicio práctico por bloque
* Código: debe usar las librerías estándar del ecosistema indicado y seguir las convenciones del ejemplo de referencia
