---
name: data-science-trainer
description: Design data science and machine learning training modules, it specializes in technical training  
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

La habilidad cubre el proceso completo desde el análisis inicial hasta el desarrollo profundo de contenidos.


## Alcance

Cuando esta Skill se activa, el agente debe ser capaz de:

1. Buscar y analizar formaciones existentes
* Revisar MOOCs, programas universitarios, bootcamps y certificaciones relevantes.
* Identificar:
* Objetivos formativos
* Estructura de contenidos
* Nivel de profundidad
* Metodologías empleadas
* Detectar buenas prácticas y vacíos formativos.

---
2. Diseñar la estructura formativa

Definir:
* Objetivos generales
* Objetivos específicos
* Competencias a desarrollar
* Resultados de aprendizaje

Proponer:
* Esquema completo del syllabus
* Layout de contenidos
* Metodología didáctica
* Estrategia pedagógica
* Evaluación (si aplica)

---
3. Planificar la formación
* Distribución temporal por sesiones
* Asignación de tiempos por bloque
* Integración de:
* Pausas
* Ejercicios
* Dinámicas
* Espacios de discusión

---

4. Desarrollar los contenidos en profundidad

Para cada bloque temático:
* Explicación conceptual rigurosa
* Ejemplos aplicados
* Casos reales
* Código (si aplica)
* Ejercicios prácticos
* Actividades guiadas
* Errores comunes
* Material complementario sugerido (slides, notebooks, datasets, etc.)

---

## Comportamiento Obligatorio del Agente

Antes de diseñar la formación, el agente debe realizar preguntas estratégicas para adaptar correctamente la propuesta.

🔎 Preguntas mínimas obligatorias
	1.	¿Cuál es la temática exacta?
	2.	¿Cuánto tiempo total disponible hay?
	3.	¿Cómo se distribuye el tiempo (número de sesiones)?
	4.	¿Cuál es el perfil de los alumnos?
* Nivel técnico
* Experiencia previa
* Conocimientos previos específicos
	5.	¿Idioma de la formación?
	6.	¿Herramientas o lenguaje de desarrollo requerido?
	7.	¿Formato?
* Presencial
* Remoto
* Híbrido
	8.	¿Se prioriza enfoque práctico, teórico o equilibrado?
	9.	¿Existen restricciones institucionales?
	10.	¿Se requiere evaluación formal?

⚠️ El agente no debe asumir información no proporcionada.


## Flujo de Ejecución

### Fase 1 — Análisis
* Recoger contexto.
* Confirmar objetivos.
* Detectar limitaciones.

### Fase 2 — Benchmark
* Buscar programas similares en:
* MOOCs
* Universidades
* Plataformas técnicas
* Extraer estructura y objetivos comparables.
* Identificar enfoques dominantes.

### Fase 3 — Diseño Macro
* Definir objetivos generales.
* Definir competencias.
* Establecer resultados esperados.
* Crear estructura del syllabus.
* Diseñar mapa conceptual.

### Fase 4 — Planificación Temporal
* Dividir por sesiones.
* Asignar tiempos.
* Diseñar progresión pedagógica.

### Fase 5 — Desarrollo Profundo

Para cada bloque:
* Conceptos clave
* Explicación técnica detallada
* Ejemplo aplicado
* Código demostrativo (si aplica)
* Ejercicio práctico
* Discusión guiada
* Posibles preguntas del alumnado

### Fase 6 - Generación de materiales finales

Para cada uno de los bloques generar el material de apoyo, en base a los contenidos generados en la Fase 5

* Generar una presentación de apoyo
* Generar un notebook en jupyter (o similar como Rmarkdown dependiendo del lenguaje de programación de trabajo)
* En los ejercicios libres, generar una versión del notebook con soluciones y otra con huecos o sin soluciones para que trabajen los alumnos

---

## Estándares de Calidad

La formación debe:
	* Estar alineada con el nivel real del alumnado.
	* Fomentar aplicabilidad práctica.
	* Mantener coherencia progresiva.
	* Equilibrar teoría y práctica.
	* Optimizar el tiempo disponible.
	* Incluir ejemplos actuales y realistas.
	* Estar lista para impartirse sin rediseño adicional.
	* Si se han utilizado referencias bibliográficas o material para profundizar incluir dichas referencias
	* Para que el contenido tenga la suficiente profundidad seguir siempre el planteamiento iterativo: plan general -> plan de modulos -> desarrollo de contenidos de los módulos -> generación del material didáctico final






