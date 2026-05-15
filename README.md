# ScoliosisSegmentation

## Grupo 18

Autores de la solucion:

- Cristian Camilo Nino Rincon
- Sandra Milena Pantoja Cárdenas
- Integrante pendiente 3
- Integrante pendiente 4

Proyecto de exploracion, entrenamiento e inferencia para segmentacion e identificacion de vertebras toracicas y lumbares en radiografias de columna. El repositorio documenta el camino completo de construccion de una solucion de inteligencia artificial: desde la revision y limpieza del dataset, hasta la definicion de una arquitectura en cascada, el analisis de errores y la consolidacion de un pipeline final de inferencia.

## Objetivo

Construir un pipeline reproducible que permita:

- localizar la columna vertebral en radiografias;
- segmentar vertebras toracicas y lumbares (`T1..T12` y `L1..L5`);
- reducir predicciones anatomicas inconsistentes en imagenes parciales;
- generar mascaras, metricas y reportes que soporten la evaluacion del modelo.

El enfoque final combina modelos supervisados y reglas anatomicas conservadoras para mejorar la utilidad del resultado en radiografias donde no siempre se observa toda la columna.

## Estructura del Repositorio

```text
ScoliosisSegmentation/
├── data/
│   └── Scoliosis_Dataset/        # Dataset crudo y metadatos base. No se versiona.
├── models/                       # Checkpoints entrenados y versionables.
├── notebooks/                    # Flujo experimental completo del proyecto.
├── outputs/                      # Entradas y resultados finales de inferencia.
├── reports/
│   ├── analysis_outputs/         # Metricas, manifests y tablas experimentales.
│   └── dataset_metadata/         # Reportes derivados del dataset.
├── src/                          # Espacio para codigo reutilizable del proyecto.
└── README.md
```

La carpeta `data/Scoliosis_Dataset/` contiene informacion confidencial y esta excluida mediante `.gitignore`. Los artefactos generados por el proceso, como modelos, reportes e inferencias, viven fuera de `data/` para que puedan versionarse cuando sea apropiado.

## Flujo de Trabajo

Los notebooks estan numerados para reflejar la evolucion tecnica del proyecto. Cada notebook produce insumos para las etapas siguientes.

### 01. Exploracion, limpieza y estrategia de cobertura

`notebooks/01_colab_thoracolumbar_coverage_strategy_clean.ipynb`

Primera etapa del proyecto. Revisa el indice del dataset, el diccionario de etiquetas y los reportes disponibles para entender que vertebras aparecen en cada mascara. Su objetivo es transformar una coleccion de datos anotados en una base experimental util para entrenamiento.

Este notebook:

- analiza la cobertura de vertebras toracicas y lumbares;
- separa el problema de interes de regiones no objetivo, como cervicales;
- construye matrices de presencia por vertebra;
- identifica muestras candidatas para revision;
- genera el manifest oficial de entrenamiento.

Salidas principales:

- `reports/analysis_outputs/training_manifest_thoracolumbar.csv`
- `reports/analysis_outputs/thoracolumbar_coverage_summary.csv`
- `reports/analysis_outputs/thoracolumbar_presence_matrix.csv`
- `reports/analysis_outputs/thoracolumbar_review_candidates.csv`

### 02. Modelo binario de localizacion de columna

`notebooks/02_colab_train_spine_binary_and_thoracolumbar.ipynb`

Entrena la primera etapa de la arquitectura: un modelo binario que diferencia columna frente a fondo. Esta etapa permite localizar la region espinal antes de intentar segmentar vertebras individuales.

Este notebook:

- define el split oficial `train/val/test`;
- entrena un modelo binario de segmentacion;
- calibra umbrales de decision;
- guarda metricas, historial y configuracion del experimento;
- exporta el checkpoint base para las siguientes etapas.

Salidas principales:

- `models/binary_spine_thoracolumbar_best.pt`
- `reports/analysis_outputs/training_runs_binary_thoracolumbar/`

### 03. Cascada binaria a multiclase thoracolumbar

`notebooks/03_colab_train_spine_cascade_binary_to_thoracolumbar_explained.ipynb`

Construye la segunda etapa del pipeline: una cascada donde el modelo binario localiza la columna y un modelo multiclase segmenta vertebras toracicas y lumbares dentro de una region de interes.

Este notebook:

- usa la prediccion binaria para definir una ROI espinal;
- entrena una red multiclase para `background + T1..T12 + L1..L5`;
- incorpora contexto espacial mediante canales de coordenadas;
- aplica pesos de clase, scheduler y early stopping;
- documenta decisiones experimentales sobre resolucion, padding y subset de entrenamiento.

Salidas principales:

- `models/thoracolumbar_partial_cascade_explained_best.pt`
- `reports/analysis_outputs/training_runs_cascade_thoracolumbar_explained/`

### 04. Inferencia y analisis de errores

`notebooks/04_colab_infer_analyze_thoracolumbar_predictions_explained.ipynb`

Evalua el comportamiento del modelo multiclase sobre el conjunto de test. Esta etapa mira mas alla de las metricas globales para entender como falla el modelo y que tipo de correcciones son necesarias.

Este notebook:

- genera predicciones sobre test;
- compara mascara real contra mascara predicha;
- calcula metricas globales, por clase y por muestra;
- estudia componentes conectados y centroides;
- identifica errores anatomicos, fusiones, omisiones y sobrepredicciones.

Salidas principales:

- `reports/analysis_outputs/thoracolumbar_inference_analysis_explained/`

### 05. Postproceso anatomico conservador

`notebooks/05_colab_postprocess_anatomical_thoracolumbar_v2_explained.ipynb`

Explora reglas de postproceso para limpiar predicciones poco plausibles sin destruir segmentaciones validas. Surge como respuesta a experimentos previos donde un recorte demasiado agresivo eliminaba informacion util.

Este notebook:

- limpia componentes pequenos;
- preserva la mayor parte de la mascara valida;
- reduce clases aisladas improbables;
- evalua consistencia vertical de las vertebras;
- compara resultados antes y despues del postproceso.

Salidas principales:

- `reports/analysis_outputs/thoracolumbar_postprocess_anatomical_v2_explained/`

### 06. Estimador de rango visible

`notebooks/06_colab_train_visible_range_estimator_and_clip_thoracolumbar_explained.ipynb`

Ataca un problema observado durante el analisis: el modelo multiclase puede predecir vertebras que no estan realmente visibles en radiografias parciales. Para corregirlo, se entrena un modelo auxiliar que estima la primera y la ultima vertebra visible.

Este notebook:

- construye targets supervisados de rango visible;
- entrena un estimador de primera y ultima vertebra visible;
- aplica clipping anatomico sobre la mascara multiclase;
- compara prediccion cruda, clipping estimado y clipping ideal.

Salidas principales:

- `models/visible_range_estimator_thoracolumbar_best.pt`
- `reports/analysis_outputs/visible_range_estimator_thoracolumbar_explained/`

### 07. Estimador especializado de ultima vertebra visible

`notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb`

Refina la etapa anterior al concentrarse en el principal cuello de botella: estimar correctamente la ultima vertebra visible. El proyecto encontro que la primera vertebra visible suele ser mas estable, mientras que el modelo tiende a extender la prediccion hacia niveles lumbares no visibles.

Este notebook:

- construye el target `last_visible_idx`;
- combina evidencia visual de la ROI con features anatomicas de la prediccion multiclase;
- entrena un estimador especializado;
- compara la mascara cruda, el clipping por rango, el clipping por ultima vertebra y el clipping ideal.

Salidas principales:

- `models/last_visible_estimator_thoracolumbar_best.pt`
- `reports/analysis_outputs/last_visible_estimator_thoracolumbar_explained/`

### 08. Pipeline final de inferencia

`notebooks/08_colab_final_inference_pipeline_thoracolumbar_explained.ipynb`

Empaqueta la mejor solucion experimental en un flujo de inferencia reproducible para nuevas radiografias.

Pipeline final:

1. modelo binario localiza la columna;
2. modelo multiclase segmenta vertebras toracicas y lumbares;
3. estimador especializado predice la ultima vertebra visible;
4. clipping anatomico recorta etiquetas fuera del rango plausible;
5. se exportan mascaras, vistas previas y tabla de resultados.

Entradas y salidas principales:

- entrada: `outputs/inference_inputs/`
- salida: `outputs/final_inference_pipeline_thoracolumbar/`

### 09. Resumen tecnico final del proyecto

`notebooks/09_colab_final_project_summary_thoracolumbar_explained.ipynb`

Consolida la historia tecnica del proyecto y genera tablas finales para documentar decisiones, arquitectura, experimentos y resultados.

Este notebook:

- resume el objetivo del proyecto;
- organiza las decisiones de arquitectura;
- compara las etapas experimentales;
- consolida metricas relevantes;
- deja tablas exportables para documentacion y presentacion.

Salidas principales:

- `reports/analysis_outputs/final_project_summary_thoracolumbar/`

## Arquitectura de la Solucion

La solucion final es una cascada de cuatro niveles:

```text
Radiografia
   ↓
Modelo binario de columna
   ↓
ROI espinal
   ↓
Modelo multiclase thoracolumbar
   ↓
Estimador de ultima vertebra visible
   ↓
Clipping anatomico
   ↓
Mascara final + reporte de inferencia
```

Esta arquitectura separa el problema en subtareas mas manejables:

- localizacion gruesa de columna;
- segmentacion fina de vertebras;
- correccion anatomica de imagenes parciales;
- exportacion reproducible de resultados.

## Datos y Versionamiento

El dataset crudo esta en:

```text
data/Scoliosis_Dataset/
```

Esta carpeta esta ignorada por Git porque contiene informacion confidencial. Para ejecutar los notebooks, se debe disponer localmente de esa carpeta con la estructura esperada.

Los modelos y resultados generados estan separados del dataset:

- `models/`: checkpoints `.pt`;
- `reports/analysis_outputs/`: metricas, historiales, manifests y tablas;
- `outputs/`: entradas y resultados finales de inferencia.

Algunos modelos pueden superar el tamano recomendado por GitHub para archivos normales. Para evolucionar el proyecto con multiples checkpoints grandes, se recomienda usar Git LFS.

## Orden Recomendado de Ejecucion

Antes de ejecutar el flujo, se debe copiar localmente la carpeta confidencial del dataset dentro de `data/`, de forma que la estructura quede asi:

```text
ScoliosisSegmentation/
└── data/
    └── Scoliosis_Dataset/
```

Esta carpeta no esta versionada en GitHub por confidencialidad, pero es requerida para que los notebooks puedan leer imagenes, mascaras, indice y diccionario de etiquetas.

Ejecutar los notebooks en orden numerico:

1. `01_colab_thoracolumbar_coverage_strategy_clean.ipynb`
2. `02_colab_train_spine_binary_and_thoracolumbar.ipynb`
3. `03_colab_train_spine_cascade_binary_to_thoracolumbar_explained.ipynb`
4. `04_colab_infer_analyze_thoracolumbar_predictions_explained.ipynb`
5. `05_colab_postprocess_anatomical_thoracolumbar_v2_explained.ipynb`
6. `06_colab_train_visible_range_estimator_and_clip_thoracolumbar_explained.ipynb`
7. `07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb`
8. `08_colab_final_inference_pipeline_thoracolumbar_explained.ipynb`
9. `09_colab_final_project_summary_thoracolumbar_explained.ipynb`

Los notebooks estan preparados para resolver rutas relativas a la estructura actual del repositorio.
