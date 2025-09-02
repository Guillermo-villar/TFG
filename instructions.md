🇪🇸 -ES
# Guía de Usuario Detallada del Framework de Experimentación

Este documento proporciona una guía paso a paso sobre cómo utilizar la interfaz gráfica (GUI) para ejecutar y gestionar experimentos de Machine Learning.

## 1. Pantalla de Configuración Inicial

Al ejecutar `GUI/connector.py`, la primera ventana que aparece te permite configurar tu experiencia.

<!-- Reemplazar con una captura de pantalla si es posible -->
<!-- ![Initial Setup Screen](URL_A_IMAGEN_DE_LA_PANTALLA_INICIAL) -->

1.  **Selección de Idioma (Language Selection)**:
    *   Puedes elegir entre **Español** e **Inglés**. La interfaz se traducirá automáticamente según tu elección. Esta configuración se puede cambiar cada vez que inicies la aplicación.

2.  **Nivel de Experiencia en Machine Learning (ML Familiarity Level)**:
    *   **Principiante (Beginner)**: Esta opción simplifica la interfaz, ocultando algunos de los detalles técnicos más complejos. Es ideal si solo quieres cargar un dataset y obtener un resultado sin preocuparte por los hiperparámetros.
    *   **Avanzado (Advanced)**: Muestra toda la información disponible, incluyendo logs detallados, configuraciones de los modelos y métricas en tiempo real. Es la opción recomendada para un análisis en profundidad.

Una vez configurado, pulsa **"Comenzar" (Start)** para acceder a la ventana principal.

## 2. Interfaz Principal

La ventana principal está organizada en pestañas y paneles para facilitar la gestión de los experimentos.

<!-- Reemplazar con una captura de pantalla si es posible -->
<!-- ![Main GUI](URL_A_IMAGEN_DE_LA_GUI_PRINCIPAL) -->

### Panel Superior: Controles del Experimento

Aquí es donde configuras y lanzas un nuevo experimento.

1.  **Fuente de Datos (Data Source)**:
    *   **Sintético (Synthetic)**: Utiliza un generador de datos interno para crear un dataset con parámetros específicos (número de muestras, características, etc.). Ideal para pruebas rápidas.
    *   **Fichero CSV (CSV File)**: Permite seleccionar tu propio fichero `.csv` desde tu sistema de archivos.

2.  **Selección de Archivo CSV (Select CSV File)**:
    *   Si elegiste "Fichero CSV", este botón se activará. Al pulsarlo, se abrirá un explorador de archivos para que elijas tu dataset.
    *   El sistema analizará el fichero y detectará si es un problema **binario** o **multiclase**.

3.  **Estrategia Multiclase (Multiclass Strategy)**:
    *   Este menú solo se activa si se detecta un problema multiclase.
    *   **OVA (One-vs-All)**: Entrena un clasificador para cada clase contra todas las demás.
    *   **OVO (One-vs-One)**: Entrena un clasificador para cada par de clases.

4.  **Nivel de Capilaridad (Capillarity Level)**:
    *   Este parámetro controla la "profundidad" del estudio. Un nivel más alto implica una búsqueda más exhaustiva de hiperparámetros, lo que se traduce en un mayor tiempo de ejecución pero resultados potencialmente mejores.

5.  **Botones de Acción**:
    *   **RUN**: Inicia el experimento con la configuración seleccionada.
    *   **STOP**: Detiene un experimento que está en curso.
    *   **DELETE PROGRESS**: Borra los resultados y el progreso de un experimento asociado al dataset seleccionado, permitiendo volver a ejecutarlo desde cero.

### Panel Central: Estado y Progreso

Este panel te da información en tiempo real sobre el estado del dataset seleccionado.

*   **Estado del Dataset (Dataset Status)**: Muestra si un experimento para el dataset seleccionado es **Nuevo (New)**, **En Progreso (In Progress)** o **Completado (Completed)**.
*   **Progreso (Progress)**:
    *   Para experimentos binarios, verás el progreso de **Stage 1** (búsqueda amplia) y **Stage 2** (búsqueda refinada).
    *   Para experimentos multiclase, la barra de progreso representa las **dicotomías** (clasificadores binarios) que se han completado.
*   **Mejor Métrica (Best Metric So Far)**: Muestra el mejor valor de la métrica objetivo (ej. Balanced Accuracy) encontrado hasta el momento.
*   **Mejor Configuración (Best Runner)**: Muestra los hiperparámetros del modelo que logró la mejor métrica.
*   **Información de Tiempo (Timing Info)**:
    *   **ETA**: Tiempo estimado para la finalización de la etapa actual.
    *   **Avg. Time/Run**: Tiempo medio por cada configuración probada.
    *   **Runs/Hour**: Rendimiento del experimento en configuraciones por hora.

### Pestañas Inferiores: Información Detallada

1.  **Log**:
    *   Muestra un registro en tiempo real de lo que está haciendo el sistema: guardando ficheros, entrenando modelos, calculando métricas, etc. Es la mejor fuente de información para depurar problemas.

2.  **Config Preview**:
    *   Muestra el contenido del fichero `config.yaml` que se usará para el experimento. Permite verificar que los parámetros son los correctos antes de lanzar la ejecución.

3.  **Report**:
    *   Esta pestaña se activa cuando un experimento ha finalizado.
    *   Pulsa el botón **"Generar Informe" (Generate Report)** para crear un documento PDF con un resumen completo del experimento, incluyendo gráficos, la matriz de confusión del mejor modelo y tablas de resultados.
    *   Aparecerá una ventana emergente para previsualizar y guardar el informe.

## 3. Flujo de Trabajo Típico

1.  **Inicio**: Ejecuta `python3 GUI/connector.py`.
2.  **Configuración**: Elige idioma y nivel de familiaridad.
3.  **Selección de Datos**: En la ventana principal, elige "Fichero CSV" y selecciona tu dataset.
4.  **Configuración de Experimento**: Si es multiclase, elige una estrategia (OVA/OVO). Selecciona un nivel de capilaridad.
5.  **Ejecución**: Pulsa **RUN**.
6.  **Monitorización**: Observa el panel de estado para ver el progreso y la ETA. Revisa la pestaña "Log" para ver detalles técnicos.
7.  **Finalización**: Una vez el estado sea "Completed", ve a la pestaña "Report".
8.  **Generación de Informe**: Pulsa "Generate Report" para obtener el análisis final en PDF.

Este framework está diseñado para ser una herramienta potente y flexible. ¡Experimenta con diferentes datasets y configuraciones para sacarle el máximo partido!

---

🇬🇧/🇺🇸 -EN
# Detailed User Guide for the Experimentation Framework

This document provides a step-by-step guide on how to use the Graphical User Interface (GUI) to run and manage Machine Learning experiments.

## 1. Initial Setup Screen

When you run `GUI/connector.py`, the first window that appears allows you to configure your experience.

<!-- Replace with a screenshot if possible -->
<!-- ![Initial Setup Screen](URL_TO_INITIAL_SCREEN_IMAGE) -->

1.  **Language Selection**:
    *   You can choose between **English** and **Spanish**. The interface will be automatically translated based on your choice. This setting can be changed every time you start the application.

2.  **ML Familiarity Level**:
    *   **Beginner**: This option simplifies the interface by hiding some of the more complex technical details. It is ideal if you just want to load a dataset and get a result without worrying about hyperparameters.
    *   **Advanced**: Displays all available information, including detailed logs, model configurations, and real-time metrics. This is the recommended option for in-depth analysis.

Once configured, click **"Start"** to access the main window.

## 2. Main Interface

The main window is organized into tabs and panels to facilitate experiment management.

<!-- Replace with a screenshot if possible -->
<!-- ![Main GUI](URL_TO_MAIN_GUI_IMAGE) -->

### Top Panel: Experiment Controls

This is where you configure and launch a new experiment.

1.  **Data Source**:
    *   **Synthetic**: Uses an internal data generator to create a dataset with specific parameters (number of samples, features, etc.). Ideal for quick tests.
    *   **CSV File**: Allows you to select your own `.csv` file from your file system.

2.  **Select CSV File**:
    *   If you chose "CSV File," this button will become active. Clicking it will open a file explorer for you to choose your dataset.
    *   The system will analyze the file and detect if it is a **binary** or **multi-class** problem.

3.  **Multiclass Strategy**:
    *   This menu is only enabled if a multi-class problem is detected.
    *   **OVA (One-vs-All)**: Trains one classifier for each class against all other classes.
    *   **OVO (One-vs-One)**: Trains one classifier for each pair of classes.

4.  **Capillarity Level**:
    *   This parameter controls the "depth" of the study. A higher level implies a more exhaustive search for hyperparameters, resulting in longer execution time but potentially better results.

5.  **Action Buttons**:
    *   **RUN**: Starts the experiment with the selected configuration.
    *   **STOP**: Stops an experiment that is in progress.
    *   **DELETE PROGRESS**: Deletes the results and progress of an experiment associated with the selected dataset, allowing it to be run again from scratch.

### Central Panel: Status and Progress

This panel provides real-time information about the status of the selected dataset.

*   **Dataset Status**: Shows whether an experiment for the selected dataset is **New**, **In Progress**, or **Completed**.
*   **Progress**:
    *   For binary experiments, you will see the progress of **Stage 1** (broad search) and **Stage 2** (refined search).
    *   For multi-class experiments, the progress bar represents the completed **dichotomies** (binary classifiers).
*   **Best Metric So Far**: Displays the best value of the target metric (e.g., Balanced Accuracy) found so far.
*   **Best Runner**: Shows the hyperparameters of the model that achieved the best metric.
*   **Timing Info**:
    *   **ETA**: Estimated time for the completion of the current stage.
    *   **Avg. Time/Run**: Average time per tested configuration.
    *   **Runs/Hour**: Experiment performance in configurations per hour.

### Bottom Tabs: Detailed Information

1.  **Log**:
    *   Displays a real-time log of what the system is doing: saving files, training models, calculating metrics, etc. It is the best source of information for debugging issues.

2.  **Config Preview**:
    *   Shows the content of the `config.yaml` file that will be used for the experiment. It allows you to verify that the parameters are correct before launching the run.

3.  **Report**:
    *   This tab becomes active when an experiment is finished.
    *   Click the **"Generate Report"** button to create a PDF document with a complete summary of the experiment, including charts, the confusion matrix of the best model, and results tables.
    *   A pop-up window will appear to preview and save the report.

## 3. Typical Workflow

1.  **Start**: Run `python3 GUI/connector.py`.
2.  **Setup**: Choose your language and familiarity level.
3.  **Data Selection**: In the main window, choose "CSV File" and select your dataset.
4.  **Experiment Configuration**: If it's a multi-class problem, choose a strategy (OVA/OVO). Select a capillarity level.
5.  **Execution**: Click **RUN**.
6.  **Monitoring**: Watch the status panel to see the progress and ETA. Check the "Log" tab for technical details.
7.  **Completion**: Once the status is "Completed," go to the "Report" tab.
8.  **Report Generation**: Click "Generate Report" to get the final analysis in PDF format.

This framework is designed to be a powerful and flexible tool. Experiment with different datasets and configurations to get the most out
