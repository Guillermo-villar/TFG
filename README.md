🇪🇸
# Framework de Experimentación de Machine Learning

Este proyecto es un framework de software diseñado para automatizar la ejecución de experimentos de clasificación de Machine Learning. La herramienta proporciona una interfaz gráfica para gestionar, ejecutar y analizar el rendimiento de modelos de ensamblaje bayesianos en problemas de clasificación tanto binaria como multiclase.

## Características Principales

*   **Ejecución Automatizada**: Lanza estudios completos para encontrar la configuración óptima de modelos de ensamblaje bayesianos.
*   **Soporte Biclase y Multiclase**: Acepta ficheros CSV para problemas de clasificación con dos o más clases.
*   **Preprocesamiento Automático**: Incluye una fase de preprocesamiento de datos que adapta los ficheros CSV de entrada. Este proceso es robusto, pero puede encontrar dificultades con formatos de datos no estándar.
*   **Estrategias ECOC**: Para problemas multiclase, implementa estrategias de codificación de salida (ECOC) para descomponer el problema en clasificadores binarios.
*   **Generación de Informes**: Al finalizar un experimento, el sistema genera automáticamente un informe detallado con las métricas de rendimiento, la mejor configuración encontrada y visualizaciones.
*   **Interfaz Gráfica Bilingüe**: Toda la gestión de los experimentos se realiza a través de una GUI intuitiva. La aplicación está disponible tanto en **español** como en **inglés**.

## Requisitos

*   **Python**: Se requiere una versión de **Python 3.10** o superior.
*   **Dependencias**: Todas las librerías necesarias están listadas en el fichero `requirements.txt`.
*   **Hardware**: El software es compatible con cualquier equipo, pero su rendimiento mejora drásticamente en máquinas preparadas para Machine Learning. Para obtener la máxima velocidad, se recomienda encarecidamente el uso de una **GPU NVIDIA compatible con CUDA**.

## Instalación

Para poner en marcha el entorno de trabajo, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL-del-repositorio>
    cd <nombre-del-directorio>
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar las dependencias:**
    Asegúrate de tener el entorno virtual activado y ejecuta:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

El punto de entrada principal de la aplicación es el script `GUI/connector.py`.

1.  **Lanzar la aplicación:**
    Ejecuta el siguiente comando desde la raíz del directorio `Small`:
    ```bash
    python3 GUI/connector.py
    ```

2.  **Seleccionar un Dataset:**
    Desde la interfaz gráfica, podrás seleccionar un fichero `.csv` que contenga tu dataset. El sistema detectará automáticamente si se trata de un problema de clasificación binaria o multiclase.

3.  **Ejecutar el Experimento:**
    Inicia el estudio. El framework comenzará el proceso de preprocesamiento, entrenamiento de los diferentes modelos de ensamblaje bayesianos y la búsqueda de la mejor estrategia.

4.  **Visualizar el Informe:**
    Una vez finalizado el análisis, se generará un informe con los resultados, que podrás visualizar directamente desde la aplicación.

---

🇬🇧/🇺🇸
# Machine Learning Experimentation Framework

This project is a software framework designed to automate the execution of Machine Learning classification experiments. The tool provides a graphical user interface to manage, run, and analyze the performance of Bayesian ensemble models on both binary and multi-class classification problems.

## Main Features

*   **Automated Execution**: Launches comprehensive studies to find the optimal configuration for Bayesian ensemble models.
*   **Binary and Multi-class Support**: Accepts CSV files for classification problems with two or more classes.
*   **Automatic Preprocessing**: Includes a data preprocessing stage that adapts the input CSV files. This process is robust but may encounter issues with non-standard data formats.
*   **ECOC Strategies**: For multi-class problems, it implements Error-Correcting Output Codes (ECOC) strategies to decompose the problem into binary classifiers.
*   **Report Generation**: Upon completion of an experiment, the system automatically generates a detailed report with performance metrics, the best configuration found, and visualizations.
*   **Bilingual Graphical Interface**: All experiment management is handled through an intuitive GUI. The application is available in both **English** and **Spanish**.

## Requirements

*   **Python**: **Python 3.10** or a higher version is required.
*   **Dependencies**: All necessary libraries are listed in the `requirements.txt` file.
*   **Hardware**: The software is compatible with any computer, but its performance is dramatically improved on machines equipped for Machine Learning. For maximum speed, the use of a **CUDA-compatible NVIDIA GPU** is strongly recommended.

## Installation

To set up the working environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-URL>
    cd <directory-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Make sure the virtual environment is activated and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point for the application is the `GUI/connector.py` script.

1.  **Launch the application:**
    Run the following command from the root of the `Small` directory:
    ```bash
    python3 GUI/connector.py
    ```

2.  **Select a Dataset:**
    From the graphical interface, you can select a `.csv` file containing your dataset. The system will automatically detect whether it is a binary or multi-class classification problem.

3.  **Run the Experiment:**
    Start the study. The framework will begin the process of preprocessing, training the different Bayesian ensemble models, and searching for the best strategy.

4.  **View the Report:**
    Once the analysis is complete, a report with the results will be generated, which you can view directly from the application.
