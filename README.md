🇪🇸 -ES
# Framework de Experimentación de Machine Learning

El objetivo principal de este proyecto es ofrecer un framework de software robusto que agiliza y automatiza la experimentación con Machine Learning. Su propósito es abordar uno de los desafíos más comunes en la clasificación: el desbalance de clases en los datasets. El sistema está diseñado específicamente para encontrar las configuraciones óptimas de modelos de ensamblaje bayesianos, permitiendo al usuario analizar cómo diferentes grados de desbalance afectan al rendimiento y cuál es la mejor estrategia para mitigarlo.

Los modelos utilizados son **Ensamblajes de Perceptrones Multicapa (MLP) Bayesianos**. Estos modelos combinan la capacidad predictiva de los ensamblajes (utilizando múltiples modelos para mejorar la robustez) con las fortalezas de los métodos bayesianos, que permiten cuantificar la incertidumbre en las predicciones. Esta característica es especialmente valiosa cuando se trabaja con datos desbalanceados, ya que el modelo puede ser más "honesto" sobre su confianza al clasificar las clases minoritarias.

## Características Principales

*   **Ejecución Automatizada**: Lanza estudios completos para encontrar la configuración óptima de modelos de ensamblaje bayesianos.
*   **Soporte Biclase y Multiclase**: Acepta ficheros CSV para problemas de clasificación con dos o más clases.
*   **Preprocesamiento Automático**: Incluye una fase de preprocesamiento de datos que adapta los ficheros CSV de entrada. Este proceso es robusto, pero puede encontrar dificultades con formatos de datos no estándar.
*   **Estrategias ECOC**: Para problemas multiclase, implementa estrategias de codificación de salida (ECOC) para descomponer el problema en clasificadores binarios.
*   **Generación de Informes**: Al finalizar un experimento, el sistema genera automáticamente un informe detallado con las métricas de rendimiento, la mejor configuración encontrada y visualizaciones.
*   **Interfaz Gráfica Bilingüe**: Toda la gestión de los experimentos se realiza a través de una GUI intuitiva. La aplicación está disponible tanto en **español** como en **inglés**.

## Estructura de Archivos

El código fuente principal del proyecto se encuentra en el directorio `CodeStructure/`. Esta carpeta contiene todos los módulos clave de la aplicación:

*   `CodeStructure/GUI/`: Contiene la interfaz gráfica de usuario. El punto de entrada es `connector.py`.
*   `CodeStructure/model/`: Incluye la implementación de los modelos de ensamblaje bayesianos y la lógica de entrenamiento.
*   `CodeStructure/report_generation/`: Módulos encargados de generar los informes en PDF.
*   `CodeStructure/ecoc.py`: Implementación de las estrategias ECOC para clasificación multiclase.

## Requisitos

*   **Python**: Se requiere la versión específica **Python 3.10.16**. El uso de esta versión es crucial para garantizar la compatibilidad total con las librerías del proyecto.
*   **Dependencias**: Todas las librerías necesarias están listadas en el fichero `requirements.txt`. Es fundamental instalar las **versiones exactas** especificadas en este fichero para asegurar el correcto funcionamiento del software.
*   **Hardware**: El software es compatible con cualquier equipo, pero su rendimiento mejora drásticamente en máquinas preparadas para Machine Learning. Para obtener la máxima velocidad, se recomienda encarecidamente el uso de una **GPU NVIDIA compatible con CUDA**.
*   **Documentación Adicional**: Para un entendimiento completo y detallado del funcionamiento de la aplicación, es fundamental leer el fichero `INSTRUCTIONS.md`.

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

El punto de entrada principal de la aplicación es el script `CodeStructure/GUI/connector.py`.

1.  **Lanzar la aplicación:**
    Ejecuta el siguiente comando desde la raíz del directorio del proyecto:
    ```bash
    python3 CodeStructure/GUI/connector.py
    ```

2.  **Seleccionar un Dataset:**
    Desde la interfaz gráfica, podrás seleccionar un fichero `.csv` que contenga tu dataset. El sistema detectará automáticamente si se trata de un problema de clasificación binaria o multiclase.

3.  **Ejecutar el Experimento:**
    Inicia el estudio. El framework comenzará el proceso de preprocesamiento, entrenamiento de los diferentes modelos de ensamblaje bayesianos y la búsqueda de la mejor estrategia.

4.  **Visualizar el Informe:**
    Una vez finalizado el análisis, se generará un informe con los resultados, que podrás visualizar directamente desde la aplicación.

---

🇬🇧/🇺🇸 -EN
# Machine Learning Experimentation Framework

The main goal of this project is to provide a robust software framework that streamlines and automates Machine Learning experimentation. Its purpose is to address one of the most common challenges in classification: class imbalance in datasets. The system is specifically designed to find the optimal configurations for Bayesian ensemble models, allowing the user to analyze how different degrees of imbalance affect performance and to determine the best strategy to mitigate it.

The models at the core of this framework are **Bayesian Multi-Layer Perceptron (MLP) Ensembles**. These models combine the predictive power of ensembles (using multiple models to improve robustness) with the strengths of Bayesian methods, which allow for the quantification of uncertainty in predictions. This feature is particularly valuable when dealing with imbalanced data, as the model can be more "honest" about its confidence when classifying minority classes.

## Main Features

*   **Automated Execution**: Launches comprehensive studies to find the optimal configuration for Bayesian ensemble models.
*   **Binary and Multi-class Support**: Accepts CSV files for classification problems with two or more classes.
*   **Automatic Preprocessing**: Includes a data preprocessing stage that adapts the input CSV files. This process is robust but may encounter issues with non-standard data formats.
*   **ECOC Strategies**: For multi-class problems, it implements Error-Correcting Output Codes (ECOC) strategies to decompose the problem into binary classifiers.
*   **Report Generation**: Upon completion of an experiment, the system automatically generates a detailed report with performance metrics, the best configuration found, and visualizations.
*   **Bilingual Graphical Interface**: All experiment management is handled through an intuitive GUI. The application is available in both **English** and **Spanish**.

## File Structure

The main source code for the project is located in the `CodeStructure/` directory. This folder contains all the key modules for the application:

*   `CodeStructure/GUI/`: Contains the graphical user interface. The entry point is `connector.py`.
*   `CodeStructure/model/`: Includes the implementation of the Bayesian ensemble models and the training logic.
*   `CodeStructure/report_generation/`: Modules responsible for generating PDF reports.
*   `CodeStructure/ecoc.py`: Implementation of ECOC strategies for multi-class classification.

## Requirements

*   **Python**: The specific version **Python 3.10.16** is required. Using this version is crucial to ensure full compatibility with the project's libraries.
*   **Dependencies**: All necessary libraries are listed in the `requirements.txt` file. It is essential to install the **exact versions** specified in this file to ensure the software works correctly.
*   **Hardware**: The software is compatible with any computer, but its performance is dramatically improved on machines equipped for Machine Learning. For maximum speed, the use of a **CUDA-compatible NVIDIA GPU** is strongly recommended.
*   **Additional Documentation**: For a complete and detailed understanding of the application's functionality, it is essential to read the `INSTRUCTIONS.md` file.

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

The main entry point for the application is the `CodeStructure/GUI/connector.py` script.

1.  **Launch the application:**
    Run the following command from the project's root directory:
    ```bash
    python3 CodeStructure/GUI/connector.py
    ```

2.  **Select a Dataset:**
    From the graphical interface, you can select a `.csv` file containing your dataset. The system will automatically detect whether it is a binary or multi-class classification problem.

3.  **Run the Experiment:**
    Start the study. The framework will begin the process of preprocessing, training the different Bayesian ensemble models, and searching for the best strategy.

4.  **View the Report:**
    Once the analysis is complete, a report with the results will be generated, which you can view directly from
