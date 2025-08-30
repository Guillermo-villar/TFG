"""
report_html_generator.py
Generates HTML + CSS for experiment reports, ready for PDF export with WeasyPrint.
"""
import os
from datetime import datetime
from jinja2 import Template

class ReportHTMLGenerator:
    def __init__(self, experiment_data):
        self.data = experiment_data

    def render_html(self):
        # Get current timestamp
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Simple, modern HTML template with CSS
        template = Template('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Experiment Report</title>
            <style>
                @page { size: A4; margin: 1in; }
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    background: #ffffff; 
                    color: #222; 
                    margin: 0; 
                    padding: 20px;
                    line-height: 1.6;
                }
                .container { 
                    max-width: 100%; 
                    margin: 0 auto; 
                    background: #fff; 
                    padding: 0;
                }
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                    border-bottom: 2px solid #3a3a7a;
                    padding-bottom: 20px;
                }
                h1 { 
                    color: #2d2d2d; 
                    font-size: 2.5em; 
                    margin: 0;
                    font-weight: 300;
                }
                .dataset-name {
                    font-size: 2.5em;
                    color: #3a3a7a;
                    margin: 10px 0 0 0;
                    font-weight: 300;
                }
                h2 { 
                    color: #3a3a7a; 
                    margin-top: 2em;
                    margin-bottom: 0.5em;
                    font-size: 1.5em;
                }
                .section { 
                    margin-bottom: 2em; 
                    page-break-inside: avoid;
                }
                .explanation {
                    background: #f8f9fa;
                    border-left: 4px solid #007acc;
                    padding: 15px;
                    margin: 15px 0;
                    font-style: italic;
                    color: #555;
                }
                .label { font-weight: bold; color: #444; }
                .imbalance { 
                    color: #b36a00; 
                    font-weight: bold; 
                    font-size: 1.1em;
                }
                .imbalance-level {
                    padding: 5px 10px;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                    display: inline-block;
                    margin-left: 10px;
                }
                .low { background-color: #28a745; }
                .moderate { background-color: #ffc107; color: #000; }
                .high { background-color: #fd7e14; }
                .extreme { background-color: #dc3545; }
                .mono { 
                    font-family: 'Consolas', monospace; 
                    background: #f3f3f7; 
                    padding: 8px 12px; 
                    border-radius: 6px;
                    display: block;
                    margin: 5px 0;
                }
                .card { 
                    background: #f8f9fa; 
                    border: 1px solid #dee2e6;
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 15px 0;
                }
                .emoji { font-size: 1.3em; margin-right: 0.5em; }
                .footer { 
                    color: #888; 
                    font-size: 0.9em; 
                    margin-top: 3em; 
                    text-align: center;
                    border-top: 1px solid #dee2e6;
                    padding-top: 20px;
                }
                .class-item {
                    display: inline-block;
                    background: #e9ecef;
                    padding: 8px 15px;
                    margin: 5px;
                    border-radius: 20px;
                    font-weight: bold;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }
                .metric-item {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #007acc;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><span class="emoji">📊</span>Experiment Report</h1>
                    <div class="dataset-name">{{ data.dataset_name }}</div>
                    <p style="margin: 15px 0 0 0; color: #666; font-size: 1.1em;">
                        <strong>Experiment Type:</strong> {{ 'Multiclass Classification' if data.is_multiclass else 'Binary Classification' }}
                    </p>
                </div>

                <div class="section">
                    <h2><span class="emoji">🏷️</span>Class Distribution</h2>
                    <div class="explanation">
                        <strong>What is Class Distribution?</strong><br>
                        Class distribution shows how many samples (data points) belong to each category in your dataset. 
                        A balanced dataset has roughly equal numbers in each class, while an imbalanced dataset has 
                        some classes with many more samples than others.
                    </div>
                    
                    {% if data.class_distribution %}
                        <div style="margin: 20px 0;">
                        {% for k, v in data.class_distribution.items() %}
                            <div class="class-item">{{ k }}: {{ v }} samples</div>
                        {% endfor %}
                        </div>
                        
                        {% if data.imbalance_ratio %}
                        <div style="margin: 20px 0;">
                            <span class="imbalance">⚖️ Imbalance Ratio: {{ data.imbalance_ratio }}</span>
                            {% set ratio = data.imbalance_ratio|float %}
                            {% if ratio <= 1.5 %}
                                <span class="imbalance-level low">Well Balanced</span>
                            {% elif ratio <= 3.0 %}
                                <span class="imbalance-level moderate">Moderately Imbalanced</span>
                            {% elif ratio <= 10.0 %}
                                <span class="imbalance-level high">Highly Imbalanced</span>
                            {% else %}
                                <span class="imbalance-level extreme">Extremely Imbalanced</span>
                            {% endif %}
                        </div>
                        <div class="explanation">
                            <strong>Understanding Imbalance Ratio:</strong><br>
                            • <strong>1.0 - 1.5:</strong> Well balanced dataset - all classes have similar representation<br>
                            • <strong>1.5 - 3.0:</strong> Moderately imbalanced - some classes are underrepresented<br>
                            • <strong>3.0 - 10.0:</strong> Highly imbalanced - significant disparity between classes<br>
                            • <strong>10.0+:</strong> Extremely imbalanced - some classes are severely underrepresented
                        </div>
                        {% endif %}
                    {% else %}
                        <div class="card">No class distribution data available.</div>
                    {% endif %}
                </div>

                <div class="section">
                    <h2><span class="emoji">🧪</span>Machine Learning Strategy</h2>
                    <div class="explanation">
                        <strong>What is Multiclass Decomposition?</strong><br>
                        When dealing with multiple classes (like A, B, C, D), we break down the complex problem into simpler 
                        binary (two-class) problems called "dichotomies." This makes it easier for the computer to learn patterns
                        and make accurate predictions.
                    </div>
                    
                    {% if data.is_multiclass %}
                    <div class="card">
                        <h3 style="margin-top: 0; color: #3a3a7a;">Multiclass Strategy: Dichotomy Decomposition</h3>
                        
                        {% if data.strategy == 'ovo' %}
                        <p><strong>🥊 One-vs-One (OVO) Strategy:</strong></p>
                        <div class="explanation">
                            This approach creates a separate binary classifier for every possible pair of classes. 
                            For {{ data.class_distribution|length }} classes, we created {{ ((data.class_distribution|length * (data.class_distribution|length - 1)) / 2)|int }} binary classifiers.
                            Each classifier becomes an "expert" at distinguishing between just two specific classes.
                            
                            <br><br><strong>Why OVO?</strong>
                            <ul style="margin: 10px 0;">
                                <li>Each classifier focuses on a simpler, more specific task</li>
                                <li>Better handling of class boundaries and overlapping regions</li>
                                <li>More robust to class imbalance in individual pairs</li>
                                <li>Higher accuracy when classes have complex decision boundaries</li>
                            </ul>
                            
                            <br><strong>Class Pairs Analyzed:</strong><br>
                            {% set class_list = data.class_distribution.keys()|list %}
                            {% for i in range(class_list|length) %}
                                {% for j in range(i+1, class_list|length) %}
                                    <span style="background: #e3f2fd; padding: 3px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 0.9em;">
                                        {{ class_list[i] }} vs {{ class_list[j] }}
                                    </span>
                                {% endfor %}
                            {% endfor %}
                        </div>
                        
                        {% elif data.strategy == 'ova' %}
                        <p><strong>🎯 One-vs-All (OVA) Strategy:</strong></p>
                        <div class="explanation">
                            This approach creates {{ data.class_distribution|length }} binary classifiers, where each one learns to distinguish 
                            one specific class from all the others combined. For example: "Is this an A?" vs "Is this not an A?"
                            
                            <br><br><strong>Why OVA?</strong>
                            <ul style="margin: 10px 0;">
                                <li>Computationally efficient with fewer classifiers needed</li>
                                <li>Each classifier learns the unique characteristics of one class</li>
                                <li>Good for problems where classes have distinct features</li>
                                <li>Scales well with increasing number of classes</li>
                            </ul>
                            
                            <br><strong>Classifiers Created:</strong><br>
                            {% for class_name in data.class_distribution.keys() %}
                                <span style="background: #f3e5f5; padding: 3px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 0.9em;">
                                    {{ class_name }} vs All Others
                                </span>
                            {% endfor %}
                        </div>
                        
                        {% else %}
                        <p><strong>🔄 Multiclass Decomposition Strategy:</strong></p>
                        <div class="explanation">
                            A sophisticated multiclass decomposition approach was used to break down this {{ data.class_distribution|length }}-class problem 
                            into simpler binary classification tasks. Each binary classifier was individually optimized for maximum performance.
                        </div>
                        {% endif %}
                        
                        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #ff9800;">
                            <h4 style="margin-top: 0; color: #f57c00;">🏆 Final Prediction</h4>
                            <p>To make a final prediction, all binary classifiers "vote" and the system combines their results using 
                            sophisticated algorithms to determine the most likely class for each new data point.</p>
                        </div>
                    </div>
                    {% else %}
                    <div class="card">
                        <h3 style="margin-top: 0; color: #3a3a7a;">Binary Classification Strategy</h3>
                        <p>This experiment used a direct binary classification approach, as there are only two classes to distinguish between. 
                        The model was optimized specifically for this two-class problem using:</p>
                        <ul>
                            <li>Advanced neural network architecture with multiple experts</li>
                            <li>Custom loss functions designed for binary classification</li>
                            <li>Specialized techniques to handle class imbalance</li>
                        </ul>
                    </div>
                    {% endif %}
                </div>

                <div class="section">
                    <h2><span class="emoji">🔧</span>Technical Implementation</h2>
                    <div class="explanation">
                        <strong>Advanced Techniques Used:</strong><br>
                        The model employs cutting-edge machine learning techniques including ensemble methods,
                        regularization, and custom optimization strategies.
                    </div>
                    <div class="card">
                        <h3 style="margin-top: 0;">Key Technical Features:</h3>
                        <ul>
                            <li><strong>Expert Ensemble:</strong> Multiple specialized neural networks working together</li>
                            <li><strong>Dropout Regularization:</strong> Prevents overfitting and improves generalization</li>
                            <li><strong>Custom Loss Functions:</strong> Optimized for handling imbalanced datasets</li>
                            <li><strong>Batch Processing:</strong> Efficient training with optimized batch sizes</li>
                            <li><strong>Adaptive Learning:</strong> Dynamic adjustment of learning parameters</li>
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <h2><span class="emoji">🏆</span>Detailed Results by Dichotomy</h2>
                    <div class="explanation">
                        <strong>Performance Analysis:</strong><br>
                        Each dichotomy represents a binary classification problem. Below are the results
                        in sequential order, showing which classes were compared and what the performance metrics mean.
                    </div>
                    
                    {% if data.is_multiclass and data.dichotomy_mapping %}
                        {% for i in range(1, (data.dichotomy_mapping|length) + 1) %}
                            {% set d_key = 'dichotomy_{:02d}'.format(i) %}
                            {% set mapping = data.dichotomy_mapping.get(d_key) %}
                            {% set metrics = data.metrics.get(d_key) %}
                            {% set config = data.best_model.get(d_key, {}).get('best_runner', {}) %}
                            
                            {% if mapping %}
                            <div class="card" style="margin-bottom: 25px;">
                                <h3 style="margin-top: 0; color: #3a3a7a; border-bottom: 2px solid #e9ecef; padding-bottom: 10px;">
                                    📊 Dicotomía {{ i }}: {{ mapping.description }}
                                </h3>
                                
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                                    <h4 style="margin-top: 0; color: #495057;">🎯 Problema Binario:</h4>
                                    <p style="margin: 5px 0;"><strong>Clase Positiva:</strong> {{ mapping.class_1 }}</p>
                                    <p style="margin: 5px 0;"><strong>Clase Negativa:</strong> {{ mapping.class_2 }}</p>
                                    <p style="margin: 5px 0; font-style: italic;">
                                        El clasificador aprendió a distinguir entre estas dos clases específicamente.
                                    </p>
                                </div>
                                
                                {% if metrics %}
                                <h4 style="color: #28a745;">📈 Métricas de Rendimiento:</h4>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                                    {% for k, v in metrics.items() %}
                                        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                                            <strong style="color: #155724;">{{ k.replace('_', ' ').title() }}</strong><br>
                                            <span style="font-size: 1.3em; color: #155724; font-weight: bold;">{{ v }}</span><br>
                                            <small style="color: #6c757d;">
                                                {% if 'accuracy' in k.lower() %}
                                                    Porcentaje de predicciones correctas
                                                {% elif 'precision' in k.lower() %}
                                                    De las predicciones positivas, cuántas fueron correctas
                                                {% elif 'recall' in k.lower() %}
                                                    De los casos positivos reales, cuántos se detectaron
                                                {% elif 'f1' in k.lower() %}
                                                    Promedio armónico entre precisión y recall
                                                {% elif 'auc' in k.lower() %}
                                                    Área bajo la curva ROC (0.5 = aleatorio, 1.0 = perfecto)
                                                {% else %}
                                                    Métrica de evaluación del modelo
                                                {% endif %}
                                            </small>
                                        </div>
                                    {% endfor %}
                                </div>
                                {% endif %}
                                
                                {% if config %}
                                <h4 style="color: #007acc;">⚙️ Configuración Óptima Encontrada:</h4>
                                <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 15px 0;">
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                                        {% if config.get('activation_fn') %}
                                        <div>
                                            <strong>🔄 Función de Activación:</strong> <code>{{ config.activation_fn.upper() }}</code><br>
                                            <small style="color: #6c757d;">
                                                {% if config.activation_fn == 'relu' %}
                                                    ReLU: Rápida y efectiva, evita gradientes negativos
                                                {% elif config.activation_fn == 'tanh' %}
                                                    Tanh: Salida entre -1 y 1, buena para datos centrados
                                                {% elif config.activation_fn == 'sigmoid' %}
                                                    Sigmoid: Salida entre 0 y 1, clásica para probabilidades
                                                {% else %}
                                                    Función que determina cómo se activan las neuronas
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                        
                                        {% if config.get('num_experts') %}
                                        <div>
                                            <strong>👥 Número de Expertos:</strong> <span style="font-size: 1.2em; color: #007acc;">{{ config.num_experts }}</span><br>
                                            <small style="color: #6c757d;">
                                                Redes neuronales paralelas que trabajan juntas. 
                                                {% if config.num_experts|int < 15 %}
                                                    Configuración conservadora, menos complejidad.
                                                {% elif config.num_experts|int < 25 %}
                                                    Configuración balanceada, buena capacidad.
                                                {% else %}
                                                    Configuración avanzada, alta capacidad de aprendizaje.
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                        
                                        {% if config.get('hidden_size') %}
                                        <div>
                                            <strong>🧠 Tamaño de Capa Oculta:</strong> <span style="font-size: 1.2em; color: #007acc;">{{ config.hidden_size }}</span><br>
                                            <small style="color: #6c757d;">
                                                Neuronas en la capa principal. 
                                                {% if config.hidden_size|int < 30 %}
                                                    Modelo simple, menos riesgo de sobreajuste.
                                                {% elif config.hidden_size|int < 60 %}
                                                    Modelo equilibrado, buena capacidad.
                                                {% else %}
                                                    Modelo complejo, alta capacidad de patrones.
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                        
                                        {% if config.get('alpha') %}
                                        <div>
                                            <strong>📊 Tasa de Aprendizaje (α):</strong> <span style="font-size: 1.2em; color: #007acc;">{{ config.alpha }}</span><br>
                                            <small style="color: #6c757d;">
                                                Qué tan rápido aprende el modelo. 
                                                {% if config.alpha|float < 0.05 %}
                                                    Aprendizaje conservador y estable.
                                                {% elif config.alpha|float < 0.15 %}
                                                    Aprendizaje equilibrado.
                                                {% else %}
                                                    Aprendizaje rápido, puede ser inestable.
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                        
                                        {% if config.get('drop_out') %}
                                        <div>
                                            <strong>🛡️ Tasa de Dropout:</strong> <span style="font-size: 1.2em; color: #007acc;">{{ (config.drop_out * 100)|round(1) }}%</span><br>
                                            <small style="color: #6c757d;">
                                                Regularización para evitar sobreajuste. 
                                                {% if config.drop_out|float < 0.1 %}
                                                    Poca regularización, modelo más directo.
                                                {% elif config.drop_out|float < 0.3 %}
                                                    Regularización moderada, equilibrio.
                                                {% else %}
                                                    Alta regularización, previene sobreajuste.
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                        
                                        {% if config.get('n_epoch') %}
                                        <div>
                                            <strong>🔄 Épocas de Entrenamiento:</strong> <span style="font-size: 1.2em; color: #007acc;">{{ config.n_epoch }}</span><br>
                                            <small style="color: #6c757d;">
                                                Veces que el modelo vio todos los datos. 
                                                {% if config.n_epoch|int < 50 %}
                                                    Entrenamiento rápido, puede necesitar más.
                                                {% elif config.n_epoch|int < 150 %}
                                                    Entrenamiento equilibrado.
                                                {% else %}
                                                    Entrenamiento extensivo, muy optimizado.
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                        
                                        {% if config.get('n_batch') %}
                                        <div>
                                            <strong>📦 Tamaño de Lote:</strong> <span style="font-size: 1.2em; color: #007acc;">{{ config.n_batch }}</span><br>
                                            <small style="color: #6c757d;">
                                                Muestras procesadas antes de actualizar. 
                                                {% if config.n_batch|int < 64 %}
                                                    Lotes pequeños, actualizaciones frecuentes.
                                                {% elif config.n_batch|int < 256 %}
                                                    Lotes equilibrados, buen rendimiento.
                                                {% else %}
                                                    Lotes grandes, actualizaciones estables.
                                                {% endif %}
                                            </small>
                                        </div>
                                        {% endif %}
                                    </div>
                                </div>
                                {% endif %}
                            </div>
                            {% endif %}
                        {% endfor %}
                    {% else %}
                        <!-- Fallback para datos sin mapeo de dicotomías -->
                        {% for d, m in data.metrics.items() %}
                            <div class="card">
                                <h3 style="margin-top: 0; color: #3a3a7a;">{{ d.replace('_', ' ').title() }}</h3>
                                <div class="metrics-grid">
                                {% for k, v in m.items() %}
                                    <div class="metric-item">
                                        <strong>{{ k.replace('_', ' ').title() }}</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ v }}</span>
                                    </div>
                                {% endfor %}
                                </div>
                            </div>
                        {% endfor %}
                    {% endif %}
                </div>

                <div class="section">
                    <h2><span class="emoji">🛠️</span>Model Configuration Details</h2>
                    <div class="explanation">
                        <strong>Optimized Parameters:</strong><br>
                        These are the best-performing configurations discovered through systematic testing.
                        Each dichotomy was individually optimized for maximum performance.
                    </div>
                    
                    {% for d, bm in data.best_model.items() %}
                        <div class="card">
                            <h3 style="margin-top: 0; color: #3a3a7a;">{{ d.replace('_', ' ').title().replace('Dichotomy', 'Binary Classifier') }}</h3>
                            {% if bm.get('best_runner') %}
                                {% set config = bm.best_runner %}
                                <div class="metrics-grid">
                                    {% if config.get('activation_fn') %}
                                    <div class="metric-item">
                                        <strong>Activation Function</strong><br>
                                        <span style="font-family: monospace;">{{ config.activation_fn|upper }}</span>
                                        <small style="display: block; color: #666;">Neural network activation type</small>
                                    </div>
                                    {% endif %}
                                    {% if config.get('num_experts') %}
                                    <div class="metric-item">
                                        <strong>Expert Networks</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ config.num_experts }}</span>
                                        <small style="display: block; color: #666;">Parallel neural networks</small>
                                    </div>
                                    {% endif %}
                                    {% if config.get('hidden_size') %}
                                    <div class="metric-item">
                                        <strong>Network Complexity</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ config.hidden_size }} neurons</span>
                                        <small style="display: block; color: #666;">Hidden layer size</small>
                                    </div>
                                    {% endif %}
                                    {% if config.get('n_epoch') %}
                                    <div class="metric-item">
                                        <strong>Training Iterations</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ config.n_epoch }} epochs</span>
                                        <small style="display: block; color: #666;">Learning cycles completed</small>
                                    </div>
                                    {% endif %}
                                    {% if config.get('n_batch') %}
                                    <div class="metric-item">
                                        <strong>Batch Size</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ config.n_batch }} samples</span>
                                        <small style="display: block; color: #666;">Data processed per update</small>
                                    </div>
                                    {% endif %}
                                    {% if config.get('drop_out') %}
                                    <div class="metric-item">
                                        <strong>Dropout Rate</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ (config.drop_out * 100)|round(1) }}%</span>
                                        <small style="display: block; color: #666;">Regularization strength</small>
                                    </div>
                                    {% endif %}
                                    {% if config.get('alpha') %}
                                    <div class="metric-item">
                                        <strong>Learning Rate (α)</strong><br>
                                        <span style="font-size: 1.2em; color: #007acc;">{{ config.alpha }}</span>
                                        <small style="display: block; color: #666;">Optimization speed</small>
                                    </div>
                                    {% endif %}
                                </div>
                                
                                {% if config.get('mode') or config.get('loss_fn') %}
                                <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 6px;">
                                    <strong>Optimization Details:</strong>
                                    {% if config.get('mode') %}
                                    <span style="margin-left: 10px;">Training Mode: <code>{{ config.mode|title }}</code></span>
                                    {% endif %}
                                    {% if config.get('loss_fn') %}
                                    <span style="margin-left: 10px;">Loss Function: <code>{{ config.loss_fn|title }}</code></span>
                                    {% endif %}
                                </div>
                                {% endif %}
                            {% else %}
                                <div style="padding: 15px; background: #f8f9fa; border-radius: 6px;">
                                    <p>Configuration details not available for this classifier.</p>
                                </div>
                            {% endif %}
                        </div>
                    {% endfor %}
                </div>

                <div class="section">
                    <h2><span class="emoji">💡</span>Interpretation & Recommendations</h2>
                    <div class="card">
                        <h3 style="margin-top: 0;">Key Findings:</h3>
                        {% if data.imbalance_ratio %}
                            {% set ratio = data.imbalance_ratio|float %}
                            {% if ratio <= 1.5 %}
                                <p>✅ <strong>Excellent balance:</strong> Your dataset has good representation across all classes, which typically leads to reliable model performance.</p>
                            {% elif ratio <= 3.0 %}
                                <p>⚠️ <strong>Moderate imbalance:</strong> Some classes are underrepresented. The model may perform better on majority classes.</p>
                            {% elif ratio <= 10.0 %}
                                <p>🚨 <strong>High imbalance:</strong> Significant class disparity detected. Special techniques were applied to improve minority class performance.</p>
                            {% else %}
                                <p>🔴 <strong>Extreme imbalance:</strong> Severe class disparity. Results should be interpreted carefully, especially for minority classes.</p>
                            {% endif %}
                        {% endif %}
                        
                        <h3>What This Means:</h3>
                        <ul>
                            <li>The model was trained using advanced techniques designed for {{ 'multiclass' if data.is_multiclass else 'binary' }} classification</li>
                            <li>{{ 'Multiple binary classifiers were combined to handle the multiclass problem' if data.is_multiclass else 'A single binary classifier was optimized for this two-class problem' }}</li>
                            <li>Performance metrics indicate how well the model can make predictions on new, unseen data</li>
                            {% if data.imbalance_ratio and data.imbalance_ratio|float > 3.0 %}
                            <li>Due to class imbalance, pay special attention to metrics for minority classes</li>
                            {% endif %}
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <h2><span class="emoji">🎯</span>Strategy Recommendations</h2>
                    <div class="explanation">
                        <strong>Optimization Analysis:</strong><br>
                        Based on the performance across all dichotomies, here are the recommended configurations
                        for future experiments and production deployment.
                    </div>
                    
                    {% if data.best_model and data.is_multiclass %}
                    <div class="card">
                        <h3 style="margin-top: 0; color: #3a3a7a;">📊 Performance Analysis Across Dichotomies</h3>
                        
                        {% set all_configs = [] %}
                        {% set performance_data = [] %}
                        
                        {% for d, bm in data.best_model.items() %}
                            {% if bm.get('best_runner') %}
                                {% set config = bm.best_runner %}
                                {% set _ = all_configs.append(config) %}
                                {% set metric_value = data.metrics.get(d, {}).get('accuracy', 'N/A') %}
                                {% if metric_value != 'N/A' %}
                                    {% set _ = performance_data.append({'dichotomy': d, 'accuracy': metric_value|float, 'config': config}) %}
                                {% endif %}
                            {% endif %}
                        {% endfor %}
                        
                        {% if performance_data %}
                        <div style="margin: 15px 0;">
                            <h4>🏆 Best Performing Dichotomy:</h4>
                            {% set best_performer = performance_data|max(attribute='accuracy') %}
                            <div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                <strong>{{ best_performer.dichotomy.replace('_', ' ').title() }}</strong><br>
                                <span style="font-size: 1.2em; color: #155724;">Accuracy: {{ "%.3f"|format(best_performer.accuracy) }}</span><br>
                                <small>This configuration achieved the highest performance</small>
                            </div>
                            
                            <h4>⚠️ Lowest Performing Dichotomy:</h4>
                            {% set worst_performer = performance_data|min(attribute='accuracy') %}
                            <div style="background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                <strong>{{ worst_performer.dichotomy.replace('_', ' ').title() }}</strong><br>
                                <span style="font-size: 1.2em; color: #721c24;">Accuracy: {{ "%.3f"|format(worst_performer.accuracy) }}</span><br>
                                <small>This may indicate class imbalance or difficult separation</small>
                            </div>
                        </div>
                        {% endif %}
                        
                        <h4>🔧 Recommended Configuration for Future Experiments:</h4>
                        {% if performance_data %}
                            {% set best_config = best_performer.config %}
                            {% if data.imbalance_ratio and data.imbalance_ratio|float > 3.0 %}
                                {% set worst_config = worst_performer.config %}
                                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px;">
                                    <p><strong>⚖️ High Imbalance Detected:</strong> Using configuration from lowest performer to handle difficult cases</p>
                                    <div class="metrics-grid" style="margin-top: 15px;">
                                        <div class="metric-item">
                                            <strong>Recommended Experts</strong><br>
                                            <span style="font-size: 1.2em;">{{ worst_config.get('num_experts', 'N/A') }}</span>
                                            <small style="display: block;">More experts for complex boundaries</small>
                                        </div>
                                        <div class="metric-item">
                                            <strong>Hidden Layer Size</strong><br>
                                            <span style="font-size: 1.2em;">{{ worst_config.get('hidden_size', 'N/A') }}</span>
                                            <small style="display: block;">Adequate complexity</small>
                                        </div>
                                        <div class="metric-item">
                                            <strong>Learning Rate (α)</strong><br>
                                            <span style="font-size: 1.2em;">{{ worst_config.get('alpha', 'N/A') }}</span>
                                            <small style="display: block;">Conservative learning</small>
                                        </div>
                                        <div class="metric-item">
                                            <strong>Dropout Rate</strong><br>
                                            <span style="font-size: 1.2em;">{{ (worst_config.get('drop_out', 0) * 100)|round(1) }}%</span>
                                            <small style="display: block;">Regularization strength</small>
                                        </div>
                                    </div>
                                </div>
                            {% else %}
                                <div style="background: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 8px;">
                                    <p><strong>📈 Balanced Dataset:</strong> Using best performer configuration for optimal results</p>
                                    <div class="metrics-grid" style="margin-top: 15px;">
                                        <div class="metric-item">
                                            <strong>Recommended Experts</strong><br>
                                            <span style="font-size: 1.2em;">{{ best_config.get('num_experts', 'N/A') }}</span>
                                            <small style="display: block;">Proven optimal count</small>
                                        </div>
                                        <div class="metric-item">
                                            <strong>Hidden Layer Size</strong><br>
                                            <span style="font-size: 1.2em;">{{ best_config.get('hidden_size', 'N/A') }}</span>
                                            <small style="display: block;">Optimal complexity</small>
                                        </div>
                                        <div class="metric-item">
                                            <strong>Learning Rate (α)</strong><br>
                                            <span style="font-size: 1.2em;">{{ best_config.get('alpha', 'N/A') }}</span>
                                            <small style="display: block;">Optimal learning speed</small>
                                        </div>
                                        <div class="metric-item">
                                            <strong>Dropout Rate</strong><br>
                                            <span style="font-size: 1.2em;">{{ (best_config.get('drop_out', 0) * 100)|round(1) }}%</span>
                                            <small style="display: block;">Proven regularization</small>
                                        </div>
                                    </div>
                                </div>
                            {% endif %}
                        {% endif %}
                        
                        <h4>📋 Parameter Interpretation:</h4>
                        <div style="margin: 15px 0;">
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                                <p><strong>Number of Experts:</strong> More experts can capture complex patterns but increase computational cost. 
                                Range: 10-50 (higher for complex datasets)</p>
                                
                                <p><strong>Hidden Layer Size:</strong> Controls model capacity. Larger sizes can learn more complex patterns 
                                but risk overfitting. Range: 20-100 neurons</p>
                                
                                <p><strong>Learning Rate (α):</strong> Controls how fast the model learns. Higher values learn faster but may miss optima. 
                                Range: 0.001-0.3 (lower for stable learning)</p>
                                
                                <p><strong>Dropout Rate:</strong> Prevents overfitting by randomly disabling neurons during training. 
                                Range: 0.1-0.5 (higher for more regularization)</p>
                            </div>
                        </div>
                    </div>
                    {% else %}
                    <div class="card">
                        <p>Strategy recommendations require completed experiments with performance metrics.</p>
                    </div>
                    {% endif %}
                </div>

                <div class="footer">
                    <p>Generated by <strong>Guillermo Villar Sánchez</strong> • All Rights Reserved</p>
                    <p>Report created on {{ current_time }}</p>
                    <p style="font-size: 0.8em; color: #aaa; margin-top: 10px;">© 2025 UC3M</p>
                </div>
            </div>
        </body>
        </html>
        ''')
        return template.render(data=self.data, current_time=current_time)
