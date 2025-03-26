# TFG-
Hay que renombrar New_Code por SmartNOC


Modelo q no tiene pq ser MLP;

Trabajar con datos más pequeños (otros datasets). 

El 

Probar otros metodos de optimizacion del LBFGS. 

Trabajando en un codigo unico por su parte, LBFGS. Yo extracción de parámetros, a un high-level. SIEMPRE aplicando algo del rollo de la tanh comprimida q teniamos. La función que minimizamos está puesta en el loss,

GridSearchCV


Apuntes primeras runs small: 

Para analizar small primero - 1 Analisis de parámetros en teoría, 2 Analisis de la implementación en código, 3 analisis de base de datos, 4 analisis coherencia de todos los anteriores. 

Una vez hecho esto, (TODO after 3 y 4) pensar y proponer nuevas bases de datos. Analisis previo a resultados, y repetir analisis de un dataset (1-4) anteriores. 

Con esto hecho dos veces, reunión y seguimos.


LSEnsemble

Hay muchas cosas q comentar

Por ejemplo, weights es clave. Todo el estudio de desbalances (Y mas adelante habria q entrar en estrategias)
Es absolutamente relativo a los desbalances q tengas. El estudio se tiene que ajustar del todo a los IR que se tengan!!

Los runs iterativos por favor!! Hay q hacer un estudio de iteraciones. Como? Q es mejor? 

Ya tengo alguna idea..

Idea principal - Runs y parametros en entorno reducido. De ahí extrapolar, habiendo sacado info. 

Comparación teoría, resultados de "laboratorio" y resultados de verdad. Implicaciones principalmente y con enfoque a... IR!! Desbalances.

Tener en cuenta las distintas estrategias. Lista: 

Label switching, SMOTE, QRBC (a 0, pero está). 


labelswitching (nuevo) sacar parámetro de trabajadores mirar data_loading (yo creo q no aplica mucho), extraer al yaml  