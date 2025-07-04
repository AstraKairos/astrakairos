# Análisis del Proper Motion de Estrellas Dobles

Este proyecto consta de un script diseñado para analizar el movimiento angular total (`v_total`) (aka. "total observed angular velocity") de sistemas de estrellas dobles. El script automatiza el proceso de:
1.  Cargar una lista de estrellas desde un archivo CSV **exportado directamente desde Stelle Doppie**.
2.  Realizar web scraping de forma masiva y concurrente en la base de datos de [Stelle Doppie](https://www.stelledoppie.it/).
3.  Extraer datos astrométricos clave de la primera y última observación registrada para cada estrella.
4.  Calcular el movimiento angular total basándose en una fórmula astrométrica.
5.  Presentar un ranking de las estrellas con mayor movimiento angular.

Todo esto con el objetivo de facilitar marginalmente el proceso de selección de candidatos de observación.

El uso de `asyncio` y `aiohttp` permite procesar un gran número de estrellas en una fracción del tiempo que tomaría un script síncrono.

## Características Principales

-   **Procesamiento Concurrente:** Utiliza `asyncio` y `aiohttp` para realizar múltiples peticiones web en paralelo, optimizando significativamente el tiempo de ejecución.
-   **Integración Directa:** Trabaja directamente con los datos exportados desde la función "Export to Excel" del sitio `stelledoppie.it`.
-   **Web Scraping Robusto:** Implementa una lógica de scraping que maneja los diferentes tipos de respuesta del servidor, incluyendo redirecciones y páginas de resultados.
-   **Cálculo de la v_total:** Aplica una fórmula para calcular la velocidad angular total a partir del Ángulo de Posición (PA) y la Separación (Sep).
-   **Ranking de Resultados:** Ordena las estrellas procesadas de mayor a menor `v_total` y presenta un resumen claro.

## Metodología de Cálculo

La velocidad angular total (`v_total`), expresada en arcosegundos por año, se calcula para determinar el desplazamiento relativo de los componentes de un sistema doble a lo largo del tiempo. Muy útil en sistemas con movimiento lento, cuya órbita es muy difícil de computar. Ahora, un poco de la explicación de esta fórmula:

Las coordenadas polares de cada observación son (ρ, θ), donde:
-   **ρ (rho)** es la separación angular entre las estrellas (Sep).
-   **θ (theta)** es el ángulo de posición (PA).

Estas coordenadas se convierten a un sistema cartesiano (x, y) para calcular la distancia euclidiana entre la primera y la última observación:

-   `x = ρ * sin(θ)`
-   `y = ρ * cos(θ)`

La fórmula para la velocidad angular total es:

**v_total = √( (x₂ - x₁)² + (y₂ - y₁)² ) / (t₂ - t₁)**

Donde:
-   `(x₁, y₁)` son las coordenadas cartesianas en el tiempo `t₁` (primera observación).
-   `(x₂, y₂)` son las coordenadas cartesianas en el tiempo `t₂` (última observación).

Esta metodología está basada en el trabajo presentado en el siguiente paper:

> Rubina & Hilburn (2025). *Astrometric Observations Suggest that WDS 09483-5904 is not Physically Associated*. [Enlace al Paper en ResearchGate](https://www.researchgate.net/profile/Martin-Rubina/publication/391600528_Astrometric_Observations_Suggest_that_WDS_09483-5904_is_not_Physically_Associated/links/681e623fbd3f1930dd6f5669/Astrometric-Observations-Suggest-that-WDS-09483-5904-is-not-Physically-Associated.pdf)

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/TehMartinXz/stelle-doppie-filtering.git
    cd stelle-doppie-filtering
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    Crea un archivo `requirements.txt` con el siguiente contenido:
    ```
    pandas
    aiohttp
    beautifulsoup4
    ```    
    Y luego instálalo con pip:
    ```bash
    pip install -r requirements.txt
    ```

    o más directo,
    ```bash
    pip install pandas aiohttp beautifulsoup4
    ``

## Uso

El script está diseñado para trabajar con el archivo de datos que se obtiene directamente del sitio web de Stelle Doppie.

**Paso 1: Exportar los datos desde Stelle Doppie**

1.  Ve al sitio web [Stelle Doppie](https://www.stelledoppie.it/index2.php?section=2).
2.  Realiza la búsqueda o aplica los filtros que desees.
3.  En el menú de "ACTIONS" (Acciones) a la izquierda, haz clic en **"Export to Excel"**. Esto descargará el archivo .csv necesario para el análisis por este script.

**Paso 2: Ejecutar el Script**

Usa el archivo descargado como entrada para el script. Ejecútalo desde la línea de comandos, proporcionando la ruta al archivo y el número de estrellas que deseas procesar (se seleccionarán las que tengan más observaciones).

**Sintaxis:**
```bash
python main.py <archivo_csv_exportado> <numero_estrellas>
```

**Ejemplo:**
```bash
python main.py "Stelle Doppie.csv" 50
```
Este comando procesará las 50 estrellas con el mayor número de observaciones del archivo `Stelle Doppie.csv` que descargaste.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
