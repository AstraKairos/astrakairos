import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import time
import math
import re
import sys
from urllib.parse import urljoin, urlparse, parse_qs

def load_csv_data(csv_file):
    """
    Carga el archivo CSV con los datos de estrellas dobles
    """
    try:
        # Intentar diferentes delimitadores
        df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
        print(f"CSV cargado exitosamente. Filas: {len(df)}")
        return df
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        try:
            # Intentar con otro delimitador
            df = pd.read_csv(csv_file, delimiter=',', encoding='utf-8')
            print(f"CSV cargado con delimitador coma. Filas: {len(df)}")
            return df
        except Exception as e2:
            print(f"Error con delimitador coma: {e2}")
            return None

def clean_wds_name(wds_name):
    """
    Limpia el nombre WDS para usar en la URL
    """
    if pd.isna(wds_name) or wds_name == '':
        return None
    
    # Reemplazar espacios en la URL con +
    wds_clean = str(wds_name).strip().replace(' ', '+')
    return wds_clean

async def get_star_data(wds_name, session, max_retries=3):
    """
    Obtiene los datos de una estrella desde stelledoppie.it (versión asíncrona)
    """
    if not wds_name:
        return None
    
    wds_q = clean_wds_name(wds_name)
    if not wds_q:
        return None
    
    print(f"Buscando datos para: {wds_name} -> {wds_q}")
    
    # URL de búsqueda inicial
    search_url = f"https://www.stelledoppie.it/index2.php?cerca_database={wds_q}&azione=cerca_testo_database&nofilter=1&section=2&ricerca=+Search+"
    

    # Por motivos que no me molestaré en entender, StelleDoppie a veces redirige a una URL con un formato incorrecto.
    # Haciendo pruebas, encontré algunos de estos casos. Este for intenta solucionar los errores de scraping.
    for attempt in range(max_retries):
        try:
            # Hacer la búsqueda inicial
            async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                content = await response.text()
                
                # Verificar si hay redirección o si necesitamos buscar el ID
                final_url = str(response.url)
                
                # Si la URL final contiene 'iddoppia', extraer los datos
                if 'iddoppia=' in final_url:
                    # Corregir la URL si tiene el formato incorrecto
                    if 'index2.php&iddoppia=' in final_url:
                        final_url = final_url.replace('index2.php&iddoppia=', 'index2.php?iddoppia=')
                    
                    print(f"Redirigido a: {final_url}")
                    
                    # Hacer nueva request con la URL corregida
                    if final_url != str(response.url):
                        async with session.get(final_url, timeout=aiohttp.ClientTimeout(total=10)) as corrected_response:
                            corrected_response.raise_for_status()
                            content = await corrected_response.text()
                    
                    return extract_star_data(content, wds_name)
                
                # Si no hay redirección directa, buscar en la página de resultados
                soup = BeautifulSoup(content, 'html.parser')
                
                # Buscar enlaces que contengan iddoppia
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if 'iddoppia=' in href:
                        full_url = urljoin('https://www.stelledoppie.it/', href)
                        
                        # Filtrar enlaces que no sean de informes/reportes
                        if 'report' in href or 'section=4' in href:
                            continue
                        
                        print(f"Encontrado enlace: {full_url}")
                        
                        # Obtener la página de detalles
                        async with session.get(full_url, timeout=aiohttp.ClientTimeout(total=10)) as detail_response:
                            detail_response.raise_for_status()
                            detail_content = await detail_response.text()
                            
                            return extract_star_data(detail_content, wds_name)
                
                # Si no encontramos enlaces directos, buscar el primer enlace válido
                for link in links:
                    href = link['href']
                    if 'iddoppia=' in href and 'report' not in href:
                        full_url = urljoin('https://www.stelledoppie.it/', href)
                        print(f"Intentando enlace alternativo: {full_url}")
                        
                        # Obtener la página de detalles
                        async with session.get(full_url, timeout=aiohttp.ClientTimeout(total=10)) as detail_response:
                            detail_response.raise_for_status()
                            detail_content = await detail_response.text()
                            
                            result = extract_star_data(detail_content, wds_name)
                            if result and any(result[key] is not None for key in ['date_first', 'pa_first', 'sep_first']):
                                return result
                
                print(f"No se encontraron datos para {wds_name}")
                return None
                
        except asyncio.TimeoutError:
            print(f"Timeout en intento {attempt + 1} para {wds_name}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial
            else:
                print(f"Falló después de {max_retries} intentos")
                return None
        except aiohttp.ClientError as e:
            print(f"Error de cliente en intento {attempt + 1} para {wds_name}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial
            else:
                print(f"Falló después de {max_retries} intentos")
                return None
        except Exception as e:
            print(f"Error inesperado para {wds_name}: {e}")
            return None

def extract_star_data(html_content, wds_name):
    """
    Extrae los datos específicos de la página de detalles
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    data = {
        'wds_name': wds_name,
        'date_first': None,
        'pa_first': None,
        'sep_first': None,
        'date_last': None,
        'pa_last': None,
        'sep_last': None,
        'obs': None
    }
    
    try:
        text = soup.get_text()
        
        # Patrones para buscar los datos
        patterns = [
            # Patrones originales
            (r'Date first\s*(\d+)', 'date_first'),
            (r'Pa first\s*([\d.-]+)', 'pa_first'),
            (r'Sep first\s*([\d.-]+)', 'sep_first'),
            (r'Date last\s*(\d+)', 'date_last'),
            (r'Pa last\s*([\d.-]+)', 'pa_last'),
            (r'Sep last\s*([\d.-]+)', 'sep_last'),
            (r'Obs\s*(\d+)', 'obs'),
            
            # Patrones alternativos con diferentes espacios y formatos
            (r'Date\s+first\s*(\d+)', 'date_first'),
            (r'Pa\s+first\s*([\d.-]+)', 'pa_first'),
            (r'Sep\s+first\s*([\d.-]+)', 'sep_first'),
            (r'Date\s+last\s*(\d+)', 'date_last'),
            (r'Pa\s+last\s*([\d.-]+)', 'pa_last'),
            (r'Sep\s+last\s*([\d.-]+)', 'sep_last'),
            
            # Patrones con números después de los dos puntos
            (r'Date first\s*:\s*(\d+)', 'date_first'),
            (r'Pa first\s*:\s*([\d.-]+)', 'pa_first'),
            (r'Sep first\s*:\s*([\d.-]+)', 'sep_first'),
            (r'Date last\s*:\s*(\d+)', 'date_last'),
            (r'Pa last\s*:\s*([\d.-]+)', 'pa_last'),
            (r'Sep last\s*:\s*([\d.-]+)', 'sep_last'),
        ]
        
        # Aplicar patrones
        for pattern, field in patterns:
            if data[field] is None:  # Solo si no hemos encontrado el valor
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        if field in ['date_first', 'date_last', 'obs']:
                            data[field] = int(match.group(1))
                        else:
                            data[field] = float(match.group(1))
                    except ValueError:
                        continue
        
        # Buscar en tablas HTML si los patrones de texto no funcionan
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    cell_text = cells[0].get_text().strip().lower()
                    value_text = cells[1].get_text().strip()
                    
                    # Mapear campos de la tabla
                    if 'date first' in cell_text and data['date_first'] is None:
                        try:
                            data['date_first'] = int(re.search(r'(\d+)', value_text).group(1))
                        except:
                            pass
                    elif 'pa first' in cell_text and data['pa_first'] is None:
                        try:
                            data['pa_first'] = float(re.search(r'([\d.-]+)', value_text).group(1))
                        except:
                            pass
                    elif 'sep first' in cell_text and data['sep_first'] is None:
                        try:
                            data['sep_first'] = float(re.search(r'([\d.-]+)', value_text).group(1))
                        except:
                            pass
                    elif 'date last' in cell_text and data['date_last'] is None:
                        try:
                            data['date_last'] = int(re.search(r'(\d+)', value_text).group(1))
                        except:
                            pass
                    elif 'pa last' in cell_text and data['pa_last'] is None:
                        try:
                            data['pa_last'] = float(re.search(r'([\d.-]+)', value_text).group(1))
                        except:
                            pass
                    elif 'sep last' in cell_text and data['sep_last'] is None:
                        try:
                            data['sep_last'] = float(re.search(r'([\d.-]+)', value_text).group(1))
                        except:
                            pass
        
        # Contar cuántos datos válidos encontramos
        valid_data = sum(1 for k, v in data.items() if v is not None and k != 'wds_name')
        print(f"Encontrados {valid_data} campos válidos de 7 posibles")
        
        return data
        
    except Exception as e:
        print(f"Error extrayendo datos para {wds_name}: {e}")
        return data

def calculate_vtotal(data):
    """
    Calcula v_total usando la fórmula proporcionada
    """
    try:
        # Extraer valores necesarios
        p1 = data['pa_first']  # Pa first (θ1)
        p2 = data['pa_last']   # Pa last (θ2)
        s1 = data['sep_first'] # Sep first (ρ1)
        s2 = data['sep_last']  # Sep last (ρ2)
        t1 = data['date_first'] # Date first
        t2 = data['date_last']  # Date last
        
        # Verificar que tenemos todos los datos necesarios
        if None in [p1, p2, s1, s2, t1, t2]:
            print(f"Datos incompletos para {data['wds_name']}: p1={p1}, p2={p2}, s1={s1}, s2={s2}, t1={t1}, t2={t2}")
            return None
        
        # Convertir ángulos a radianes
        theta1_rad = math.radians(p1)
        theta2_rad = math.radians(p2)
        
        # Velocidad total observada:
        # v_total = sqrt((ρ2*sin(θ2) - ρ1*sin(θ1))² + (ρ2*cos(θ2) - ρ1*cos(θ1))²) / (t2 - t1)
        
        # Calcular componentes
        x1 = s1 * math.sin(theta1_rad)  # ρ1 * sin(θ1)
        y1 = s1 * math.cos(theta1_rad)  # ρ1 * cos(θ1)
        x2 = s2 * math.sin(theta2_rad)  # ρ2 * sin(θ2)
        y2 = s2 * math.cos(theta2_rad)  # ρ2 * cos(θ2)
        
        # Calcular el numerador
        numerator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calcular el denominador
        denominator = t2 - t1
        
        if denominator == 0:
            print(f"División por cero para {data['wds_name']} (misma fecha)")
            return None
        
        # Calcular v_total
        v_total = numerator / denominator
        
        print(f"Cálculo para {data['wds_name']}:")
        print(f"  ρ1={s1}, θ1={p1}° -> x1={x1:.3f}, y1={y1:.3f}")
        print(f"  ρ2={s2}, θ2={p2}° -> x2={x2:.3f}, y2={y2:.3f}")
        print(f"  Δt={denominator} años")
        print(f"  v_total={v_total:.6f} arcsec/año")
        
        return v_total
        
    except Exception as e:
        print(f"Error calculando v_total para {data['wds_name']}: {e}")
        return None

async def process_star(row, session, semaphore):
    """
    Procesa una estrella individual con limitación de concurrencia
    """
    async with semaphore:
        print(f"\n--- Procesando {row['wds_name']} (obs: {row['obs']}) ---")
        
        # Obtener datos del webscraping
        star_data = await get_star_data(row['wds_name'], session)
        
        if star_data and all(star_data[key] is not None for key in ['date_first', 'pa_first', 'sep_first', 'date_last', 'pa_last', 'sep_last']):
            # Calcular v_total
            v_total = calculate_vtotal(star_data)
            
            if v_total is not None:
                result = {
                    'wds_name': row['wds_name'],
                    'obs_csv': row['obs'],
                    'obs_web': star_data['obs'],
                    'date_first': star_data['date_first'],
                    'date_last': star_data['date_last'],
                    'pa_first': star_data['pa_first'],
                    'pa_last': star_data['pa_last'],
                    'sep_first': star_data['sep_first'],
                    'sep_last': star_data['sep_last'],
                    'v_total': v_total
                }
                
                print(f"✓ v_total calculado: {v_total:.6f}")
                return result
            else:
                print("✗ Error calculando v_total")
        else:
            print("✗ Datos incompletos del webscraping")
        
        return None

async def main():
    if len(sys.argv) != 3:
        print("Uso: python script.py <archivo_csv> <numero_estrellas>")
        print("Ejemplo: python script.py stars_data.csv 10")
        return
    
    csv_file = sys.argv[1]
    try:
        n_best = int(sys.argv[2])
    except ValueError:
        print("Error: El número de estrellas debe ser un entero")
        return
    
    # Cargar datos del CSV
    print(f"Cargando datos del CSV: {csv_file}")
    df = load_csv_data(csv_file)
    
    if df is None:
        print("No se pudo cargar el archivo CSV")
        return
    
    # Filtrar y ordenar por obs (descendente)
    df_filtered = df[df['wds_name'].notna() & (df['wds_name'] != '')].copy()
    df_filtered = df_filtered.sort_values('obs', ascending=False)
    
    # Tomar los N mejores
    df_best = df_filtered.head(n_best)
    
    print(f"Procesando las {len(df_best)} estrellas con más observaciones...")
    
    # Configurar sesión HTTP asíncrona
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=30)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # "Semáforo" para limitar concurrencia (máximo 5 requests simultáneos)
    # Ojalá el autor de StelleDoppie no me odie... lol
    # me sorry bro
    semaphore = asyncio.Semaphore(5)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
        # Crear tareas para todas las estrellas
        tasks = []
        for index, row in df_best.iterrows():
            task = process_star(row, session, semaphore)
            tasks.append(task)
        
        # Ejecutar todas las tareas concurrentemente
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filtrar resultados válidos
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    
    # Mostrar resultados finales
    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    
    for result in valid_results:
        print(f"\nEstrella: {result['wds_name']}")
        print(f"Obs (CSV): {result['obs_csv']}, Obs (Web): {result['obs_web']}")
        print(f"Período: {result['date_first']} - {result['date_last']}")
        print(f"PA: {result['pa_first']}° - {result['pa_last']}°")
        print(f"Sep: {result['sep_first']}\" - {result['sep_last']}\"")
        print(f"v_total: {result['v_total']:.6f} arcsec/year")
    
    # Ordenar por v_total y mostrar ranking
    if valid_results:
        results_sorted = sorted(valid_results, key=lambda x: x['v_total'], reverse=True)
        
        print("\n" + "="*80)
        print("RANKING POR v_total (MAYOR A MENOR)")
        print("="*80)
        
        for i, result in enumerate(results_sorted, 1):
            print(f"{i:2d}. {result['wds_name']:<20} v_total = {result['v_total']:.6f} arcsec/año")
        
        print(f"\nProcesadas {len(valid_results)} estrellas exitosamente de {len(df_best)} intentos.")
        print(f"La estrella con mayor v_total es: {results_sorted[0]['wds_name']} ({results_sorted[0]['v_total']:.6f} arcsec/año)")
    else:
        print("\nNo se pudieron procesar estrellas exitosamente.")

if __name__ == "__main__":
    
    # Flexear la rapidez
    inicio = time.time()
    
    # Ejecutar la función main asíncrona
    asyncio.run(main())
    
    fin = time.time()
    print(f"\nTiempo total de ejecución: {fin - inicio:.2f} segundos")