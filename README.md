# compare_exels - Streamlit
Aplicación web para comparar dos archivos Excel y listar diferencia, duplicados, etc

## Características
- Interfaz web con Streamlit
- Comparación de archivos Excel por clave
- Mapeo de claves entre Excels
- Comparación 1:1 o por selección de campos
- Mapeo de campos a comparar 
- Generación de reportes en Excel

## Instalación

### Instalación con virtual env en Windows

1. Abre una terminal (CMD o PowerShell)
2. Navega hasta la carpeta de tu proyecto:
   ```bash
   cd ruta/a/tu/proyecto
   ```
3. Crea el entorno virtual:
   ```bash
   python -m venv venv
   ```
4. Activa el entorno virtual:
   ```bash
   venv\Scripts\activate
   ```
5. Confirma que está activado (verás (venv) al inicio de la línea de comandos)


-----------------------------

1. Clona el repositorio

2. Instala las dependencias(idealmente con venv activado):
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Inicia la aplicación:
   ```bash
   streamlit run app.py
   ```

2. Accede a la aplicación en tu navegador (por defecto en http://localhost:8501)

3. Sube los dos archivos Excel que deseas comparar

4. Selecciona las columnas clave para la comparación

5. Visualiza las diferencias y descarga el reporte en Excel

## Estructura del Proyecto

app.py - Código principal de la aplicación
requirements.txt - Dependencias del proyecto
.env - Variables de entorno
