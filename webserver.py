from flask import Flask, request, render_template, send_from_directory
from threading import Thread, Timer
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
from data_plotting import get_options_data
import re
from zoneinfo import ZoneInfo
from os import environ, makedirs
import os, shutil


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Crear un event loop global para manejar las tareas asíncronas


HTML_TEMPLATE = "index.html"

def delete_plots_directory():
    try:
        shutil.rmtree('plots')
        # Recrear el directorio vacío para la próxima solicitud
        os.makedirs('plots', exist_ok=True)
    except Exception as e:
        print(f"Error deleting directory 'plots/': {str(e)}")

@app.route('/', methods=['GET', 'POST'])
async def index():
    try:
        if request.method == 'POST':
            ticker = request.form['ticker']
            exp = request.form['exp']
            greek = request.form['greek']
            
        #try:
            # Validar entradas
            if not ticker.isalnum() or len(ticker) > 10:
                return render_template(HTML_TEMPLATE, error="Ticker not valid", greek=greek)
            if greek not in ['delta', 'gamma', 'vanna', 'charm']:
                return render_template(HTML_TEMPLATE, error="Greek not valid", greek=greek)
            if not re.match(r'^\d{4}-\d{2}-\d{2}$|^(0dte|1dte|weekly|monthly|opex|all)$', exp):
                return render_template(HTML_TEMPLATE, error="Formato de expiración inválido (YYYY-MM-DD o 0dte/1dte/monthly/opex/all)", greek=greek)
            
            # Llamar a get_options_data de forma asíncrona
            #print(f"Calling get_options_data with ticker={ticker}, exp={exp}, greek={greek}")
            filenames = await get_options_data(ticker, exp, greek)
            
            # Verificar y loggear los paths de las imágenes
            images = []

            for i in filenames:
                for filename in i:  
                    # Normalizar el path para evitar duplicados o prefijos incorrectos
                    normalized_filename = os.path.normpath(filename)
                    # Asegurarse de que el path sea relativo a 'plots'
                    #if not normalized_filename.startswith('plots/'):
                    #    normalized_filename = os.path.join('plots', normalized_filename.lstrip('/'))
                    #print("normalized_filename:", normalized_filename)
                    if os.path.exists(normalized_filename):
                        # Crear el path para el navegador
                        web_path = f"/plots/{filename.lstrip('plots/').lstrip('/')}"
                        images.append(web_path)
                    else:
                        print(f"Image not found: {filename}")
            
            if not images:
                print("No images found for the request")
                return render_template(HTML_TEMPLATE, error="Error plot not found, retry in a minute", greek=greek)
            
            response = render_template(HTML_TEMPLATE, images=images, greek=greek)
            Timer(5.0, delete_plots_directory).start()
            
            return response
        
            #except Exception as e:
            #    print(f"Error processing request: ", e)
            #    return render_template(HTML_TEMPLATE, error=f"Error al cargar los gráficos: {str(e)}", greek=greek)
        
        return render_template(HTML_TEMPLATE, greek='')
    except Exception as e:
        print("Error in index()")
        print(e)
        return render_template(HTML_TEMPLATE, error="Ticker invalid or failed download, retry in a minute", greek=greek)


@app.route('/plots/<path:filename>')
def serve_plot(filename):
    try:
        #print(f"Attempting to serve file: plots/: {filename}")
        return send_from_directory("plots", filename)
    except Exception as e:
        print(f"Error serving file plots/{filename}: {str(e)}")
        return "Archivo no encontrado", 404



if __name__ == "__main__":
    app.run(debug=False)
