# Predicción de cancelaciones de reservas de hotel

Proyecto del segundo corte de IA006 (Inteligencia Artificial). Clasificación binaria supervisada con el dataset Hotel Booking Demand (Antonio, Almeida y Nunes, 2019): dada una reserva, predecir si se va a cancelar.

Autores: Rafael Antonio Arias Monsalve y Cristian Julián Peña.

Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand (se carga desde el espejo de TidyTuesday).

## Archivos

| Archivo | Descripción |
|---|---|
| `IA006_Proyecto_Cancelaciones.ipynb` | Notebook de Colab: limpieza, EDA, ingeniería de variables, PCA y LDA, modelos, evaluación y conclusiones |
| `app.py` | Aplicación de Streamlit |
| `model.py` | Funciones que usa la app para preparar las reservas y predecir |
| `models/modelo.joblib` | Mejor modelo (Random Forest) con el escalador y la información para preparar los datos |
| `models/resultados.csv` | Métricas de los 9 modelos evaluados |
| `models/importancias.csv` | Variables más importantes del Random Forest |
| `data/muestra_test.csv` | 1.000 reservas del conjunto de prueba para la demostración |
| `requirements.txt` | Librerías |

## Notebook

Se puede abrir directamente en Google Colab, ya tiene las salidas. Si se vuelve a ejecutar completo tarda unos 20 minutos por la búsqueda de hiperparámetros, y al final genera de nuevo las carpetas `models/` y `data/`.

## Ejecutar la app

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows se activa con `.venv\Scripts\activate`. Se fija `scikit-learn==1.6.1` porque es la versión con la que se guardó el modelo.

## Cómo se conecta el modelo con la app

El notebook guarda en `models/modelo.joblib` el modelo entrenado junto con el `StandardScaler`, las columnas del one-hot, los países y agentes más frecuentes y los límites del IQR de `adr`. `model.py` repite los mismos pasos del notebook (limpieza, variables nuevas, logaritmo, one-hot y escalado) y `app.py` lo usa para predecir una reserva escrita en el formulario o varias reservas de un CSV.
