# Predicción de cancelaciones de reservas de hotel

Proyecto del segundo corte de IA006 (Inteligencia Artificial). Clasificación binaria supervisada sobre el dataset *Hotel Booking Demand* (Antonio, Almeida y Nunes, 2019; CC BY 4.0): dada una reserva, estimar si terminará cancelada.

Autores: Rafael Antonio Arias Monsalve y Cristian Julián Peña.

## Estructura

| Archivo | Descripción |
|---|---|
| `IA006_Proyecto_Cancelaciones.ipynb` | Cuaderno de Colab: limpieza, EDA, ingeniería de variables, PCA y LDA, modelado, evaluación y conclusiones |
| `model.py` | Funciones de limpieza, ingeniería de variables, preprocesamiento, validación cruzada, entrenamiento y predicción |
| `app.py` | Aplicación de Streamlit |
| `requirements.txt` | Dependencias |
| `models/` | Modelo entrenado (`best_model.joblib`), metadatos para el formulario y tabla de resultados |
| `data/muestra_reservas_test.csv` | 1.000 reservas del conjunto de prueba para la demostración |

## Cuaderno

Abrir el `.ipynb` en Google Colab y ejecutar todo. La primera celda escribe `model.py`; la última guarda `models/` y `data/`. Tarda unos 20 a 25 minutos por la búsqueda de hiperparámetros (nueve combinaciones de modelo y reducción).

## Ejecutar la aplicación en local (VS Code)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

(En Mac/Linux: `source .venv/bin/activate`.) El modelo se guardó con scikit-learn 1.9.1, por eso se fija esa versión.

## Despliegue en Streamlit Community Cloud

1. Subir a GitHub `app.py`, `model.py`, `requirements.txt`, `models/` y `data/` (no subir `.venv`).
2. En https://share.streamlit.io elegir el repositorio y la rama, y `app.py` como archivo principal.
3. Desplegar.

## Cómo se conectan modelo y aplicación

`app.py` importa de `model.py` las mismas funciones que usa el cuaderno, así cada reserva se transforma igual que en el entrenamiento. `models/best_model.joblib` guarda el preprocesamiento ya ajustado (límites del IQR, escalado, categorías frecuentes) junto con el modelo; `predict_risk` aplica ambos y devuelve la probabilidad de cancelación.
