import json
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from model import RAW_COLS, clean_uploaded, evaluate_threshold, load_bundle, predict_risk

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

st.set_page_config(page_title="Riesgo de cancelación", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle("models/best_model.joblib")


@st.cache_data
def get_meta():
    with open("models/app_metadata.json", encoding="utf-8") as f:
        return json.load(f)


bundle = get_bundle()
meta = get_meta()
options = meta["opciones"]
numeric = meta["numericas"]
base_rate = meta["tasa_cancelacion_base"]

st.title("Riesgo de cancelación de reservas de hotel")
st.write("Proyecto IA006 · Dataset Hotel Booking Demand · Clasificación binaria supervisada")

st.header("Datos de la reserva")
col1, col2, col3 = st.columns(3)

with col1:
    hotel = st.selectbox("Hotel", options["hotel"])
    arrival = st.date_input("Fecha de llegada", value=date.today() + timedelta(days=60))
    lead_time = st.slider("Días de antelación (lead time)", 0, int(numeric["lead_time"]["p99"]) + 50, 60)
    weekend_nights = st.number_input("Noches de fin de semana", 0, 10, 1)
    week_nights = st.number_input("Noches entre semana", 0, 25, 2)
    meal = st.selectbox("Régimen de comidas", options["meal"], index=options["meal"].index("BB"))
    room = st.selectbox("Tipo de habitación reservada", options["reserved_room_type"],
                        index=options["reserved_room_type"].index("A"))

with col2:
    adults = st.number_input("Adultos", 1, 10, 2)
    children = st.number_input("Niños", 0, 5, 0)
    babies = st.number_input("Bebés", 0, 3, 0)
    customer_type = st.selectbox("Tipo de cliente", options["customer_type"],
                                 index=options["customer_type"].index("Transient"))
    repeated = st.checkbox("Huésped repetido")
    previous_cancellations = st.number_input("Cancelaciones previas", 0, 30, 0)
    previous_ok = st.number_input("Reservas previas no canceladas", 0, 80, 0)
    country = st.selectbox("País de origen", meta["paises_frecuentes"] + ["Otro país"])

with col3:
    market_segment = st.selectbox("Segmento de mercado", options["market_segment"],
                                  index=options["market_segment"].index("Online TA"))
    channel = st.selectbox("Canal de distribución", options["distribution_channel"],
                           index=options["distribution_channel"].index("TA/TO"))
    deposit_type = st.selectbox("Tipo de depósito", options["deposit_type"])
    agent = st.selectbox("Agente", meta["agentes_frecuentes"] + ["Otro agente"])
    adr = st.number_input("Tarifa diaria (adr)", 0.0, 500.0, float(round(numeric["adr"]["mediana"])), step=5.0)
    booking_changes = st.number_input("Cambios a la reserva", 0, 20, 0)
    waiting_days = st.number_input("Días en lista de espera", 0, 400, 0)
    parking = st.number_input("Parqueaderos requeridos", 0, 8, 0)
    requests = st.number_input("Solicitudes especiales", 0, 5, 0)

booking = pd.DataFrame([{
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_year": arrival.year,
    "arrival_date_month": MONTHS[arrival.month - 1],
    "arrival_date_week_number": int(arrival.isocalendar()[1]),
    "arrival_date_day_of_month": arrival.day,
    "stays_in_weekend_nights": weekend_nights,
    "stays_in_week_nights": week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "country": "XXX" if country == "Otro país" else country,
    "market_segment": market_segment,
    "distribution_channel": channel,
    "is_repeated_guest": int(repeated),
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": previous_ok,
    "reserved_room_type": room,
    "booking_changes": booking_changes,
    "deposit_type": deposit_type,
    "agent": "9999" if agent == "Otro agente" else agent,
    "days_in_waiting_list": waiting_days,
    "customer_type": customer_type,
    "adr": adr,
    "required_car_parking_spaces": parking,
    "total_of_special_requests": requests,
}])

st.header("Predicción")
risk = float(predict_risk(booking, bundle)[0])

c1, c2 = st.columns(2)
c1.metric("Riesgo de cancelación", f"{100 * risk:.1f}%",
          delta=f"{100 * (risk - base_rate):+.1f} puntos frente al promedio ({100 * base_rate:.1f}%)",
          delta_color="inverse")
if risk >= 0.5:
    c2.error("Cancelación probable: el modelo clasifica esta reserva como cancelada.")
else:
    c2.success("Cancelación poco probable: el modelo clasifica esta reserva como no cancelada.")
st.progress(min(max(risk, 0.0), 1.0))
st.caption("El puntaje sirve para ordenar reservas por riesgo; no es una probabilidad calibrada. El umbral de decisión es 50%.")

st.subheader("¿Qué cambiaría el riesgo?")
g1, g2, g3 = st.columns(3)

days = list(range(0, int(numeric["lead_time"]["p99"]) + 50, 15))
copies = pd.concat([booking] * len(days), ignore_index=True)
copies["lead_time"] = days
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(days, predict_risk(copies, bundle) * 100)
ax.scatter([lead_time], [risk * 100], color="red", zorder=3, label="Reserva actual")
ax.axhline(base_rate * 100, color="gray", ls="--")
ax.set_xlabel("Días de antelación")
ax.set_ylabel("Riesgo (%)")
ax.set_title("Riesgo según la antelación")
ax.legend()
g1.pyplot(fig)

deposits = options["deposit_type"]
copies = pd.concat([booking] * len(deposits), ignore_index=True)
copies["deposit_type"] = deposits
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.barh(deposits, predict_risk(copies, bundle) * 100)
ax.axvline(base_rate * 100, color="gray", ls="--")
ax.set_xlabel("Riesgo (%)")
ax.set_title("Riesgo según el tipo de depósito")
g2.pyplot(fig)

segments = options["market_segment"]
copies = pd.concat([booking] * len(segments), ignore_index=True)
copies["market_segment"] = segments
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.barh(segments, predict_risk(copies, bundle) * 100)
ax.axvline(base_rate * 100, color="gray", ls="--")
ax.set_xlabel("Riesgo (%)")
ax.set_title("Riesgo según el segmento de mercado")
g3.pyplot(fig)
st.caption("La línea punteada marca la tasa de cancelación promedio del dataset.")

st.header("Modelo en uso")
st.write(f"{meta['modelo']} · {meta['hiperparametros']}")
cols = st.columns(len(meta["metricas_test"]))
for col, (name, value) in zip(cols, meta["metricas_test"].items()):
    col.metric(name.replace(" test", ""), f"{value:.3f}")
st.caption("Métricas en el conjunto de prueba (20% de los datos, no usado para entrenar ni elegir el modelo).")

st.subheader("Comparación de los nueve modelos evaluados")
results = pd.read_csv("models/resultados_modelos.csv")
st.dataframe(results[["Modelo", "Reducción", "F1 validación (CV)", "Accuracy", "Precision", "Recall", "F1 test", "ROC-AUC"]])

importances = pd.Series(meta["importancias"]).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(importances.index, importances.values)
ax.set_title("Random Forest: importancia de variables")
st.pyplot(fig)

st.header("Puntuar muchas reservas")
source = st.radio("Origen de los datos", ["Muestra del conjunto de prueba", "Subir un CSV"])

if source == "Subir un CSV":
    file = st.file_uploader("CSV con las columnas del dataset Hotel Booking Demand", type=["csv"])
    if file is None:
        st.info("Esperando un archivo.")
        st.stop()
    df = pd.read_csv(file, keep_default_na=False, na_values=[""])
else:
    df = pd.read_csv("data/muestra_reservas_test.csv", keep_default_na=False, na_values=[""])
    amount = st.slider("Cantidad de reservas", 50, len(df), 500, step=50)
    df = df.head(amount)

missing = [c for c in RAW_COLS if c not in df.columns]
if missing:
    st.error("Faltan columnas: " + ", ".join(missing))
    st.stop()

clean = clean_uploaded(df)
scores = predict_risk(clean, bundle)
threshold = st.slider("Umbral de decisión", 0.20, 0.80, 0.50, step=0.05)

result = clean.copy()
result["riesgo"] = scores
result["prediccion"] = np.where(scores >= threshold, "Cancelada", "Se mantiene")

if "is_canceled" in df.columns:
    metrics, pred = evaluate_threshold(df["is_canceled"].astype(int), scores, threshold)
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ConfusionMatrixDisplay.from_predictions(df["is_canceled"].astype(int), pred,
                                            display_labels=["Se mantiene", "Cancelada"], colorbar=False, ax=ax)
    ax.set_title("Matriz de confusión")
    ax.grid(False)
    st.pyplot(fig)
else:
    st.metric("Reservas marcadas como cancelación probable", f"{int((scores >= threshold).sum())} de {len(df)}")

st.subheader("Reservas ordenadas por riesgo")
columns = ["hotel", "lead_time", "deposit_type", "market_segment", "country", "adr", "riesgo", "prediccion"]
st.dataframe(result[columns].sort_values("riesgo", ascending=False))
