import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from model import load_model, predict

MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December']

st.set_page_config(page_title='Cancelaciones de reservas', layout='wide')

@st.cache_resource
def get_model():
    return load_model('models/modelo.joblib')

art = get_model()
options = art['options']

st.title('Prediccion de cancelaciones de reservas de hotel')
st.write('Proyecto IA006 - Dataset Hotel Booking Demand - Clasificacion binaria supervisada')

st.header('Datos de la reserva')
col1, col2, col3 = st.columns(3)

with col1:
    hotel = st.selectbox('Hotel', options['hotel'])
    arrival = st.date_input('Fecha de llegada', date(2026, 12, 15))
    lead_time = st.slider('Dias de antelacion (lead time)', 0, 500, 60)
    weekend_nights = st.number_input('Noches de fin de semana', 0, 10, 1)
    week_nights = st.number_input('Noches entre semana', 0, 20, 2)
    meal = st.selectbox('Regimen de comidas', options['meal'])
    room = st.selectbox('Tipo de habitacion', options['reserved_room_type'])

with col2:
    adults = st.number_input('Adultos', 1, 10, 2)
    children = st.number_input('Ninos', 0, 5, 0)
    babies = st.number_input('Bebes', 0, 5, 0)
    customer_type = st.selectbox('Tipo de cliente', options['customer_type'])
    repeated = st.checkbox('Huesped repetido')
    previous_cancellations = st.number_input('Cancelaciones previas', 0, 30, 0)
    previous_ok = st.number_input('Reservas previas no canceladas', 0, 80, 0)

with col3:
    market_segment = st.selectbox('Segmento de mercado', options['market_segment'])
    channel = st.selectbox('Canal de distribucion', options['distribution_channel'])
    deposit_type = st.selectbox('Tipo de deposito', options['deposit_type'])
    country = st.selectbox('Pais', art['top_country'] + ['Otro'])
    agent = st.selectbox('Agente', art['top_agent'] + ['Otro'])
    adr = st.number_input('Tarifa diaria (adr)', 0.0, 500.0, 100.0)
    booking_changes = st.number_input('Cambios a la reserva', 0, 20, 0)
    waiting = st.number_input('Dias en lista de espera', 0, 400, 0)
    parking = st.number_input('Parqueaderos requeridos', 0, 5, 0)
    requests = st.number_input('Solicitudes especiales', 0, 5, 0)

booking = pd.DataFrame([{
    'hotel': hotel,
    'lead_time': lead_time,
    'arrival_date_year': arrival.year,
    'arrival_date_month': MONTHS[arrival.month - 1],
    'arrival_date_day_of_month': arrival.day,
    'stays_in_weekend_nights': weekend_nights,
    'stays_in_week_nights': week_nights,
    'adults': adults,
    'children': children,
    'babies': babies,
    'meal': meal,
    'country': country,
    'market_segment': market_segment,
    'distribution_channel': channel,
    'is_repeated_guest': int(repeated),
    'previous_cancellations': previous_cancellations,
    'previous_bookings_not_canceled': previous_ok,
    'reserved_room_type': room,
    'booking_changes': booking_changes,
    'deposit_type': deposit_type,
    'agent': 0 if agent == 'None' else (9999 if agent == 'Otro' else int(agent)),
    'days_in_waiting_list': waiting,
    'customer_type': customer_type,
    'adr': adr,
    'required_car_parking_spaces': parking,
    'total_of_special_requests': requests
}])

st.header('Resultado')
prob = predict(booking, art)[0]

c1, c2 = st.columns(2)
c1.metric('Probabilidad de cancelacion', f'{prob * 100:.1f}%')
if prob >= 0.5:
    c1.error('El modelo predice que la reserva se va a cancelar')
else:
    c1.success('El modelo predice que la reserva se mantiene')

fig, ax = plt.subplots(figsize=(6,2.5))
ax.barh(['Promedio del dataset', 'Esta reserva'], [art['base_rate'] * 100, prob * 100], color=['gray', 'tab:red' if prob >= 0.5 else 'tab:green'])
ax.axvline(50, color='black', linestyle='--')
ax.set_xlim(0, 100)
ax.set_xlabel('Probabilidad de cancelacion (%)')
c2.pyplot(fig)

st.header('Modelo')
st.write('Modelo en uso:', art['model_name'])
st.write('Hiperparametros:', art['params'])
results = pd.read_csv('models/resultados.csv')
st.subheader('Comparacion de modelos (conjunto de prueba)')
st.dataframe(results)

importance = pd.read_csv('models/importancias.csv')
fig, ax = plt.subplots(figsize=(7,5))
ax.barh(importance['variable'][::-1], importance['importancia'][::-1])
ax.set_title('Variables mas importantes (Random Forest)')
st.pyplot(fig)

st.header('Predecir varias reservas')
source = st.radio('Datos', ['Muestra del conjunto de prueba', 'Subir un CSV'])

if source == 'Subir un CSV':
    file = st.file_uploader('CSV con las columnas del dataset Hotel Booking Demand', type=['csv'])
    if file is None:
        st.stop()
    df = pd.read_csv(file)
else:
    df = pd.read_csv('data/muestra_test.csv')

missing = [col for col in booking.columns if col not in df.columns]
if missing:
    st.error('Faltan columnas: ' + ', '.join(missing))
    st.stop()

df['probabilidad'] = predict(df, art)
df['prediccion'] = (df['probabilidad'] >= 0.5).astype(int)

st.write('Reservas predichas como canceladas:', int(df['prediccion'].sum()), 'de', len(df))

if 'is_canceled' in df.columns:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Accuracy', f"{accuracy_score(df['is_canceled'], df['prediccion']):.3f}")
    m2.metric('Precision', f"{precision_score(df['is_canceled'], df['prediccion']):.3f}")
    m3.metric('Recall', f"{recall_score(df['is_canceled'], df['prediccion']):.3f}")
    m4.metric('F1', f"{f1_score(df['is_canceled'], df['prediccion']):.3f}")
    fig, ax = plt.subplots(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(df['is_canceled'], df['prediccion'], display_labels=['Se mantiene', 'Cancelada'], ax=ax, colorbar=False)
    st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6,3))
ax.hist(df['probabilidad'] * 100, bins=20)
ax.set_xlabel('Probabilidad de cancelacion (%)')
ax.set_ylabel('Reservas')
st.pyplot(fig)

st.dataframe(df[['hotel', 'lead_time', 'deposit_type', 'market_segment', 'adr', 'probabilidad', 'prediccion']].sort_values('probabilidad', ascending=False))
