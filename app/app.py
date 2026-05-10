import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import feature_columns


@st.cache_resource
def load_model():
    data = joblib.load(ROOT / 'models' / 'final_model.pkl')
    return data['pipeline'], data['features']


def default_row():
    return {
        'lead_time': 60,
        'arrival_date_year': 2017,
        'arrival_month_num': 6,
        'arrival_date_week_number': 25,
        'arrival_date_day_of_month': 15,
        'arrival_weekday': 2,
        'stays_in_weekend_nights': 1,
        'stays_in_week_nights': 2,
        'total_nights': 3,
        'adults': 2,
        'children': 0,
        'babies': 0,
        'total_guests': 2,
        'is_repeated_guest': 0,
        'previous_cancellations': 0,
        'previous_bookings_not_canceled': 0,
        'booking_changes': 0,
        'days_in_waiting_list': 0,
        'adr': 100.0,
        'adr_per_person': 50.0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 1,
        'has_children': 0,
        'is_family': 0,
        'room_changed': 0,
        'has_company': 0,
        'agent': 9,
        'hotel': 'City Hotel',
        'meal': 'BB',
        'market_segment': 'Online TA',
        'distribution_channel': 'TA/TO',
        'reserved_room_type': 'A',
        'assigned_room_type': 'A',
        'deposit_type': 'No Deposit',
        'customer_type': 'Transient',
        'season': 'summer',
        'country': 'PRT',
    }


st.set_page_config(page_title='Отмена брони', layout='centered')
st.title('Прогноз отмены бронирования')
st.caption('Модель HistGradientBoosting, обучена на Hotel Booking Demand (2015-2017).')

model, cols = load_model()

st.subheader('Параметры брони')
c1, c2 = st.columns(2)

with c1:
    hotel = st.selectbox('Отель', ['City Hotel', 'Resort Hotel'])
    lead_time = st.number_input('Lead time (дней до заезда)', 0, 720, 60)
    adults = st.number_input('Взрослых', 1, 10, 2)
    children = st.number_input('Детей', 0, 10, 0)
    babies = st.number_input('Младенцев', 0, 5, 0)
    week_nights = st.number_input('Будних ночей', 0, 30, 2)
    weekend_nights = st.number_input('Выходных ночей', 0, 10, 1)
    adr = st.number_input('ADR (средняя цена за ночь)', 0.0, 700.0, 100.0)

with c2:
    deposit_type = st.selectbox('Депозит', ['No Deposit', 'Non Refund', 'Refundable'])
    market = st.selectbox('Market segment',
                          ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate',
                           'Groups', 'Complementary', 'Aviation', 'Undefined'])
    channel = st.selectbox('Distribution channel',
                           ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined'])
    customer = st.selectbox('Customer type',
                            ['Transient', 'Transient-Party', 'Contract', 'Group'])
    country = st.text_input('Country (ISO код)', 'PRT').upper()
    prev_cancel = st.number_input('Прошлых отмен у клиента', 0, 50, 0)
    special = st.number_input('Спец-запросов', 0, 10, 1)
    parking = st.number_input('Мест парковки', 0, 5, 0)


def _season(m):
    if m in (12, 1, 2):
        return 'winter'
    if m in (3, 4, 5):
        return 'spring'
    if m in (6, 7, 8):
        return 'summer'
    return 'autumn'


if st.button('Посчитать', type='primary'):
    row = default_row()
    row.update({
        'hotel': hotel,
        'lead_time': lead_time,
        'adults': adults, 'children': children, 'babies': babies,
        'stays_in_week_nights': week_nights,
        'stays_in_weekend_nights': weekend_nights,
        'total_nights': week_nights + weekend_nights,
        'total_guests': adults + children + babies,
        'has_children': int((children + babies) > 0),
        'is_family': int(adults > 0 and (children + babies) > 0),
        'adr': adr,
        'adr_per_person': adr / max(adults + children + babies, 1),
        'deposit_type': deposit_type,
        'market_segment': market,
        'distribution_channel': channel,
        'customer_type': customer,
        'country': country or 'Unknown',
        'previous_cancellations': prev_cancel,
        'total_of_special_requests': special,
        'required_car_parking_spaces': parking,
    })
    row['season'] = _season(row['arrival_month_num'])

    X = pd.DataFrame([row])[cols]
    proba = float(model.predict_proba(X)[0, 1])

    if proba >= 0.6:
        st.error(f'Высокий риск отмены: {proba:.1%}')
    elif proba >= 0.35:
        st.warning(f'Средний риск отмены: {proba:.1%}')
    else:
        st.success(f'Низкий риск отмены: {proba:.1%}')

    st.progress(proba)

    with st.expander('Что влияет на предсказание'):
        st.write('Главные факторы в этой модели (по permutation importance): '
                 '`lead_time`, `country`, `deposit_type`, `total_of_special_requests`, `market_segment`. '
                 'Non Refund почти всегда даёт высокий прогноз - это особенность датасета.')
