import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import feature_columns


# страны которые встречались в train (как минимум 50 раз), отсортированы по частоте
COUNTRIES = [
    'PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD',
    'USA', 'CHE', 'CN', 'AUT', 'SWE', 'CHN', 'POL', 'RUS', 'NOR', 'ROU',
    'Unknown', 'FIN', 'ISR', 'DNK', 'AUS', 'AGO', 'LUX', 'MAR', 'ARG', 'TUR',
    'HUN', 'JPN', 'CZE', 'IND', 'GRC', 'KOR', 'HRV', 'EST', 'IRN', 'DZA',
    'ZAF', 'MEX', 'LTU', 'BGR', 'NZL', 'COL', 'CHL',
]


SCENARIOS = {
    'высокий риск (Online TA + Non Refund + PRT, длинный lead_time)': {
        'hotel': 'City Hotel',
        'lead_time': 200,
        'adults': 2, 'children': 0, 'babies': 0,
        'week_nights': 3, 'weekend_nights': 1,
        'adr': 95.0,
        'deposit_type': 'Non Refund',
        'market': 'Online TA',
        'channel': 'TA/TO',
        'customer': 'Transient',
        'country': 'PRT',
        'prev_cancel': 1,
        'special': 0,
        'parking': 0,
    },
    'низкий риск (Direct + спецзапросы + парковка)': {
        'hotel': 'Resort Hotel',
        'lead_time': 14,
        'adults': 2, 'children': 1, 'babies': 0,
        'week_nights': 2, 'weekend_nights': 2,
        'adr': 130.0,
        'deposit_type': 'No Deposit',
        'market': 'Direct',
        'channel': 'Direct',
        'customer': 'Transient',
        'country': 'GBR',
        'prev_cancel': 0,
        'special': 3,
        'parking': 1,
    },
    'средний риск (Group, средний lead_time)': {
        'hotel': 'City Hotel',
        'lead_time': 90,
        'adults': 2, 'children': 0, 'babies': 0,
        'week_nights': 2, 'weekend_nights': 0,
        'adr': 80.0,
        'deposit_type': 'No Deposit',
        'market': 'Groups',
        'channel': 'TA/TO',
        'customer': 'Transient-Party',
        'country': 'PRT',
        'prev_cancel': 0,
        'special': 0,
        'parking': 0,
    },
    'дефолтный (всё среднее)': None,  # просто используем дефолты формы
}


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


def _season(m):
    if m in (12, 1, 2):
        return 'winter'
    if m in (3, 4, 5):
        return 'spring'
    if m in (6, 7, 8):
        return 'summer'
    return 'autumn'


def predict(model, cols, params):
    row = default_row()
    row.update({
        'hotel': params['hotel'],
        'lead_time': params['lead_time'],
        'adults': params['adults'],
        'children': params['children'],
        'babies': params['babies'],
        'stays_in_week_nights': params['week_nights'],
        'stays_in_weekend_nights': params['weekend_nights'],
        'total_nights': params['week_nights'] + params['weekend_nights'],
        'total_guests': params['adults'] + params['children'] + params['babies'],
        'has_children': int((params['children'] + params['babies']) > 0),
        'is_family': int(params['adults'] > 0 and (params['children'] + params['babies']) > 0),
        'adr': params['adr'],
        'adr_per_person': params['adr'] / max(params['adults'] + params['children'] + params['babies'], 1),
        'deposit_type': params['deposit_type'],
        'market_segment': params['market'],
        'distribution_channel': params['channel'],
        'customer_type': params['customer'],
        'country': params['country'] or 'Unknown',
        'previous_cancellations': params['prev_cancel'],
        'total_of_special_requests': params['special'],
        'required_car_parking_spaces': params['parking'],
    })
    row['season'] = _season(row['arrival_month_num'])
    X = pd.DataFrame([row])[cols]
    return float(model.predict_proba(X)[0, 1])


st.set_page_config(page_title='Отмена брони', layout='centered')
st.title('Прогноз отмены бронирования')
st.caption('Модель HistGradientBoosting, обучена на Hotel Booking Demand (2015-2017).')

model, cols = load_model()

# --- быстрые сценарии вверху, чтобы можно было сразу кликнуть и увидеть результат ---
st.subheader('Быстрый тест')
st.caption('Готовые наборы параметров. Жмёшь - получаешь предсказание сразу.')
sc_cols = st.columns(len(SCENARIOS))
for (name, params), col in zip(SCENARIOS.items(), sc_cols):
    with col:
        if st.button(name):
            if params is None:
                p = default_row()
                X = pd.DataFrame([p])[cols]
                proba = float(model.predict_proba(X)[0, 1])
            else:
                proba = predict(model, cols, params)
            st.session_state['proba'] = proba
            st.session_state['scenario_name'] = name

st.divider()

# --- ручной ввод ---
st.subheader('Параметры брони')
c1, c2 = st.columns(2)

with c1:
    hotel = st.selectbox('Тип отеля', ['City Hotel', 'Resort Hotel'])
    lead_time = st.number_input(
        'Дней до заезда',
        0, 720, 60,
        help='Тестировать стоит 7, 60, 200. Чем больше - тем выше риск отмены.',
    )
    adults = st.number_input('Взрослых', 1, 10, 2, help='Обычно 1-3.')
    children = st.number_input('Детей', 0, 10, 0, help='Обычно 0-2.')
    babies = st.number_input('Младенцев', 0, 5, 0, help='Обычно 0-1.')
    week_nights = st.number_input(
        'Будних ночей',
        0, 30, 2,
        help='Обычно 1-5.',
    )
    weekend_nights = st.number_input(
        'Выходных ночей',
        0, 10, 1,
        help='Обычно 0-2.',
    )
    adr = st.number_input(
        'Средняя цена за ночь (ADR), евро',
        0.0, 700.0, 100.0,
        help='Типичный диапазон 50-200. Resort обычно дороже.',
    )

with c2:
    deposit_type = st.selectbox(
        'Тип депозита',
        ['No Deposit', 'Non Refund', 'Refundable'],
        help='Non Refund в этом датасете почти всегда даёт отмену - особенность данных.',
    )
    market = st.selectbox(
        'Канал продажи',
        ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate',
         'Groups', 'Complementary', 'Aviation', 'Undefined'],
    )
    channel = st.selectbox(
        'Канал дистрибуции',
        ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined'],
    )
    customer = st.selectbox(
        'Тип клиента',
        ['Transient', 'Transient-Party', 'Contract', 'Group'],
    )
    country = st.selectbox(
        'Страна клиента',
        COUNTRIES,
        index=0,
        help='Список ограничен странами, которые встречались в обучающих данных.',
    )
    prev_cancel = st.number_input(
        'Прошлых отмен у клиента',
        0, 50, 0,
        help='Обычно 0. Стоит попробовать 1-3 для повышения риска.',
    )
    special = st.number_input(
        'Спец-запросов',
        0, 10, 1,
        help='Обычно 0-3. Больше запросов - меньше отмен.',
    )
    parking = st.number_input(
        'Запрошено мест парковки',
        0, 5, 0,
        help='Обычно 0 или 1. 1 заметно снижает риск.',
    )


if st.button('Посчитать', type='primary'):
    params = dict(
        hotel=hotel, lead_time=lead_time,
        adults=adults, children=children, babies=babies,
        week_nights=week_nights, weekend_nights=weekend_nights,
        adr=adr, deposit_type=deposit_type, market=market, channel=channel,
        customer=customer, country=country, prev_cancel=prev_cancel,
        special=special, parking=parking,
    )
    st.session_state['proba'] = predict(model, cols, params)
    st.session_state['scenario_name'] = 'ручной ввод'


if 'proba' in st.session_state:
    proba = st.session_state['proba']
    label = st.session_state.get('scenario_name', '')
    st.markdown(f'**Сценарий:** {label}')
    if proba >= 0.6:
        st.error(f'Высокий риск отмены: {proba:.1%}')
    elif proba >= 0.35:
        st.warning(f'Средний риск отмены: {proba:.1%}')
    else:
        st.success(f'Низкий риск отмены: {proba:.1%}')
    st.progress(proba)

    with st.expander('Что влияет на предсказание'):
        st.write(
            'Главные факторы по permutation importance: lead_time, country, deposit_type, '
            'total_of_special_requests, market_segment. '
            'Non Refund почти всегда даёт высокий прогноз - это известная особенность датасета.'
        )

st.divider()

# --- подсказка для защиты: чем гарантированно продемонстрировать модель ---
st.subheader('Что показать на защите')
st.markdown(
    '''
    1. **Высокий риск.** City Hotel, lead_time=200, deposit=Non Refund, страна PRT,
       Online TA, без спецзапросов. Модель выдаёт около 0.95+.
    2. **Низкий риск.** Resort, lead_time=14, Direct, страна GBR, 3 спецзапроса,
       парковка=1, семья с ребёнком. Модель около 0.05.
    3. **Средний.** Группа от 90 дней, Groups канал, без спецзапросов.
       Модель в районе 0.4-0.6 - показывает что граница не острая.
    4. Можно менять только lead_time (например 7 -> 60 -> 200) - вероятность отмены
       растёт почти монотонно. Это самый сильный признак.
    '''
)
