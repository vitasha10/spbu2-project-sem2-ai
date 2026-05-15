import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import feature_columns


# страны, встречавшиеся в обучении (>=50 раз), с русскими подписями.
# Первый код — то, что ожидает модель; вторая часть — для отображения.
COUNTRY_RU = {
    'PRT': 'Португалия', 'GBR': 'Великобритания', 'FRA': 'Франция',
    'ESP': 'Испания', 'DEU': 'Германия', 'ITA': 'Италия',
    'IRL': 'Ирландия', 'BEL': 'Бельгия', 'BRA': 'Бразилия',
    'NLD': 'Нидерланды', 'USA': 'США', 'CHE': 'Швейцария',
    'CN': 'Китай (CN)', 'CHN': 'Китай', 'AUT': 'Австрия',
    'SWE': 'Швеция', 'POL': 'Польша', 'RUS': 'Россия',
    'NOR': 'Норвегия', 'ROU': 'Румыния', 'Unknown': 'неизвестно',
    'FIN': 'Финляндия', 'ISR': 'Израиль', 'DNK': 'Дания',
    'AUS': 'Австралия', 'AGO': 'Ангола', 'LUX': 'Люксембург',
    'MAR': 'Марокко', 'ARG': 'Аргентина', 'TUR': 'Турция',
    'HUN': 'Венгрия', 'JPN': 'Япония', 'CZE': 'Чехия',
    'IND': 'Индия', 'GRC': 'Греция', 'KOR': 'Южная Корея',
    'HRV': 'Хорватия', 'EST': 'Эстония', 'IRN': 'Иран',
    'DZA': 'Алжир', 'ZAF': 'ЮАР', 'MEX': 'Мексика',
    'LTU': 'Литва', 'BGR': 'Болгария', 'NZL': 'Новая Зеландия',
    'COL': 'Колумбия', 'CHL': 'Чили',
}
COUNTRY_CODES = list(COUNTRY_RU.keys())
COUNTRY_LABELS = {f'{c} — {COUNTRY_RU[c]}': c for c in COUNTRY_CODES}

# мэппинг русских подписей в значения, которые видела модель
HOTEL = {'Городской отель': 'City Hotel', 'Курортный отель': 'Resort Hotel'}
DEPOSIT = {
    'Без депозита': 'No Deposit',
    'Невозвратный': 'Non Refund',
    'Возвратный': 'Refundable',
}
MARKET = {
    'Онлайн-агентство': 'Online TA',
    'Офлайн-агентство': 'Offline TA/TO',
    'Прямое бронирование': 'Direct',
    'Корпоративное': 'Corporate',
    'Группы': 'Groups',
    'Подарок от отеля': 'Complementary',
    'Авиакомпания': 'Aviation',
    'Не указано': 'Undefined',
}
CHANNEL = {
    'Через агентство': 'TA/TO',
    'Прямой': 'Direct',
    'Корпоративный': 'Corporate',
    'Глобальная система бронирования': 'GDS',
    'Не указано': 'Undefined',
}
CUSTOMER = {
    'Обычный гость': 'Transient',
    'Гость в составе группы': 'Transient-Party',
    'Контрактный': 'Contract',
    'Группа': 'Group',
}


SCENARIOS = {
    'Высокий риск (онлайн-агентство, невозвратный депозит, Португалия, 200 дней)': {
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
    'Низкий риск (прямое бронирование, спецзапросы, парковка, семья)': {
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
    'Средний риск (групповая бронь, 90 дней)': {
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
    'Параметры по умолчанию': None,
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

model, cols = load_model()

# --- готовые сценарии: вертикально, по одной кнопке в строке ---
st.subheader('Быстрый тест')
st.caption('Готовые наборы параметров. Жмёшь — получаешь предсказание сразу.')
for name, params in SCENARIOS.items():
    if st.button(name, use_container_width=True):
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
    hotel_ru = st.selectbox('Тип отеля', list(HOTEL.keys()))
    lead_time = st.number_input(
        'Дней до заезда',
        0, 720, 60,
        help='Тестировать стоит 7, 60, 200. Чем больше — тем выше риск отмены.',
    )
    adults = st.number_input('Взрослых', 1, 10, 2, help='Обычно 1–3.')
    children = st.number_input('Детей', 0, 10, 0, help='Обычно 0–2.')
    babies = st.number_input('Младенцев', 0, 5, 0, help='Обычно 0–1.')
    week_nights = st.number_input(
        'Будних ночей',
        0, 30, 2,
        help='Обычно 1–5.',
    )
    weekend_nights = st.number_input(
        'Выходных ночей',
        0, 10, 1,
        help='Обычно 0–2.',
    )
    adr = st.number_input(
        'Средняя цена за ночь, евро',
        0.0, 700.0, 100.0,
        help='Типичный диапазон 50–200. Курортный обычно дороже.',
    )

with c2:
    deposit_ru = st.selectbox(
        'Тип депозита',
        list(DEPOSIT.keys()),
        help='Невозвратный депозит в этих данных почти всегда даёт отмену — особенность датасета.',
    )
    market_ru = st.selectbox('Канал продажи', list(MARKET.keys()))
    channel_ru = st.selectbox('Канал дистрибуции', list(CHANNEL.keys()))
    customer_ru = st.selectbox('Тип клиента', list(CUSTOMER.keys()))
    country_label = st.selectbox(
        'Страна клиента',
        list(COUNTRY_LABELS.keys()),
        index=0,
        help='В списке только страны, которые встречались в обучающих данных.',
    )
    prev_cancel = st.number_input(
        'Прошлых отмен у клиента',
        0, 50, 0,
        help='Обычно 0. Попробуй 1–3, чтобы повысить риск.',
    )
    special = st.number_input(
        'Спецзапросов от клиента',
        0, 10, 1,
        help='Обычно 0–3. Больше запросов — меньше отмен.',
    )
    parking = st.number_input(
        'Запрошено мест парковки',
        0, 5, 0,
        help='Обычно 0 или 1. Значение 1 заметно снижает риск.',
    )


if st.button('Посчитать', type='primary'):
    params = dict(
        hotel=HOTEL[hotel_ru],
        lead_time=lead_time,
        adults=adults, children=children, babies=babies,
        week_nights=week_nights, weekend_nights=weekend_nights,
        adr=adr,
        deposit_type=DEPOSIT[deposit_ru],
        market=MARKET[market_ru],
        channel=CHANNEL[channel_ru],
        customer=CUSTOMER[customer_ru],
        country=COUNTRY_LABELS[country_label],
        prev_cancel=prev_cancel,
        special=special, parking=parking,
    )
    st.session_state['proba'] = predict(model, cols, params)
    st.session_state['scenario_name'] = 'Ручной ввод'


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
            'Главные признаки по значимости: срок до заезда, страна клиента, тип депозита, '
            'число спецзапросов, канал продажи. '
            'Невозвратный депозит в этих данных почти всегда означает отмену — '
            'это известная особенность набора.'
        )

st.divider()

# --- чеклист сценариев для защиты ---
st.subheader('Чеклист')
st.markdown(
    '''
    1. **Высокий риск.** Городской отель, срок 200 дней, невозвратный депозит,
       клиент из Португалии, онлайн-агентство, без спецзапросов.
       Модель выдаёт около 0.95+.
    2. **Низкий риск.** Курортный отель, срок 14 дней, прямое бронирование,
       клиент из Великобритании, 3 спецзапроса, парковка, семья с ребёнком.
       Модель около 0.05.
    3. **Средний риск.** Групповая бронь, срок 90 дней, без спецзапросов.
       Модель примерно 0.4–0.6 — граница риска не острая.
    4. Меняем только срок до заезда (7 → 60 → 200) — вероятность отмены растёт
       почти монотонно. Это самый сильный признак.
    '''
)
