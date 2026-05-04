import numpy as np
import pandas as pd


MONTHS = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}


def _season(m):
    if m in (12, 1, 2):
        return 'winter'
    if m in (3, 4, 5):
        return 'spring'
    if m in (6, 7, 8):
        return 'summer'
    return 'autumn'


def add_features(df):
    df = df.copy()
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['has_children'] = ((df['children'] + df['babies']) > 0).astype(int)
    df['is_family'] = ((df['adults'] > 0) & ((df['children'] + df['babies']) > 0)).astype(int)
    df['room_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

    df['arrival_month_num'] = df['arrival_date_month'].map(MONTHS)
    df['season'] = df['arrival_month_num'].map(_season)

    arrival = pd.to_datetime(dict(
        year=df['arrival_date_year'],
        month=df['arrival_month_num'],
        day=df['arrival_date_day_of_month'],
    ), errors='coerce')
    df['arrival_weekday'] = arrival.dt.weekday

    safe_guests = df['total_guests'].replace(0, np.nan)
    df['adr_per_person'] = (df['adr'] / safe_guests).fillna(df['adr'])

    return df


NUMERIC_COLS = [
    'lead_time', 'arrival_date_year', 'arrival_month_num',
    'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_weekday',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights',
    'adults', 'children', 'babies', 'total_guests',
    'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list',
    'adr', 'adr_per_person',
    'required_car_parking_spaces', 'total_of_special_requests',
    'has_children', 'is_family', 'room_changed', 'has_company',
    'agent',
]

LOW_CARD_CAT = [
    'hotel', 'meal', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type',
    'customer_type', 'season',
]

HIGH_CARD_CAT = ['country']


def feature_columns():
    return NUMERIC_COLS + LOW_CARD_CAT + HIGH_CARD_CAT
