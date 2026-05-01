# Прогнозирование отмены бронирования отеля

Проект по курсу "Технологии ИИ", СПбГУ, 2 курс, весна 2026.

Команда: Сухоплечев Виталий, Столярова Полина.

## Задача
Бинарная классификация: по параметрам брони предсказать, будет ли она отменена до заезда.

## Данные
Hotel Booking Demand (Antonio, Almeida, Nunes, 2019), 119 390 записей по двум отелям Португалии за 2015-2017 годы.

## Запуск

1. Скачать датасет:
   ```
   python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv', 'data/hotels.csv')"
   ```
2. Установить зависимости: `pip install -r requirements.txt`
3. Пройти ноутбуки в порядке `notebooks/01_eda.ipynb` -> ... -> `05_error_analysis.ipynb`.
4. Запустить дашборд: `streamlit run app/app.py`.

## Результаты
Финальная модель - HistGradientBoosting. ROC-AUC около 0.93 на отложенном тесте.

Подробности - в `design_document.md`.
