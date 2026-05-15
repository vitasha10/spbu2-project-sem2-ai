"""Генерация графиков и сборка презентации .pptx из шаблона СПбГУ.

Использует пустой шаблон base_empty.pptx из соседнего проекта - там сохранён
дизайн СПбГУ (титульный, рабочие, закрывающий), но без примеров слайдов.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pptx import Presentation
from pptx.util import Pt, Emu

import joblib
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_raw, clean
from src.features import add_features

TEMPLATE = 'C:/it/test-faceswap-2026/presentation/output/base_empty.pptx'
OUT_DIR = ROOT / 'docs'
IMG_DIR = OUT_DIR / 'pptx_assets'
OUT_PPTX = OUT_DIR / 'presentation.pptx'

sns.set_theme(style='whitegrid', font_scale=0.9)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'

MONTHS_RU = ['янв', 'фев', 'мар', 'апр', 'май', 'июн',
             'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']


def save_charts():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw(str(ROOT / 'data' / 'hotels.csv'))
    df = clean(df_raw)
    df = add_features(df)

    # 1. баланс классов
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
    counts = df_raw['is_canceled'].value_counts().rename({0: 'Доехали', 1: 'Отменили'})
    counts.plot(kind='bar', ax=ax, color=['#4c72b0', '#dd8452'])
    ax.set_title('Сколько броней отменяют')
    ax.set_ylabel('кол-во броней')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}'.replace(',', ' '),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'balance.png')
    plt.close(fig)

    # 2. срок до заезда
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
    sns.boxplot(data=df_raw, x='is_canceled', y='lead_time', ax=ax,
                palette=['#4c72b0', '#dd8452'])
    ax.set_xticklabels(['Доехали', 'Отменили'])
    ax.set_xlabel('')
    ax.set_ylabel('дней до заезда')
    ax.set_title('Срок до заезда у отменивших и доехавших')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'lead_time.png')
    plt.close(fig)

    # 3. депозит
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
    name_map = {'No Deposit': 'Без депозита', 'Refundable': 'Возвратный',
                'Non Refund': 'Невозвратный'}
    g = df_raw.groupby('deposit_type')['is_canceled'].mean()
    g.index = [name_map.get(x, x) for x in g.index]
    g = g.sort_values()
    g.plot(kind='barh', ax=ax, color='#c44e52')
    ax.set_xlabel('доля отмен')
    ax.set_ylabel('')
    ax.set_title('Доля отмен по типу депозита')
    for i, v in enumerate(g.values):
        ax.text(v + 0.01, i, f'{v:.0%}', va='center', fontsize=9)
    ax.set_xlim(0, 1.1)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'deposit.png')
    plt.close(fig)

    # 4. топ стран
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
    top = df_raw['country'].value_counts().head(10).index
    sub = df_raw[df_raw['country'].isin(top)]
    g = sub.groupby('country')['is_canceled'].mean().loc[top].sort_values()
    g.plot(kind='barh', ax=ax, color='#4c72b0')
    ax.set_xlabel('доля отмен')
    ax.set_ylabel('')
    ax.set_title('Доля отмен, топ-10 стран')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'country.png')
    plt.close(fig)

    # 5. сезонность
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    by_month = df_raw.groupby('arrival_date_month')['is_canceled'].mean().reindex(months_order)
    by_month.plot(kind='line', marker='o', ax=ax, color='#c44e52')
    ax.set_ylabel('доля отмен')
    ax.set_xlabel('')
    ax.set_title('Отмены по месяцам заезда')
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS_RU, rotation=0)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'season.png')
    plt.close(fig)

    # 6. сравнение моделей
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=160)
    models = ['Случайный\nответ', 'Логистич.\nрегрессия',
              'Метод k\nсоседей', 'Решающее\nдерево',
              'Случайный\nлес', 'Градиентный\nбустинг']
    aucs = [0.500, 0.840, 0.851, 0.881, 0.913, 0.920]
    colors = ['#999999', '#4c72b0', '#55a868', '#8172b3', '#dd8452', '#c44e52']
    bars = ax.bar(models, aucs, color=colors)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('Качество ROC-AUC')
    ax.set_title('Сравнение моделей (5-фолдовая кросс-валидация)')
    ax.axhline(0.5, color='gray', ls='--', lw=0.8)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'models.png')
    plt.close(fig)

    # 7. значимость признаков
    split = joblib.load(ROOT / 'models' / '_split.pkl')
    X_test = split['X_test']
    y_test = split['y_test']
    final = joblib.load(ROOT / 'models' / 'final_model.pkl')
    hgb = final['pipeline']

    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(X_test), 3000, replace=False)
    Xs = X_test.iloc[sub_idx]
    ys = y_test.iloc[sub_idx]
    imp = permutation_importance(hgb, Xs, ys, scoring='roc_auc',
                                 n_repeats=3, random_state=42, n_jobs=-1)
    importance_ru = {
        'lead_time': 'срок до заезда',
        'country': 'страна клиента',
        'deposit_type': 'тип депозита',
        'total_of_special_requests': 'число спец-запросов',
        'market_segment': 'канал продажи',
        'adr': 'цена за ночь',
        'arrival_date_year': 'год заезда',
        'arrival_date_week_number': 'неделя заезда',
        'arrival_date_day_of_month': 'день заезда',
        'previous_cancellations': 'прошлые отмены клиента',
        'agent': 'агентство',
        'customer_type': 'тип клиента',
        'required_car_parking_spaces': 'запрос парковки',
        'booking_changes': 'число изменений брони',
        'room_changed': 'сменили номер при заезде',
        'assigned_room_type': 'выданный номер',
        'reserved_room_type': 'запрошенный номер',
        'distribution_channel': 'канал дистрибуции',
        'adr_per_person': 'цена на гостя',
    }
    ser = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False).head(10)
    ser.index = [importance_ru.get(x, x) for x in ser.index]
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=160)
    ser.iloc[::-1].plot(kind='barh', ax=ax, color='#4c72b0')
    ax.set_xlabel('падение качества ROC-AUC при перетасовке признака')
    ax.set_ylabel('')
    ax.set_title('Значимость признаков (топ-10)')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'importance.png')
    plt.close(fig)

    # 8. калибровка
    rf = joblib.load(ROOT / 'models' / '_rf.pkl')
    lr = joblib.load(ROOT / 'models' / '_logreg.pkl')
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=160)
    for name, m, color in [
        ('Логистическая регрессия', lr, '#4c72b0'),
        ('Случайный лес', rf, '#dd8452'),
        ('Градиентный бустинг', hgb, '#c44e52'),
    ]:
        CalibrationDisplay.from_estimator(m, X_test, y_test, n_bins=10,
                                          ax=ax, name=name, color=color)
    ax.set_title('Калибровка вероятностей')
    ax.set_xlabel('предсказанная вероятность')
    ax.set_ylabel('фактическая доля отмен')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'calibration.png')
    plt.close(fig)

    # тестовые AUC
    test_auc = {
        'LogReg': roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]),
        'RandomForest': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
        'HistGB': roc_auc_score(y_test, hgb.predict_proba(X_test)[:, 1]),
    }
    print('test AUC:', {k: round(v, 4) for k, v in test_auc.items()})


# ----------------- сборка pptx -----------------

def _set_text(ph, text, size=None, bold=None):
    tf = ph.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def _set_bullets(ph, bullets, size=13):
    tf = ph.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)


def _add_image(slide, path, left, top, width, height):
    return slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def _placeholder(slide, idx):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            return ph
    return None


def _textbox(slide, left, top, width, height):
    """Текстовое поле в свободном месте, не привязанное к placeholder."""
    return slide.shapes.add_textbox(left, top, width, height)


def _set_box_text(box, text, size=14, bold=False):
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold


def _set_box_bullets(box, bullets, size=13):
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)


def _set_title(slide, text, size=22):
    """Заголовок слайда (placeholder idx=0)."""
    ph = _placeholder(slide, 0)
    if ph is not None:
        _set_text(ph, text, size=size, bold=True)


def build():
    save_charts()
    pres = Presentation(TEMPLATE)
    layouts = {l.name: l for l in pres.slide_layouts}
    L_title = layouts['TITLE']
    L_work = layouts['1_Рабочий слайд с фотографией']
    L_close = layouts['Закрывающий слайд']

    # стандартные зоны на рабочем слайде (16:9 = 9144000 x 5143500 EMU)
    LEFT_MARGIN = Emu(323528)
    TOP_BELOW_TITLE = Emu(900000)
    # Текстовая колонка слева, картинка справа
    TXT_W = Emu(4200000)
    IMG_LEFT = Emu(4700000)
    IMG_W = Emu(4200000)
    IMG_H = Emu(2800000)
    CONTENT_H = Emu(4000000)
    FULL_W = Emu(8500000)
    SLIDE_W = Emu(9144000)
    SLIDE_H = Emu(5143500)

    # 1. титульный
    s = pres.slides.add_slide(L_title)
    _set_text(_placeholder(s, 0),
              'Прогнозирование отмены\nбронирования отеля', size=30, bold=True)
    _set_bullets(_placeholder(s, 1), [
        'Сухоплечев Виталий, Столярова Полина',
        'Курс «Технологии ИИ», СПбГУ, 2 курс, весна 2026',
    ], size=15)

    # вспомогательные функции для рабочего слайда: title через placeholder,
    # подзаголовок и контент - через textbox с явными координатами

    def work_slide(title, subtitle=None):
        s = pres.slides.add_slide(L_work)
        _set_title(s, title, size=22)
        if subtitle:
            sub = _textbox(s, LEFT_MARGIN, Emu(770000), FULL_W, Emu(380000))
            _set_box_text(sub, subtitle, size=14, bold=False)
        return s

    # 2. задача и зачем это нужно
    s = work_slide('Задача и зачем это нужно',
                   'Предсказать, отменит ли клиент бронь до заезда')
    box = _textbox(s, LEFT_MARGIN, Emu(1200000), FULL_W, CONTENT_H)
    _set_box_bullets(box, [
        '• Что предсказываем: 0 (доехал) или 1 (отменил)',
        '',
        '• Зачем отелю.',
        '   Отмены — это потерянные деньги. Если знать риск заранее,',
        '   можно попросить депозит, держать запас по перебронированию',
        '   или прозвонить клиента и подтвердить визит.',
        '',
        '• Чем меряем качество модели.',
        '   ROC-AUC (площадь под ROC-кривой) — основная метрика.',
        '   Она не зависит от выбранного порога принятия решения.',
        '',
        '   Полнота по классу «отмена» — дополнительная.',
        '   Пропустить отмену хуже, чем ложно её предсказать.',
    ], size=14)

    # 3. данные
    s = work_slide('Данные')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        '• Открытый набор Hotel Booking',
        '   Demand (Antonio и соавт., 2019)',
        '',
        '• 119 390 броней',
        '• 32 признака',
        '',
        '• 2 отеля в Португалии:',
        '   городской и курортный',
        '',
        '• Период 2015 – 2017 гг.',
        '',
        '• Доля отмен — 37 %',
    ], size=14)
    _add_image(s, IMG_DIR / 'balance.png', IMG_LEFT, Emu(1059582), IMG_W, IMG_H)

    # 4. проблемы данных
    s = work_slide('Что почистили в данных',
                   'Без чистки модель ловит мусор и утечку')
    box = _textbox(s, LEFT_MARGIN, Emu(1200000), FULL_W, CONTENT_H)
    _set_box_bullets(box, [
        '• Пропуски:',
        '   страна клиента — 488 пустых → заменили на «неизвестно»',
        '   агентство — 16 340 пустых → заменили на «нет агентства»',
        '   компания — 112 593 пустых (94 % колонки!) →',
        '      превратили в бинарный признак «бронь от компании»',
        '',
        '• Цена за ночь:',
        '   были значения от −6 € до 5 400 € (а обычно около 100 €)',
        '   убрали записи с отрицательной ценой и аномально высокой',
        '',
        '• Дубли: около 32 тысяч одинаковых строк → удалили',
        '• Брони с 0 гостей: удалили',
        '',
        '• Утечка целевой переменной:',
        '   колонки «статус брони» и «дата статуса» —',
        '   это прямой ответ. Удалили до обучения.',
    ], size=13)

    # 5. главный сигнал
    s = work_slide('Главный сигнал: срок до заезда')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        '• У отменивших медианный',
        '   срок до заезда почти в 2 раза',
        '   больше, чем у доехавших',
        '',
        '• Логика простая:',
        '   чем раньше человек бронирует,',
        '   тем больше шансов передумать',
        '',
        '• Это самый сильный',
        '   отдельный признак в данных',
    ], size=14)
    _add_image(s, IMG_DIR / 'lead_time.png', IMG_LEFT, Emu(1059582), IMG_W, IMG_H)

    # 6. депозит и страны
    s = work_slide('Депозит и страна клиента')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        '• «Невозвратный» депозит:',
        '   почти всегда отменяют.',
        '   Особенность датасета,',
        '   описана в оригинальной статье.',
        '',
        '• Португальцы (PRT)',
        '   отменяют чаще остальных.',
        '',
        '• Логично: отели в Португалии,',
        '   местные жители бронируют',
        '   без особых обязательств.',
    ], size=12)
    _add_image(s, IMG_DIR / 'deposit.png', IMG_LEFT, Emu(900000), IMG_W, Emu(1900000))
    _add_image(s, IMG_DIR / 'country.png', IMG_LEFT, Emu(2900000), IMG_W, Emu(1900000))

    # 7. сезонность
    s = work_slide('Сезонность')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        '• Летом доля отмен выше',
        '',
        '• Январь – апрель — самые',
        '   спокойные месяцы',
        '',
        '• Из месяца сделали три',
        '   признака для модели:',
        '   – номер месяца',
        '   – время года',
        '   – день недели заезда',
    ], size=14)
    _add_image(s, IMG_DIR / 'season.png', IMG_LEFT, Emu(1059582), IMG_W, IMG_H)

    # 8. подготовка данных и признаков
    s = work_slide('Подготовка данных и новые признаки',
                   'Код вынесен в модули src/data.py и src/features.py')
    box = _textbox(s, LEFT_MARGIN, Emu(1200000), FULL_W, CONTENT_H)
    _set_box_bullets(box, [
        '• Новые признаки, которые мы посчитали:',
        '   общее число ночей и общее число гостей,',
        '   признак «есть ли дети» и «семейная бронь»,',
        '   «сменили номер при заезде» (снижает риск отмены),',
        '   месяц, время года, день недели заезда,',
        '   цена на одного гостя.',
        '',
        '• Категориальные признаки переводим в числа:',
        '   признаки с малым числом значений (тип отеля, питание, канал) —',
        '      one-hot кодирование для линейных моделей,',
        '      порядковое кодирование для деревьев и бустинга.',
        '   страна (около 175 значений) — целевое кодирование',
        '      (значение признака = средняя доля отмен по стране,',
        '      считается без утечки — отдельно для каждого фолда).',
        '',
        '• Разбиение: 80 % обучение / 20 % тест, со стратификацией.',
    ], size=12)

    # 9. сравнение моделей
    s = work_slide('Сравнение моделей')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        '5-фолдовая стратифицированная',
        'кросс-валидация',
        '',
        '• Случайный ответ — 0.50',
        '• Логистическая регрессия — 0.84',
        '• k ближайших соседей — 0.85',
        '• Решающее дерево — 0.88',
        '• Случайный лес — 0.91',
        '• Градиентный бустинг — 0.92',
        '',
        'Разброс по фолдам ≈ 0.002,',
        'результаты устойчивые',
    ], size=12)
    _add_image(s, IMG_DIR / 'models.png', IMG_LEFT, Emu(1059582), IMG_W, IMG_H)

    # 10. финальная модель
    s = work_slide('Финальная модель — градиентный бустинг')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        'HistGradientBoosting из',
        'библиотеки scikit-learn',
        '',
        '• Качество на тесте:',
        '   ROC-AUC = 0.917',
        '   F1-мера = 0.77',
        '   Полнота = 0.67',
        '',
        '• Почему выиграл:',
        '   ловит нелинейные связи,',
        '   быстрее случайного леса,',
        '   стабилен по фолдам.',
        '',
        '• Параметры: 300 итераций,',
        '   скорость обучения 0.05,',
        '   глубина дерева 8.',
    ], size=11)
    _add_image(s, IMG_DIR / 'importance.png', IMG_LEFT, Emu(1059582), IMG_W, IMG_H)

    # 11. калибровка
    s = work_slide('Насколько можно верить вероятностям')
    txt = _textbox(s, LEFT_MARGIN, Emu(1059582), TXT_W, CONTENT_H)
    _set_box_bullets(txt, [
        'Хорошо откалиброванная модель:',
        'если она говорит «70 % шанс»,',
        'то в 70 % случаев так и будет.',
        '',
        '• Логистическая — почти идеально',
        '',
        '• Случайный лес — задирает',
        '   уверенность (предсказал 0.9,',
        '   реально доля = 0.7)',
        '',
        '• Градиентный бустинг —',
        '   близко к идеалу, чуть',
        '   переоценивает хвост',
        '',
        'Дополнительно выпрямляется',
        'сигмоидой при необходимости.',
    ], size=11)
    _add_image(s, IMG_DIR / 'calibration.png', IMG_LEFT, Emu(1059582), IMG_W, IMG_H)

    # 12. демо приложения
    s = work_slide('Демонстрация приложения',
                   'Запуск: run_app.bat → откроется в браузере по адресу localhost:8501')
    box = _textbox(s, LEFT_MARGIN, Emu(1200000), FULL_W, CONTENT_H)
    _set_box_bullets(box, [
        'Сверху страницы — три готовых сценария:',
        '',
        '1. ВЫСОКИЙ РИСК.',
        '   Городской отель, срок 200 дней, невозвратный депозит,',
        '   клиент из Португалии, канал — онлайн-агентство.',
        '   Модель выдаёт ≈ 99.8 % шанс отмены.',
        '',
        '2. НИЗКИЙ РИСК.',
        '   Курортный отель, срок 14 дней, прямое бронирование,',
        '   3 спецзапроса, парковка, семья с ребёнком, клиент из Великобритании.',
        '   Модель выдаёт ≈ 0 %.',
        '',
        '3. СРЕДНИЙ РИСК.',
        '   Групповая бронь, срок 90 дней, без спецзапросов.',
        '   Модель ≈ 57 % — показывает, что граница не острая.',
        '',
        'Можно вручную менять только срок до заезда (7 → 60 → 200) —',
        'вероятность отмены растёт почти монотонно.',
    ], size=12)

    # 13. деплой и риски
    s = work_slide('Как развернуть и чем рискуем',
                   'Что нужно для работы в продакшене и какие у модели слабые места')
    box = _textbox(s, LEFT_MARGIN, Emu(1200000), FULL_W, CONTENT_H)
    _set_box_bullets(box, [
        '• Развёртывание:',
        '   простой веб-сервис (например на FastAPI) с командой /predict,',
        '   модель загружается из файла в память при старте,',
        '   время ответа меньше 50 мс на бронь,',
        '   раз в сутки — пересчёт по активным бронированиям,',
        '   мониторинг смещения распределений признаков,',
        '   переобучение раз в квартал на скользящем окне.',
        '',
        '• Главные риски:',
        '   1. Перенос на другие отели. Учились на 2 отелях в Португалии —',
        '      на сетевой отель в Германии может не перенестись.',
        '   2. Смещение во времени. Данные 2015 – 2017, после ковида',
        '      клиенты ведут себя иначе.',
        '   3. Новая страна. Для страны, которой не было в обучении,',
        '      целевое кодирование даст среднее по миру — точность падает.',
    ], size=12)

    # 14. закрывающий
    s = pres.slides.add_slide(L_close)
    _set_text(_placeholder(s, 0), 'Спасибо за внимание', size=24, bold=True)
    _set_text(_placeholder(s, 1),
              'Код, ноутбуки и отчёт — в репозитории. Готовы к вопросам.',
              size=14)

    pres.save(str(OUT_PPTX))
    print('saved:', OUT_PPTX)


if __name__ == '__main__':
    build()
