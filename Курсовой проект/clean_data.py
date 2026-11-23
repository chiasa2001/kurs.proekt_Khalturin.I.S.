# -*- coding: utf-8 -*-
"""
Очистка данных о продажах автомобилей в Казахстане за 2019 год.
Запуск: python clean_data.py
На выходе: autokz2019_clean.csv (кодировка utf-8-sig — корректно открывается в Excel).
Требования: pandas, numpy. Скрипт идемпотентен: можно запускать повторно.
"""
import re
import numpy as np
import pandas as pd

SRC = 'autokz2019.csv'
DST = 'autokz2019_clean.csv'

# ---------------------------------------------------------------- загрузка
def load(path=SRC):
    # разделитель ';' и десятичный разделитель ',' обязательны
    df = pd.read_csv(path, sep=';', decimal=',')
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# ------------------------------------------------------- базовая очистка
def drop_useless(df):
    cols = ['Модификация', 'Сегмент', 'Сегментация Eng', 'Локализация производства',
            'Тип клиента', 'Форма расчета', 'Наименование дилерского центра']
    df = df.drop(columns=cols)
    return df.rename(columns={
        'Год':'year', 'Месяц':'month', 'Компания':'dealer_company', 'Бренд':'brand',
        'Модель':'model', 'Год выпуска':'prod_year', 'Страна-производитель':'prod_country',
        'Вид топлива':'fuel_type', 'Объём двиг, л,':'engine_vol', 'Коробка передач':'gearbox',
        'Тип привода':'drive', 'Регион':'region', 'Количество':'quantity',
        'Цена, USD':'price_usd', 'Продажа, USD':'total_usd', 'Область':'oblast',
        'Сегментация 2013':'segment_2013', 'Класс 2013':'class_2013'})

def drop_empty_dupes(df):
    df = df.dropna(how='all')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df = df[df['quantity'] > 0]                      # возвраты (quantity = -1) удаляем
    for c in ['price_usd', 'total_usd']:
        # запятая -> точка ДО конверсии, иначе дробные цены теряются
        df[c] = (df[c].astype(str)
                 .str.replace('\u00a0', '', regex=False)
                 .str.replace(' ', '', regex=False)
                 .str.replace(',', '.', regex=False))
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------- смещённые строки
def fix_shifted(df):
    """У части Renault в fuel_type стоит объём, а в engine_vol — коробка."""
    def is_num(s):
        try:
            float(str(s).replace(',', '.'))
            return True
        except ValueError:
            return False
    mask = (df['brand'].isin(['Renault'])) & (df['fuel_type'].apply(is_num)) & (~df['engine_vol'].apply(is_num))
    df.loc[mask, 'engine_vol'] = df.loc[mask, 'fuel_type']
    df.loc[mask, 'gearbox']    = df.loc[mask, 'drive']
    df.loc[mask, 'drive']      = np.nan
    df.loc[mask, 'fuel_type']  = np.nan
    return df

# ------------------------------------------------------------- маппинги
COUNTRY = {
    'Германия':'DEU', 'Россия':'RUS', 'Казахстан':'KAZ', 'США':'USA', 'Япония':'JPN',
    'Корея':'KOR', 'Южная Корея':'KOR', 'Китай':'CHN', 'Франция':'FRA', 'Чехия':'CZE',
    'Словакия':'SVK', 'Венгрия':'HUN', 'Турция':'TUR', 'Испания':'ESP', 'Италия':'ITA',
    'Великобритания':'GBR', 'Румыния':'ROU', 'Узбекистан':'UZB', 'Швеция':'SWE',
    'Австрия':'AUT', 'Бельгия':'BEL', 'Польша':'POL'}

FUEL = {
    'бензин':'F', 'Бензин':'F', 'БЕНЗИН':'F', 'дизель':'D', 'Дизель':'D', 'ДИЗЕЛЬ':'D',
    'электро':'E', 'Электро':'E', 'Электричество':'E', 'гибрид':'HYB', 'Гибрид':'HYB',
    'гибридный':'HYB', 'plug-in гибрид':'HYB', 'Plug-in Hybrid':'HYB',
    'гибрид бензин':'HYB', 'гибрид дизель':'HYB', 'газобаллонное':'F', 'LPG':'F'}

DRIVE = {
    '4WD':'4WD', 'AWD':'4WD', 'Полный':'4WD', 'полный':'4WD', 'ПОЛНЫЙ':'4WD',
    'Полный привод':'4WD', '4х4':'4WD', '4x4':'4WD', '4х4.2':'4WD',
    '4WD (подключаемый)':'4WD', 'Part-time':'4WD',
    'FWD':'FWD', 'передний':'FWD', 'Передний':'FWD', 'ПЕРЕДНИЙ':'FWD',
    'Передний привод':'FWD', '2WD':'FWD', '4х2':'FWD', '4x2':'FWD', '2.2':'FWD',
    'Переднеприводной':'FWD',
    'RWD':'RWD', 'задний':'RWD', 'Задний':'RWD', 'ЗАДНИЙ':'RWD',
    'Задний привод':'RWD', 'Заднеприводной':'RWD'}

def norm_gb(s):
    """83 варианта написания КПП -> Автомат / Механика (устойчиво к смешанным алфавитам)."""
    if not isinstance(s, str):
        return np.nan
    t = s.strip()
    if t == '':
        return np.nan
    u = t.upper()
    if u in ('4WD', 'FWD', 'RWD', '2WD', 'TDI', 'ТДИ'):
        return np.nan
    if re.search(r'АМТ|AMT', u):
        return 'Автомат'
    if re.search(r'М[ЕЕ]Х|МКП', u):
        return 'Механика'
    if (re.match(r'^\d*\s*[МM]\s*[/\-]?\s*[ТT]$', u)
            or re.match(r'^\d*\s*[МM]\s*$', u)
            or re.match(r'^\d*[МM][ТT]$', u)):
        return 'Механика'
    if (re.match(r'^\d*\s*[АA]?[ТT]$', u)
            or re.match(r'^\d*\s*[АA]\s*/?\s*[ТT]\s*$', u)
            or re.match(r'^\d*\s*[АA]$', u)):
        return 'Автомат'
    if re.search(r'АКП|АКПП|DSG|DCT|CVT|TRONIC|PDK|ВАРИАТОР|РОБОТ|АВТОМАТ|AUTOMATIC', u):
        return 'Автомат'
    return np.nan

def clean_vol(v, brand, model):
    """Объём двигателя: 'L', запятые, куб. см, мощность вместо объёма, аномалии Niva."""
    s = str(v).strip()
    if s in ('nan', ''):
        return np.nan
    s = s.replace(',', '.').replace('л', '').replace('L', '').strip()
    if re.search(r'л\.?с|hp|kwh', s, re.I):
        return np.nan
    try:
        x = float(s)
    except ValueError:
        return np.nan
    if x >= 1000:
        x = x / 1000
    if x > 12 and not (brand == 'Scania' or (brand == 'Mercedes-Benz' and model == 'Actros')):
        return np.nan
    if brand == 'Chevrolet' and model == 'Niva' and x != 1.7:
        x = 1.7
    return x

MONTHS = {'Январь':1, 'Февраль':2, 'Март':3, 'Апрель':4, 'Май':5, 'Июнь':6,
          'Июль':7, 'Август':8, 'Сентябрь':9, 'Октябрь':10, 'Ноябрь':11, 'Декабрь':12}

COMPANY = {
    'Mercur Autos':'Mercur Auto', 'MERCUR AUTO':'Mercur Auto',
    'Ravon Motors Kazakstan':'Равон Моторс Казахстан', 'Автокапитал':'Autokapital',
    'Каспиан Моторс':'Caspian Motors', 'ММС Рус':'MMC RUS'}

def cap_ru(s):
    if not isinstance(s, str):
        return s
    s = re.sub(r'^г\.\s*', '', s)
    return '-'.join((w[0].upper() + w[1:]) if w else w for w in s.split(' '))

# --------------------------------------------------------------- конвейер
def transform(df):
    df = fix_shifted(df)
    df['prod_country'] = df['prod_country'].map(COUNTRY).fillna(df['prod_country'])
    df['fuel_type'] = df['fuel_type'].map(lambda x: FUEL.get(x, x if x in ['F','D','E','HYB'] else np.nan))
    df['drive'] = df['drive'].map(lambda x: DRIVE.get(x, x if x in ['4WD','FWD','RWD'] else np.nan))
    df['gearbox'] = df['gearbox'].map(norm_gb)
    df['engine_vol'] = [clean_vol(v, b, m) for v, b, m in zip(df['engine_vol'], df['brand'], df['model'])]
    df['month'] = df['month'].map(MONTHS)
    df['sale_date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1)) + pd.offsets.MonthEnd(0)
    df = df.drop(columns=['year', 'month'])
    df['prod_year'] = pd.to_numeric(df['prod_year'], errors='coerce')
    df['dealer_company'] = (df['dealer_company'].replace(COMPANY)
                            .str.replace('  ', ' ', regex=False).str.strip())
    df['region'] = df['region'].map(cap_ru)
    df['oblast'] = df['oblast'].map(cap_ru)
    for c in ['fuel_type', 'gearbox', 'drive', 'segment_2013', 'class_2013']:
        df[c] = df[c].astype('category')
    return df

def main():
    df = transform(drop_empty_dupes(drop_useless(load())))
    df.to_csv(DST, index=False, encoding='utf-8-sig', sep=';')
    print(f'Готово: {DST}, {df.shape[0]} строк x {df.shape[1]} столбцов')

if __name__ == '__main__':
    main()
