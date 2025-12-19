import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Сравнение цен (Точное)", page_icon="🎯", layout="wide")

st.title("🎯 Сравнение цен (Улучшенная точность)")
st.markdown("""
Этот алгоритм лучше понимает сокращения и цифры (например, видит, что **"100г"** и **"100 г"** — это одно и то же).
Коды товаров в итоговую таблицу не добавляются.
""")

def aggressive_clean_name(name):
    """
    Улучшенная очистка для повышения точности:
    1. Нижний регистр.
    2. Замена 'ё' на 'е'.
    3. Разделение букв и цифр (100г -> 100 г).
    4. Удаление спецсимволов.
    """
    if not isinstance(name, str):
        return ""
    
    # 1. Базовая очистка
    name = name.lower()
    name = name.replace('ё', 'е')
    
    # 2. Важно: Разделяем цифры и буквы (чтобы "1кг" и "1 кг" стали одинаковыми)
    # Вставляет пробел между цифрой и буквой (100г -> 100 г)
    name = re.sub(r'(?<=\d)(?=[а-яa-z])', ' ', name)
    # Вставляет пробел между буквой и цифрой (№1 -> № 1)
    name = re.sub(r'(?<=[а-яa-z])(?=\d)', ' ', name)
    
    # 3. Заменяем запятые в цифрах на точки (3,2% -> 3.2%)
    name = name.replace(',', '.')
    
    # 4. Удаляем всё, кроме букв, цифр и пробелов (убираем кавычки, скобки, №, /)
    name = re.sub(r'[^a-zа-я0-9\s\.]', ' ', name)
    
    # 5. Убираем лишние пробелы
    return re.sub(r'\s+', ' ', name).strip()

def find_best_column(df, target_type):
    """
    Умный поиск колонок (игнорирует артикулы при поиске названий).
    """
    cols = df.columns
    cols_lower = [str(c).lower() for c in cols]
    
    if target_type == 'name':
        # 1. Приоритет: явное "наименование"
        strict_keywords = ['наименование', 'название', 'name']
        for k in strict_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name:
                    return cols[i]
        
        # 2. Поиск "товар", исключая "код"
        soft_keywords = ['товар', 'продукт', 'product', 'item']
        for k in soft_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name and not any(x in col_name for x in ['код', 'id', 'sku', 'code', 'art']):
                    return cols[i]

    elif target_type == 'price':
        keywords = ['цена', 'price', 'cost', 'сумма', 'rub', 'руб']
        for k in keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name:
                    return cols[i]
    return None

def process_files(file1, file2, threshold):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    # Поиск колонок
    name_col1 = find_best_column(df1, 'name')
    price_col1 = find_best_column(df1, 'price')
    
    name_col2 = find_best_column(df2, 'name')
    price_col2 = find_best_column(df2, 'price')

    st.info(f"""
    **Найдено:**
    Файл 1: Товар='{name_col1}'
    Файл 2: Товар='{name_col2}'
    """)

    if not name_col1 or not price_col1 or not name_col2 or not price_col2:
        st.error("Не удалось найти колонки. Проверьте заголовки.")
        return None

    # Применяем УЛУЧШЕННУЮ очистку
    df1['clean_name'] = df1[name_col1].apply(aggressive_clean_name)
    df2['clean_name'] = df2[name_col2].apply(aggressive_clean_name)
    
    # Векторизация (настраиваем n-граммы для лучшего поиска частей слов)
    # ngram_range=(2, 4) - ищет совпадения по частям слов длиной от 2 до 4 букв
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    
    try:
        tfidf_matrix1 = vectorizer.fit_transform(df1['clean_name'].astype(str))
        tfidf_matrix2 = vectorizer.transform(df2['clean_name'].astype(str))
    except ValueError:
        st.error("Ошибка обработки текста. Возможно, файлы пустые или содержат некорректные данные.")
        return None
    
    cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
    
    matches = []
    
    progress_bar = st.progress(0)
    total_items = len(df1)
    
    for i in range(total_items):
        if i % (total_items // 10 + 1) == 0:
            progress_bar.progress(i / total_items)
            
        best_idx = cosine_sim[i].argmax()
        score = cosine_sim[i][best_idx]
        
        if score > threshold:
            matches.append({
                'Товар (Наш)': df1.iloc[i][name_col1],
                'Товар (Конкурент)': df2.iloc[best_idx][name_col2],
                'Сходство (%)': round(score * 100, 1),
                'Цена (Наша)': df1.iloc[i][price_col1],
                'Цена (Конкурент)': df2.iloc[best_idx][price_col2]
            })
            
    progress_bar.progress(100)
    
    if not matches:
        return None
        
    res_df = pd.DataFrame(matches)
    res_df['Разница'] = res_df['Цена (Наша)'] - res_df['Цена (Конкурент)']
    
    cols_order = ['Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    return res_df[cols_order]

# --- UI ---

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Файл 1 (Основной)", type=['xlsx', 'xls'], key="f1")
with col2:
    file2 = st.file_uploader("Файл 2 (Конкурент)", type=['xlsx', 'xls'], key="f2")

# Добавили пояснение к слайдеру
threshold_val = st.slider(
    "Порог сходства", 
    min_value=0.0, max_value=1.0, value=0.60, step=0.05,
    help="Рекомендуемое значение: 0.60 - 0.70. Если ставить ниже, будет много ошибок. Если выше — найдет только полные копии."
)

if file1 and file2:
    if st.button("🚀 Сравнить", type="primary"):
        res = process_files(file1, file2, threshold_val)
        if res is not None and not res.empty:
            st.success(f"Найдено совпадений: {len(res)}")
            st.dataframe(res, use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                res.to_excel(writer, index=False)
            
            st.download_button("Скачать Excel", buffer.getvalue(), "result_exact.xlsx")
        else:
            st.warning("Ничего не найдено. Попробуйте уменьшить порог сходства.")
