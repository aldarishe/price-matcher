import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Сравнение цен", page_icon="⚖️", layout="wide")

st.title("⚖️ Сравнение цен")
st.markdown("""
не обещаю 100% совпадения, просто я так увидел:-)
ИНСТРУКЦИЯ:
- файл 1 это наш прайс-лист, файл 2 - прай-лист конкурента
- загружаемые файлы должны содержать столбцы: наименование и цена
- наименование должно быть полным (бренд, вес, объём и т.д.), а цена в числовом формате (например 2,44)
- точность совпадений регулируйте ползунком
,
""")

def aggressive_clean_name(name):
    """Очистка для векторизации (текст)"""
    if not isinstance(name, str):
        return ""
    name = name.lower().replace('ё', 'е')
    name = name.replace(',', '.')
    # Разделяем цифры и буквы для лучшего чтения
    name = re.sub(r'(?<=\d)(?=[а-яa-z])', ' ', name)
    name = re.sub(r'(?<=[а-яa-z])(?=\d)', ' ', name)
    return re.sub(r'[^a-zа-я0-9\s\.]', ' ', name).strip()

def extract_weight(text):
    """
    Вытаскивает вес из строки и переводит в граммы/единицы.
    Возвращает число или None, если вес не найден.
    """
    if not isinstance(text, str):
        return None
    
    text = text.lower().replace(',', '.')
    
    # Шаблоны для поиска (число + пробел? + единица)
    patterns = {
        r'(\d+\.?\d*)\s*кг': 1000,   # кг -> г
        r'(\d+\.?\d*)\s*г(?!\w)': 1, # г -> г
        r'(\d+\.?\d*)\s*л': 1000,    # л -> мл
        r'(\d+\.?\d*)\s*мл': 1       # мл -> мл
    }
    
    for pattern, multiplier in patterns.items():
        match = re.search(pattern, text)
        if match:
            try:
                val = float(match.group(1))
                return val * multiplier
            except:
                continue
    return None

def find_best_column(df, target_type):
    cols = [str(c) for c in df.columns]
    cols_lower = [c.lower() for c in cols]
    
    if target_type == 'name':
        keywords = ['наименование', 'название', 'name']
        for k in keywords:
            for i, col in enumerate(cols_lower):
                if k in col: return cols[i]
        for i, col in enumerate(cols_lower):
            if 'товар' in col and 'код' not in col: return cols[i]

    elif target_type == 'price':
        keywords = ['цена', 'price', 'rub']
        for k in keywords:
            for i, col in enumerate(cols_lower):
                if k in col: return cols[i]
    return None

def process_files(file1, file2, threshold):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    name_col1 = find_best_column(df1, 'name')
    price_col1 = find_best_column(df1, 'price')
    name_col2 = find_best_column(df2, 'name')
    price_col2 = find_best_column(df2, 'price')

    if not all([name_col1, price_col1, name_col2, price_col2]):
        st.error("Не найдены колонки.")
        return None

    # 1. Текстовая подготовка
    df1['clean_name'] = df1[name_col1].apply(aggressive_clean_name)
    df2['clean_name'] = df2[name_col2].apply(aggressive_clean_name)
    
    # 2. Извлечение веса (из оригинальных названий)
    df1['weight_val'] = df1[name_col1].apply(extract_weight)
    df2['weight_val'] = df2[name_col2].apply(extract_weight)

    # 3. Векторизация и базовый поиск
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    try:
        tfidf_matrix1 = vectorizer.fit_transform(df1['clean_name'].astype(str))
        tfidf_matrix2 = vectorizer.transform(df2['clean_name'].astype(str))
    except:
        st.error("Ошибка данных.")
        return None
    
    cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
    
    matches = []
    
    progress = st.progress(0)
    total = len(df1)
    
    for i in range(total):
        if i % (total // 10 + 1) == 0: progress.progress(i / total)
            
        # Берем ТОП-3 кандидата по тексту
        best_candidates = cosine_sim[i].argsort()[-3:][::-1]
        
        final_best_idx = -1
        final_best_score = 0
        
        weight1 = df1.iloc[i]['weight_val']
        
        for idx in best_candidates:
            score = cosine_sim[i][idx]
            
            # --- ЛОГИКА ПРОВЕРКИ ВЕСА ---
            weight2 = df2.iloc[idx]['weight_val']
            
            # Если у обоих товаров определился вес
            if weight1 is not None and weight2 is not None:
                # Считаем разницу в процентах
                diff_percent = abs(weight1 - weight2) / max(weight1, weight2)
                
                # Если разница больше 10% (например 800г и 1000г = 20% разницы)
                # То это РАЗНЫЕ товары, даже если текст совпадает
                if diff_percent > 0.1:
                    score = 0.0 # Обнуляем сходство
            
            if score > final_best_score:
                final_best_score = score
                final_best_idx = idx
        
        if final_best_score > threshold:
            matches.append({
                'Товар (Наш)': df1.iloc[i][name_col1],
                'Товар (Конкурент)': df2.iloc[final_best_idx][name_col2],
                'Сходство (%)': round(final_best_score * 100, 1),
                'Цена (Наша)': df1.iloc[i][price_col1],
                'Цена (Конкурент)': df2.iloc[final_best_idx][price_col2]
            })
            
    progress.progress(100)
    
    if not matches:
        return None
        
    res = pd.DataFrame(matches)
    res['Разница'] = res['Цена (Наша)'] - res['Цена (Конкурент)']
    cols = ['Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    return res[cols]

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Файл 1", type=['xlsx', 'xls'], key="f1")
with col2:
    file2 = st.file_uploader("Файл 2", type=['xlsx', 'xls'], key="f2")

threshold_val = st.slider("Порог", 0.0, 1.0, 0.60, 0.05)

if file1 and file2:
    if st.button("🚀 Сравнить", type="primary"):
        r = process_files(file1, file2, threshold_val)
        if r is not None and not r.empty:
            st.success(f"Найдено: {len(r)}")
            st.dataframe(r, use_container_width=True)
            b = io.BytesIO()
            with pd.ExcelWriter(b, engine='xlsxwriter') as w: r.to_excel(w, index=False)
            st.download_button("Скачать", b.getvalue(), "result_checked.xlsx")
        else:
            st.warning("Ничего не найдено.")







