import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Сравнение цен (Оптимальный)", page_icon="⚖️", layout="wide")

st.title("⚖️ Сравнение цен (Баланс точности)")
st.markdown("""
Этот вариант находит "золотую середину": он различает бренды лучше, чем первый вариант, 
но не отбрасывает товары с опечатками, как второй.
""")

def aggressive_clean_name(name):
    """
    Качественная очистка: разделяет слипшиеся цифры и буквы, убирает мусор.
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().replace('ё', 'е')
    
    # Разделяем буквы и цифры (100г -> 100 г, №1 -> № 1)
    name = re.sub(r'(?<=\d)(?=[а-яa-z])', ' ', name)
    name = re.sub(r'(?<=[а-яa-z])(?=\d)', ' ', name)
    
    # Заменяем запятые в дробях на точки
    name = name.replace(',', '.')
    
    # Оставляем только буквы, цифры и точки
    return re.sub(r'[^a-zа-я0-9\s\.]', ' ', name).strip()

def find_best_column(df, target_type):
    cols = df.columns
    cols_lower = [str(c).lower() for c in cols]
    
    if target_type == 'name':
        keywords = ['наименование', 'название', 'name']
        for k in keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name: return cols[i]
        
        soft_keywords = ['товар', 'продукт', 'product', 'item']
        for k in soft_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name and not any(x in col_name for x in ['код', 'id', 'sku']):
                    return cols[i]

    elif target_type == 'price':
        keywords = ['цена', 'price', 'cost', 'сумма']
        for k in keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name: return cols[i]
    return None

def process_files(file1, file2, threshold, show_all):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    name_col1 = find_best_column(df1, 'name')
    price_col1 = find_best_column(df1, 'price')
    name_col2 = find_best_column(df2, 'name')
    price_col2 = find_best_column(df2, 'price')

    if not all([name_col1, price_col1, name_col2, price_col2]):
        st.error("Ошибка поиска колонок. Проверьте заголовки.")
        return None

    df1['clean_name'] = df1[name_col1].apply(aggressive_clean_name)
    df2['clean_name'] = df2[name_col2].apply(aggressive_clean_name)
    
    # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: ngram_range=(3, 5) ---
    # Мы ищем совпадения блоков по 3, 4 и 5 букв.
    # Это сильно повышает точность: "Увелка" и "Пассим" не имеют общих блоков по 3 буквы.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    
    try:
        tfidf_matrix1 = vectorizer.fit_transform(df1['clean_name'].astype(str))
        tfidf_matrix2 = vectorizer.transform(df2['clean_name'].astype(str))
    except ValueError:
        st.error("Файлы пустые или данные некорректны.")
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
        
        # Если включено "Показать все", мы добавляем даже плохие совпадения (с пометкой)
        if score > threshold or show_all:
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
    
    # Сортируем: сначала самые похожие
    res_df = res_df.sort_values('Сходство (%)', ascending=False)
    
    cols_order = ['Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    return res_df[cols_order]

# --- UI ---

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Файл 1", type=['xlsx', 'xls'], key="f1")
with col2:
    file2 = st.file_uploader("Файл 2", type=['xlsx', 'xls'], key="f2")

col_set1, col_set2 = st.columns(2)
with col_set1:
    # Понизили дефолтный порог, чтобы не терять данные
    threshold_val = st.slider("Порог уверенности", 0.0, 1.0, 0.50, 0.05)
with col_set2:
    st.write("") 
    st.write("") 
    # Галочка спасения
    show_all_rows = st.checkbox("Показать даже сомнительные совпадения", value=False)

if file1 and file2:
    if st.button("🚀 Сравнить", type="primary"):
        res = process_files(file1, file2, threshold_val, show_all_rows)
        if res is not None and not res.empty:
            st.success(f"Найдено пар: {len(res)}")
            st.dataframe(res, use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                res.to_excel(writer, index=False)
            
            st.download_button("Скачать Excel", buffer.getvalue(), "result_balanced.xlsx")
        else:
            st.warning("Ничего не найдено. Попробуйте поставить галочку 'Показать даже сомнительные'.")
