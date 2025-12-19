import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Сравнение цен (Simple)", page_icon="⚖️", layout="wide")

st.title("⚖️ Сравнение цен (Только товары)")
st.markdown("""
Загрузите два прайс-листа. Скрипт найдет похожие товары и сравнит цены.
Коды товаров в итоговую таблицу не добавляются.
""")

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    return re.sub(r'[\s\W_]+', ' ', name).strip()

def find_best_column(df, target_type):
    """
    Умный поиск колонок.
    Игнорирует колонки с кодами при поиске названий, чтобы не терять совпадения.
    """
    cols = df.columns
    cols_lower = [c.lower() for c in cols]
    
    if target_type == 'name':
        # 1. Ищем явное "наименование" или "название"
        strict_keywords = ['наименование', 'название', 'name']
        for k in strict_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name:
                    return cols[i]
        
        # 2. Ищем "товар", НО строго исключаем "код", "id", "sku"
        # Это решает проблему, когда "Код товара" считался названием
        soft_keywords = ['товар', 'продукт', 'product', 'item']
        for k in soft_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name and not any(x in col_name for x in ['код', 'id', 'sku', 'code']):
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
    
    # Ищем только Название и Цену
    name_col1 = find_best_column(df1, 'name')
    price_col1 = find_best_column(df1, 'price')
    
    name_col2 = find_best_column(df2, 'name')
    price_col2 = find_best_column(df2, 'price')

    # Диагностика для пользователя
    st.info(f"""
    **Распознанные колонки:**
    Файл 1: Товар='{name_col1}', Цена='{price_col1}'
    Файл 2: Товар='{name_col2}', Цена='{price_col2}'
    """)

    if not name_col1 or not price_col1 or not name_col2 or not price_col2:
        st.error("Не удалось найти колонки с названием или ценой. Проверьте заголовки файлов.")
        return None

    df1['clean_name'] = df1[name_col1].apply(clean_name)
    df2['clean_name'] = df2[name_col2].apply(clean_name)
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix1 = vectorizer.fit_transform(df1['clean_name'].astype(str))
    tfidf_matrix2 = vectorizer.transform(df2['clean_name'].astype(str))
    
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
    
    # Итоговый порядок колонок (без кодов)
    cols_order = ['Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    return res_df[cols_order]

# --- UI ---

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Файл 1 (Основной)", type=['xlsx', 'xls'], key="f1")
with col2:
    file2 = st.file_uploader("Файл 2 (Конкурент)", type=['xlsx', 'xls'], key="f2")

threshold_val = st.slider("Порог сходства", 0.0, 1.0, 0.65, 0.05)

if file1 and file2:
    if st.button("🚀 Сравнить", type="primary"):
        try:
            res = process_files(file1, file2, threshold_val)
            if res is not None and not res.empty:
                st.success(f"Найдено совпадений: {len(res)}")
                st.dataframe(res, use_container_width=True)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    res.to_excel(writer, index=False)
                
                st.download_button("Скачать Excel", buffer.getvalue(), "result.xlsx")
            else:
                st.warning("Ничего не найдено.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
