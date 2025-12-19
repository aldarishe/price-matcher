import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Сравнение цен v2.1", page_icon="⚖️", layout="wide")

st.title("⚖️ Сравнение цен (Исправленное)")
st.markdown("""
Загрузите файлы. Алгоритм теперь умнее определяет, где Название, а где Код товара, 
чтобы не терять совпадения.
""")

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    return re.sub(r'[\s\W_]+', ' ', name).strip()

def find_best_column(df, target_type):
    """
    Умный поиск колонок.
    target_type может быть: 'code', 'name', 'price'
    """
    cols = df.columns
    cols_lower = [c.lower() for c in cols]
    
    if target_type == 'code':
        # Приоритет: артикул, код, id, sku
        keywords = ['артикул', 'код', 'sku', 'id', 'art', 'code']
        for k in keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name:
                    return cols[i]
                    
    elif target_type == 'name':
        # Сначала ищем явное "наименование" или "название"
        strict_keywords = ['наименование', 'название', 'name']
        for k in strict_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name:
                    return cols[i]
        
        # Если не нашли, ищем "товар" или "продукт", НО исключаем колонки, где есть "код" или "id"
        soft_keywords = ['товар', 'продукт', 'product', 'item']
        for k in soft_keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name and 'код' not in col_name and 'id' not in col_name:
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
    
    # 1. Определяем колонки
    code_col1 = find_best_column(df1, 'code')
    name_col1 = find_best_column(df1, 'name')
    price_col1 = find_best_column(df1, 'price')
    
    name_col2 = find_best_column(df2, 'name')
    price_col2 = find_best_column(df2, 'price')

    # Диагностика (показываем пользователю, что нашел скрипт)
    st.info(f"""
    **Распознанные колонки:**
    Файл 1: Название='{name_col1}', Цена='{price_col1}', Код='{code_col1}'
    Файл 2: Название='{name_col2}', Цена='{price_col2}'
    """)

    if not name_col1 or not price_col1 or not name_col2 or not price_col2:
        st.error("Не удалось найти необходимые колонки (Название или Цена). Проверьте заголовки файлов.")
        return None

    df1['clean_name'] = df1[name_col1].apply(clean_name)
    df2['clean_name'] = df2[name_col2].apply(clean_name)
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix1 = vectorizer.fit_transform(df1['clean_name'].astype(str))
    tfidf_matrix2 = vectorizer.transform(df2['clean_name'].astype(str))
    
    cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
    
    matches = []
    
    # Прогресс бар
    progress_bar = st.progress(0)
    total_items = len(df1)
    
    for i in range(total_items):
        if i % (total_items // 10 + 1) == 0:
            progress_bar.progress(i / total_items)
            
        best_idx = cosine_sim[i].argmax()
        score = cosine_sim[i][best_idx]
        
        if score > threshold:
            row = {
                'Товар (Наш)': df1.iloc[i][name_col1],
                'Товар (Конкурент)': df2.iloc[best_idx][name_col2],
                'Сходство (%)': round(score * 100, 1),
                'Цена (Наша)': df1.iloc[i][price_col1],
                'Цена (Конкурент)': df2.iloc[best_idx][price_col2]
            }
            if code_col1:
                row['Код товара'] = df1.iloc[i][code_col1]
            else:
                row['Код товара'] = '—'
            matches.append(row)
            
    progress_bar.progress(100)
    
    if not matches:
        return None
        
    res_df = pd.DataFrame(matches)
    res_df['Разница'] = res_df['Цена (Наша)'] - res_df['Цена (Конкурент)']
    
    # Сортировка колонок
    cols_order = ['Код товара', 'Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    final_cols = [c for c in cols_order if c in res_df.columns]
    
    return res_df[final_cols]

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Файл 1 (Основной)", type=['xlsx', 'xls'], key="f1")
with col2:
    file2 = st.file_uploader("Файл 2 (Конкурент)", type=['xlsx', 'xls'], key="f2")

threshold_val = st.slider("Порог сходства", 0.0, 1.0, 0.65, 0.05)

if file1 and file2:
    if st.button("🚀 Сравнить", type="primary"):
        res = process_files(file1, file2, threshold_val)
        if res is not None and not res.empty:
            st.success(f"Найдено совпадений: {len(res)}")
            st.dataframe(res, use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                res.to_excel(writer, index=False)
            
            st.download_button("Скачать Excel", buffer.getvalue(), "result.xlsx")
        else:
            st.warning("Ничего не найдено. Проверьте колонки (см. синий блок выше).")
