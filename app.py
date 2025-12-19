import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Сравнение цен (Бренд-контроль)", page_icon="🛡️", layout="wide")

st.title("🛡️ Сравнение цен (с учетом Брендов)")
st.markdown("""
Этот алгоритм строже относится к несовпадению слов. 
Если в названиях **разные бренды** (например, Увелка и Пассим), сходство будет сильно снижено.
""")

def aggressive_clean_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().replace('ё', 'е')
    name = re.sub(r'(?<=\d)(?=[а-яa-z])', ' ', name)
    name = re.sub(r'(?<=[а-яa-z])(?=\d)', ' ', name)
    name = name.replace(',', '.')
    return re.sub(r'[^a-zа-я0-9\s\.]', ' ', name).strip()

def get_tokens(text):
    """Разбивает текст на множество слов (set)"""
    return set(text.split())

def calculate_similarity_with_penalty(df1, df2, name_col1, name_col2):
    """
    Вычисляет сходство, но штрафует, если в названиях есть разные уникальные слова.
    """
    # 1. Базовый TF-IDF (как раньше)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    
    corpus = df1['clean_name'].tolist() + df2['clean_name'].tolist()
    vectorizer.fit(corpus)
    
    tfidf1 = vectorizer.transform(df1['clean_name'])
    tfidf2 = vectorizer.transform(df2['clean_name'])
    
    cosine_sim = cosine_similarity(tfidf1, tfidf2)
    
    # 2. Штраф за несовпадение слов (Бренд-контроль)
    # Если слова отличаются, мы уменьшаем score
    
    matches = []
    
    # Чтобы ускорить, предварительно разобьем все на слова
    tokens1 = [get_tokens(n) for n in df1['clean_name']]
    tokens2 = [get_tokens(n) for n in df2['clean_name']]
    
    total_items = len(df1)
    progress_bar = st.progress(0)

    for i in range(total_items):
        if i % (total_items // 10 + 1) == 0:
            progress_bar.progress(i / total_items)

        # Берем топ-5 кандидатов по TF-IDF, чтобы не перебирать всех
        # (это ускоряет работу)
        best_candidates_indices = cosine_sim[i].argsort()[-5:][::-1]
        
        best_score = 0
        best_match_idx = -1
        
        for idx in best_candidates_indices:
            raw_score = cosine_sim[i][idx]
            
            # Логика штрафа:
            # Находим слова, которые есть в одном названии, но нет в другом
            t1 = tokens1[i]
            t2 = tokens2[idx]
            
            # Симметричная разность (слова, которые не совпали)
            diff = t1.symmetric_difference(t2)
            
            # Если "разных" слов слишком много относительно длины названия, штрафуем
            # Увелка vs Пассим -> diff = {'увелка', 'пассим'} (2 слова)
            penalty = 0.0
            
            if len(diff) > 0:
                # Штрафуем на 15% за каждое несовпадающее слово
                # Но не больше 50%
                penalty = min(len(diff) * 0.15, 0.5)
            
            final_score = raw_score - penalty
            
            if final_score > best_score:
                best_score = final_score
                best_match_idx = idx
        
        matches.append((best_match_idx, best_score))
            
    progress_bar.progress(100)
    return matches

def find_best_column(df, target_type):
    cols = df.columns
    cols_lower = [str(c).lower() for c in cols]
    if target_type == 'name':
        keywords = ['наименование', 'название', 'name', 'товар', 'product']
        # Исключаем коды
        for k in keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name and not any(x in col_name for x in ['код', 'id', 'sku']):
                    return cols[i]
    elif target_type == 'price':
        keywords = ['цена', 'price', 'rub']
        for k in keywords:
            for i, col_name in enumerate(cols_lower):
                if k in col_name:
                    return cols[i]
    return None

def process_files(file1, file2, threshold):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    name_col1 = find_best_column(df1, 'name')
    price_col1 = find_best_column(df1, 'price')
    name_col2 = find_best_column(df2, 'name')
    price_col2 = find_best_column(df2, 'price')

    if not all([name_col1, price_col1, name_col2, price_col2]):
        st.error("Ошибка поиска колонок")
        return None

    df1['clean_name'] = df1[name_col1].apply(aggressive_clean_name)
    df2['clean_name'] = df2[name_col2].apply(aggressive_clean_name)
    
    # Запускаем умное сравнение
    results = calculate_similarity_with_penalty(df1, df2, name_col1, name_col2)
    
    final_matches = []
    for i, (best_idx, score) in enumerate(results):
        if score > threshold:
            final_matches.append({
                'Товар (Наш)': df1.iloc[i][name_col1],
                'Товар (Конкурент)': df2.iloc[best_idx][name_col2],
                'Сходство (%)': round(score * 100, 1),
                'Цена (Наша)': df1.iloc[i][price_col1],
                'Цена (Конкурент)': df2.iloc[best_idx][price_col2]
            })
            
    if not final_matches:
        return None
        
    res_df = pd.DataFrame(final_matches)
    res_df['Разница'] = res_df['Цена (Наша)'] - res_df['Цена (Конкурент)']
    cols_order = ['Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    return res_df[cols_order]

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Файл 1", type=['xlsx', 'xls'], key="f1")
with col2:
    file2 = st.file_uploader("Файл 2", type=['xlsx', 'xls'], key="f2")

threshold_val = st.slider("Порог сходства", 0.0, 1.0, 0.60, 0.05)

if file1 and file2:
    if st.button("🚀 Сравнить", type="primary"):
        res = process_files(file1, file2, threshold_val)
        if res is not None and not res.empty:
            st.success(f"Найдено: {len(res)}")
            st.dataframe(res, use_container_width=True)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                res.to_excel(writer, index=False)
            st.download_button("Скачать Excel", buffer.getvalue(), "result.xlsx")
        else:
            st.warning("Ничего не найдено.")
