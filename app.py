import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Настройка страницы
st.set_page_config(page_title="Сравнение цен + Код", page_icon="⚖️", layout="wide")

st.title("⚖️ Сравнение цен (с артикулами)")
st.markdown("""
Загрузите два файла:
1. **Основной файл** (содержит: Код, Наименование, Цена)
2. **Файл конкурента** (содержит: Наименование, Цена)
""")

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[\s\W_]+', ' ', name).strip()
    return name

def find_column(df, keywords):
    """Ищет колонку, название которой содержит одно из ключевых слов"""
    cols = df.columns.str.lower()
    for keyword in keywords:
        # Ищем точное или частичное совпадение
        found = [c for c in df.columns if keyword in c.lower()]
        if found:
            return found[0]
    return None

def process_files(file1, file2, threshold):
    # Читаем файлы
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    # --- Поиск колонок в Файле 1 (Основной) ---
    # 1. Ищем Код (Артикул)
    code_col1 = find_column(df1, ['код', 'code', 'sku', 'артикул', 'id', 'art'])
    # 2. Ищем Наименование
    name_col1 = find_column(df1, ['наим', 'name', 'товар', 'product'])
    # 3. Ищем Цену
    price_col1 = find_column(df1, ['цен', 'price', 'cost', 'sum'])
    
    # --- Поиск колонок в Файле 2 (Конкурент) ---
    name_col2 = find_column(df2, ['наим', 'name', 'товар', 'product'])
    price_col2 = find_column(df2, ['цен', 'price', 'cost', 'sum'])

    # Проверка, все ли нашлось
    if not name_col1 or not price_col1:
        st.error(f"В файле 1 не найдены колонки Наименования или Цены. Найденные колонки: {list(df1.columns)}")
        return None
    if not name_col2 or not price_col2:
        st.error(f"В файле 2 не найдены колонки Наименования или Цены. Найденные колонки: {list(df2.columns)}")
        return None

    # Очистка имен для поиска
    df1['clean_name'] = df1[name_col1].apply(clean_name)
    df2['clean_name'] = df2[name_col2].apply(clean_name)
    
    # Векторизация и поиск
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
            # Формируем строку результата
            row = {
                'Товар (Наш)': df1.iloc[i][name_col1],
                'Товар (Конкурент)': df2.iloc[best_idx][name_col2],
                'Сходство (%)': round(score * 100, 1),
                'Цена (Наша)': df1.iloc[i][price_col1],
                'Цена (Конкурент)': df2.iloc[best_idx][price_col2]
            }
            # Если нашли колонку с кодом, добавляем её в начало
            if code_col1:
                row['Код товара'] = df1.iloc[i][code_col1]
            else:
                row['Код товара'] = '—'
                
            matches.append(row)
            
    progress_bar.progress(100)
    
    if not matches:
        return None
        
    res_df = pd.DataFrame(matches)
    
    # Считаем разницу
    res_df['Разница'] = res_df['Цена (Наша)'] - res_df['Цена (Конкурент)']
    
    # Красивый порядок колонок
    cols = ['Код товара', 'Товар (Наш)', 'Товар (Конкурент)', 'Цена (Наша)', 'Цена (Конкурент)', 'Разница', 'Сходство (%)']
    # Если каких-то колонок нет (вдруг ошибка), берем только те, что есть
    final_cols = [c for c in cols if c in res_df.columns]
    res_df = res_df[final_cols]
    
    return res_df

# --- Интерфейс ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Основной файл (с кодами)")
    file1 = st.file_uploader("Загрузить файл", type=['xlsx', 'xls'], key="f1")

with col2:
    st.subheader("2. Файл для сравнения")
    file2 = st.file_uploader("Загрузить файл", type=['xlsx', 'xls'], key="f2")

with st.expander("⚙️ Настройки точности поиска"):
    threshold_val = st.slider(
        "Минимальный порог сходства", 
        min_value=0.0, max_value=1.0, value=0.65, step=0.05
    )

if file1 and file2:
    if st.button("🚀 Начать сравнение", type="primary"):
        with st.spinner('Ищем совпадения и коды...'):
            try:
                result_df = process_files(file1, file2, threshold_val)
                
                if result_df is not None and not result_df.empty:
                    st.success(f"Готово! Найдено совпадений: {len(result_df)}")
                    
                    total_diff = result_df['Разница'].sum()
                    st.metric("Общая разница в цене", f"{total_diff:.2f} ₽")
                    
                    st.dataframe(result_df, use_container_width=True)
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Сравнение')
                    
                    st.download_button(
                        label="📥 Скачать таблицу с кодами (Excel)",
                        data=buffer.getvalue(),
                        file_name="sravnenie_s_kodami.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    st.warning("Совпадений не найдено. Попробуйте уменьшить порог.")
            except Exception as e:
                st.error(f"Ошибка: {e}")
