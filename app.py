import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Настройка страницы
st.set_page_config(page_title="Сравнение цен", page_icon="⚖️", layout="wide")

st.title("⚖️ Сравнение цен в двух прайс-листах")
st.markdown("""
Загрузите два Excel файла, и алгоритм автоматически найдет одинаковые товары 
(даже если названия написаны немного по-разному) и сравнит их цены.
""")

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[\s\W_]+', ' ', name).strip()
    return name

def process_files(file1, file2, threshold):
    # Читаем файлы
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    # Пытаемся автоматически найти колонки
    cols1 = df1.columns.str.lower()
    cols2 = df2.columns.str.lower()
    
    # Ищем колонки с названиями (содержат "наим", "name", "товар")
    name_col1 = df1.columns[cols1.str.contains('наим|name|товар|product')][0]
    price_col1 = df1.columns[cols1.str.contains('цен|price|cost')][0]
    
    name_col2 = df2.columns[cols2.str.contains('наим|name|товар|product')][0]
    price_col2 = df2.columns[cols2.str.contains('цен|price|cost')][0]

    # Очистка
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
        # Обновляем прогресс бар каждые 10%
        if i % (total_items // 10 + 1) == 0:
            progress_bar.progress(i / total_items)
            
        best_idx = cosine_sim[i].argmax()
        score = cosine_sim[i][best_idx]
        
        if score > threshold:
            matches.append({
                'Товар (Файл 1)': df1.iloc[i][name_col1],
                'Товар (Файл 2)': df2.iloc[best_idx][name_col2],
                'Сходство (%)': round(score * 100, 1),
                'Цена (Файл 1)': df1.iloc[i][price_col1],
                'Цена (Файл 2)': df2.iloc[best_idx][price_col2]
            })
            
    progress_bar.progress(100)
    
    if not matches:
        return None
        
    res_df = pd.DataFrame(matches)
    res_df['Разница'] = res_df['Цена (Файл 1)'] - res_df['Цена (Файл 2)']
    res_df = res_df.sort_values('Сходство (%)', ascending=False)
    
    return res_df

# --- Интерфейс ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Файл 1 (например, Лента)")
    file1 = st.file_uploader("Загрузить первый Excel", type=['xlsx', 'xls'], key="f1")

with col2:
    st.subheader("📁 Файл 2 (например, Мега)")
    file2 = st.file_uploader("Загрузить второй Excel", type=['xlsx', 'xls'], key="f2")

# Настройки точности (в скрываемом блоке)
with st.expander("⚙️ Настройки точности поиска"):
    threshold_val = st.slider(
        "Минимальный порог сходства названий", 
        min_value=0.0, max_value=1.0, value=0.65, step=0.05,
        help="Чем выше значение, тем строже поиск. 1.0 - полное совпадение."
    )

if file1 and file2:
    if st.button("🚀 Начать сравнение", type="primary"):
        with st.spinner('Анализируем прайс-листы...'):
            try:
                result_df = process_files(file1, file2, threshold_val)
                
                if result_df is not None and not result_df.empty:
                    st.success(f"Готово! Найдено совпадений: {len(result_df)}")
                    
                    # Метрики
                    total_diff = result_df['Разница'].sum()
                    st.metric("Общая разница в цене (по найденным товарам)", f"{total_diff:.2f} ₽")
                    
                    # Показ таблицы
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Кнопка скачивания
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Comparison')
                    
                    st.download_button(
                        label="📥 Скачать результат в Excel",
                        data=buffer.getvalue(),
                        file_name="sravnenie_result.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    st.warning("Совпадений не найдено. Попробуйте уменьшить порог сходства в настройках.")
            except Exception as e:
                st.error(f"Произошла ошибка при обработке: {e}")
                st.info("Убедитесь, что в файлах есть колонки с названием 'наименование'/'name' и ценой 'цена'/'price'")
