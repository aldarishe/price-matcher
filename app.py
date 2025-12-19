import pandas as pd
import numpy as np
import re
import io
import ipywidgets as widgets
from IPython.display import display, clear_output, FileLink
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Логика обработки (скрыта от пользователя в функции) ---

def clean_name(name):
    if not isinstance(name, str):
        return ""
    return re.sub(r'[\s\W_]+', ' ', name.lower()).strip()

def find_columns(df):
    """Автоматический поиск колонок с названием и ценой"""
    cols = df.columns.str.lower()
    name_col = next((c for c in df.columns if 'наим' in c.lower() or 'name' in c.lower() or 'товар' in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if 'цен' in c.lower() or 'price' in c.lower()), df.columns[1])
    return name_col, price_col

def process_comparison(file1_content, file2_content, threshold):
    # Чтение файлов из байтов (памяти)
    df1 = pd.read_excel(io.BytesIO(file1_content))
    df2 = pd.read_excel(io.BytesIO(file2_content))
    
    # Определение колонок
    name1, price1 = find_columns(df1)
    name2, price2 = find_columns(df2)
    
    # Подготовка данных
    df1['clean'] = df1[name1].apply(clean_name)
    df2['clean'] = df2[name2].apply(clean_name)
    
    # Векторизация и поиск (TF-IDF)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix1 = vectorizer.fit_transform(df1['clean'].astype(str))
    tfidf_matrix2 = vectorizer.transform(df2['clean'].astype(str))
    
    cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
    
    matches = []
    
    for i in range(len(df1)):
        best_idx = cosine_sim[i].argmax()
        score = cosine_sim[i][best_idx]
        
        if score > threshold:
            matches.append({
                'Товар_1': df1.iloc[i][name1],
                'Товар_2': df2.iloc[best_idx][name2],
                'Сходство': round(score, 2),
                'Цена_1': df1.iloc[i][price1],
                'Цена_2': df2.iloc[best_idx][price2]
            })
            
    if not matches:
        return None
        
    res_df = pd.DataFrame(matches)
    res_df['Разница'] = res_df['Цена_1'] - res_df['Цена_2']
    res_df = res_df.sort_values('Сходство', ascending=False)
    return res_df

# --- Интерфейс (Виджеты) ---

style = {'description_width': 'initial'}

# Виджеты загрузки
uploader_1 = widgets.FileUpload(accept='.xlsx', multiple=False, description='Загрузить файл 1 (Лента)')
uploader_2 = widgets.FileUpload(accept='.xlsx', multiple=False, description='Загрузить файл 2 (Мега)')

# Слайдер точности
slider_threshold = widgets.FloatSlider(
    value=0.65, min=0.1, max=1.0, step=0.05, 
    description='Порог точности:', style=style,
    layout=widgets.Layout(width='50%')
)

# Кнопка запуска
btn_run = widgets.Button(
    description='Сравнить цены',
    button_style='primary', # 'success', 'info', 'warning', 'danger' or ''
    icon='check'
)

# Область вывода результатов
output = widgets.Output()

def on_button_clicked(b):
    with output:
        clear_output()
        
        if not uploader_1.value or not uploader_2.value:
            print("⚠️ Пожалуйста, загрузите оба файла!")
            return
            
        print("⏳ Идет анализ данных... Пожалуйста, подождите.")
        
        try:
            # Получение контента файлов (для ipywidgets >= 8.0)
            content1 = uploader_1.value[0]['content'] if isinstance(uploader_1.value, tuple) else list(uploader_1.value.values())[0]['content']
            content2 = uploader_2.value[0]['content'] if isinstance(uploader_2.value, tuple) else list(uploader_2.value.values())[0]['content']
            
            result_df = process_comparison(content1, content2, slider_threshold.value)
            
            clear_output()
            
            if result_df is not None:
                total_diff = result_df['Разница'].sum()
                color = "green" if total_diff < 0 else "red"
                
                display(widgets.HTML(f"<h3>✅ Готово! Найдено совпадений: {len(result_df)}</h3>"))
                display(widgets.HTML(f"<b>Общая разница: <span style='color:{color}'>{total_diff:.2f} ₽</span></b>"))
                
                # Показать таблицу (первые 10 строк)
                display(result_df.head(10))
                
                # Сохранение файла
                filename = 'comparison_result.xlsx'
                result_df.to_excel(filename, index=False)
                display(FileLink(filename, result_html=f'<h3>📥 <a href="{filename}" download>Скачать полный результат (Excel)</a></h3>'))
                
            else:
                print("🤷‍♂️ Совпадений не найдено. Попробуйте уменьшить порог точности.")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("Проверьте структуру файлов (нужны колонки с названием и ценой).")

btn_run.on_click(on_button_clicked)

# Компоновка интерфейса
ui = widgets.VBox([
    widgets.HBox([uploader_1, uploader_2]),
    slider_threshold,
    btn_run,
    output
])

display(ui)
