import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack


MODEL0_PATH = "04_task_classification/router_model0.pkl"
MODEL_MATH_PATH = "04_task_classification/router_model_math.pkl"
MODEL_PHYS_PATH = "04_task_classification/router_model_phys.pkl"
TEXT_VECTORIZER_PATH = "04_task_classification/text_vectorizer.pkl"
FORM_VECTORIZER_PATH = "04_task_classification/form_vectorizer.pkl"
LE_BIN_PATH = "04_task_classification/le_bin.pkl"
LE_FULL_PATH = "04_task_classification/le_full.pkl"
MATH_LE_PATH = "04_task_classification/math_le_final.pkl"
PHYS_LE_PATH = "04_task_classification/phys_le_final.pkl"
TOPIC_PATH = "04_task_classification/topics.csv"

# Natasha
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

# Для LaTeX -> sympy
import sympy
from sympy.parsing.latex import parse_latex

# --- Загрузка моделей и векторизаторов ---
@st.cache_resource
def load_models_and_encoders():
    final_model0 = joblib.load(MODEL0_PATH)
    final_model_math = joblib.load(MODEL_MATH_PATH)
    final_model_phys = joblib.load(MODEL_PHYS_PATH)
    text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
    form_vectorizer = joblib.load(FORM_VECTORIZER_PATH)
    math_le_final = joblib.load(MATH_LE_PATH)
    phys_le_final = joblib.load(PHYS_LE_PATH)
    le_bin = joblib.load(LE_BIN_PATH)
    le_full = joblib.load(LE_FULL_PATH)
    topic_df = pd.read_csv(TOPIC_PATH)
    return (final_model0, final_model_math, final_model_phys,
            text_vectorizer, form_vectorizer, le_bin,
            math_le_final, phys_le_final, le_full, topic_df)

(final_model0, final_model_math, final_model_phys,
 text_vectorizer, form_vectorizer, le_bin,
 math_le_final, phys_le_final, le_full, topic_df) = load_models_and_encoders()

# Определяем числовые значения для математики и физики
if le_bin is not None:
    mapping = {cls: le_bin.transform([cls])[0] for cls in le_bin.classes_}
    math_val = mapping.get(1, 1)
    phys_val = mapping.get(193, 193)
else:
    math_val = 1
    phys_val = 193


# --- Функции предобработки ---
def preprocess_russian_text_natasha(text: str) -> str:
    # Удаляем LaTeX формулы
    text_no_latex = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text_no_latex = re.sub(r'\$.*?\$', '', text_no_latex, flags=re.DOTALL)
    text_no_latex = re.sub(r'\\\(.+?\\\)', '', text_no_latex, flags=re.DOTALL)
    text_lower = text_no_latex.lower()
    text_clean = re.sub(r'[^a-zа-яё0-9]+', ' ', text_lower).strip()
    doc = Doc(text_clean)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    lemmas = [token.lemma for token in doc.tokens]
    return " ".join(lemmas)
    
def parse_latex_formula(formula_str: str) -> str:
    formula_str = formula_str.replace('\x0crac', r'\frac')
    replacements = [(r'\\left', ''), (r'\\right', ''), (r'\\text', '')]
    cleaned = formula_str
    for pat, repl in replacements:
        cleaned = re.sub(pat, repl, cleaned)
    try:
        expr = parse_latex(cleaned)
        return str(expr)
    except Exception:
        return cleaned
        
def extract_formulas_advanced(text: str) -> str:
    pattern = r'(?:\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)|\\begin\{equation\}(.*?)\\end\{equation\})'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    formulas_found = []
    for triple in matches:
        items = [s for s in triple if s.strip() != ""]
        if not items:
            continue
        f_str = items[0].strip().replace("\\ ", " ")
        parsed = parse_latex_formula(f_str)
        formulas_found.append(parsed)
    return " ".join(formulas_found) if formulas_found else ""

# --- Функция маршрутизации предсказаний ---
def predict_router_final(model0, model_math, model_phys, X, math_val, phys_val, math_le, phys_le):
    # Если X представлено в разреженном формате, преобразуем его в плотный массив
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    y0_pred = np.array(model0.predict(X_dense))
    st.success(f"1:{y0_pred}")
    y1_pred = np.empty(len(X_dense), dtype=object)  # Используем тип object, чтобы сохранить декодированные метки
    
    st.success(f'math_val :{math_val}, phys_val :{phys_val}')
    # Индексы, где базовая модель определила математику
    math_idx = np.where(y0_pred == math_val)[0]
    st.success(f"2:{math_idx}")
    # Индексы, где базовая модель определила физику
    phys_idx = np.where(y0_pred == phys_val)[0]
    st.success(f"3:{phys_idx}")
    
    # Для математики: получаем закодированные предсказания и декодируем их
    if len(math_idx) > 0:
        math_pred_enc = model_math.predict(X_dense[math_idx])
        # inverse_transform ожидает 1D-массив, поэтому передаём массив
        y1_pred[math_idx] = math_le.inverse_transform(math_pred_enc)
        st.success(f"4:{y1_pred[math_idx]}")
    
    # Для физики: аналогично
    if len(phys_idx) > 0:
        phys_pred_enc = model_phys.predict(X_dense[phys_idx])
        y1_pred[phys_idx] = phys_le.inverse_transform(phys_pred_enc)
        st.success(f"5:{y1_pred[phys_idx]}")
    
    # Для остальных индексов оставляем исходное предсказание базовой модели
    other_idx = np.where((y0_pred != math_val) & (y0_pred != phys_val))[0]
    if len(other_idx) > 0:
        y1_pred[other_idx] = y0_pred[other_idx]
    
    return y1_pred

# --- Интерфейс Streamlit ---
st.title("Классификация текстовых задач")

st.markdown("Введите текст задачи, и модель предскажет тему (уровень 0) и подтему (уровень 1).")

user_text = st.text_area("Введите текст задачи:")

if st.button("Предсказать"):
    if user_text.strip():
        # Применяем функции предобработки
        processed_text = preprocess_russian_text_natasha(user_text)
        processed_form = extract_formulas_advanced(user_text)
        X_text = text_vectorizer.transform([processed_text])
        X_form = form_vectorizer.transform([processed_form])
        X = hstack([X_text, X_form])

        # Получаем предсказание для уровня 0
        level0_pred = final_model0.predict(X)
        st.success(f"level0_pred {level0_pred[0]}")
        # Используем маршрутизатор для получения подтемы (уровень 1)
        level1_pred = predict_router_final(final_model0, final_model_math, final_model_phys,
                                           X, math_val, phys_val, math_le_final, phys_le_final)
        st.success(f"level1_pred {level1_pred[0]}")
        # Декодируем числовые идентификаторы в наименования тем, используя таблицу topic_df.
        # Предполагается, что в таблице topic_df есть столбцы 'id' и 'name' для уровня 0 и уровня 1.
        # Если используется одна таблица для обоих уровней, то может потребоваться различать их по диапазону id
        # или использовать две разные таблицы.
        def decode_topic(topic_id):
            row = topic_df[topic_df['id'] == topic_id]
            st.success(f'row:{row}')
            return row['name'].values[0] if not row.empty else "Неизвестная тема"
        
        # Декодируем результаты
        st.success(f'dftype:{type(topic_df['id'][0])}')
        st.success(f'encoder1: {int(le_bin.inverse_transform([level0_pred[0]])[0])}')
        topic_level0_name = decode_topic(int(le_bin.inverse_transform([level0_pred[0]])[0]))
        st.success(f"topic_level0_name {topic_level0_name}")
        topic_level1_name = decode_topic(int(le_full.inverse_transform([level1_pred[0]])[0]))
        st.success(f"topic_level1_name {topic_level1_name}")
        
        st.success(f"Тема уровня 0: **{topic_level0_name}**")
        st.success(f"Подтема уровня 1: **{topic_level1_name}**")
    else:
        st.error("Пожалуйста, введите текст задачи.")
