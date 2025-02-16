import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack


MODEL0_PATH = "router_model0.pkl"
MODEL_MATH_PATH = "router_model_math.pkl"
MODEL_PHYS_PATH = "router_model_phys.pkl"
TEXT_VECTORIZER_PATH = "text_vectorizer.pkl"
FORM_VECTORIZER_PATH = "form_vectorizer.pkl"
LE_BIN_PATH = "le_bin.pkl"
LE_FULL_PATH = "le_full.pkl"
MATH_LE_PATH = "math_le_final.pkl"
PHYS_LE_PATH = "phys_le_final.pkl"
TOPIC_PATH = "topics.csv"

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
def predict_router_final(model0, model_math, model_phys, X, math_val, phys_val, math_le_final, phys_le_final):
    """
    Сначала предсказывается основной класс с model0. Если предсказанный класс соответствует
    математике или физике, используется специализированная модель для уточнения.
    """
    y0 = model0.predict(X)
    y_final = []
    # Для каждой строки данных
    for i in range(X.shape[0]):
        if y0[i] == math_val:
            pred = model_math.predict(X[i])
            y_final.append(pred[0])
        elif y0[i] == phys_val:
            pred = model_phys.predict(X[i])
            y_final.append(pred[0])
        else:
            y_final.append(y0[i])
    return np.array(y_final)

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

        # Получаем предсказание (числовой id темы)
        y_pred = predict_router_final(final_model0, final_model_math, final_model_phys,
                                      X, math_val, phys_val, math_le_final, phys_le_final)

        # Декодируем предсказанный id в наименование темы
        # Если в таблице topic_df столбцы называются "id" и "name"
        topic_row = topic_df[topic_df['id'] == y_pred[0]]
        if not topic_row.empty:
            topic_name = topic_row['name'].values[0]
        else:
            topic_name = "Неизвестная тема"

        st.success(f"Предсказанная тема: **{topic_name}**")
    else:
        st.error("Пожалуйста, введите текст задачи.")
