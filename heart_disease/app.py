import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np

st.header('Оценка риска сердечных заболеваний')

# установим слайдер и кнопки для ввода показателей
gender = st.selectbox('Пол', ['М','Ж'],key='gender')

age = st.slider('Возраст', 1, 120, key='age')
height = st.slider('Рост', 1, 230, key='height')
weight = st.slider('Вес', 1, 200, key='weight')

ap_lo = st.slider('Нижнее давление', 1, 130, key='ap_lo')
ap_hi = st.slider('Верхнее давление', 50, 160, key='ap_hi')

lc, mc, rc, = st.columns(3)

with lc:
    chol = st.radio('Холестерин', (1,2,3),key='cholesterol')
with mc:
    gluc = st.radio('Глюкоза', (1, 2, 3), key='gluc')
with rc:
    smoke = st.checkbox('Курение', key='smoking')
    alco = st.checkbox('Алкоголь', key='alcohol')
    active = st.checkbox('Спорт', key='active')



#средние значения показателей, полученных на train, для обработки выбросов
mean_weight = 73.59976024455928
mean_height = 164.4003348133052
mean_aphi = 126.70251109978892
mean_aplo = 81.39544362762938

lb_height = 126
ub_height = 203
lb_weight = 31
ub_height = 116

lb_aphi = 60
ub_aphi = 200
lb_aplo = 60
ub_aplo = 110


gender_meaning = {
    'М': 2,
    'Ж': 1
}

df = pd.DataFrame([[age*365, gender_meaning[gender], height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active]],
                  columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                                 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])


# обрабатываем данные также, как обрабатывали их на тестовых данных
def replace_outliers_with_mean(df, column_name, lower_bound, upper_bound, mean):
    df_cleaned = df.copy()
    outliers = (df_cleaned[column_name] < lower_bound) | (df_cleaned[column_name] > upper_bound)
    df_cleaned.loc[outliers, column_name] = mean
    return df_cleaned

df = replace_outliers_with_mean(df, 'height', lb_height, ub_height, mean_height)
df = replace_outliers_with_mean(df, 'weight', lb_weight, ub_height, mean_height)

df.loc[df['ap_hi'] < 0, 'ap_hi']  = -df['ap_hi']
df.loc[df['ap_lo'] < 0, 'ap_lo']  = -df['ap_lo']
df['dif'] = df['ap_hi'] - df['ap_lo']
df.loc[(df['ap_lo'] >= 1000) & (df['ap_lo'] <2000) & (df['dif'] < 0), 'ap_lo']  = (df['ap_lo'] / 10).round(0)
df.loc[(df['ap_lo'] >= 5000) & (df['ap_lo'] <10000) & (df['dif'] < 0), 'ap_lo']  = (df['ap_lo'] / 100).round(0)
df.loc[(df['ap_lo'] >= 500) & (df['ap_lo'] <1000) & (df['dif'] < 0), 'ap_lo']  = (df['ap_lo'] / 10).round(0)
df.loc[(df['ap_hi'] >= 10) & (df['ap_hi'] <20) & (df['dif'] < 0), 'ap_hi']  = (df['ap_hi'] * 10).round(0)
df.loc[(df['ap_hi'] == 20) & (df['ap_hi'] == 20) & (df['dif'] < 0), 'ap_hi']  = (df['ap_hi'] * 6).round(0)
df.loc[(df['ap_hi'] >= 10000), 'ap_hi']  = (df['ap_hi'] / 100).round(0)
df.loc[(df['ap_hi'] >= 1000), 'ap_hi']  = (df['ap_hi'] / 10).round(0)
df.loc[(df['ap_hi'] >= 700), 'ap_hi']  = (df['ap_hi'] / 10).round(0)
df.loc[(df['ap_hi'] < 60), 'ap_hi']  = (df['ap_hi'] * 10).round(0)
df.loc[(df['ap_lo'] == 1), 'ap_lo']  = (df['ap_lo'] * 100).round(0)
df.loc[(df['ap_lo'] <= 10), 'ap_lo']  = (df['ap_lo'] * 10).round(0)
df.loc[(df['ap_lo'] >= 10000), 'ap_lo']  = (df['ap_lo'] / 100).round(0)
df.loc[(df['ap_hi'] == 10), 'ap_hi']  = (df['ap_hi'] * 10).round(0)
df['dif'] = df['ap_hi'] - df['ap_lo']
df.loc[(df['dif'] <0), 'ap_hi'], df.loc[(df['dif'] <0), 'ap_lo'] = df.loc[(df['dif'] <0), 'ap_lo'], df.loc[(df['dif'] <0), 'ap_hi']
df = df.drop('dif', axis = 1)

df = replace_outliers_with_mean(df, 'ap_hi', lb_aphi, ub_aphi, mean_aphi)
df = replace_outliers_with_mean(df, 'ap_lo', lb_aplo, ub_aplo, mean_aplo)

# подгружаем модель
def load():
    with open("model.plc", "rb") as fid:
        return pickle.load(fid)

model = load()

y_pr = model.predict_proba(df)[:,1]

# определяем наиболее важные признаки
# _all - определяется самый важный признак из всех
# _control - исключены признаки age, height, gender, т.е. те на которые нельзя повлиять
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df)
feature_names_all = df.columns
max_importance_idx_all = np.abs(shap_values).mean(axis=0).argmax()
feature_names_control = df.columns[3:]
max_importance_idx_control = np.abs(shap_values).mean(axis=0)[:,3:].argmax()

feature_names = {
    'age': 'Возраст',
    'gender': 'Пол',
    'height': 'Рост',
    'weight': 'Вес',
    'ap_hi': 'Верхнее давление',
    'ap_lo': 'Нижнее давление',
    'cholesterol': 'Холестерин',
    'gluc': 'Глюкоза',
    'smoke': 'Курение',
    'alco': 'Алкоголь',
    'active': 'Спорт'
}



st.success("Риск возникновения сердечных заболеваний: {:.2%}".format(y_pr[0]))
st.write('Наиболее важный показатель:', feature_names[feature_names_all[max_importance_idx_all]])
st.write('На что обратить внимание:', feature_names[feature_names_control[max_importance_idx_control]])