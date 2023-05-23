import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os




# Obtener la ruta absoluta del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Nombre del archivo que deseas subir
base = "base.csv"
model_ft = 'tree_full.pkl'
model_s = 'tree.pkl'

# Ruta completa del archivo
ruta_base = os.path.join(script_dir, base)
ruta_model = os.path.join(script_dir, model_ft)
ruta_model_s = os.path.join(script_dir, model_s)
base = pd.read_csv(ruta_base)


@st.cache_resource
def load_ftree():
    f_tree = pickle.load(open(ruta_model, 'rb'))
    return f_tree

@st.cache_resource
def load_mintree():
    tree = pickle.load(open(ruta_model_s, 'rb'))
    return tree


def clean(data):
    data['Sexo']=  data['Sexo'].str.lower()
    data.loc[data['Régimen de tenencia Vivienda']=='<No Registra>', ['Régimen de tenencia Vivienda']] = '<No Aplica>'
    data.loc[data['DesagregadoDesembolsoBIE'].isin(['Culminado con agotamiento de tiempo para acceder a BIE','Culminado sin agotamiento de tiempo para acceder a BIE']), ['DesagregadoDesembolsoBIE']] = 'Culminado'
    data = pd.get_dummies(data, columns = [ 'Sexo', 'Régimen de tenencia Vivienda',  'Nivel Educativo',  'OcupacionEconomica', 'Estado ISUN', 'Posee Servicio Social?', 'Tipo de ASS Vinculada',
        'Régimen de salud', 'DesagregadoDesembolsoBIE'])
    data.drop([ 'Sexo_femenino','Nivel Educativo_Por Establecer', 'OcupacionEconomica_<No Registra>',  'Estado ISUN_<No Aplica>', 'Posee Servicio Social?_Posee Certificación de Servicio Social',
            'Posee Servicio Social?_Se encuentra vinculado a Servicio Social', 'Tipo de ASS Vinculada_<No Aplica>', 
            'Régimen de salud_S - SUBSIDIADO', 'DesagregadoDesembolsoBIE_Culminado'], axis=1, inplace =True)

    return data



def pred_ft(nrow, base = base):

    classes = {
    0: "Abandono de proceso.",
    1: "Culminación de proceso."
    }

    model = load_ftree()
    pred =  base.append(nrow, ignore_index = True)
    
    pred = clean(pred)
    pred = pred.iloc[[10]]

    # Realizamos una gráfica mostrando los porcentajes de confianza de los modelos
    fig = make_subplots()
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')


    p = model.predict(pred)[0] # Obtenemos la clase del modelo según la predicción
    proba = model.predict_proba(pred)[0] # Obtenemos las probabiidades de pertenencia de cada clase
    st.markdown(f'La clasificación es  *{classes[p]}*') # Mostramos el resultado apra cada uno de los modelos seleccionados

    # Añadimos cada predicción según su probabilidad.
    fig.add_trace(
        go.Bar(
            x=list(classes.values()),
            y=proba,
            name='Árbol completo'
        ))

    fig.update_yaxes(range=[0,1])
    fig.update_xaxes(title_text="Clase")
    st.plotly_chart(fig,use_container_width=True)




def pred_s(nrow):

    classes = {
    0: "Abandono de proceso.",
    1: "Culminación de proceso."
    }
    
    d = {
        'DesagregadoDesembolsoBIE': ['Posee desembolso BIE','No está en Proceso',
        'Culminado','Culminado',
        'Culminado','Culminado'],
        'Estado de la vinculación ASS':['Certificado','<No Aplica>',
        'Vinculado', 'Abandono sin justa causa','Abandono con justa causa',
        'No vinculado por limitaciones físicas o mentales permanentes certificadas']
        }
    base = pd.DataFrame(data=d)

    model = load_mintree()
    pred =  base.append(nrow, ignore_index = True)
    #pred = pred.iloc[[6]]
    model(pred)
    #pred = clean(pred)
    #pred = pred.iloc[[6]]
    #model(pred)
    # Realizamos una gráfica mostrando los porcentajes de confianza de los modelos
    fig = make_subplots()
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')


    p = model.predict(pred)[0] # Obtenemos la clase del modelo según la predicción
    proba = model.predict_proba(pred)[0] # Obtenemos las probabiidades de pertenencia de cada clase
    st.markdown(f'La clasificación es  *{classes[p]}*') # Mostramos el resultado apra cada uno de los modelos seleccionados

    # Añadimos cada predicción según su probabilidad.
    fig.add_trace(
        go.Bar(
            x=list(classes.values()),
            y=proba,
            name='Árbol completo'
        ))

    fig.update_yaxes(range=[0,1])
    fig.update_xaxes(title_text="Clase")
    st.plotly_chart(fig,use_container_width=True)