import streamlit as st

import pandas as pd
import numpy as np


from datetime import date
from models.full_tree import pred_ft
from models.full_tree import pred_s

today = date.today()
#import pyautogui

Width = 700
H = 600


st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: black;'>Proyecto final</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: black;'>Análisis con Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: black;'>Modelo de clasificación basado en árboles de decision</h3>", unsafe_allow_html=True)

modelo = st.radio("¿Que modelo quiere utilizar?", ('2-variables', 'Multivariables'))

if modelo == '2-variables':

    col1, col2 = st.columns(2)

    with col1:
        des_bie = st.selectbox('Desagregado Desembolso BIE', ('Posee desembolso BIE','No está en Proceso', 'Culminado con agotamiento de tiempo para acceder a BIE', 'No posee desembolso BIE', 'Culminado sin agotamiento de tiempo para acceder a BIE' ))

    with col2:
        vinc_ass = st.selectbox("Estado de la vinculación ASS", ("Certificado", "<No Aplica>", "Vinculado", "No vinculado por limitaciones físicas o mentales permanentes certificadas", "Abandono sin justa causa", "Abandono con justa causa"))

else:
    col1, col2, col3 = st.columns(3)

    with col1:
        serv_soc = st.selectbox('Posee Servicio Social?', ('Posee Certificación de Servicio Social','No está vinculado a Servicio Social'))
        ocupacion = st.selectbox('Ocupación económica', ('Ocupados en el sector Informal','<No Registra>', 'No Aplica',  'Población Económicamente Inactiva', 'Desocupados'))
        ass = st.selectbox('Tipo de ASS Vinculada', ('<No Aplica>','Embellecimiento de Espacio Publico','Recuperación Ambiental', 'Generación de espacios de recreación, Arte, Cultura y Deporte',
            'Aporte de habilidades Especiales que le participante ponga a disposición de la comunidad','Acompañamiento a la atención en Salud y atención Alimentaria a comunidades vulnerable',
            'Multiplicadores del Conocimiento'))
        grup_et = st.selectbox('Grupo Etario',('<No Registra>','Entre 18 y 25 años', 'Entre 26 y 40 años','Entre 41 y 60 años', 'Mayor de 60 años' ))

    with col2:
        des_bie = st.selectbox('Desagregado Desembolso BIE', ('Posee desembolso BIE','No está en Proceso', 'Culminado'))
        reg_vivienda = st.selectbox('Régimen de tenencia Vivienda', ('<No Aplica>','Con permiso del propietario, sin pago alguno', 'En arriendo o subarriendo', 'Es usufructo', 
            'Familiar', 'Otra forma de tenencia (posesión sin título, ocupante de hecho, propiedad colectiva, etc)', 'Posesión sin título (ocupante de hecho) o propiedad colectiva',
            'Propia, la están pagando', 'Propia, totalmente pagada', 'Sana posesión con título'))
        sexo = st.selectbox('Sexo', ('Masculino', 'Femenino'))
        val_hijos = st.radio("¿Tiene hijos?", ('Si', 'No'))

    with col3:
        isun = st.selectbox("Estado ISUN",("<No Aplica>", "En Funcionamiento", "Cerrado", 'Pendiente por visita ISUN'))
        niv_educativo = st.selectbox("Nivel Educativo",("Alfabetización", "Bachiller", "Básica Primaria", 'Básica Secundaria', 'Por Establecer'))
        reg_salud = st.selectbox("Régimen de salud",("S - SUBSIDIADO", "<No Registra>", "C - CONTRIBUTIVO"))

    if val_hijos == 'Si':
        hijos = st.slider('Número de hijos ', 1, 10, (1))
    else:
        hijos = -1

clasif = st.button('Realizar Clasificación')


a= False

if (clasif):
    if modelo == '2-variables':
        nrow =  {'DesagregadoDesembolsoBIE': des_bie, 'Estado de la vinculación ASS': vinc_ass}
        pred_ft(nrow)
        
    else:
        nrow =  {'Posee Servicio Social?': serv_soc, 'DesagregadoDesembolsoBIE': des_bie, 'Estado ISUN': isun, 'OcupacionEconomica': ocupacion, 'Régimen de tenencia Vivienda':reg_vivienda, 
            'Nivel Educativo': niv_educativo, 'Tipo de ASS Vinculada': ass, 'Sexo': sexo, 'Régimen de salud': reg_salud, 'N° de Hijos': hijos}

        pred_ft(nrow)
      
