from streamlit import button, cache_resource, columns, markdown, plotly_chart, radio, selectbox, slider
from datetime import date
from joblib import load
from pandas import DataFrame
from plotly.subplots import make_subplots
from plotly.graph_objects import Bar
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, include_all: bool) -> None:
        super().__init__()
        self.include_all = include_all

    def fit(self, x, y=None):
        return self

    def transform(self, x: DataFrame, y: DataFrame | None = None):
        x['DesagregadoDesembolsoBIE'].replace(
            'Culminado con agotamiento de tiempo para acceder a BIE', 'Culminado sin desembolso', inplace=True)
        x['DesagregadoDesembolsoBIE'].replace(
            'Culminado sin agotamiento de tiempo para acceder a BIE', 'Culminado sin desembolso', inplace=True)
        if not self.include_all:
            minimum = ['DesagregadoDesembolsoBIE',
                       'Estado de la vinculación ASS']
            x.drop(
                columns=[col for col in x.columns if col not in minimum], inplace=True)
            return x
        x['Grupo Etario'].replace(
            'Entre 18 y 25 años', 'Entre 18 y 40 años', inplace=True)
        x['Grupo Etario'].replace(
            'Entre 26 y 40 años', 'Entre 18 y 40 años', inplace=True)
        x['Sexo'] = x['Sexo'].str.upper()
        x['OcupacionEconomica'].replace(
            'No Aplica', '<No Registra>', inplace=True)
        x['Posee Cónyuge o Compañero(a)?'].replace(
            '<No Registra>', '<No Aplica>', inplace=True)
        x['N° de Hijos'].replace(-2, -1, inplace=True)
        exclude = ['Ex Grupo', 'Año desmovilización', 'Ingresó/No ingresó', 'Año de Independización/Ingreso',
                   'Municipio de residencia', 'BeneficioTRV', 'BeneficioFA', 'BeneficioFPT', 'BeneficioPDT',
                   'Desembolso BIE', 'Estado ISUN', 'Posee Servicio Social?', 'Posee Censo de Familia?',
                   'Posee Censo de Habitabilidad?', 'Clasificación Componente Específico', 'FechaCorte', 'FechaActualizacion']
        x.drop(
            columns=[col for col in x.columns if col in exclude], inplace=True)
        return x


@cache_resource
def load_models():
    return {
        'Simple': load(f'./models/tree.pkl'),
        'Completo': load(f'./models/tree.pkl')
    }


models = load_models()


def predict(data: DataFrame, model_name: str):
    classes = {
        0: 'Culminará el proceso',
        1: 'Abandonará el proceso'
    }
    model = models[model_name]
    fig = make_subplots()
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    print(model.steps[0][1].transform(data))
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]
    markdown(f'La clasificación es  *{classes[prediction]}*')
    fig.add_trace(
        Bar(
            x=list(classes.values()),
            y=probability,
            name='Árbol completo'
        ))
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title_text='Clase')
    plotly_chart(fig, use_container_width=True)


today = date.today()
Width = 700
H = 600
markdown("<h1 style='text-align: center'>Análisis de Factores de Riesgo que influyen en la deserción de desmovilizados que han ingresado al proceso de reintegración</h1>", True)
markdown("<h2 style='text-align: center'>Oscar Julián Castañeda\nJuan David Ayala</h5>", True)
modelo = radio('¿Que modelo quiere utilizar?', ('Simple', 'Completo'))
desembolso_bie_options = (
    'Posee desembolso BIE',
    'No está en Proceso',
    'Culminado con agotamiento de tiempo para acceder a BIE',
    'No posee desembolso BIE',
    'Culminado sin agotamiento de tiempo para acceder a BIE')
vinculacion_ass_options = (
    'Certificado',
    '<No Aplica>',
    'Vinculado',
    'No vinculado por limitaciones físicas o mentales permanentes certificadas',
    'Abandono sin justa causa',
    'Abandono con justa causa')

if modelo == 'Simple':
    inputs = columns(2)
    with inputs[0]:
        desembolso_bie = selectbox(
            'Desagregado Desembolso BIE', desembolso_bie_options)
    with inputs[1]:
        vinculacion_ass = selectbox(
            'Estado de la vinculación ASS', vinculacion_ass_options)
    data = DataFrame({
        'DesagregadoDesembolsoBIE': [desembolso_bie],
        'Estado de la vinculación ASS': [vinculacion_ass]})
else:
    inputs = columns(3)
    with inputs[0]:
        serv_soc = selectbox('Posee Servicio Social?', (
            'Posee Certificación de Servicio Social',
            'No está vinculado a Servicio Social'))
        ocupacion = selectbox('Ocupación económica', (
            'Ocupados en el sector Informal',
            '<No Registra>',
            'No Aplica',
            'Población Económicamente Inactiva',
            'Desocupados'))
        ass = selectbox('Tipo de ASS Vinculada', (
            '<No Aplica>',
            'Embellecimiento de Espacio Publico',
            'Recuperación Ambiental',
            'Generación de espacios de recreación, Arte, Cultura y Deporte',
            'Aporte de habilidades Especiales que le participante ponga a disposición de la comunidad',
            'Acompañamiento a la atención en Salud y atención Alimentaria a comunidades vulnerable',
            'Multiplicadores del Conocimiento'))
        grup_et = selectbox('Grupo Etario', ('<No Registra>', 'Entre 18 y 25 años',
                                             'Entre 26 y 40 años', 'Entre 41 y 60 años', 'Mayor de 60 años'))
    with inputs[1]:
        des_bie = selectbox('Desagregado Desembolso BIE', (
            'Posee desembolso BIE',
            'No está en Proceso',
            'Culminado'))
        reg_vivienda = selectbox('Régimen de tenencia Vivienda', (
            '<No Aplica>',
            'Con permiso del propietario, sin pago alguno',
            'En arriendo o subarriendo', 'Es usufructo',
            'Familiar', 'Otra forma de tenencia (posesión sin título, ocupante de hecho, propiedad colectiva, etc)',
            'Posesión sin título (ocupante de hecho) o propiedad colectiva',
            'Propia, la están pagando',
            'Propia, totalmente pagada',
            'Sana posesión con título'))
        sexo = selectbox('Sexo', ('Masculino', 'Femenino'))
        val_hijos = radio('¿Tiene hijos?', ('Si', 'No'))
    with inputs[2]:
        isun = selectbox('Estado ISUN', (
            '<No Aplica>',
            'En Funcionamiento',
            'Cerrado',
            'Pendiente por visita ISUN'))
        niv_educativo = selectbox('Nivel Educativo', (
            'Alfabetización',
            'Bachiller',
            'Básica Primaria',
            'Básica Secundaria',
            'Por Establecer'))
        reg_salud = selectbox('Régimen de salud', (
            'S - SUBSIDIADO',
            '<No Registra>',
            'C - CONTRIBUTIVO'))
    if val_hijos == 'Si':
        hijos = slider('Número de hijos ', 1, 10, (1))
    else:
        hijos = -1
    data = DataFrame({
        'Posee Servicio Social?': [serv_soc],
        'DesagregadoDesembolsoBIE': [des_bie],
        'Estado ISUN': [isun],
        'OcupacionEconomica': [ocupacion],
        'Régimen de tenencia Vivienda': [reg_vivienda],
        'Nivel Educativo': [niv_educativo],
        'Tipo de ASS Vinculada': [ass],
        'Sexo': [sexo],
        'Régimen de salud': [reg_salud],
        'N° de Hijos': [hijos]})
if (button('Realizar Clasificación')):
    predict(data, modelo)
