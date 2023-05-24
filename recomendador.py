from streamlit import button, cache_resource, columns, markdown, plotly_chart, radio, selectbox, slider
from datetime import date
from joblib import load
from pandas import DataFrame
from plotly.subplots import make_subplots
from plotly.graph_objects import Bar
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, include_all) -> None:
        super().__init__()
        self.include_all = include_all

    def fit(self, x, y=None):
        return self

    def transform(self, x, y = None):
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
        grup_et = selectbox('Grupo Etario', ('<No Registra>', 'Entre 18 y 25 años',
                                             'Entre 26 y 40 años', 'Entre 41 y 60 años', 'Mayor de 60 años')) #si
        dep_res = selectbox('Departamento de residencia',(
           'Meta', 'Putumayo', 'Antioquia', 'Norte de Santander',
           'Bogotá D.C.', 'Córdoba', 'Bolívar', 'Magdalena', 'Cesar',
           'Cundinamarca', 'Nariño', 'Santander', 'Cauca', 'Chocó',
           '<No Registra>', 'Caldas', 'Valle del Cauca', 'Casanare',
           'Atlántico', 'Huila', 'Caquetá', 'Tolima', 'Quindio', 'La Guajira',
           'Boyacá', 'Guaviare', 'Sucre', 'Vaupés', 'Risaralda', 'Guainía',
           'Arauca', 'Amazonas', 'Vichada',
           'Archipiélago de San Andrés. Providencia y Santa Catalina'))
        tip_bie = selectbox('Tipo de BIE Accedido',
                           ('<No Aplica>', 'Plan de Negocio', 'Vivienda', 'Educación Superior'))
        pos_con = selectbox('Posee Cónyuge o Compañero(a)?', (
            '<No Aplica>', 'No', 'Sí', '<No Registra>'))
        tip_viv = selectbox('Tipo de Vivienda', (
            '<No Aplica>', 'Casa', 'Apartamento', 'Casa-Lote', 'Habitación',
           'Finca', 'Rancho', 'Otro', 'Cuarto(s)', 'Vivienda (casa) indígena',
           'Otro tipo de vivienda (carpa, tienda, vagón, embarcación, cueva, refugio natural, puente, calle, etc.)',
           '<No Registra>'))
    
    with inputs[1]:

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
        Max_fpt = selectbox('Máximo Nivel FpT Reportado',(
           '<No Aplica>', 'Complementario', 'Técnico', 'Semicalificado',
           'Tecnológico', 'Operario', 'Transversal', 'Técnico Profesional',
           'Técnico Laboral', 'Auxiliar', 'Técnico Laboral por Competencias',
           'Especialización Tecnológica',
           'Certificación por Evaluación de Competencias',
           'Especialización Técnica'
            ))
        est_ass = selectbox('Estado de la vinculación ASS',
            ('<No Aplica>', 'Certificado', 'Abandono sin justa causa',
            'Abandono con justa causa', 'Vinculado',
            'No vinculado por limitaciones físicas o mentales permanentes certificadas'))
        int_gf = selectbox('Total Integrantes grupo familiar',
                          ())
        val_hijos = radio('¿Tiene hijos?', ('Si', 'No'))
        if val_hijos == 'Si':
            hijos = slider('Número de hijos ', 1, 10, (1))
        else:
            hijos = -1 
    
    with inputs[2]:

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
        tipo_desmov = selectbox('Tipo de Desmovilización', (
            'Colectiva',
            'Individual))
        lin_fpt = selectbox('Línea de FpT para el Máx., Nivel',
            '<No Aplica>', '<No Registra>', 'OTROS',
            'OPERADORES DE MAQUINAS, EQUIPO Y TRANSPORTE', 'SERVICIOS',
            'CARPINTERIA Y EBANISTERIA', 'SALUD',
            'MECANICA AUTOMOTRIZ Y DE MOTOS', 'SISTEMAS',
            'FINANZAS Y ADMINISTRACION', 'AGROPECUARIA', 'ALIMENTOS Y BEBIDAS',
            'ELECTRICIDAD', 'CONSTRUCCION', 'MERCADEO Y VENTAS',
            'MECANICA INDUSTRIAL', 'TRANSVERSAL', 'ELECTRONICA',
            'CONFECCION, MARROQUINERIA Y CALZADO', 'AMBIENTAL',
            'ARTESANIAS Y JOYERIA', 'ESTETICA',
            'EXPLOTACION MINERA, PETROLEO Y GAS', 'DISEÑO Y ARTES GRAFICAS')
        grup_f = slider('Integrantes grupo familiar', 0, 20, (1))
        if grup_f ==0:
            grup_f =-1
        serv_p = selectbox('Posee Serv. Públicos Básicos',('<No Aplica>', 'No', 'Sí'))
        ocup = selectbox('Ocupacion económica', ('Ocupados en el sector Informal', 'No Aplica', '<No Registra>',
            'Población Económicamente Inactiva', 'Desocupados'))
        tip_ass = seelctbox('Tipo de ASS Vinculada',(
            '<No Aplica>',
           'Acompañamiento a la atención en Salud y atención Alimentaria a comunidades vulnerables',
           'Embellecimiento de Espacio Publico',
           'Aporte de habilidades Especiales que le participante ponga a disposición de la comunidad',
           'Multiplicadores del Conocimiento',
           'Generación de espacios de recreación, Arte, Cultura y Deporte',
           'Recuperación Ambiental'))
            
            
    data = DataFrame({ 
            'Tipo de Desmovilización':[tipo_desmov], 
            'Grupo Etario':[grup_et], 
            'Sexo':[sexo], 
            'Departamento de residencia':[dep_res], 
            'Nivel Educativo':[niv_educativo], 
            'Máximo Nivel FpT Reportado':[Max_fpt], 
            'Línea de FpT para el Máx.':[lin_fpt], 
            'Nivel OcupacionEconomica':[ocup], 
            'Tipo de BIE Accedido':[tip_bie], 
            'Estado de la vinculación ASS':[est_ass], 
            'Tipo de ASS Vinculada':[tip_ass], 
            'Posee Cónyuge o Compañero(a)?':[pos_con], 
            'N° de Hijos':[val_hijos], 
            'Total Integrantes grupo familiar':[grup_f], 
            'Tipo de Vivienda':[tip_viv], 
            'Régimen de tenencia Vivienda':[reg_vivienda], 
            'Posee Serv. Públicos Básicos':[serv_p], 
            'Régimen de salud':[reg_salud]
                     })
if (button('Realizar Clasificación')):
    predict(data, modelo)
