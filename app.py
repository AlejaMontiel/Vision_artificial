import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd

# Cargar el modelo previamente entrenado
model = yolov5.load('yolov5s.pt')

# Establecer parámetros del modelo
model.conf = 0.25  # Umbral de confianza NMS
model.iou = 0.45  # Umbral IoU NMS
model.agnostic = False  # NMS clase-agnóstica
model.multi_label = False  # NMS múltiples etiquetas por caja
model.max_det = 1000  # Número máximo de detecciones por imagen

# Interfaz de la aplicación
st.title("Detección de Objetos en Imágenes")

# Parámetros de configuración
with st.sidebar:
    st.subheader('Parámetros de Configuración')
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0)
    st.write('IOU:', model.iou)
    model.conf = st.slider('Seleccione el Confidence', 0.0, 1.0)
    st.write('Conf:', model.conf)

# Capturar imagen con la cámara
picture = st.camera_input("Capturar foto", label_visibility='visible')

if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Realizar inferencia
    results = model(cv2_img)

    # Parsear resultados
    predictions = results.pred[0]
    boxes = predictions[:, :4] 
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    col1, col2 = st.columns(2)

    with col1:
        # Mostrar imagen con detecciones
        results.render()
        st.image(cv2_img, channels='BGR')

    with col2:
        # Obtener nombres de las etiquetas
        label_names = model.names
        category_count = {}

        # Contar categorías
        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        data = []
        # Imprimir recuentos de categorías y etiquetas
        for category, count in category_count.items():
            label = label_names[int(category)]            
            data.append({"Categoría": label, "Cantidad": count})
        
        data2 = pd.DataFrame(data)
        
        # Agrupar los datos por la columna "Categoría" y sumar las cantidades
        df_sum = data2.groupby('Categoría')['Cantidad'].sum().reset_index()
        st.write(df_sum)
