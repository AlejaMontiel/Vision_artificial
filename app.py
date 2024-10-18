import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd

# Cargar el modelo YOLOv5 preentrenado
model = yolov5.load('yolov5s.pt')

# Parámetros del modelo
model.conf = 0.25  # Umbral de confianza
model.iou = 0.45  # Umbral de IoU
model.agnostic = False  # Clases agnósticas
model.multi_label = False  # Etiquetas múltiples por caja
model.max_det = 1000  # Máximo de detecciones por imagen

# Título de la aplicación
st.title("🔍 Detección de Objetos en Imágenes")

# Barra lateral para configuraciones
with st.sidebar:
    st.subheader('⚙️ Parámetros de Configuración')
    
    # Ajustar IoU
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0, model.iou)
    st.write(f'**IoU seleccionado:** {model.iou}')
    
    # Ajustar confianza
    model.conf = st.slider('Seleccione el Umbral de Confianza', 0.0, 1.0, model.conf)
    st.write(f'**Confianza seleccionada:** {model.conf}')

# Capturar imagen desde la cámara
picture = st.camera_input("📸 Captura una imagen")

if picture:
    # Convertir la imagen capturada en formato OpenCV
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
  
    # Realizar la inferencia con el modelo
    results = model(cv2_img)

    # Extraer las predicciones
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # Coordenadas de las cajas de detección
    scores = predictions[:, 4]  # Confianza de detección
    categories = predictions[:, 5]  # Clases detectadas

    # Layout de columnas para mostrar resultados
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Imagen con Detecciones")
        # Renderizar las detecciones sobre la imagen
        results.render()
        # Mostrar la imagen con las cajas de detección
        st.image(cv2_img, channels='BGR')

    with col2:
        st.subheader("📊 Resultados de la Detección")
        
        # Obtener los nombres de las etiquetas
        label_names = model.names
        
        # Contar las categorías detectadas
        category_count = {}
        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        # Crear una lista para almacenar los resultados
        data = []        
        for category, count in category_count.items():
            label = label_names[int(category)]
            data.append({"Categoría": label, "Cantidad": count})
        
        # Convertir los resultados en un DataFrame de pandas
        data_df = pd.DataFrame(data)
        
        # Agrupar las categorías y sumar las cantidades
        df_sum = data_df.groupby('Categoría')['Cantidad'].sum().reset_index()
        
        # Mostrar los resultados en una tabla
        st.dataframe(df_sum)

