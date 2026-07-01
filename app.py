import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuração visual da página
st.set_page_config(page_title="Sunflower Vision", page_icon="🌻", layout="centered")

st.title("🌻 Sunflower Vision")
st.write("### Artefato Tecnológico — Disciplina de Inteligência Artificial")
st.write("Demonstração Web inspirada no ecossistema assistivo para crianças com TEA.")

# Mapeamento oficial das emoções (Ajuste a ordem conforme o seu class_map.json original)
CLASSES_EMOCOES = ["Raiva", "Desprezo", "Nojo", "Medo", "Feliz", "Neutro", "Triste", "Surpresa"]

@st.cache_resource
def carregar_modelo():
    try:
        # Carrega o interpretador do arquivo TFLite na raiz do repositório
        interpreter = tf.lite.Interpreter(model_path="modelo_emocoes_quant.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de IA: {e}")
        return None

interpreter = carregar_modelo()

# Interface para upload da foto
arquivo_img = st.file_uploader("Escolha uma imagem ou foto de uma expressão facial...", type=["jpg", "jpeg", "png"])

if arquivo_img is not None and interpreter is not None:
    # Abre e exibe a imagem original na tela
    imagem_original = Image.open(arquivo_img).convert('RGB')
    st.image(imagem_original, caption='Imagem enviada para análise', width=300)
    
    with st.spinner("🤖 Executando inferência com Sunflower Vision..."):
        # 1. Redimensiona para o tamanho exato exigido pelo modelo (96x96)
        img_redimensionada = imagem_original.resize((96, 96))
        
        # 2. Converte para array numérico e normaliza (de 0-255 para o range -1.0 a 1.0)
        img_array = np.array(img_redimensionada, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        
        # 3. Adiciona a dimensão do batch (de 96x96x3 para 1x96x96x3)
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Detalhes das entradas e saídas do interpretador TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 4. Injeta os dados da imagem no tensor de entrada
        interpreter.set_tensor(input_details[0]['index'], img_tensor)
        
        # 5. Roda a IA de forma síncrona
        interpreter.invoke()
        
        # 6. Coleta os resultados probabilísticos da camada de saída
        resultados = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Encontra o índice com maior valor (probabilidade mais alta)
        indice_predicao = np.argmax(resultados)
        confianca = resultados[indice_predicao]
        
        # Mapeia o índice para a string legível correspondente
        emocao_final = CLASSES_EMOCOES[indice_predicao] if indice_predicao < len(CLASSES_EMOCOES) else "Desconhecida"
        
    # Exibe o resultado final de maneira elegante em um bloco de sucesso
    st.success("🎉 Análise Concluída!")
    st.metric(label="Expressão Facial Identificada", value=f"🌻 {emocao_final}")
    st.info(f"Grau de Confiança do Modelo: {confianca * 100:.1f}%")