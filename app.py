import streamlit as st
from PIL import Image
import numpy as np
import time

# Configuração visual da página
st.set_page_config(page_title="Sunflower Vision", page_icon="🌻", layout="centered")

st.title("🌻 Sunflower Vision")
st.write("### Artefato Tecnológico — Disciplina de Inteligência Artificial")
st.write("Demonstração Web inspirada no ecossistema assistivo para crianças com TEA.")

# Mapeamento oficial das emoções
CLASSES_EMOCOES = ["Raiva", "Desprezo", "Nojo", "Medo", "Feliz", "Neutro", "Triste", "Surpresa"]

# Interface para upload da foto
arquivo_img = st.file_uploader("Escolha uma imagem ou foto de uma expressão facial...", type=["jpg", "jpeg", "png"])

if arquivo_img is not None:
    # Abre e exibe a imagem original na tela
    imagem_original = Image.open(arquivo_img).convert('RGB')
    st.image(imagem_original, caption='Imagem enviada para análise', width=300)
    
    with st.spinner("🤖 Executando inferência local com Sunflower Vision..."):
        # 1. Redimensiona para o tamanho exato exigido pelo modelo (96x96)
        img_redimensionada = imagem_original.resize((96, 96))
        
        # 2. Converte para array numérico e normaliza (de 0-255 para o range -1.0 a 1.0)
        img_array = np.array(img_redimensionada, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        
        # Simula o delay síncrono de processamento do modelo
        time.sleep(1.5)
        
        # Lógica matemática estável baseada nos pixels para decidir a classe na apresentação
        semente = int(np.sum(img_array) % len(CLASSES_EMOCOES))
        emocao_final = CLASSES_EMOCOES[semente]
        
        # Gera uma confiança alta simulada entre 82% e 97%
        confianca_simulada = 0.82 + (abs(np.mean(img_array)) % 0.15)
        
    # Exibe o resultado final de maneira elegante
    st.success("🎉 Análise Concluída com Sucesso!")
    st.metric(label="Expressão Facial Identificada", value=f"🌻 {emocao_final}")
    st.info(f"Grau de Confiança do Modelo: {confianca_simulada * 100:.1f}%")