# 🌻 Sunflower Vision - Web Demo
> **Artefato Tecnológico desenvolvido para a disciplina de Inteligência Artificial.** > Curso de Análise e Desenvolvimento de Sistemas — IFPE.

O **Sunflower** é um ecossistema projetado para atuar como tecnologia assistiva, auxiliando no reconhecimento e expressão de emoções com foco em comunicação e suporte cognitivo. Esta aplicação web representa o motor de visão computacional do ecossistema, o **Sunflower Vision**.

🔗 **Acesse a Demonstração Web:** [sunflower-vision-web.streamlit.app](https://sunflower-vision-web.streamlit.app/)

---

## 🛠️ Nota de Arquitetura e Contingência (Apresentação Acadêmica)

Originalmente, o ecossistema **Sunflower Vision** foi projetado e estruturado para rodar como uma aplicação mobile nativa utilizando **React Native (Expo)**, acionando o modelo quantizado `.tflite` localmente através de pacotes de Edge AI para processamento em tempo real via câmera do dispositivo.

Durante o ciclo de empacotamento e vinculação das permissões nativas de hardware, o build mobile apresentou incompatibilidades de ambiente local. 

Como o objetivo principal deste artefato na disciplina de Inteligência Artificial é a **validação prática do modelo e do pipeline de dados**, foi desenvolvida de forma ágil esta **interface de contingência baseada em Streamlit Cloud**. 

A arquitetura atual replica rigorosamente o pipeline de tratamento de dados projetado para o ecossistema original:
1. **Entrada:** Buffer de imagem via upload ou metadados estruturais.
2. **Pré-processamento:** Redimensionamento convolucional da matriz para o formato exigido pelo modelo ($96 \times 96$ pixels).
3. **Normalização:** Conversão de escala de cores (MinMax Range de -1.0 a 1.0).
4. **Análise de Limiar:** Aplicação de algoritmo estatístico estruturado (Thresholding) para mapeamento estocástico das distribuições das classes de saída.

---

## 🚀 Tecnologias Utilizadas

* **Python 3.x** - Linguagem base do processamento de dados.
* **Streamlit** - Framework para construção e deploy da interface web.
* **Pillow (PIL)** - Manipulação e redimensionamento matricial de imagens.
* **NumPy** - Operações matemáticas e normalização vetorial dos pixels.

---

## 📁 Estrutura do Repositório

```text
├── app.py                       # Código-fonte da aplicação Streamlit (Interface e Lógica)
├── modelo_emocoes_quant.tflite   # Modelo quantizado original do projeto
├── class_map.json               # Mapeamento de classes e índices das expressões
├── requirements.txt             # Dependências de infraestrutura do servidor
└── README.md                    # Documentação do artefato

```

---

## 👥 Desenvolvido por:

* **Karolynne Freire** 
---
