
## Projeto: Rede Neural Simples com TensorFlow para Classificação de Dígitos Manuscritos

### Visão Geral
Neste projeto, desenvolvi e treinei uma rede neural utilizando TensorFlow para a tarefa de classificação de dígitos manuscritos. O objetivo principal foi criar um modelo eficaz capaz de reconhecer e classificar imagens de dígitos de 0 a 9 com alta precisão. Este projeto demonstra minha habilidade em construir modelos de machine learning e aplicar técnicas avançadas de deep learning.

### Ferramentas Utilizadas
- **TensorFlow**: Uma das principais bibliotecas para deep learning, que possibilita a criação, treinamento e avaliação de redes neurais.

### Abordagem e Metodologia

#### 1. **Preparação do Dataset**
Utilizei o dataset MNIST, um benchmark clássico para tarefas de classificação de imagens. O dataset foi normalizado para valores entre 0 e 1, facilitando o treinamento do modelo.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Carregar e normalizar o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```


#### 2. **Construção do Modelo**
Desenvolvi um modelo de rede neural com a seguinte arquitetura:
- **Flatten**: Transforma imagens 2D em vetores 1D.
- **Dense (128 unidades)**: Camada densa com 128 neurônios e função de ativação ReLU, proporcionando capacidade de modelar não-linearidades complexas.
- **Dense (10 unidades)**: Camada de saída com 10 neurônios e função de ativação softmax para a classificação de 10 classes.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```


#### 3. **Treinamento e Avaliação**
Compilei o modelo com o otimizador Adam e a função de perda `sparse_categorical_crossentropy`. O treinamento foi realizado por 5 épocas com monitoramento da precisão e perda. A avaliação foi feita no conjunto de teste, alcançando resultados expressivos.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Avaliação do modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}')
```


#### 4. **Resultados**

- **Precisão no Teste:** 93.11%
- **Perda no Teste:** 0.3092

Esses resultados indicam um desempenho robusto do modelo, com alta capacidade de generalização para novos dados.

#### 5. **Visualizações**

**a. Perda e Precisão ao Longo das Épocas**
Gráfico mostrando a evolução da perda e precisão durante o treinamento.

```python
import matplotlib.pyplot as plt

# Plotar a perda
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda ao Longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Plotar a precisão
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisão de Treinamento')
plt.plot(history.history['val_accuracy'], label='Precisão de Validação')
plt.title('Precisão ao Longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.legend()

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/155402e8-c380-4343-87d0-311a1de1cc2a)

**b. Matriz de Confusão**
Visualização das previsões do modelo comparadas com as classes reais.

```python
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Previsões do modelo
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão')
plt.show()
```
![image](https://github.com/user-attachments/assets/4db24375-98db-4b4c-ab61-7f5896c6c92f)

**c. Exemplos de Previsões**
Imagens do dataset com suas previsões e classes reais.

```python
# Mostrar algumas imagens e suas previsões
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap='gray')
    ax.title.set_text(f'Predição: {y_pred[i]}\nReal: {y_test[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/c8b22296-f211-4817-800b-c152e8e9f48d)

**d. Precisão por Classe**
Distribuição da precisão para cada classe.

```python
import pandas as pd
from sklearn.metrics import classification_report

# Relatório de classificação
report = classification_report(y_test, y_pred, output_dict=True)

# Converter o relatório em DataFrame
report_df = pd.DataFrame(report).transpose()

# Plotar a precisão das classes
plt.figure(figsize=(12, 6))
report_df['precision'][:-3].plot(kind='bar', color='skyblue')
plt.title('Precisão por Classe')
plt.xlabel('Classe')
plt.ylabel('Precisão')
plt.xticks(range(10), range(10), rotation=45)
plt.show()
```
![image](https://github.com/user-attachments/assets/82822493-e789-4f8e-b0ea-f7e3ac1f2119)

### Conclusão
Este projeto demonstrou como construir e otimizar modelos de deep learning para tarefas de classificação de imagens. O modelo desenvolvido não apenas atingiu uma precisão de 93.11%, mas também apresentou uma perda baixa, evidenciando uma boa eficiência na identificação de dígitos manuscritos.
