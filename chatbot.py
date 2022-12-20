#instalar o natural language toolkit
#instalar o numpy (responsável pelo processamento de arrys, matrizes & funções matematicas)
#instalar o keras (responsável pelos models)
#instalar o tensorflow (kerar requer o tf)
#instalar o flask (framework pra webapp em python)
import nltk
import json
import pickle
import random
#nltk vai ajudar a processar as palavras
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#tokenizer divide o texto em uma lista de sentenças
nltk.download('punkt')
#lexical database de inglês, ajuda a interpretar relacionamentos conceituais entre os termos 
nltk.download('wordnet')
nltk.download('omw-1.4')


words=[]
classes=[]
documents=[]
ignore_words=['?','!']
data_file=open('deco_intents.json', encoding='utf-8').read()
intents = json.loads(data_file)
lemmatizer = WordNetLemmatizer()

#percorre o json de intents para processar o pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#quebra a palavra em um significado único: captura as formas dos verbos e compreende o significado
#  atribuindo-o às diversas formas: ex: cry -> crying, cried, cries, cry           
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#inicialização do treinamento dos dados
training=[]
output_empty = [0] * len(classes)
for doc in documents:
    bag = []

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # classificando as palavras a partir do pattern: se a palavra possui o lema, ensinando o significado 
    # da palavra associado ao lema passado
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[(classes.index(doc[1]))] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

#cria um conjunto de testes e de treino onde X -> pattern e Y -> intents pra fitar o modelo
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

##cria o modelo: 3 camadas 3Layers: 1L -> 128 neuronios, 2L -> 64 neuronios, 3L contem a saída com o 
# numero de neuronios igual ao numero das intenções para prever a saida da intenção
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #rectified linear unit
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compila o modelo: otimizador gradiente estocástico descendente com gradiente acelerado de Nesterov (coisas dos modelos matemáticos pra ajudar a dar bons resultados)
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fittando e salvando o modelo -> a cada alteração dos intents ou configuração de acurácia essa função deve ser rodada
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
#modelo treinado e salvo que será importado na webapp em flask :)
model.save('chatbot_models.h5', hist)

print("Model created")