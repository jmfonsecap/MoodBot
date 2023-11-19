from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import string
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np

import pytholog as pl

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


nltk.download('twitter_samples')
nltk.download('stopwords')
app = Flask(__name__)
CORS(app)
@app.route('/edit_tweet', methods=['POST'])
def predict_sentiment_api():
    global tweet
    global polarity
    global pol
    global tokenization
    global dictionary
    global frequent_word
    global sorted_dictionary
    try:
        data = request.get_json()
        new_tweet_data = data.get('tweet')
        polarity, tokenization, dictionary = pln(new_tweet_data)
        polarity = round(float(polarity[0]) if isinstance(polarity, np.ndarray) else float(polarity),2)
        if not dictionary:
            raise ValueError("Dictionary is empty")
        frequent={}
        for element in dictionary.keys():
            if element in dataEmotion:
                frequent[element]=dictionary[element]
        frequent_word = max(frequent, key=frequent.get)
        sorted_dictionary = sorted(frequent.items(), key=lambda x: x[1], reverse=True)
        tweet = new_tweet_data
        
        if float(polarity)>0.5:
            pol="positive"
        else:
            pol="negative"
        response_data = {
            "tweet": tweet,
            "polarity": polarity,
            "pol": pol,
            "frequent_word": frequent_word
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/deepen_emotion', methods = ['POST'])
def deepen_emotion():
    deepening = deeper(frequent_word)
    data = request.get_json()
    deep = data.get('deep')
    if (deepening):
        result = deepening[deep-1]
    else:
        result = False  
    return jsonify({"deepen": result})
@app.route('/get_emotions', methods = ['GET'])
def get_emotions():
    
    
    if len(sorted_dictionary)==1:
        response_data = {
        "emotion1": sorted_dictionary[0][0],
        }
    elif len(sorted_dictionary)==2:
        response_data = {
        "emotion1": sorted_dictionary[0][0],
        "emotion2": sorted_dictionary[1][0]
        }
    
    else:
        response_data = {
        "emotion1": sorted_dictionary[0][0],
        "emotion2": sorted_dictionary[1][0],
        "emotion3": sorted_dictionary[2][0]
        }
    return jsonify(response_data)
@app.route('/explore_emotion', methods = ['POST'])
def explore_emotion():
    data = request.get_json()
    t_op = data.get('t_op')
    task = data.get('task')
    if t_op==1:
        t_op=sorted_dictionary[0][0]
    elif t_op==2:
        t_op=sorted_dictionary[1][0]
    elif t_op==3:
        t_op=sorted_dictionary[2][0]
    result = prolog(t_op,task)
    if(result[0]=='No'):
        return jsonify ({"result":'No'})
    else:
        return jsonify ({"result":result[0]['What']})
@app.route('/suddenness', methods = ['POST'])
def suddenness():
    data = request.get_json()
    inten = int(data.get('intensity'))
    freq = int(data.get('frequency'))
    sudden = suddeness(inten,freq)
    result= sudden
    return jsonify({"result": result})

@app.route('/get_tweet', methods =['GET'])
def get_tweet():
    return jsonify({"data": tweet})
def process_tweet(tweet):

    #Remover caracteres de RT
    tweet2 = re.sub(r'^RT[\s]','', tweet)

    #Remover hipervinculos
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*','', tweet2)

    #Remover hastags
    #Only removing the hash # sign from the word
    tweet2 = re.sub(r'#','',tweet2)

    #print("Tweet sin simbolos/palabras relleno:",tweet2)

    # Crear clase para tokenizar
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    # Tokenizar los tweets
    tweet_tokens = tokenizer.tokenize(tweet2)

    #print("Tweet tokenizado:",tweet_tokens)

    #Importar la librería de stop words
    stopwords_english = stopwords.words('english')

    #Lista de palabras relevantes sin stopwords
    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweets_clean.append(word)

    #print("Tweet sin stopwords:",tweets_clean)
    #Clase de stemming
    stemmer = PorterStemmer()

    #Llevar cada una de las palabras a su raíz
    tweets_stem = []
    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)

    #print("Tweet con palabras en raíz:",tweets_stem)


    return tweets_stem

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs

def sigmoid(z):
    # Sigmoide de z
    h = 1/(1 + np.exp(-z))

    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)

    for i in range(0, num_iters):

        # Z es igual al producto punto de x y theta
        z = np.dot(x,theta)

        # obtener sigmoide de z
        h = sigmoid(z)

        # costo de la funcion
        J = (-1/m)*(np.dot(y.T,np.log(h)) + np.dot((1-y).T,np.log(1-h)))

        # actualizar los pesos de theta
        theta = theta - (alpha/m)*np.dot(x.T, h-y)

    J = float(J)
    return J, theta

def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)

    # vector
    x = np.zeros((1, 3))

    x[0,0] = 1


    for word in word_l:

        # incrementar el conteo positivo de 1
        x[0,1] += freqs.get((word,1),0)

        # incrementar el conteo negativo de 0
        x[0,2] += freqs.get((word,0),0)

    assert(x.shape == (1, 3))
    return x

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#Separar los datos en datos de entrenamiento y datos de prueba
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]
train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)


#Recopilación de las características de x en una matriz X
freqs = build_freqs(train_x, train_y)
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)
Y = train_y
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

def predict_tweet(tweet, freqs, theta):
    # extraer características del texto
    x = extract_features(tweet, freqs)

    # hacer la prediccion usando x y thetha
    z = np.dot(x,theta)
    y_pred = sigmoid(z)


    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta):
    # lista de predicciones
    y_hat = []

    for tweet in test_x:
        # obtener la predicción del tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    y_hat = np.array(y_hat)
    test_y = test_y.reshape(-1)
    accuracy = np.sum((test_y == y_hat).astype(int))/len(test_x)

    return accuracy

tweet = 'Had a terrible experience with customer service today. They were so rude and unhelpful. '
#tweet_clasificacion = 0
#diccionario = {("sad",0):1,("sad",1):0}

predict_tweet(tweet, freqs, theta)
test_logistic_regression(test_x, test_y, freqs, theta)

#Retorna:
#polaridad (float): polaridad del tweet (negativa < 0.5 < positiva)
#stem (array): lista de palabras tokenizadas y con stemming
#dic (dictionary): diccionario de frecuencia de palabras que están en stem
def pln(texto):
    polaridad=predict_tweet(texto,freqs,theta)
    stem=process_tweet(texto)
    dic={}
    for element in stem:
        if element not in dic.keys():
            dic[element]=0
        else:
            dic[element]+=1
    return polaridad,stem,dic

#Binary search
def binary_search(lista,goal):
    left,right = 0, len(lista)-1

    while left <= right:
        mid = (left+right)//2
        mid_word = lista[mid]

        if mid_word == goal:
            return mid
        elif mid_word < goal:
            left = mid+1
        else:
            right = mid-1
    return None

#Search a word
def search_word(lista,goal):
    lista.sort()
    idx=binary_search(lista,goal)

    if idx != None:
        return True
    else:
        return False

#Define tree
class Nodo:
    def __init__(self, emocion, palabras=None, child=None, nivel=None):
        self.emocion = emocion
        self.palabras = palabras if palabras else []
        self.child = child if child  else []
        self.nivel = nivel

def agregar_nodo(emocion, palabras=None, child=None, nivel=None):
    return Nodo(emocion, palabras, child, nivel)


def conectar_nodos(padre, hijos):
    padre.child.extend(hijos)

#BFS
def bfs(start, goal, level):
    queue = []
    queue.append((start, []))
    visited = set()

    while queue:
        current, path = queue.pop(0)
        exist=binary_search(current.palabras,goal)

        if exist != None and level == current.nivel:
            return True, path + [current.emocion]

        visited.add(current)

        for child in current.child:
            if child not in visited:
                queue.append((child, path + [current.emocion]))
    return False, []

#Definición de niveles del arbol
niveles = {
    "Nivel 0": 0,
    "Nivel 1": 1,
    "Nivel 2": 2,
    "Nivel 3": 3
}
dataEmotion=['joy', 'breath', 'encourag', 'joy', 'happi', 
    'happi', 'celebratori', 'full', 'fulfil', 'smile', 'smile','grate', 'appreci', 'appreci', 
    'reliabl', 'confid', 'hope', 'hope', 'gratitud', 'inspir', 'inspir', 'motiv', 'motiv', 'optim', 'optimist', 
    'satisfi', 'satisfact','love', 'love', 'generos', 'gener','friendship', 
    'friendli', 'empathi', 'empathet','anguish', 'anguish', 'depress', 'disillusion', 'disillus', 'pain', 'pain', 
    'lament', 'lament', 'sad', 'sad','discourag', 'discourag', 'desper', 'desper', 'scare', 'disast', 'disastr', 'fail', 
    'failur', 'frustrat', 'frustrat', 'fear','bitter', 'bitter', 'disdain', 'disdain', 'doubt', 'doubt', 'enviou', 
    'envi', 'insecur', 'insecur', 'hate', 'hatr', 'resent', 'resent','angri']
nodos = {
    'Tweet': agregar_nodo('Tweet'),

    'positive_words': agregar_nodo('positive_words', palabras=['joy', 'breath', 'encourag', 'joy', 'happi',
    'happi', 'celebratori', 'full', 'fulfil', 'smile', 'smile','grate', 'appreci', 'appreci',
    'reliabl', 'confid', 'hope', 'hope', 'gratitud', 'inspir', 'inspir', 'motiv', 'motiv', 'optim', 'optimist',
    'satisfi', 'satisfact','love', 'love', 'generos', 'gener','friendship',
    'friendli', 'empathi', 'empathet' ], nivel=niveles['Nivel 1']),

    'optimism': agregar_nodo('optimism', palabras=['joy', 'breath', 'encourag', 'joy', 'happi',
    'happi', 'celebratori', 'full', 'fulfil', 'smile', 'smile','grate', 'appreci', 'appreci',
    'reliabl', 'confid', 'hope', 'hope', 'gratitud', 'inspir', 'inspir', 'motiv', 'motiv', 'optim', 'optimist',
    'satisfi', 'satisfact'], nivel=niveles['Nivel 2']),

    'relationship_emotion': agregar_nodo('relationship_emotion', palabras=['love', 'love', 'generos', 'gener','friendship',
    'friendli', 'empathi', 'empathet'], nivel=niveles['Nivel 2']),

    'happiness': agregar_nodo('happiness', palabras=['joy', 'breath', 'encourag', 'joy', 'happi',
    'happi', 'celebratori', 'full', 'fulfil', 'smile', 'smile'], nivel=niveles['Nivel 3']),

    'motivation': agregar_nodo('motivation', palabras=['grate', 'appreci', 'appreci',
    'reliabl', 'confid', 'hope', 'hope', 'gratitud', 'inspir', 'inspir', 'motiv', 'motiv', 'optim', 'optimist',
    'satisfi', 'satisfact'], nivel=niveles['Nivel 3']),

    'love': agregar_nodo('love', palabras=['love', 'love', 'generos', 'gener'], nivel=niveles['Nivel 3']),

    'empathy': agregar_nodo('empathy', palabras=['friendship', 'friendli', 'empathi', 'empathet'], nivel=niveles['Nivel 3']),



    'negative_words': agregar_nodo('negative_words', palabras=['anguish', 'anguish', 'depress', 'disillusion', 'disillus', 'pain', 'pain',
    'lament', 'lament', 'sad', 'sad','discourag', 'discourag', 'desper', 'desper', 'scare', 'disast', 'disastr', 'fail',
    'failur', 'frustrat', 'frustrat', 'fear','bitter', 'bitter', 'disdain', 'disdain', 'doubt', 'doubt', 'enviou',
    'envi', 'insecur', 'insecur', 'hate', 'hatr', 'resent', 'resent'], nivel=niveles['Nivel 1']),

    'discouragement': agregar_nodo('discouragement', palabras=['anguish', 'anguish', 'depress', 'disillusion', 'disillus', 'pain', 'pain',
    'lament', 'lament', 'sad', 'sad','discourag', 'discourag', 'desper', 'desper' ], nivel=niveles['Nivel 2']),

    'difficulties': agregar_nodo('difficulties', palabras=['scare', 'disast', 'disastr', 'fail',
    'failur', 'frustrat', 'frustrat', 'fear','bitter', 'bitter', 'disdain', 'disdain', 'doubt', 'doubt', 'enviou',
    'envi', 'insecur', 'insecur', 'hate', 'hatr', 'resent', 'resent'], nivel=niveles['Nivel 2']),

    'sadness': agregar_nodo('sadness', palabras=['anguish', 'anguish', 'depress', 'disillusion', 'disillus', 'pain', 'pain',
    'lament', 'lament', 'sad', 'sad'], nivel=niveles['Nivel 3']),

    'despair': agregar_nodo('despair', palabras=['discourag', 'discourag', 'desper', 'desper'], nivel=niveles['Nivel 3']),

    'negativity': agregar_nodo('negativity', palabras=['bitter', 'bitter', 'disdain', 'disdain', 'doubt', 'doubt', 'enviou',
    'envi', 'insecur', 'insecur', 'hate', 'hatr', 'resent', 'resent'], nivel=niveles['Nivel 3']),

    'frustration': agregar_nodo('frustration', palabras=['scare', 'disast', 'disastr', 'fail',
    'failur', 'frustrat', 'frustrat', 'fear'], nivel=niveles['Nivel 3']),
}

conectar_nodos(nodos['Tweet'], [nodos['positive_words'], nodos['negative_words']])
conectar_nodos(nodos['positive_words'], [nodos['optimism'], nodos['relationship_emotion']])
conectar_nodos(nodos['negative_words'], [nodos['discouragement'], nodos['difficulties']])
conectar_nodos(nodos['optimism'], [nodos['happiness'], nodos['motivation']])
conectar_nodos(nodos['relationship_emotion'], [nodos['love'], nodos['empathy']])
conectar_nodos(nodos['discouragement'], [nodos['sadness'], nodos['despair']])
conectar_nodos(nodos['difficulties'], [nodos['negativity'], nodos['frustration']])

#Retorna:
#emocion_objetivo: emoción del nivel al que pertenece
#path: lista
def deeper(objetivo):
    encontrado, camino = bfs(nodos['Tweet'], objetivo, niveles['Nivel 3'])

    if encontrado:
        emocion_objetivo = camino[-1]
        path=camino[1:]
        return path
    else:
        return False



#Para que el usuario utilice esta funcionalidad, solo debe decir que tan específica quiere que se arroje su emoción
#ingresa numero del 1 al 3 y este input-1 es la posición en path que hay que mostrar

happiness=['joy', 'breath', 'encourag', 'joy', 'happi',
    'celebratori', 'full', 'fulfil', 'smile']

motivation=['grate', 'appreci',
    'reliabl', 'confid', 'hope', 'gratitud', 'inspir', 'inspir', 'motiv', 'optim', 'optimist',
    'satisfi', 'satisfact']

love=['love', 'generos', 'gener']

empathy=['friendship', 'friendli', 'empathi', 'empathet']

sadness=['anguish', 'depress', 'disillusion', 'disillus', 'pain',
    'lament', 'sad']

despair=['discourag', 'discourag', 'desper', 'desper']

negativity=['bitter', 'disdain','doubt', 'enviou',
    'envi', 'insecur', 'hate', 'hatr', 'resent']

frustration=['scare', 'disast', 'disastr', 'fail',
    'failur', 'frustrat','fear']

#Crear base de conocimiento
general = []

#Creamos el hecho "sentimiento" el cual relaciona a la palabra con su sentimiento correspondiente
for element in happiness:
  general.append("sentimiento("+element.lower()+", happiness)")
for element in motivation:
  general.append("sentimiento("+element.lower()+", motivation)")
for element in love:
  general.append("sentimiento("+element.lower()+", love)")
for element in empathy:
  general.append("sentimiento("+element.lower()+", empathy)")
for element in sadness:
  general.append("sentimiento("+element.lower()+", sadness)")
for element in despair:
  general.append("sentimiento("+element.lower()+", despair)")
for element in negativity:
  general.append("sentimiento("+element.lower()+", negativity)")
for element in frustration:
  general.append("sentimiento("+element.lower()+", frustration)")

#Creamos el hecho "pertenece" que establece la relación entre las emociones y su emoción general asociada. Tal cuál se hace en el árbol binario de la iteración #2
pertenece = [
      "pertenece(happiness, optimism)",
       "pertenece(motivation, optimism)",
       "pertenece(love, relationship_emotion)",
       "pertenece(empathy, relationship_emotion)",
       "pertenece(sadness, discouragement)",
       "pertenece(despair, discouragement)",
       "pertenece(negativity, difficulties)",
       "pertenece(frustration, difficulties)",
       "pertenece(optimism, positive_words)",
       "pertenece(relationship_emotion, positive_words)",
       "pertenece(discouragement, negative_words)",
       "pertenece(difficulties, negative_words)",
       ]

#Creamos el hecho "recomendacion" que relaciona el nombre de la playlist con su emoción asociada
recomendacion = ["recomendacion(playlist1, happiness)",
                 "recomendacion(playlist2, motivation)",
                 "recomendacion(playlist3, love)",
                 "recomendacion(playlist4, empathy)",
                 "recomendacion(playlist5, sadness)",
                 "recomendacion(playlist6, despair)",
                 "recomendacion(playlist7, negativity)",
                 "recomendacion(playlist8, frustration)",
                 "recomendacion(playlist1, sadness)",
                 "recomendacion(playlist2, negativity)",]

#Creamos el hecho "link" que relaciona el link de una playlist con su nombre correspondiente
link = ["link(https://open.spotify.com/playlist/37i9dQZF1EVJSvZp5AOML2?si=505da5307df044b7, playlist1)",
        "link(https://open.spotify.com/playlist/3TNs8P4SYLlgzF4LwsjAiq?si=f3216b9ef08a43c5, playlist2)",
        "link(https://open.spotify.com/playlist/5z5QwyYbszJaoaUGu27oxh?si=8f47a9af97334acd, playlist3)",
        "link(https://open.spotify.com/playlist/37i9dQZF1DX2HouCUyhCTv?si=f06d78476dc74b0f, playlist4)",
        "link(https://open.spotify.com/playlist/37i9dQZF1DXdZjf8WgcTKM?si=3b3a59c3e1644c66, playlist5)",
        "link(https://open.spotify.com/playlist/37i9dQZF1DWWEJlAGA9gs0?si=c380b68aeb5942bf, playlist6)",
        "link(https://open.spotify.com/playlist/3I7tAUuDnqBp4Tz9cfbUD7?si=1929188880104738, playlist7)",
        "link(https://open.spotify.com/playlist/3vXUEGi4ip1EhI9OtdgdCy?si=510feaf4bdb4446b, playlist8)",
        ]

#Creamos las reglas:
# "polaridad" permite encontrar la polaridad de una palabra
# "playlist" permite encontrar el link de una playlist de acuerdo con la palabra dada
reglas = ["polaridad(X,Y):- sentimiento(X,P), pertenece(P,L), pertenece(L,Y)",
          "playlist(X,Y):- sentimiento(X,R), recomendacion(L,R), link(Y,L)",
          ]
# Se agregan los hechos y reglas a la base de conocimiento
general += pertenece + recomendacion + link +  reglas

new_kb = pl.KnowledgeBase("MoodBot")
new_kb(general)

#Obtener emoción en base a la palabra
new_kb.query(pl.Expr("sentimiento(bitter, What)"))

#Obtener polaridad de una palabra
new_kb.query(pl.Expr("polaridad(happi, What)"))

#Obtener playlist en base a la palabra
new_kb.query(pl.Expr("playlist(resent, What)"))

def prolog(palabra,option):
    if option==1: #Retorna emoción en base a la palabra
        return new_kb.query(pl.Expr(f"sentimiento({palabra}, What)"))
    elif option==2: #Retorna polaridad de una palabra
        return new_kb.query(pl.Expr(f"polaridad({palabra}, What)"))
    elif option==3: #Retorna playlist en base a la palabra
        return new_kb.query(pl.Expr(f"playlist({palabra}, What)"))

prolog("happi",3)

#Definir antecedentes y consecuentes
intensidad = ctrl.Antecedent(np.arange(0, 11, 1), 'intensity')
intensidad['low'] = fuzz.trimf(intensidad.universe, [0, 0, 5])
intensidad['medium'] = fuzz.trimf(intensidad.universe, [0, 5, 10])
intensidad['high'] = fuzz.trimf(intensidad.universe, [8, 10, 10])

frecuencia = ctrl.Antecedent(np.arange(0, 11, 1), 'frecuency')
frecuencia['low'] = fuzz.trimf(frecuencia.universe, [0, 0, 3])
frecuencia['medium'] = fuzz.trimf(frecuencia.universe, [0, 5, 10])
frecuencia['high'] = fuzz.trimf(frecuencia.universe, [7, 10, 10])

repentinidad = ctrl.Consequent(np.arange(0, 11, 1), 'suddeness')
repentinidad['low'] = fuzz.trimf(repentinidad.universe, [0, 0, 5])
repentinidad['medium'] = fuzz.trimf(repentinidad.universe, [0, 5, 10])
repentinidad['high'] = fuzz.trimf(repentinidad.universe, [5, 10, 10])

#Definir reglas difusas
rule11 = ctrl.Rule(intensidad['high'] & frecuencia['high'], repentinidad['medium'])
rule12 = ctrl.Rule(intensidad['high'] & frecuencia['medium'], repentinidad['medium'])
rule13 = ctrl.Rule(intensidad['high'] & frecuencia['low'], repentinidad['high'])

rule21 = ctrl.Rule(intensidad['medium'] & frecuencia['high'], repentinidad['low'])
rule22 = ctrl.Rule(intensidad['medium'] & frecuencia['medium'], repentinidad['low'])
rule23 = ctrl.Rule(intensidad['medium'] & frecuencia['low'], repentinidad['medium'])

rule31 = ctrl.Rule(intensidad['low'] & frecuencia['high'], repentinidad['low'])
rule32 = ctrl.Rule(intensidad['low'] & frecuencia['medium'], repentinidad['low'])
rule33 = ctrl.Rule(intensidad['low'] & frecuencia['low'], repentinidad['medium'])

#Definir sistemas de control
tipping_ctrl = ctrl.ControlSystem([rule11, rule12, rule13, rule21, rule22, rule23, rule31, rule32, rule33])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

#Retorna el valor asignado de repentinidad (1 a 10)

def suddeness(inten,frecu): #Recibe la intensidad y la frecuencia por el usuario
    tipping.input['intensity'] = inten
    tipping.input['frecuency'] = frecu
    tipping.compute()
    return round(tipping.output['suddeness'],2)


if __name__ == "__main__":
    app.run(debug=True) 


