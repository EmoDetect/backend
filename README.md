<h1 align="center"> Emotion Kids </h1>

<!-- <p align="center"> <img src="https://user-images.githubusercontent.com/74990858/122049132-bbb25500-cdea-11eb-9e5c-c2ce6c1649f4.jpg"
     alt="Markdown Monster icon"/>
</p> -->


---

## Functionalitati aplicatie

- Logare utilizator (copil + educator)
- Rezolvarea unei activitati educative de catre copil
- Captura poza copil si identificare emotii pe baza acesteia
- Prezentarea de emotii ale fiecarui copil in pagina educatorului

## Descriere problema rezolvata cu ajutorul AI

- Recunoasterea de emotii in imagini ale fetei

## Related work

- [kaggle - emotion recognition](https://www.kaggle.com/anisbouaziz/emotion-recognition)
- [Vision AI](https://cloud.google.com/vision)

## Useful tools and technologies

- Librarii
  - tensorflow
  - keras
- Google Colab pentru scris si executat cod

## Informatii despre datele folosite

- Baza de date: [fer2013](https://www.kaggle.com/deadskull7/fer2013)
- Structura baza de date:
  - fiecare intrare din fisier reprezinta o matrice de pixeli a unei imagini si are atasata o emotie(eticheta intre 0 si 6)
  - etichete:
    - 0 = 'Angry'
    - 1 = 'Disgust'
    - 2 = 'Fear'
    - 3 = 'Happy'
    - 4 = 'Sad'
    - 5 = 'Surprise'
    - 6 = 'Neutral'

## Descriere algoritm inteligent

- Prelucrare date
  - Extragere matrice de pixeli pentru fiecare imagine de dimensiune 48 x 48
  - Augmentare date
  - Impartire prin shuffle in train, test si validation astfel:
    - train = 70%
    - test = 9%
    - validation = 21% 

- Utilizarea unei arhitecturi simple de retea neuronala convolutionala
- [Modelul CNN este inspirat de la Goodfellow, I.J., et. al.(2013).](https://arxiv.org/pdf/1307.0414.pdf)
<img src="https://user-images.githubusercontent.com/74990858/141697405-8d09c76c-b76b-4707-8bd2-e6e316f73b5c.png"
     alt="Markdown Monster icon"/>

- Reteaua neuronala foloseste un layer de input, un layer de output, 4 layer-e convolutionale si 2 layer-e finale. 
- Functii de activare
  - 'softmax' - pentru layer-ul de output
  - 'relu' - pentru toate celalalte layere
- Compilare model
  - optimizatorul stocastic Adam 
  - functia de loss de cross-entropie categorica
  - acuratetea ca metrica de performanta
