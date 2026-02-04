#🧁 Projet Cheffe Muffin

Bienvenue ! Ce projet est une application d'IA (RAG) capable de discuter avec vous pour trouver la recette de muffin parfaite dans son grimoire de recettes scrapées sur Marmiton.

## Guide d'utilisation

Suivez ces étapes dans l'ordre pour lancer l'application sur votre ordinateur :

1. Ouvrir le terminal et récupérer le dossier du projet :

```bash
git clone [https://github.com/clemencejung/rag_a_muffin.git]
cd rag_a_muffin
```

2. Installer les dépendances
Utilisez le fichier requirements.txt pour installer toutes les bibliothèques d'un coup :

```bash
pip install -r requirements.txt
```

3. Démarrer l'application Streamlit

```bash
streamlit run stream_app.py
```

## Utilisation

Une fois l'application ouverte dans votre navigateur, allez dans la barre latérale à gauche.

Collez votre clé API Mistral (disponible gratuitement sur console.mistral.ai).

Posez votre question à la Cheffe (ex: "J'ai envie de quelque chose de fruité" ou "Un muffin salé avec du fromage").