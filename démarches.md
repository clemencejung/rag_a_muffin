# rag_a_muffin
Projet NLP, école des Mines de Paris

1) Création de la base de données

J'ai utilisé les données de Marmiton pour récolter le plus de recettes de muffins possibles (sucrés ou salés), en utilisant le package Python recipe-scrapers. La création de la base de données se trouve dans le notebook données_recettes.ipynb, et le fichier de données final se trouve dans le fichier json base_de_donnees.json.

J'ai intégré directement dans la base de donnée pour chaque recette 'text_for_embedding', qui contient le titre de la recette ainsi que la liste des ingrédients (sans les quantités). Je l'utilise ensuite pour la vectorisation.

2) Application Streamlit -> stream_app.py

ChromaDB pour le stockage et la récupération des recettes

Embedding : utilisation du modèle "paraphrase-multilingual-MiniLM-L12-v2" comme conseillé

LLM : J'ai choisi l'API Mistral (modèle small-latest) car elle me donnait une réponse instantanée sans surcharger mon ordinateur (très faible), et reste gratuite (premier palier gratuit). Par sécurité, l'application ne contient aucune clé privée, il suffit de coller sa clé API mistral dans la barre latérale pour activer la Cheffe à muffin. J'ai personnalisé la réponse de ma cheffe (cf script stream_app.py) dans le prompt. 

Interface : j'ai utilisé Streamlit