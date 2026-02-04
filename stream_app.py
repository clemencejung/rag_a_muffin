import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os
from mistralai import Mistral



EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "ma_collection_muffins"


@st.cache_resource
def initialiser_base_donnees():
    
    # Chargement de la base de données de recette JSON, créé par le fichier données_recettes.ipynb
    if not os.path.exists('base_de_donnees.json'):
        st.error("Fichier 'base_de_donnees.json' introuvable sur GitHub !")
        st.stop()
        
    df = pd.read_json('base_de_donnees.json')
    df_copy = df.copy().fillna("")
    
    # Nettoyage des listes pour ChromaDB
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
    
    # Création du modèle d'embedding
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(df_copy["text_for_embedding"].tolist(), normalize_embeddings=True).tolist() # On utilise le text_for_embedding  créé dans la base de données

    # Client Chroma 
    client = chromadb.Client()
    
    # Création de la collection
    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        documents=df_copy["text_for_embedding"].tolist(),
        embeddings=embeddings,
        metadatas=df_copy.to_dict(orient='records'),
        ids=[str(uuid.uuid4()) for _ in range(len(df_copy))]
    )
    return collection, model



with st.sidebar:
    st.title("Configuration 🔑")
    user_api_key = st.text_input("Entre ta clé API Mistral :", type="password")
    st.info("Tu peux obtenir une clé sur console.mistral.ai")


# Génération de la réponse par le Chef Muffin
# Génération de texte avec une clé Mistral API
def generer_reponse_chef(query, results, api_key):
    if not api_key:
        return "Oups ! Il me manque ta clé API dans la barre latérale pour pouvoir cuisiner... 🧁"
    
    client = Mistral(api_key=api_key) # On utilise la clé API Mistral fournie par l'utilisateur
    
    # On construit le contexte à partir des résultats de ChromaDB
    # Version plus structurée pour l'IA
    contexte = ""
    for m in results['metadatas'][0]:
        contexte += f"""
        ---
        RECETTE : {m['titre']}
        INGRÉDIENTS : {m.get('ingredients', 'Non listés')}
        INSTRUCTIONS : {m.get('instructions', 'Non précisées')}
        DESCRIPTION : {m.get('description', '')}
        """
    # Instructions pour mon prompt
    prompt = f"""TU ES UNE CHEFFE MUFFIN, UNE ASSISTANTE CULINAIRE OBSESSIONNELLE MAIS SYMPATHIQUE.
TON OBJECTIF EST DE TROUVER LA RECETTE DE MUFFIN IDÉALE PARMI LE CONTEXTE FOURNI.

### TES DIRECTIVES (GUARDRAILS) :
1. OBSESSION : Tu ne cuisines QUE des muffins. Si on te demande des lasagnes ou une pizza, REFUSE poliment avec humour.
2. ANCRAGE : Utilise UNIQUEMENT les recettes fournies dans le bloc [CONTEXTE]. N'invente rien.
3. LANGUE : Réponds toujours en français courant.
4. CORRECTION : si l'utilisateur te demande de cuisiner avec des choses qui ne sont pas des aliments, réponds lui avec humour que tu n'es pas mécanicien, ou magicien etc... 
5. Il y a plusieurs cas, si l'utilistaeur te donne des ingrédients/à une requête qui correspond très bien avec l'une des 3 recettes de results, alors ne renvoit que cette recette à l'utilisateur,
si les 3 propositions sont proches mais ne correspondent pas exactement, dis à l'utilisateur que tu n'as pas en stock une recette qui correspond parfaitement à ses attentes mais propose
lui les trois recettes en suggestions, pour que ça l'inspire ! Attention, ces recettes doivent quand même contenir au moins l'un des ingrédient demandé, ou bien être dans la même famille d'aliment :
par exemple si je demande courgettes tu dois proposer au moins un muffin avec un autre légume. Si tu considères que l'une des propositions ne correspond pas, ne la propose pas!

Si les 3 propositions n'ont rien à voir alors ne rien renvoyer, et demander à l'utilisateur une requête moins originale. 

Si l'utilisateur te donne des ingrédients pour une recette salée, ne lui propose surtout pas les recettes sucrées et inversement, il vaut mieux ne rien répondre stp.

### STRUCTURE DE RÉPONSE STRICTE (À RESPECTER LIGNE PAR LIGNE) :
Pour chaque recette, respecte scrupuleusement cet affichage, tu dois renvoyer tels qu'ils sont dans le [CONTEXTE] exactement, le titre, les ingrédients et les instructions :

📍 **[TITRE DE LA RECETTE]**



🛒 **Ingrédients :**
- [Ingrédient 1]
- [Ingrédient 2]



👨‍🍳 **Instructions :** 
[Recopie ici TOUTES les instructions détaillées fournies dans le contexte, sans rien résumer et en gardant le ton original.]



✨ *Le mot de la Cheffe :*
[Ton commentaire humoristique]


Dans tous les cas, réponds toujours avec bonne humeur, entrain et humour ! Tu es une fan inconditionnel de muffins. Ne finis juste pas par une question. 

[CONTEXTE]
{contexte}
[QUESTION]
{query} """
    chat_response = client.chat.complete(
          model="mistral-small-latest", 
          messages=[
              {
                  "role": "user",
                  "content": prompt,
              },
          ]
      )
      
    return chat_response.choices[0].message.content

# Interface utilisateur = Application Streamlit
st.set_page_config(page_title="Cheffe Muffin", page_icon="🧁")

st.title("Rag à muffins 👩🏼‍🍳")
st.markdown(":rainbow[Bienvenue !] Je suis la cheffe muffin, je possède dans mon grimoire tout un tas de recettes de muffins, plus délicieuses les unes que les autres ! Des envies particulières aujourd'hui ? Je vous trouverai LA recette la plus adaptée.")

# Initialisation au chargement de la page
with st.spinner("La Cheffe prépare sa cuisine... (Initialisation)"):
    collection, model_embed = initialiser_base_donnees()

# Champ de saisie
query = st.text_input("Quelle envie avez-vous aujourd'hui ?", placeholder="Ex: J'ai très envie de fromage ce soir")

if st.button("Demander à la Cheffe"):
    if not user_api_key:
        st.error("N'oubliez pas de saisir votre clé API dans la barre latérale ! 👈")

    elif query:
        with st.spinner("Recherche de la meilleure recette dans mon grimoire..."):
            # Recherche vectorielle
            query_vector = model_embed.encode([query], normalize_embeddings=True).tolist()
            res = collection.query(query_embeddings=query_vector, n_results=3)
            
            # Appel à l'IA
            reponse = generer_reponse_chef(query, res, user_api_key)
            
            # Affichage
            st.chat_message("assistant").write(reponse)
            with st.expander("🔍 Vérifier les sources du grimoire"):
                for m in res['metadatas'][0]:
                    st.write(f"📖 **{m['titre']}**")
            
    else:
        st.warning("Dites-moi quelque chose, je ne lis pas encore dans les pensées ! 🧁")