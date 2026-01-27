import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os
from mistralai import Mistral

# --- 1. CONFIGURATION & SÉCURITÉ ---
# On récupère la clé depuis les "Secrets" de Streamlit
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except:
    st.error("La clé MISTRAL_API_KEY est manquante dans les Secrets Streamlit !")
    st.stop()

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "muffin_pro_deploy"

# --- 2. LOGIQUE COEUR (CHROMA + EMBEDDINGS) ---
@st.cache_resource
def initialiser_base_donnees():
    """Charge les données et crée la base vectorielle une seule fois."""
    # Chargement du fichier JSON local
    if not os.path.exists('base_de_donnees.json'):
        st.error("Fichier 'base_de_donnees.json' introuvable sur GitHub !")
        st.stop()
        
    df = pd.read_json('base_de_donnees.json')
    df_copy = df.copy().fillna("")
    
    # Nettoyage des listes pour Chroma
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
    
    # Création du modèle d'embedding
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(df_copy["text_for_embedding"].tolist(), normalize_embeddings=True).tolist()

    # Client Chroma éphémère (parfait pour Streamlit Cloud)
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

# --- 3. FONCTION DE GÉNÉRATION ---
# Génération de texte avec Mistral API
def generer_reponse_chef(query, results):
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    # On construit le contexte à partir des résultats de ChromaDB
    contexte = "\n".join([f"- {m['titre']}: {m['description']}" for m in results['metadatas'][0]])
    

    # Instructions pour mon prompt
    prompt = f"""TU ES CHEF MUFFIN, UN ASSISTANT CULINAIRE OBSESSIONNEL MAIS SYMPATHIQUE.
TON OBJECTIF EST DE TROUVER LA RECETTE DE MUFFIN IDÉALE PARMI LE CONTEXTE FOURNI.

### TES DIRECTIVES (GUARDRAILS) :
1. OBSESSION : Tu ne cuisines QUE des muffins. Si on te demande des lasagnes ou une pizza, REFUSE poliment avec humour.
2. ANCRAGE : Utilise UNIQUEMENT les recettes fournies dans le bloc [CONTEXTE]. N'invente rien.
3. LANGUE : Réponds toujours en français courant et appétissant.
4. CORRECTION : si l'utilisateur te demande de cuisiner avec des choses qui ne sont pas des aliments, réponds lui avec humour que tu n'es pas mécanicien, ou magicien etc... 
5. Il y a plusieurs cas, si l'utilistaeur te donne des ingrédients/à une requête qui correspond très bien avec l'une des 3 recettes de results, alors ne renvoit que cette recette à l'utilisateur,
si les 3 propositions sont proches mais ne correspondent pas exactement, dis à l'utilisateur que tu n'as pas en stock une recette qui correspond parfaitement à ses attentes mais propose
lui les trois recettes en suggestions, pour que ça l'inspire ! Attention, ces recettes doivent quand même contejnir au moins l'un des ingrédient demandé, ou bien être dans la même famille d'aliment :
par exemple si je demande courgettes il me propose au moins un muffin avec un autre légume. Si les 3 propositions n'ont rien à voir alors ne rien renvoyer. 
Si l'utilisateur te donne des ingrédients pour une recette salée, ne lui propose pas les recettes sucrées.

Dans tous les cas, réponds toujours avec bonne humeur, entrain et humour ! Tu es un fan inconditionnel de muffins.

[CONTEXTE]
{contexte}
[QUESTION]
{query} """
    chat_response = client.chat.complete(
          model="mistral-small-latest", # Modèle équilibré et efficace
          messages=[
              {
                  "role": "user",
                  "content": prompt,
              },
          ]
      )
      
    return chat_response.choices[0].message.content

# --- 4. INTERFACE UTILISATEUR (STREAMLIT) ---
st.set_page_config(page_title="Chef Muffin", page_icon="🧁")

st.title("👨‍🍳 Le Royaume du Chef Muffin")
st.markdown("Bienvenue ! Je suis le Chef, posez-moi vos questions sur les muffins !")

# Initialisation au chargement de la page
with st.spinner("Le Chef prépare sa cuisine... (Initialisation)"):
    collection, model_embed = initialiser_base_donnees()

# Champ de saisie
query = st.text_input("Quelle envie avez-vous aujourd'hui ?", placeholder="Ex: Un muffin salé avec du fromage")

if st.button("Demander au Chef"):
    if query:
        with st.spinner("Recherche de la meilleure recette..."):
            # Recherche vectorielle
            query_vector = model_embed.encode([query], normalize_embeddings=True).tolist()
            res = collection.query(query_embeddings=query_vector, n_results=3)
            
            # Appel à l'IA
            reponse = generer_reponse_chef(query, res)
            
            # Affichage
            st.chat_message("assistant").write(reponse)
            
            # Optionnel : Afficher les sources pour vérifier
            with st.expander("Voir les recettes trouvées par le moteur"):
                for m in res['metadatas'][0]:
                    st.write(f"📍 {m['titre']}")
    else:
        st.warning("Dites-moi quelque chose, je ne lis pas encore dans les pensées ! 🧁")