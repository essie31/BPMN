import streamlit as st
import pandas as pd
import plotly.express as px
import xml.etree.ElementTree as ET
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================
# 0. CONFIGURATION & MOT DE PASSE
# ==========================================
st.set_page_config(page_title="Assistant BPMN SAP B1", page_icon="⚙️", layout="wide")

def check_password():
    """Vérifie si l'utilisateur a le bon mot de passe."""
    def password_entered():
        expected_password = st.secrets.get("APP_PASSWORD", "admin123") 
        if st.session_state["password"] == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Entrez le mot de passe pour accéder à l'application", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Entrez le mot de passe pour accéder à l'application", type="password", on_change=password_entered, key="password")
        st.error("😕 Mot de passe incorrect")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 1. INITIALISATION (Modèles & Base de données)
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=st.secrets.get("PINECONE_API_KEY", "YOUR_PINECONE_KEY"))
    return pc.Index("sap-b1-knowledge")

model = load_embedding_model()
try:
    index = init_pinecone()
    pinecone_connected = True
except Exception as e:
    pinecone_connected = False
    pinecone_error = str(e)

hf_token = st.secrets.get("HF_TOKEN", "YOUR_HF_TOKEN")
hf_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

# ==========================================
# 2. FONCTION D'EXTRACTION BPMN (XML)
# ==========================================
def extract_tasks_from_bpmn(uploaded_file):
    """Parse le fichier XML/BPMN et extrait les tâches."""
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        
        # Le namespace BPMN standard
        namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        
        extracted_tasks = []
        
        # On cherche tous les types de tâches courants
        task_types = ['task', 'userTask', 'manualTask', 'serviceTask', 'sendTask', 'receiveTask']
        
        for task_type in task_types:
            for task in root.findall(f'.//bpmn:{task_type}', namespaces):
                task_name = task.get('name', 'Tâche sans nom')
                extracted_tasks.append({
                    "Étape": "Général", # Peut être amélioré en cherchant les 'lanes'
                    "Processus": task_name.strip().replace('\n', ' '),
                    "Type de Tâche": task_type
                })
                
        return pd.DataFrame(extracted_tasks)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier BPMN : {e}")
        return pd.DataFrame()

# ==========================================
# INTERFACE UTILISATEUR
# ==========================================
st.title("🏭 Hub d'Intégration : BPMN & SAP Business One 10.0")
st.markdown("Bienvenue dans votre assistant expert. Importez votre processus et naviguez via les onglets.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Processus & Tâches", 
    "🕸️ 2. Maturité Industrie 4.0", 
    "🔗 3. Intégrations SAP B1", 
    "🤖 4. Assistant IA (RAG)"
])

# --- ONGLET 1: UPLOAD & TÂCHES ---
with tab1:
    st.header("1. Analyse du Processus")
    
    # Bouton d'upload pour le BPMN
    uploaded_bpmn = st.file_uploader("📂 Importez votre fichier processus (.bpmn ou .xml)", type=["bpmn", "xml"])
    
    if uploaded_bpmn is not None:
        st.success("Fichier importé avec succès !")
        tasks_df = extract_tasks_from_bpmn(uploaded_bpmn)
        
        if not tasks_df.empty:
            st.subheader("Tableau Synthétique des Tâches (Extrait automatiquement)")
            st.dataframe(tasks_df, use_container_width=True)
            
            st.subheader("2. Description Logique du Processus")
            st.info("💡 Astuce : L'IA peut générer une description textuelle de ce flux en envoyant ces tâches au modèle.")
            # Vous pourriez ajouter un bouton ici pour demander à Mistral de rédiger un résumé basé sur 'tasks_df'
        else:
            st.warning("Aucune tâche reconnue trouvée dans ce fichier BPMN.")
    else:
        st.info("Veuillez importer un fichier BPMN pour voir la liste des tâches.")

# --- ONGLET 2: MATURITÉ ---
with tab2:
    st.header("Évaluation Indice de Maturité I4.0")
    st.markdown("*(Note : Ces scores sont actuellement basés sur l'évaluation standard de votre modèle. L'évaluation dynamique en temps réel nécessiterait un appel API à chaque upload).*")
    
    col1, col2 = st.columns([1, 1])
    radar_data = pd.DataFrame({
        "Pilier": [
            "Big Data", "Robots Autonomes", "Simulation", 
            "Intégration Systèmes", "IIoT", "Cybersécurité", 
            "Cloud", "Fabrication Additive", "Réalité Augmentée"
        ],
        "Score": [3, 2, 2, 4, 2, 4, 3, 1, 2]
    })
    
    with col1:
        st.subheader("Tableau des Scores")
        st.dataframe(radar_data, use_container_width=True)
        
    with col2:
        st.subheader("Radar de Maturité")
        fig = px.line_polar(radar_data, r='Score', theta='Pilier', line_close=True, range_r=[0,5])
        fig.update_traces(fill='toself', line_color="orange", fillcolor="rgba(255, 165, 0, 0.5)")
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
        st.plotly_chart(fig, use_container_width=True)

# --- ONGLET 3: SAP B1 ---
with tab3:
    st.header("Propositions d'Intégration SAP Business One 10.0")
    st.markdown("Recommandations standard basées sur votre analyse documentaire :")
    # (J'ai raccourci la liste ici pour la lisibilité, vous pouvez remettre toute la liste précédente)
    sap_integrations = [
        {"Tâche type": "Vérification packing", "Module": "Stocks/Achats", "Action": "Scan codes-barres pour rapprochement."},
        {"Tâche type": "Créer OF", "Module": "Production", "Action": "Création auto des OF via commandes clients."}
    ]
    st.dataframe(pd.DataFrame(sap_integrations), use_container_width=True)

# --- ONGLET 4: ASSISTANT IA ---
with tab4:
    st.header("🤖 Assistant SAP B1 (Mistral 7B)")
    st.markdown("Posez vos questions. L'IA lit uniquement vos documents vectorisés.")

    if not pinecone_connected:
        st.error(f"Erreur Pinecone : {pinecone_error}")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: Comment automatiser la création des OF ?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Recherche dans la documentation..."):
                    query_embedding = model.encode(prompt).tolist()
                    search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                    
                    if not search_results['matches']:
                        context_text = "Aucune information trouvée dans la base."
                    else:
                        contexts = [match['metadata']['text'] for match in search_results['matches']]
                        context_text = "\n\n---\n\n".join(contexts)

                    prompt_template = f"""<s>[INST] Tu es un expert SAP Business One.
Règles: 
1. Réponds UNIQUEMENT avec le contexte. 
2. Si la réponse n'y est pas, dis "Information introuvable."
3. Réponds en français.

CONTEXTE :
{context_text}

QUESTION :
{prompt} [/INST]"""

                    try:
                        answer = hf_client.text_generation(
                            prompt_template, max_new_tokens=400, temperature=0.01, return_full_text=False
                        )
                    except Exception as e:
                        answer = f"Erreur API : {str(e)}"

                st.markdown(answer)
                with st.expander("🔍 Sources (Pinecone)"):
                    st.write(context_text)
                    
            st.session_state.messages.append({"role": "assistant", "content": answer})
