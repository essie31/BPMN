import streamlit as st
import pandas as pd
import plotly.express as px
import xml.etree.ElementTree as ET
import os
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================================
# 0. CONFIGURATION & MOT DE PASSE
# ==========================================
st.set_page_config(page_title="Assistant BPMN SAP B1", page_icon="⚙️", layout="wide")

def check_password():
    def password_entered():
        expected_password = st.secrets.get("APP_PASSWORD", "admin123") 
        if st.session_state["password"] == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Entrez le mot de passe", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Entrez le mot de passe", type="password", on_change=password_entered, key="password")
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

groq_api_key = st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

# Modèle lourd pour la réflexion profonde (Description & Chat)
GROQ_MODEL_REASONING = "llama-3.3-70b-versatile" 
# Modèle ultra-rapide pour traiter des centaines de lignes sans timeout (Tableaux)
GROQ_MODEL_FAST = "llama-3.1-8b-instant"

# ==========================================
# 2. FONCTIONS D'EXTRACTION AVANCÉE
# ==========================================
def extract_bpmn_logic(uploaded_file):
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        
        elements = {}
        tasks_list = []
        
        for elem in root.iter():
            tag_name = elem.tag.lower()
            if 'id' in elem.attrib and 'name' in elem.attrib:
                elem_id = elem.attrib['id']
                name = elem.attrib['name'].replace('\n', ' ').strip()
                if name: 
                    elements[elem_id] = name
                    if 'task' in tag_name or 'subprocess' in tag_name:
                        tasks_list.append({"Processus": name, "Type": elem.tag.split('}')[-1]})

        flows = []
        for flow in root.iter():
            if 'sequenceflow' in flow.tag.lower():
                source = flow.get('sourceRef')
                target = flow.get('targetRef')
                if source in elements and target in elements:
                    flows.append(f"[{elements[source]}] ➔ [{elements[target]}]")
                    
        return pd.DataFrame(tasks_list), flows
    except Exception as e:
        st.error(f"Erreur lors de la lecture : {e}")
        return pd.DataFrame(), []

def analyze_with_groq(system_prompt, user_prompt, ai_model, max_tok=4000):
    try:
        completion = groq_client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, 
            max_tokens=max_tok
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erreur avec l'API Groq : {e}"

# ==========================================
# INTERFACE UTILISATEUR
# ==========================================
st.title("🏭 Hub d'Intégration : BPMN & SAP Business One 10.0")

st.sidebar.header("Options d'affichage")
print_mode = st.sidebar.checkbox("🖨️ Mode Impression (Vue globale)")
st.sidebar.info("Cochez cette case pour tout afficher sur une page, puis Ctrl+P.")

st.header("1. Importation & Analyse du Processus")
uploaded_bpmn = st.file_uploader("📂 Importez votre fichier processus (.bpmn ou .xml)", type=["bpmn", "xml"])

if uploaded_bpmn is not None:
    # Sécurité pour éviter le crash en cas de rechargement partiel de la page
    if ("last_uploaded_file" not in st.session_state 
        or st.session_state.last_uploaded_file != uploaded_bpmn.name
        or "flow_sequence" not in st.session_state):
        
        tasks_df, flow_sequence = extract_bpmn_logic(uploaded_bpmn)
        st.session_state.tasks_df = tasks_df
        st.session_state.flow_sequence = flow_sequence
        st.session_state.last_uploaded_file = uploaded_bpmn.name
        st.session_state.analysis_done = False 
        
    tasks_df = st.session_state.tasks_df
    flow_sequence = st.session_state.flow_sequence
    task_names = tasks_df["Processus"].tolist() if not tasks_df.empty else []
    
    if not tasks_df.empty:
        st.success(f"{len(tasks_df)} tâches et {len(flow_sequence)} connexions (flèches) trouvées !")
        
        if not st.session_state.get("analysis_done", False):
            if st.button("🧠 Lancer l'analyse intelligente du processus"):
                
                # 1. DESCRIPTION FLUIDE (Modèle Lourd)
                with st.spinner("Analyse du cheminement chronologique (flèches)..."):
                    sequence_text = "\n".join(flow_sequence)
                    desc_prompt = f"""Voici le cheminement exact du processus (les flèches) :
                    {sequence_text}
                    Rédige une description chronologique, fluide et logique de ce processus. Raconte l'histoire du flux étape par étape en suivant les flèches."""
                    st.session_state.ai_description = analyze_with_groq(
                        "Tu es un analyste métier expert en BPMN.", desc_prompt, GROQ_MODEL_REASONING
                    )
                
                # 2. MATURITÉ I4.0 PAR LOTS (Chunking avec Modèle Rapide)
                with st.spinner("Évaluation de TOUTES les tâches sur les 9 piliers (Traitement par lots pour éviter les coupures)..."):
                    chunk_size = 25 # L'IA traitera 25 tâches à la fois pour ne pas s'épuiser
                    full_mat_text = ""
                    
                    for i in range(0, len(task_names), chunk_size):
                        chunk = task_names[i:i+chunk_size]
                        tasks_text = ", ".join(chunk)
                        
                        mat_prompt = f"""Évalue UNIQUEMENT ces tâches spécifiques sur les 9 piliers de l'Industrie 4.0.
                        Tâches à évaluer : {tasks_text}.
                        FORMAT STRICT REQUIS (Génère uniquement ce tableau Markdown, rien d'autre) :
                        | Tâche | Pilier (Big Data, IIoT, etc.) | Note (1-5) | Justification (très concis) |
                        |---|---|---|---|"""
                        
                        res = analyze_with_groq("Tu es un auditeur Industrie 4.0.", mat_prompt, GROQ_MODEL_FAST)
                        full_mat_text += res + "\n\n"
                        
                    st.session_state.ai_maturity = full_mat_text
                
                # 3. RECOMMANDATIONS SAP B1 PAR LOTS
                with st.spinner("Génération des recommandations SAP B1..."):
                    chunk_size = 40 # Lots un peu plus grands car moins de colonnes
                    full_sap_text = ""
                    
                    for i in range(0, len(task_names), chunk_size):
                        chunk = task_names[i:i+chunk_size]
                        tasks_text = ", ".join(chunk)
                        
                        sap_prompt = f"""Analyse ces tâches : {tasks_text}. 
                        Propose des recommandations d'intégration techniques avec SAP Business One 10.0 uniquement pour les tâches pertinentes de cette liste.
                        FORMAT STRICT REQUIS :
                        | Tâche BPMN | Module SAP B1 concerné | Recommandation précise |
                        |---|---|---|"""
                        
                        res = analyze_with_groq("Tu es un architecte SAP.", sap_prompt, GROQ_MODEL_FAST)
                        full_sap_text += res + "\n\n"
                        
                    st.session_state.ai_sap = full_sap_text
                
                st.session_state.analysis_done = True
                st.rerun()

        # AFFICHAGE DES RÉSULTATS
        if st.session_state.get("analysis_done", False):
            
            def show_tab1():
                st.subheader("Description Logique du Flux (Suivi des flèches)")
                st.markdown(st.session_state.ai_description)
                with st.expander("Voir les tâches brutes extraites"):
                    st.dataframe(tasks_df, use_container_width=True)

            def show_tab2():
                st.subheader("Justification et Notation de CHAQUE tâche par Pilier I4.0")
                st.markdown(st.session_state.ai_maturity)

            def show_tab3():
                st.subheader("Propositions d'Intégration SAP Business One 10.0")
                st.markdown(st.session_state.ai_sap)

            def show_tab4():
                st.subheader("🤖 Assistant Expert SAP B1 (Recherche Documentaire)")
                st.markdown("L'IA synthétise vos documents SAP vectorisés pour formuler de vraies réponses construites.")
                
                if not pinecone_connected:
                    st.error(f"Erreur Pinecone : {pinecone_error}")
                else:
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if prompt := st.chat_input("Ex: Comment le MRP optimise-t-il la production ?"):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        with st.chat_message("assistant"):
                            with st.spinner("Exploration de la base de données & Synthèse..."):
                                query_embedding = model.encode(prompt).tolist()
                                search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True) 
                                
                                if not search_results['matches']:
                                    context_text = "Aucune documentation pertinente trouvée dans la base."
                                else:
                                    contexts = [match['metadata']['text'] for match in search_results['matches']]
                                    context_text = "\n\n---\n\n".join(contexts)

                                system_message = f"""Tu es un Consultant Senior SAP Business One.
                                Ton rôle est d'apporter une réponse de haute qualité, claire, experte et structurée à l'utilisateur, en te basant **exclusivement** sur la documentation ci-dessous extraite de la base de données de l'entreprise.
                                
                                RÈGLES CRITIQUES :
                                1. NE FAIS PAS de simple "copier-coller". Synthétise l'information, regroupe les idées et formule une réponse fluide et professionnelle (utilise des puces si nécessaire).
                                2. Si les extraits fournis ne permettent pas de répondre à la question, dis poliment que tu n'as pas trouvé l'information dans la documentation SAP fournie. N'invente rien.
                                3. Réponds toujours en français.

                                DOCUMENTS DE LA BASE DE DONNÉES :
                                {context_text}"""

                                # Utilisation du modèle lourd (Reasoning) pour une belle réponse de consultant
                                answer = analyze_with_groq(system_message, prompt, GROQ_MODEL_REASONING)

                            st.markdown(answer)
                            with st.expander("🔍 Voir les extraits bruts tirés de la base de données"):
                                st.write(context_text)
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer})

            if print_mode:
                st.warning("🖨️ Mode Impression activé. Ctrl+P pour imprimer.")
                st.markdown("---")
                show_tab1()
                st.markdown("---")
                show_tab2()
                st.markdown("---")
                show_tab3()
                st.markdown("---")
                show_tab4()
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Processus Chronologique", "🕸️ Matrice Maturité", "🔗 Intégrations SAP", "🤖 Assistant RAG"])
                with tab1: show_tab1()
                with tab2: show_tab2()
                with tab3: show_tab3()
                with tab4: show_tab4()

    else:
        st.warning("Aucune tâche reconnue dans ce fichier.")
