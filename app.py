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

# Initialisation du client Groq
groq_api_key = st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
GROQ_MODEL = "llama3-70b-8192" # Modèle très puissant pour l'analyse

# ==========================================
# 2. FONCTIONS D'EXTRACTION ET D'ANALYSE DYNAMIQUE
# ==========================================
def extract_tasks_from_bpmn(uploaded_file):
    """Parse le fichier XML/BPMN et extrait les tâches."""
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        extracted_tasks = []
        task_types = ['task', 'userTask', 'manualTask', 'serviceTask', 'sendTask', 'receiveTask', 'subProcess']
        
        for task_type in task_types:
            for task in root.findall(f'.//bpmn:{task_type}', namespaces):
                task_name = task.get('name')
                if task_name:
                    extracted_tasks.append({
                        "Étape": "Général",
                        "Processus": task_name.strip().replace('\n', ' '),
                        "Type de Tâche": task_type
                    })
        return pd.DataFrame(extracted_tasks)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier BPMN : {e}")
        return pd.DataFrame()

def analyze_with_groq(system_prompt, user_prompt):
    """Fonction générique pour appeler l'API Groq."""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Température basse pour être factuel
            max_tokens=2000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erreur avec l'API Groq : {e}"

# ==========================================
# INTERFACE UTILISATEUR
# ==========================================
st.title("🏭 Hub d'Intégration : BPMN & SAP Business One 10.0")

# Option pour le mode impression (retire les onglets)
st.sidebar.header("Options d'affichage")
print_mode = st.sidebar.checkbox("🖨️ Mode Impression (Vue globale)")
st.sidebar.info("Cochez cette case pour afficher tout le contenu sur une seule page, puis faites **Ctrl+P** pour imprimer en PDF.")

st.header("1. Analyse du Processus")
uploaded_bpmn = st.file_uploader("📂 Importez votre fichier processus (.bpmn ou .xml)", type=["bpmn", "xml"])

if uploaded_bpmn is not None:
    # On extrait les tâches si c'est un nouveau fichier
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_bpmn.name:
        st.session_state.tasks_df = extract_tasks_from_bpmn(uploaded_bpmn)
        st.session_state.last_uploaded_file = uploaded_bpmn.name
        st.session_state.analysis_done = False # Reset l'analyse
        
    tasks_df = st.session_state.tasks_df
    
    if not tasks_df.empty:
        st.success(f"{len(tasks_df)} tâches trouvées dans le fichier.")
        
        # Bouton pour lancer l'analyse dynamique
        if not st.session_state.get("analysis_done", False):
            if st.button("🧠 Lancer l'analyse complète par IA (Groq)"):
                tasks_text = ", ".join(tasks_df["Processus"].tolist())
                
                with st.spinner("Génération de la description logique..."):
                    desc_prompt = f"Voici les tâches extraites d'un processus industriel : {tasks_text}. Rédige une description logique de ce processus divisée en 3 ou 4 phases claires avec des bullet points."
                    st.session_state.ai_description = analyze_with_groq("Tu es un expert BPMN.", desc_prompt)
                
                with st.spinner("Évaluation des 9 piliers de l'Industrie 4.0..."):
                    mat_prompt = f"""Évalue ces tâches BPMN : {tasks_text}.
                    Pour CHACUN des 9 piliers suivants (Big Data, Robots Autonomes, Simulation, Intégration Systèmes, IIoT, Cybersécurité, Cloud, Fabrication Additive, Réalité Augmentée), donne une note de 1 à 5 et une très brève justification.
                    FORMAT STRICT REQUIS :
                    Pilier | Note | Justification
                    Big Data | 3 | Explication...
                    (Génère uniquement ce tableau)"""
                    st.session_state.ai_maturity = analyze_with_groq("Tu es un auditeur Industrie 4.0.", mat_prompt)
                
                with st.spinner("Génération des recommandations SAP B1..."):
                    sap_prompt = f"""Pour les tâches suivantes : {tasks_text}. 
                    Propose 7 à 10 recommandations d'intégration spécifiques avec SAP Business One 10.0.
                    FORMAT STRICT REQUIS :
                    Tâche BPMN | Module SAP B1 | Recommandation
                    Tâche X | Achats | Explication...
                    (Génère uniquement ce tableau markdown)"""
                    st.session_state.ai_sap = analyze_with_groq("Tu es un architecte SAP Business One.", sap_prompt)
                
                st.session_state.analysis_done = True
                st.rerun()

        # AFFICHAGE DES RÉSULTATS (Avec ou sans onglets selon le Mode Impression)
        if st.session_state.get("analysis_done", False):
            
            # --- FONCTIONS D'AFFICHAGE ---
            def show_tab1():
                st.subheader("Tableau Synthétique des Tâches")
                st.dataframe(tasks_df, use_container_width=True)
                st.subheader("Description Logique du Processus (Générée par l'IA)")
                st.markdown(st.session_state.ai_description)

            def show_tab2():
                st.subheader("Évaluation Indice de Maturité I4.0")
                # Parsing dynamique du texte retourné par Groq pour faire le graphique
                lines = st.session_state.ai_maturity.split('\n')
                pilier_data = {"Pilier": [], "Score": []}
                for line in lines:
                    if "|" in line and "Pilier" not in line and "---" not in line:
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 3:
                            # Cleanup du texte pour extraire le score
                            try:
                                score = int(re.search(r'\d+', parts[1]).group())
                                pilier_data["Pilier"].append(parts[0])
                                pilier_data["Score"].append(score)
                            except:
                                pass
                
                if len(pilier_data["Score"]) > 0:
                    radar_df = pd.DataFrame(pilier_data)
                    fig = px.line_polar(radar_df, r='Score', theta='Pilier', line_close=True, range_r=[0,5])
                    fig.update_traces(fill='toself', line_color="orange", fillcolor="rgba(255, 165, 0, 0.5)")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Justifications Détaillées")
                st.markdown(st.session_state.ai_maturity)

            def show_tab3():
                st.subheader("Propositions d'Intégration SAP Business One 10.0")
                st.markdown(st.session_state.ai_sap)

            def show_tab4():
                st.subheader("🤖 Assistant SAP B1 (RAG via Groq)")
                if not pinecone_connected:
                    st.error(f"Erreur Pinecone : {pinecone_error}")
                else:
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if prompt := st.chat_input("Ex: Comment configurer le MRP ?"):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        with st.chat_message("assistant"):
                            with st.spinner("Recherche (Pinecone) et Génération (Groq)..."):
                                query_embedding = model.encode(prompt).tolist()
                                search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                                
                                if not search_results['matches']:
                                    context_text = "Aucune information trouvée dans la base."
                                else:
                                    contexts = [match['metadata']['text'] for match in search_results['matches']]
                                    context_text = "\n\n---\n\n".join(contexts)

                                system_message = f"""Tu es un expert SAP Business One.
                                1. Réponds UNIQUEMENT avec le contexte ci-dessous. 
                                2. Si l'info n'y est pas, dis "Information introuvable."
                                3. Réponds en français.
                                CONTEXTE :
                                {context_text}"""

                                try:
                                    response = groq_client.chat.completions.create(
                                        model=GROQ_MODEL,
                                        messages=[
                                            {"role": "system", "content": system_message},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.01
                                    )
                                    answer = response.choices[0].message.content
                                except Exception as e:
                                    answer = f"Erreur API Groq : {str(e)}"

                            st.markdown(answer)
                            with st.expander("🔍 Sources (Pinecone)"):
                                st.write(context_text)
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer})

            # --- LOGIQUE D'AFFICHAGE SELON LE MODE ---
            if print_mode:
                st.warning("🖨️ Mode Impression activé. Faites **Ctrl+P** pour imprimer ou sauvegarder en PDF. Décochez l'option dans la barre latérale pour retrouver les onglets.")
                st.markdown("---")
                show_tab1()
                st.markdown("---")
                show_tab2()
                st.markdown("---")
                show_tab3()
                st.markdown("---")
                show_tab4()
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Processus", "🕸️ Maturité I4.0", "🔗 Intégrations SAP", "🤖 Chat RAG"])
                with tab1: show_tab1()
                with tab2: show_tab2()
                with tab3: show_tab3()
                with tab4: show_tab4()

    else:
        st.warning("Aucune tâche reconnue dans ce fichier.")
