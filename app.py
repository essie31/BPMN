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
# 2. FONCTIONS D'EXTRACTION ET DE SCORING
# ==========================================
def extract_tasks_from_bpmn(uploaded_file):
    """Parse le fichier XML/BPMN et extrait les tâches dynamiquement."""
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

def generate_detailed_scoring(tasks_df, selected_pillar):
    """Génère des justifications et scores détaillés pour chaque tâche en fonction du pilier."""
    # Base de connaissances simplifiée issue de votre PDF pour justifier sans appel API payant
    rules = {
        "Big Data": {"keywords": ["vérification", "stock", "calcul", "sap", "donnée", "mesure", "tva"], "score": 3, "just": "Collecte et analyse de données essentielles pour l'optimisation des processus."},
        "Robots Autonomes": {"keywords": ["coupe", "confection", "emballage", "retrait", "déchargement"], "score": 3, "just": "Potentiel d'automatisation via des machines spécialisées ou des AGV."},
        "Simulation": {"keywords": ["planning", "planifier", "tracé", "délavage", "teinture"], "score": 3, "just": "Simulation des paramètres pour optimiser les temps de cycle et le rendement matière."},
        "Intégration Systèmes": {"keywords": ["sap", "commande", "facture", "tva", "paiement"], "score": 4, "just": "Intégration critique entre l'ERP, les outils de production et les systèmes financiers."},
        "IIoT": {"keywords": ["machine", "teinture", "délavage", "confection", "stock"], "score": 3, "just": "Utilisation de capteurs connectés pour remonter les paramètres en temps réel."},
        "Cybersécurité": {"keywords": ["sap", "tva", "paiement", "facture", "dossier technique"], "score": 5, "just": "Protection maximale requise pour ces données financières ou de propriété intellectuelle."},
        "Cloud": {"keywords": ["sap", "commande", "client", "fournisseur", "stock"], "score": 4, "just": "Déploiement sur plateforme Cloud pour un accès partagé et en temps réel."},
        "Fabrication Additive": {"keywords": ["prototype", "échantillon", "modèle"], "score": 3, "just": "Utilisation potentielle de l'impression 3D pour la création d'accessoires ou de gabarits."},
        "Réalité Augmentée": {"keywords": ["vérification", "mesure", "retrait", "confection", "coupe"], "score": 2, "just": "Possibilité d'utiliser la RA pour assister visuellement les opérateurs sur poste."}
    }
    
    results = []
    pillar_rule = rules.get(selected_pillar, {"keywords": [], "score": 1, "just": "Peu de pertinence directe pour ce pilier."})
    
    for _, row in tasks_df.iterrows():
        task_name = row["Processus"].lower()
        score = 1
        justification = "Non pertinent ou peu d'impact direct."
        
        # Attribution logique des scores basée sur les mots clés du processus
        for kw in pillar_rule["keywords"]:
            if kw in task_name:
                score = pillar_rule["score"]
                # Boost pour SAP qui est le coeur du système
                if "sap" in task_name and selected_pillar in ["Big Data", "Intégration Systèmes", "Cloud", "Cybersécurité"]:
                    score = 5
                justification = pillar_rule["just"]
                break
                
        results.append({"Tâche BPMN": row["Processus"], "Score (1-5)": score, "Justification": justification})
        
    return pd.DataFrame(results)

# ==========================================
# INTERFACE UTILISATEUR
# ==========================================
st.title("🏭 Hub d'Intégration : BPMN & SAP Business One 10.0")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Processus & Tâches", 
    "🕸️ 2. Maturité Industrie 4.0", 
    "🔗 3. Intégrations SAP B1", 
    "🤖 4. Assistant IA (RAG)"
])

# --- ONGLET 1: PROCESSUS & TÂCHES ---
with tab1:
    st.header("1. Analyse du Processus")
    
    uploaded_bpmn = st.file_uploader("📂 Importez votre fichier processus (.bpmn ou .xml)", type=["bpmn", "xml"])
    
    if uploaded_bpmn is not None:
        st.success("Fichier importé avec succès !")
        tasks_df = extract_tasks_from_bpmn(uploaded_bpmn)
        st.session_state['tasks_df'] = tasks_df # Sauvegarde pour l'onglet 2
        
        if not tasks_df.empty:
            st.subheader("Tableau Synthétique des Tâches (Extrait du fichier)")
            st.dataframe(tasks_df, use_container_width=True)
            
            st.subheader("2. Description Logique du Processus")
            st.markdown("""
            Ce processus global semble couvrir l'intégralité du cycle de vie d'une commande client dans une entreprise de confection, depuis la réception de la demande jusqu'au paiement final et l'archivage, en passant par la conception, la production et le contrôle qualité. Il est segmenté en plusieurs phases interconnectées.
            
            **1. Phase Initiale (Commande & Planification):**
            * Le processus débute par la Réception de la commande client ou la Réception du Demande de prototype.
            * La création de la commande client, des nomenclatures, et des ordres de fabrication (OF) sont des étapes clés. 
            * La négociation de la date de réception de Matière Première (MP) avec les fournisseurs et la vérification de l'état du stock sont cruciales avant d'approvisionner.
            * La planification de la production est réalisée via MS Project, suivie de la vérification de la capacité des chaînes.
            
            **2. Phase de Conception et Prototypage:**
            * L'envoi du dossier technique aux Bureaux d'Études (BE) et modélistes marque le début de la conception.
            * Le Prototypage est une étape majeure incluant lavage, réception d'échantillons et boucles d'approbation avec le client. En cas de refus, les paramètres sont ajustés et l'échantillon refait.
            * Le Délavage, Grattage et Teinture sont des processus clés pour le traitement.
            
            **3. Phase de Production (Confection):**
            * Une fois approuvés, le processus enchaîne avec la réalisation des tracés. La Confection représente la production réelle.
            * Des tâches de contrôle et mesure avant lavage et après emballage valident la qualité (ou requièrent l'ajustement des défauts). L'Emballage est l'étape finale avant l'expédition.
            
            **4. Phase Logistique (Réception MP & Expédition):**
            * La réception fourniture initie le pesage pour vérification. Le déchargement par nuances et la saisie SAP assurent la gestion des stocks.
            * L'établissement des listes de colisage et la réception packing sont liés à l'expédition finale.
            
            **5. Phase Administrative et Financière:**
            * Rapprochement crucial entre Bon de commande et Bon de livraison.
            * Saisie des factures, paiements aux façonniers (80% puis 20% restants). Déclarations (TVA, délais de paiement) et préparation des dossiers de remboursement.
            """)
        else:
            st.warning("Aucune tâche reconnue trouvée dans ce fichier BPMN.")
    else:
        st.info("Veuillez importer un fichier BPMN pour voir la liste des tâches et l'analyse.")

# --- ONGLET 2: MATURITÉ ---
with tab2:
    st.header("Évaluation Indice de Maturité I4.0 (9 Piliers)")
    
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
        st.subheader("Scores Globaux")
        st.dataframe(radar_data, use_container_width=True)
        
    with col2:
        st.subheader("Radar de Maturité")
        fig = px.line_polar(radar_data, r='Score', theta='Pilier', line_close=True, range_r=[0,5])
        fig.update_traces(fill='toself', line_color="orange", fillcolor="rgba(255, 165, 0, 0.5)")
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Détail des Justifications par Tâche")
    if 'tasks_df' in st.session_state and not st.session_state['tasks_df'].empty:
        selected_pillar = st.selectbox("Sélectionnez un pilier pour voir le détail de l'évaluation :", radar_data["Pilier"].tolist())
        detailed_scores_df = generate_detailed_scoring(st.session_state['tasks_df'], selected_pillar)
        st.dataframe(detailed_scores_df, use_container_width=True, height=400)
    else:
        st.info("Veuillez importer un fichier BPMN dans l'onglet 1 pour générer les tableaux de justification par tâche.")

# --- ONGLET 3: SAP B1 ---
with tab3:
    st.header("Propositions d'Intégration SAP Business One 10.0")
    st.markdown("Recommandations complètes basées sur votre analyse documentaire :")
    
    sap_integrations = [
        {"Tâche BPMN": "Vérification des packing liste", "Module SAP B1": "Stocks/Achats", "Recommandation": "Utiliser la fonction de scan de codes-barres pour rapprocher automatiquement les articles reçus/expédiés avec la liste de colisage générée dans SAP B1. Intégrer des alertes en cas d'écart."},
        {"Tâche BPMN": "Vérification de métrage commandée VS reçue", "Module SAP B1": "Stocks/Achats", "Recommandation": "Saisie des quantités réelles avec unité de mesure (ex: mètres) lors de la réception de marchandises. SAP B1 calcule les écarts automatiquement."},
        {"Tâche BPMN": "Établir une liste de colisage", "Module SAP B1": "Ventes", "Recommandation": "Génération automatique de la liste de colisage à partir de la commande client ou de l'ordre de fabrication dans SAP B1 (poids, dimensions, articles)."},
        {"Tâche BPMN": "Déclaration de TVA", "Module SAP B1": "Comptabilité/Rapports", "Recommandation": "SAP B1 permet de générer automatiquement les rapports de TVA basés sur les transactions de vente et d'achat enregistrées."},
        {"Tâche BPMN": "Calculer les pertes", "Module SAP B1": "Production/Stocks", "Recommandation": "Suivi des quantités consommées vs. produites dans les OF. SAP B1 calcule et valorise les pertes."},
        {"Tâche BPMN": "Paiement / Paiement aux façonniers", "Module SAP B1": "Banque/Comptabilité", "Recommandation": "Création et exécution des paiements directement depuis SAP B1. Conditions de paiement (acomptes, soldes) gérées via les fiches fournisseurs."},
        {"Tâche BPMN": "Saisir les factures dans le système", "Module SAP B1": "Achats", "Recommandation": "Saisie directe des factures fournisseurs avec liaison aux bons de commande pour automatiser le rapprochement."},
        {"Tâche BPMN": "Vérification quantité commandée VS livrée", "Module SAP B1": "Achats / Stocks", "Recommandation": "Rapprochement automatique des quantités des bons de commande, réceptions et factures avec alertes configurées pour les écarts."},
        {"Tâche BPMN": "Réception de Bon de commande et de Bon de livraison", "Module SAP B1": "Achats", "Recommandation": "Saisie de la réception en référence au BC. Le système génère un BL interne et met à jour les stocks et engagements financiers."},
        {"Tâche BPMN": "Approvisionner la matière première", "Module SAP B1": "Achats", "Recommandation": "Utiliser la planification des besoins en matériel (MRP) pour générer automatiquement des recommandations d'achat."},
        {"Tâche BPMN": "créer la commande client", "Module SAP B1": "Ventes", "Recommandation": "Saisie structurée permettant de vérifier la disponibilité des stocks et la gestion des prix spécifiques aux clients."},
        {"Tâche BPMN": "Créer OF", "Module SAP B1": "Production", "Recommandation": "Création automatique des OF à partir des commandes clients. L'OF consomme les nomenclatures (BOM) et gère le routage."},
        {"Tâche BPMN": "créer des nomenclature", "Module SAP B1": "Production", "Recommandation": "Gestion centralisée des BOM. Chaque produit est lié à sa liste de composants et ressources pour le calcul des coûts."},
        {"Tâche BPMN": "Passer la commande", "Module SAP B1": "Achats", "Recommandation": "Génération des bons de commande depuis les demandes d'achat avec envoi automatisé aux fournisseurs (e-mail ou EDI)."},
        {"Tâche BPMN": "Vérifier l'état du stock", "Module SAP B1": "Stocks", "Recommandation": "Consultation en temps réel de l'état des stocks par article, entrepôt, lot/numéro de série."},
        {"Tâche BPMN": "Négocier date de réception MP", "Module SAP B1": "Achats", "Recommandation": "Enregistrement et suivi des dates de livraison confirmées dans les BC avec alertes pour les retards."},
        {"Tâche BPMN": "Préparation dossier remboursement", "Module SAP B1": "Comptabilité / Services", "Recommandation": "Utilisation des notes de crédit et régularisations comptables pour traiter les remboursements."},
        {"Tâche BPMN": "Déclaration des délais de paiement", "Module SAP B1": "Comptabilité / Rapports", "Recommandation": "Rapports de vieillissement des dettes et créances pour suivre et déclarer les délais de paiement."},
        {"Tâche BPMN": "Classement des factures", "Module SAP B1": "Gestion des documents", "Recommandation": "Attacher numériquement les factures scannées aux documents correspondants dans SAP B1."},
        {"Tâche BPMN": "Réaliser le planning dans MS project", "Module SAP B1": "Production / Add-ons", "Recommandation": "Intégration bidirectionnelle : les données des OF alimentent MS Project, et les dates planifiées remontent dans SAP B1."},
        {"Tâche BPMN": "Planifier la production", "Module SAP B1": "Production", "Recommandation": "Utiliser le module MRP pour générer des recommandations de production basées sur les commandes, prévisions et capacités."},
        {"Tâche BPMN": "Suivi de la production", "Module SAP B1": "Production", "Recommandation": "Utiliser les OF pour suivre l'avancement. Intégration possible avec un système MES pour la saisie en temps réel."},
        {"Tâche BPMN": "Saisie des quantités reçues", "Module SAP B1": "Stocks / Production", "Recommandation": "Enregistrement des MP reçues et quantités produites par OF pour la mise à jour des stocks et suivi des coûts."},
        {"Tâche BPMN": "Partager l'état du stock avec les achats", "Module SAP B1": "Stocks / Achats", "Recommandation": "Les rapports standard et le module MRP garantissent que les informations sont à jour et transparentes entre services."}
    ]
    
    st.dataframe(pd.DataFrame(sap_integrations), use_container_width=True, height=600)

# --- ONGLET 4: ASSISTANT IA ---
with tab4:
    st.header("🤖 Assistant SAP B1 (Open Source - Mistral)")
    st.markdown("Posez vos questions. L'IA lit uniquement vos documents vectorisés (Pinecone).")

    if not pinecone_connected:
        st.error(f"Erreur Pinecone : {pinecone_error}")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: Comment intégrer les bons de commande dans SAP B1 ?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Recherche et génération de la réponse..."):
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
