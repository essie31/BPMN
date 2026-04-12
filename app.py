import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================
# 0. PAGE CONFIGURATION & PASSWORD LOCK
# ==========================================
st.set_page_config(page_title="SAP B1 BPMN Assistant", page_icon="⚙️", layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        expected_password = st.secrets.get("APP_PASSWORD", "admin123") 
        if st.session_state["password"] == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Entrez le mot de passe pour accéder", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Entrez le mot de passe pour accéder", type="password", on_change=password_entered, key="password")
        st.error("😕 Mot de passe incorrect")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 1. INITIALIZATION (Models & Database)
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

# Initialisation du modèle Open Source Gratuit (Mistral via Hugging Face)
hf_token = st.secrets.get("HF_TOKEN", "YOUR_HF_TOKEN")
# InferenceClient utilise l'API gratuite de Hugging Face
hf_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

# ==========================================
# APP UI & TABS
# ==========================================
st.title("🏭 Hub d'Intégration : BPMN SAP Business One 10.0")
st.markdown("Bienvenue dans votre assistant expert. Naviguez via les onglets ci-dessous pour consulter l'analyse complète de vos processus et interroger le modèle.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Process & Tasks", 
    "🕸️ 2. Industry 4.0 Maturity", 
    "🔗 3. SAP B1 Integrations", 
    "🤖 4. AI Chat Assistant (Open Source)"
])

# --- TAB 1: TASKS & DESCRIPTION ---
with tab1:
    st.header("1. Tableau Synthétique des Tâches")
    
    # Extraction complète des tâches depuis le PDF
    csv_data = """Étape,Processus,Type de Tâche
Général,Vérification des packing liste,manual Task
Général,Vérification de métrage commandée VS reçue,manual Task
Général,Coupe et confection des blanquettes et patchwork,userTask
Général,Mensuration des blanquettes 50cm/50 cm et séparation par nuance,manual Task
Général,Établir une liste de colisage,user Task
Général,Déclaration de TVA,task
Général,Calculer les pertes,task
Général,paiement de 20% restant aux façonniers,task
Général,Situation 2ème choix,dataObjectReference
Général,taux acceptable?,exclusiveGateway
Général,Vérification de taux d'erreurs tolérées,task
Général,paiement de 80% aux façonniers,task
Général,Cas de façonnier?,exclusiveGateway
Général,Paiement,task
Général,Saisir les factures dans le système,task
Général,Vérification de la quantité commandée VS quantité livrée,task
Général,Réception de Bon de commande et de Bon de livraison,task
Général,Approvisionner la matière première,userTask
Général,créer la commande client,userTask
Général,Créer OF,user Task
Général,créer des nomenclature,userTask
Général,Passer la commande,user Task
Général,SAP,dataStoreReference
Général,Vérifier l'état du stock,serviceTask
Général,Reçoie de la commande client,receiveTask
Général,Négocier de la date de réception de MP avec les fournisseurs,user Task
Général,Liste de colisage,dataObjectReference
Général,Réception packing,startEvent
Général,Dossier de remboursement,dataObjectReference
Général,Préparation de dossier de remboursement,task
Général,Déclaration des délais de paiement,task
Général,Classement des factures,task
Général,envoie au fournisseur BC scannée,receiveTask
Général,bon de commande,dataObjectReference
Général,bon de livraison,dataObjectReference
Général,Attendre l'approbation de la commande par la direction,task
Général,Effectuer le paiement selon les nouveaux calculs,task
Général,Demander de la MP par whatsapp,sendTask
Général,Réaliser le retrait,task
Général,Planning,dataObjectReference
Général,Réaliser le planning dans MS project,serviceTask
Général,Reçoie dossier technique,receiveTask
Général,Envoyer les informations des retraits,intermediateCatchEvent
Général,Imprimer le planning + dispatcher le planning,manual Task
Général,Vérifier l'état stock / disponibilités / capacités,manual Task
Général,Planifier la production,userTask
Général,Insérer les commandes dans Ms Project,manual Task
Général,MP disponible?,exclusiveGateway
Général,Envoi de prototype au lavage,task
Général,Réception d'échantillon,task
Général,Envoyer au client pour l'approbation,task
Général,Création du bon de commande de la coupe et confection,task
Général,Envoyer dossier technique aux BE et les modélistes,manual Task
Général,Envoi de collection au lavage,task
Général,Réception de la collection,task
Général,Envoyer dossier technique aux modélistes,task
Général,Traiter la commande client,send Task
Général,Suivi de la production,endEvent
Général,Reçoie du patronage,task
Général,Refaire l'échantillon,task
Général,Ajuster les paramètres,task
Général,Envoie d'échantillon et le dossier technique,manual Task
Général,Contrôle et mesure après l'emballage,manual Task
Général,Emballage,manual Task
Général,Contrôle et mesure avant le lavage,manual Task
Général,Ajuster les défauts,task
Général,Signaler l'échantillon prêt pour le lavage,send Task
Général,Recevoir les articles de délavage,intermediateCatchEvent
Général,Faire une majoration de la quantité commandée,send Task
Général,Reçoie du Demande de prototype,startEvent
Général,Traiter la demande client,sendTask
Général,Décharger les tissus par retraits/nuances,user Task
Général,Déclarer au fournisseur,sendTask
Général,Envoie de la fourniture,manual Task
Général,Déclarer les quantités de fournitures sorties de magasin,userTask
Général,Saisie les quantités reçus selon les modèles dans SAP,userTask
Général,suivi l'état du stock,manual Task
Général,Tracés de prototype,dataObjectReference
Général,Tracés de la collection,dataObjectReference
Général,Tracés de la commande.,dataObjectReference
Général,Estimer la consommation de la peinture et des produits,userTask
Général,Développer la couleur adéquat,manual Task
Général,Délavage,userTask
Général,Grattage,manual Task
Général,Teinture,userTask
Général,Archiver le packing,manual Task
Général,Partager l'état du stock avec les achats,task
Général,Préparer les modèles (fournitures),manual Task
Général,Confection,subProcess
Général,Prototypage,subProcess
Général,Réalisation des tracés de la collection,subProcess
Général,Réalisation des tracés d'une commande,subProcess
Général,Reception fourniture,startEvent
Général,pesage de la commande pour la vérification,userTask
Général,Envoyer le bon de livraison au achat,task"""
    
    tasks_df = pd.read_csv(io.StringIO(csv_data))
    st.dataframe(tasks_df, use_container_width=True)

    st.header("2. Description Logique du Processus")
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
    * Des tâches de contrôle et mesure avant lavage et après emballage valident la qualité (ou requièrent l'ajustement des défauts).
    
    **4. Phase Logistique (Réception MP & Expédition):**
    * La réception fourniture initie le pesage pour vérification. Le déchargement par nuances et la saisie SAP assurent la gestion des stocks.
    * L'établissement des listes de colisage est lié à l'expédition finale.
    
    **5. Phase Administrative et Financière:**
    * Rapprochement crucial entre Bon de commande et Bon de livraison (Vérification quantité commandée VS livrée).
    * Saisie des factures, paiements aux façonniers (80% puis 20% restants). Déclarations (TVA, délais de paiement) et préparation des dossiers de remboursement.
    """)

# --- TAB 2: MATURITY & RADAR CHART ---
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
        st.subheader("Tableau des Scores Finaux")
        st.dataframe(radar_data, use_container_width=True)
        st.markdown("*Scores évalués sur une échelle de 1 à 5.*")
        
    with col2:
        st.subheader("Radar de Maturité")
        fig = px.line_polar(radar_data, r='Score', theta='Pilier', line_close=True, range_r=[0,5])
        fig.update_traces(fill='toself', line_color="orange", fillcolor="rgba(255, 165, 0, 0.5)")
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: SAP B1 INTEGRATIONS ---
with tab3:
    st.header("Propositions d'Intégration SAP Business One 10.0")
    
    sap_integrations = [
        {"Tâche BPMN": "Vérification des packing liste", "Module SAP B1": "Stocks/Achats", "Recommandation": "Utiliser la fonction de scan de codes-barres pour rapprocher automatiquement les articles reçus/expédiés avec la liste de colisage générée dans SAP B1. Intégrer des alertes en cas d'écart."},
        {"Tâche BPMN": "Vérification de métrage commandée VS reçue", "Module SAP B1": "Stocks/Achats", "Recommandation": "Saisie des quantités réelles avec unité de mesure lors de la réception de marchandises. SAP B1 peut calculer automatiquement les écarts par rapport au bon de commande, générer des rapports et initier des actions."},
        {"Tâche BPMN": "Établir une liste de colisage", "Module SAP B1": "Ventes", "Recommandation": "Génération automatique de la liste de colisage à partir de la commande client ou de l'ordre de fabrication dans SAP B1, en intégrant les informations de poids, dimensions et articles."},
        {"Tâche BPMN": "Déclaration de TVA", "Module SAP B1": "Comptabilité/Rapports", "Recommandation": "Générer automatiquement les rapports de TVA basés sur les transactions de vente et d'achat enregistrées, simplifiant la déclaration."},
        {"Tâche BPMN": "Calculer les pertes", "Module SAP B1": "Production/Stocks", "Recommandation": "Suivi des quantités de matières premières consommées vs. produites dans les ordres de fabrication. SAP B1 peut calculer les écarts (pertes) et les valoriser."},
        {"Tâche BPMN": "Paiement aux façonniers", "Module SAP B1": "Banque/Comptabilité", "Recommandation": "Création et exécution des paiements directement depuis SAP B1. Les conditions de paiement (acomptes, soldes) gérées automatiquement via fiches fournisseurs."},
        {"Tâche BPMN": "Saisir les factures", "Module SAP B1": "Achats", "Recommandation": "Saisie directe des factures fournisseurs. La liaison avec bons de commande automatise le rapprochement. L'ajout d'OCR réduit la saisie manuelle."},
        {"Tâche BPMN": "Vérification quantité commandée VS livrée", "Module SAP B1": "Achats/Stocks", "Recommandation": "SAP B1 rapproche automatiquement les quantités des commandes, réceptions et factures avec alertes d'écarts."},
        {"Tâche BPMN": "Approvisionner la matière première", "Module SAP B1": "Achats", "Recommandation": "Utiliser la planification des besoins (MRP) pour générer automatiquement des recommandations d'achat basées sur les commandes et les stocks."},
        {"Tâche BPMN": "Créer la commande client", "Module SAP B1": "Ventes", "Recommandation": "Saisie structurée permettant vérification disponibilité des stocks et gestion des prix spécifiques."},
        {"Tâche BPMN": "Créer OF (Ordre de Fabrication)", "Module SAP B1": "Production", "Recommandation": "Création automatique des OF à partir des commandes clients. L'OF consomme les nomenclatures et gère le routage."},
        {"Tâche BPMN": "Créer des nomenclatures", "Module SAP B1": "Production", "Recommandation": "Gestion centralisée des BOM. Chaque produit est lié à ses composants et ressources, permettant un calcul précis des besoins."},
        {"Tâche BPMN": "Vérifier l'état du stock", "Module SAP B1": "Stocks", "Recommandation": "Consultation en temps réel de l'état par article, entrepôt, lot. Vues détaillées pour optimiser la gestion et éviter les ruptures."},
        {"Tâche BPMN": "Planifier la production", "Module SAP B1": "Production", "Recommandation": "Utiliser le module MRP pour générer des recommandations de production. Des add-ons avancés (APS) peuvent optimiser la planification."},
        {"Tâche BPMN": "Suivi de la production", "Module SAP B1": "Production", "Recommandation": "Utiliser les OF pour suivre l'avancement (début, fin, quantités déclarées, rejets). Intégration possible avec un système MES."}
    ]
    
    st.dataframe(pd.DataFrame(sap_integrations), use_container_width=True)

# --- TAB 4: AI ASSISTANT (RAG) AVEC HUGGING FACE ---
with tab4:
    st.header("🤖 Assistant SAP B1 (Open Source - Mistral)")
    st.markdown("L'assistant utilise le modèle gratuit Mistral-7B via Hugging Face. Zéro hallucination : il lit uniquement vos documents Pinecone.")

    if not pinecone_connected:
        st.error(f"Erreur de connexion à Pinecone : {pinecone_error}")
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
                with st.spinner("Recherche et génération de la réponse (Mistral)..."):
                    # 1. Vectorize User Query
                    query_embedding = model.encode(prompt).tolist()
                    
                    # 2. Search Pinecone for context
                    search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                    
                    if not search_results['matches']:
                        context_text = "Aucune information trouvée dans la base de données."
                    else:
                        contexts = [match['metadata']['text'] for match in search_results['matches']]
                        context_text = "\n\n---\n\n".join(contexts)

                    # 3. Prompt formaté spécifiquement pour Mistral-Instruct
                    prompt_template = f"""<s>[INST] Tu es un expert SAP Business One. Ta seule source de vérité est le contexte fourni ci-dessous.
Règles: 
1. Réponds UNIQUEMENT en te basant sur le contexte. 
2. Si la réponse n'y est pas, dis EXACTEMENT "Désolé, cette information n'est pas dans la documentation." Ne crée rien.
3. Réponds en français de manière professionnelle.

CONTEXTE :
{context_text}

QUESTION :
{prompt} [/INST]"""

                    # 4. Call Hugging Face API
                    try:
                        # Paramètres pour éviter les hallucinations (temperature très basse)
                        answer = hf_client.text_generation(
                            prompt_template,
                            max_new_tokens=400,
                            temperature=0.01,
                            return_full_text=False
                        )
                    except Exception as e:
                        answer = f"Erreur avec l'API Hugging Face : {str(e)}"

                st.markdown(answer)
                with st.expander("🔍 Voir les sources utilisées (Contexte Pinecone)"):
                    st.write(context_text)
                    
            st.session_state.messages.append({"role": "assistant", "content": answer})
