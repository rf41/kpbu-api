import streamlit as st
import requests
import json
import re
from datetime import datetime
import time

# ===================== KONFIGURASI =====================
API_BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "kpbu-matchmaker-2025"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# ===================== DATA MAPPING =====================
SEKTOR_MAPPING = {
    1: "Air dan Sanitasi", 2: "Sumber Daya Air", 3: "Jalan dan Jembatan",
    4: "Transportasi", 5: "Transportasi Darat", 6: "Transportasi Laut",
    7: "Transportasi Udara", 8: "Infrastruktur", 9: "Telekomunikasi dan Informatika",
    10: "Energi", 11: "Minyak dan Gas", 12: "Ketenagalistrikan",
    13: "Kesehatan", 14: "Pendidikan", 15: "Pariwisata",
    16: "Energi Terbarukan", 17: "Perumahan", 18: "Kawasan Industri"
}

STATUS_MAPPING = {
    1: "Perencanaan", 2: "Penyiapan", 3: "Transaksi",
    4: "Penandatanganan Kontrak", 5: "Konstruksi", 6: "Operasi", 7: "Selesai/Berakhir"
}

# ===================== HELPER FUNCTIONS =====================
def make_api_call(endpoint, method="GET", data=None):
    """Helper function to make API calls"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, headers=HEADERS, json=data, timeout=15)
        elif method == "DELETE":
            response = requests.delete(url, headers=HEADERS, timeout=15)
        else:
            response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def format_currency(value):
    """Format currency in trillions"""
    if value >= 1:
        return f"Rp {value:.2f} triliun"
    else:
        return f"Rp {value*1000:.0f} miliar"

def init_session_state():
    """Initialize session state variables"""
    if 'investor_profile' not in st.session_state:
        st.session_state.investor_profile = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_active' not in st.session_state:
        st.session_state.chat_active = False
    if 'agent_typing' not in st.session_state:
        st.session_state.agent_typing = False
    if 'pending_message' not in st.session_state:
        st.session_state.pending_message = None
    if 'latest_suggestions' not in st.session_state:
        st.session_state.latest_suggestions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False

def display_chat_message(role, content, user_name="User", suggested_questions=None, message_id=None):
    """Display a chat message with proper styling and markdown support"""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**{user_name}**")
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("**AI Investment Consultant**")
            # Process markdown for AI messages
            processed_content = content
            
            # Handle bold text
            processed_content = re.sub(r'\*\*(.*?)\*\*', r'**\1**', processed_content)
            
            # Handle numbered lists - fix single line issue
            # Split by lines and process numbered items
            lines = processed_content.split('\n')
            processed_lines = []
            
            for line in lines:
                # Check if line starts with number followed by dot and space
                if re.match(r'^\d+\.\s+', line.strip()):
                    # Add newline before numbered item if previous line wasn't empty
                    if processed_lines and processed_lines[-1].strip() != '':
                        processed_lines.append('')
                    processed_lines.append(line)
                    # Add newline after numbered item
                    processed_lines.append('')
                else:
                    processed_lines.append(line)
            
            processed_content = '\n'.join(processed_lines)
            
            # Handle bullet points
            processed_content = re.sub(r'^[-*]\s+(.+)$', r'• \1', processed_content, flags=re.MULTILINE)
            
            # Clean up excessive newlines (max 2 consecutive)
            processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)
            processed_content = processed_content.strip()
            
            st.markdown(processed_content)
            
            # Store suggested questions in session state for later display
            if suggested_questions and len(suggested_questions) > 0 and message_id:
                # Only store for the latest message
                if message_id == len(st.session_state.chat_messages) - 1:
                    st.session_state.latest_suggestions = suggested_questions
                    st.session_state.show_suggestions = True

def show_typing_indicator():
    """Display typing indicator - now using streamlit chat_message"""
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("**AI Investment Consultant**")
        st.markdown("*🤖 sedang mengetik...*")

# ===================== STREAMLIT APP =====================
st.set_page_config(
    page_title="KPBU API Test Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimal design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stContainer {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    
    /* Chat Styling - Simplified */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #667eea;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #6a1b9a;
    }
    
    .stButton > button {
        width: 100%;
    }
    
    /* Suggested Questions Styling */
    .stChatMessage .stButton > button {
        background-color: #f0f2f6;
        color: #262730;
        border: 1px solid #d4d4d4;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease;
    }
    
    .stChatMessage .stButton > button:hover {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
        transform: translateY(-1px);
    }
    
    /* Left align suggestion buttons */
    div[data-testid="column"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Specific styling for suggestion buttons */
    div[data-testid="column"] .stButton > button[kind="secondary"] {
        text-align: left !important;
        justify-content: flex-start !important;
        background-color: #f0f2f6;
        border: 1px solid #d4d4d4;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="column"] .stButton > button[kind="secondary"]:hover {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    /* Force left alignment for all suggestion buttons */
    .suggestion-container .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* Override Streamlit's default center alignment */
    .suggestion-container button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    h1 {
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

st.title("🏗️ KPBU API Test Dashboard")
st.markdown("---")

# ===================== LAYOUT =====================
col1, col2 = st.columns([1, 1])

with col1:
    # ===================== INVESTOR PROFILE SECTION =====================
    with st.container():
        st.header("👤 Profil Investor")
        
        with st.form("investor_profile_form"):
            user_name = st.text_input("Nama Lengkap", value="Test User", key="user_name_input")
            
            toleransi_risiko = st.selectbox(
                "Toleransi Risiko",
                ["Konservatif", "Moderat", "Agresif"],
                index=1,
                help="Konservatif: Risiko rendah, Moderat: Balanced, Agresif: High return"
            )
            
            preferensi_sektor = st.multiselect(
                "Preferensi Sektor (Pilih 1-5 sektor)",
                options=list(SEKTOR_MAPPING.keys()),
                format_func=lambda x: f"{x}. {SEKTOR_MAPPING[x]}",
                default=[1, 3, 13],
                help="Pilih sektor yang diminati untuk investasi"
            )
            
            horison_investasi = st.selectbox(
                "Horison Investasi",
                ["Jangka Pendek", "Jangka Menengah", "Jangka Panjang"],
                index=2,
                help="Jangka Pendek: 1-5 tahun, Menengah: 5-15 tahun, Panjang: 15+ tahun"
            )
            
            ukuran_investasi = st.selectbox(
                "Ukuran Investasi",
                ["Kecil", "Menengah", "Besar", "Semua"],
                index=2,
                help="Kecil: <1T, Menengah: 1-10T, Besar: >10T"
            )
            
            limit_rekomendasi = st.slider("Jumlah Rekomendasi", 1, 10, 5)
            
            submit_profile = st.form_submit_button("💾 Simpan Profil & Dapatkan Rekomendasi", use_container_width=True)
        
        # Process investor profile
        if submit_profile and preferensi_sektor:
            st.session_state.investor_profile = {
                "toleransi_risiko": toleransi_risiko,
                "preferensi_sektor": preferensi_sektor,
                "horison_investasi": horison_investasi,
                "ukuran_investasi": ukuran_investasi,
                "limit": limit_rekomendasi
            }
            st.session_state.user_name = user_name
            
            with st.spinner("🔍 Mencari rekomendasi proyek terbaik..."):
                success, result = make_api_call("/match", "POST", st.session_state.investor_profile)
                
                if success:
                    st.session_state.recommendations = result
                    st.success(f"✅ Profil tersimpan! Ditemukan {len(result['recommendations'])} rekomendasi")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
    
    # Display recommendations
    if st.session_state.recommendations:
        with st.container():
            st.subheader("🎯 Rekomendasi Proyek")
            recommendations = st.session_state.recommendations
            
            st.info(f"📊 {recommendations['total_projects_analyzed']} proyek dianalisis | "
                   f"Setelah filter: {recommendations['projects_after_filter']} proyek")
            
            for rec in recommendations['recommendations'][:3]:  # Show top 3
                with st.expander(f"#{rec['ranking']} {rec['nama_proyek']} - {rec['skor_kecocokan_persen']:.1f}%"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Sektor:** {rec['sektor']}")
                        st.write(f"**Risiko:** {rec['profil_risiko']}")
                        st.write(f"**Durasi:** {rec['durasi_tahun']} tahun")
                    with col_b:
                        st.write(f"**Investasi:** {format_currency(rec['nilai_investasi_triliun'])}")
                        st.write(f"**Kecocokan:** {rec['skor_kecocokan_persen']:.1f}%")
    
    # ===================== RISK PREDICTION SECTION =====================
    st.markdown("---")
    with st.container():
        st.header("🔮 Prediksi Risiko Proyek")
        
        with st.form("risk_prediction_form"):
            st.subheader("📋 Informasi Dasar Proyek")
            nama_proyek = st.text_input("Nama Proyek", value="Jalan Tol Jakarta-Bandung")
            
            col_basic_a, col_basic_b = st.columns(2)
            with col_basic_a:
                id_sektor = st.selectbox(
                    "Sektor",
                    options=list(SEKTOR_MAPPING.keys()),
                    format_func=lambda x: f"{x}. {SEKTOR_MAPPING[x]}",
                    index=2  # Default: Jalan dan Jembatan
                )
                
                id_status = st.selectbox(
                    "Status Proyek",
                    options=list(STATUS_MAPPING.keys()),
                    format_func=lambda x: f"{x}. {STATUS_MAPPING[x]}",
                    index=2  # Default: Transaksi
                )
            
            with col_basic_b:
                durasi_konsesi = st.number_input("Durasi Konsesi (tahun)", min_value=1, max_value=50, value=25)
                nilai_investasi = st.number_input("Nilai Investasi Total (triliun Rp)", min_value=0.1, max_value=100.0, value=5.5, step=0.1)
            
            st.subheader("💰 Informasi Tokenisasi")
            col_token_a, col_token_b = st.columns(2)
            with col_token_a:
                target_dana_tokenisasi = st.number_input("Target Dana Tokenisasi (triliun Rp)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
                persentase_tokenisasi = st.number_input("Persentase Tokenisasi (%)", min_value=1.0, max_value=100.0, value=36.4, step=0.1)
            
            with col_token_b:
                jenis_token_utama = st.selectbox(
                    "Jenis Token Utama",
                    ["Infrastruktur", "Energi", "Utilitas", "Transportasi", "Kesehatan", "Pendidikan"],
                    index=0
                )
                token_risk_level_ordinal = st.selectbox(
                    "Token Risk Level (1=Rendah, 5=Tinggi)",
                    [1, 2, 3, 4, 5],
                    index=2  # Default: 3
                )
            
            st.subheader("⚙️ Karakteristik Token")
            col_char_a, col_char_b = st.columns(2)
            with col_char_a:
                token_ada_jaminan_pokok = st.checkbox("Ada Jaminan Pokok", value=False)
                token_return_berbasis_kinerja = st.checkbox("Return Berbasis Kinerja", value=True)
            
            with col_char_b:
                dok_studi_kelayakan = st.checkbox("Dokumen Studi Kelayakan", value=True)
                dok_laporan_keuangan_audit = st.checkbox("Laporan Keuangan Audit", value=True)
                dok_peringkat_kredit = st.checkbox("Peringkat Kredit", value=False)
            
            submit_prediction = st.form_submit_button("🎯 Prediksi Risiko", use_container_width=True)
        
        if submit_prediction:
            prediction_data = {
                "nama_proyek": nama_proyek,
                "id_sektor": id_sektor,
                "id_status": id_status,
                "durasi_konsesi_tahun": durasi_konsesi,
                "nilai_investasi_total_idr": int(nilai_investasi * 1000000000000),  # Convert to IDR
                "target_dana_tokenisasi_idr": int(target_dana_tokenisasi * 1000000000000),  # Convert to IDR
                "persentase_tokenisasi": persentase_tokenisasi,
                "jenis_token_utama": jenis_token_utama,
                "token_risk_level_ordinal": token_risk_level_ordinal,
                "token_ada_jaminan_pokok": token_ada_jaminan_pokok,
                "token_return_berbasis_kinerja": token_return_berbasis_kinerja,
                "dok_studi_kelayakan": dok_studi_kelayakan,
                "dok_laporan_keuangan_audit": dok_laporan_keuangan_audit,
                "dok_peringkat_kredit": dok_peringkat_kredit
            }
            
            with st.spinner("🔮 Menganalisis risiko proyek..."):
                success, result = make_api_call("/predict-risk", "POST", prediction_data)
                
                if success:
                    pred = result['prediction']
                    st.success(f"✅ **Prediksi Risiko: {pred['profil_risiko']}** (Confidence: {pred['confidence_percent']:.1f}%)")
                    
                    # Show probabilities
                    probs = result['probabilities']
                    col_prob1, col_prob2, col_prob3 = st.columns(3)
                    with col_prob1:
                        st.metric("Risiko Rendah", f"{probs['Rendah']:.1f}%")
                    with col_prob2:
                        st.metric("Risiko Menengah", f"{probs['Menengah']:.1f}%")
                    with col_prob3:
                        st.metric("Risiko Tinggi", f"{probs['Tinggi']:.1f}%")
                    
                    # Debug section
                    with st.expander("🔍 Debug - API Request & Response"):
                        st.subheader("📤 Request Sent:")
                        st.json(prediction_data)
                        st.subheader("📥 Response Received:")
                        st.json(result)
                else:
                    st.error(f"❌ Error: {result}")
                    
                    # Debug section for errors
                    with st.expander("🔍 Debug - Request Data & Error"):
                        st.subheader("📤 Request Sent:")
                        st.json(prediction_data)
                        st.subheader("❌ Error Details:")
                        st.code(result)

with col2:
    # ===================== CHATBOT SECTION =====================
    with st.container():
        st.header("💬 AI Investment Consultant")
        
        # Check if profile is filled
        profile_ready = st.session_state.investor_profile is not None
        
        if not profile_ready:
            st.warning("⚠️ Silakan isi profil investor terlebih dahulu untuk mengaktifkan chat")
            st.button("🔒 Start Chat", disabled=True, use_container_width=True)
        else:
            # Start chat button
            if not st.session_state.chat_active:
                if st.button("🚀 Start Investment Chat", use_container_width=True):
                    chat_payload = {
                        "investor_profile": st.session_state.investor_profile,
                        "user_name": st.session_state.user_name
                    }
                    
                    with st.spinner("🤖 Memulai konsultasi investasi..."):
                        success, result = make_api_call("/chat/start", "POST", chat_payload)
                        
                        if success:
                            st.session_state.chat_session_id = result['session_id']
                            st.session_state.chat_active = True
                            st.session_state.chat_messages = [
                                {
                                    "role": "assistant", 
                                    "content": result['message'],
                                    "suggested_questions": result.get('suggested_questions', [])
                                }
                            ]
                            # Show initial suggestions
                            if result.get('suggested_questions'):
                                st.session_state.latest_suggestions = result.get('suggested_questions', [])
                                st.session_state.show_suggestions = True
                            st.success("✅ Chat session dimulai!")
                            st.rerun()
                        else:
                            st.error(f"❌ Error starting chat: {result}")
            
            else:
                # Active chat interface
                st.success(f"💬 Chat aktif dengan **{st.session_state.user_name}** (Session: {st.session_state.chat_session_id[:8]}...)")
                
                # Check for pending message from suggested question click FIRST
                if st.session_state.pending_message:
                    # Add the suggested question as user message
                    st.session_state.chat_messages.append({
                        "role": "user", 
                        "content": st.session_state.pending_message
                    })
                    
                    # Set typing indicator, clear pending message and hide suggestions
                    st.session_state.agent_typing = True
                    st.session_state.pending_message = None
                    st.session_state.show_suggestions = False
                    st.rerun()
                
                # Chat messages container with auto-scroll
                chat_container = st.container(height=400)
                with chat_container:
                    # Display chat messages
                    for i, msg in enumerate(st.session_state.chat_messages):
                        display_chat_message(
                            role=msg["role"], 
                            content=msg["content"], 
                            user_name=st.session_state.user_name,
                            suggested_questions=msg.get("suggested_questions", []) if msg["role"] == "assistant" else None,
                            message_id=i
                        )
                    
                    # Show typing indicator if agent is typing
                    if st.session_state.agent_typing:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown("**AI Investment Consultant**")
                            st.markdown("🤖 *sedang mengetik...*")
                            # Add subtle visual indicator
                            st.info("💭 Memproses pertanyaan Anda dan menyiapkan analisis investasi...")
                
                # Display suggested questions outside of container for better interaction
                if st.session_state.show_suggestions and st.session_state.latest_suggestions and not st.session_state.agent_typing:
                    st.markdown("---")
                    st.markdown("💡 **Pertanyaan yang disarankan:**")
                    
                    # Add custom container with CSS class for left alignment
                    st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
                    
                    for i, question in enumerate(st.session_state.latest_suggestions):
                        # Create unique key for each button
                        button_key = f"suggestion_btn_{abs(hash(question))}_{i}_{len(st.session_state.chat_messages)}"
                        
                        if st.button(
                            f"❓ {question}", 
                            key=button_key,
                            help="Klik untuk mengirim pertanyaan ini",
                            use_container_width=True
                        ):
                            # Set the suggested question to be processed
                            st.session_state.pending_message = question
                            st.session_state.show_suggestions = False  # Hide suggestions after click
                            st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Chat input
                with st.form("chat_form", clear_on_submit=True):
                    user_message = st.text_area(
                        "💬 Pesan Anda:", 
                        height=100, 
                        placeholder="Tanya tentang investasi KPBU, analisis proyek, atau konsultasi risiko...",
                        help="Gunakan markdown untuk formatting. Contoh: **tebal**, *miring*, bullet points"
                    )
                    col_send, col_end = st.columns([3, 1])
                    
                    with col_send:
                        send_message = st.form_submit_button("📤 Kirim", use_container_width=True)
                    with col_end:
                        end_chat = st.form_submit_button("🔚 End", use_container_width=True)
                
                if send_message and user_message.strip():
                    # Add user message immediately
                    st.session_state.chat_messages.append({"role": "user", "content": user_message})
                    
                    # Hide suggestions when user sends new message
                    st.session_state.show_suggestions = False
                    
                    # Show typing indicator before API call
                    st.session_state.agent_typing = True
                    st.rerun()  # Rerun to show typing indicator
                
                # Handle API call after rerun (when typing indicator is shown)
                if st.session_state.agent_typing and st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
                    # Send to API
                    last_user_message = st.session_state.chat_messages[-1]["content"]
                    chat_payload = {
                        "session_id": st.session_state.chat_session_id,
                        "message": last_user_message
                    }
                    
                    try:
                        success, result = make_api_call("/chat/message", "POST", chat_payload)
                        
                        if success:
                            st.session_state.chat_messages.append({
                                "role": "assistant", 
                                "content": result['message'],
                                "suggested_questions": result.get('suggested_questions', [])
                            })
                            # Show suggestions for the new message
                            if result.get('suggested_questions'):
                                st.session_state.latest_suggestions = result.get('suggested_questions', [])
                                st.session_state.show_suggestions = True
                        else:
                            st.session_state.chat_messages.append({
                                "role": "assistant", 
                                "content": f"❌ **Terjadi kesalahan teknis:** {result}\n\nSilakan coba lagi atau hubungi tim support.",
                                "suggested_questions": []
                            })
                    except Exception as e:
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": f"❌ **Koneksi bermasalah:** {str(e)}\n\nPastikan API server berjalan dan coba lagi.",
                            "suggested_questions": []
                        })
                    finally:
                        st.session_state.agent_typing = False
                        st.rerun()
                
                # Remove the separate typing handling as it's now integrated above
                
                if end_chat:
                    st.session_state.chat_active = False
                    st.session_state.chat_session_id = None
                    st.session_state.chat_messages = []
                    st.session_state.agent_typing = False
                    st.session_state.pending_message = None
                    st.session_state.show_suggestions = False
                    st.session_state.latest_suggestions = []
                    st.info("💤 Chat session ended. Terima kasih telah menggunakan layanan konsultasi!")
                    st.rerun()

# ===================== FOOTER =====================
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    if st.button("📊 Dataset Stats"):
        with st.spinner("📈 Loading dataset statistics..."):
            success, result = make_api_call("/dataset/stats")
            if success:
                st.json(result)
            else:
                st.error(f"Error: {result}")

with col_footer2:
    if st.button("🔧 Test Gemini API"):
        with st.spinner("🧠 Testing Gemini connection..."):
            success, result = make_api_call("/test-gemini")
            if success:
                st.success("✅ Gemini API working!")
                st.json(result)
            else:
                st.error(f"❌ Gemini API Error: {result}")

with col_footer3:
    if st.button("🧹 Cleanup Sessions"):
        with st.spinner("🗑️ Cleaning up old sessions..."):
            success, result = make_api_call("/chat/cleanup", "DELETE")
            if success:
                st.success(f"✅ {result['message']}")
            else:
                st.error(f"Error: {result}")

# API Connection Status
try:
    success, _ = make_api_call("/health")
    if success:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Disconnected")
except:
    st.sidebar.error("🔴 API Not Available")

st.sidebar.markdown(f"**Base URL:** {API_BASE_URL}")
st.sidebar.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")