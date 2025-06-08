import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# --- Konfigurasi Halaman Dashboard ---
st.set_page_config(
    page_title="AI Media Intelligence Dashboard",
    page_icon="🌿",
    layout="wide"
)

# --- Fungsi untuk Memuat Data ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Memuat data
df_original = load_data('Spirifi_cleaned.csv')

# --- SIDEBAR ---
st.sidebar.title("🌿 Pengaturan")

# --- KELOMPOK FILTER DI DALAM EXPANDER ---
with st.sidebar.expander("🔍 Filter Data", expanded=True):
    selected_platforms = st.multiselect('Pilih Platform:', options=df_original['Platform'].unique(), default=df_original['Platform'].unique())
    all_influencers = df_original['Influencer'].unique()
    selected_influencers = st.multiselect('Pilih Influencer:', options=all_influencers, default=all_influencers)
    min_date = df_original['Date'].min().date()
    max_date = df_original['Date'].max().date()
    selected_date_range = st.date_input('Pilih Rentang Tanggal:', value=(min_date, max_date), min_value=min_date, max_value=max_date)

# --- PENJELASAN INPUT API KEY YANG LEBIH BAIK ---
st.sidebar.subheader("🔑 Konfigurasi AI Assistant")
api_key = st.sidebar.text_input("Masukkan Google AI API Key Anda:", type="password", help="API Key diperlukan untuk menjalankan fitur analisis otomatis oleh AI.")
st.sidebar.markdown(
    """
    Dapatkan API Key gratis Anda dari [Google AI Studio](https://aistudio.google.com/app/apikey).
    """
)

# --- Logika Filter Data ---
start_date, end_date = selected_date_range
df_filtered = df_original[
    (df_original['Date'].dt.date >= start_date) &
    (df_original['Date'].dt.date <= end_date) &
    df_original['Platform'].isin(selected_platforms) &
    df_original['Influencer'].isin(selected_influencers)
]

# --- Judul Dashboard ---
st.title('📊 Spirifi AI-Powered Media Intelligence')
st.markdown("Dashboard ini menganalisis performa media sosial brand Spirifi. Gunakan filter di sidebar untuk menjelajahi data.")

# --- Tampilan Metrik Utama (KPIs) ---
st.subheader('Ringkasan Performa (KPIs)')
if not df_filtered.empty:
    total_engagement = int(df_filtered['Engagements'].sum())
    total_posts = len(df_filtered)
    avg_engagement_per_post = int(total_engagement / total_posts) if total_posts > 0 else 0
else:
    total_engagement, total_posts, avg_engagement_per_post = 0, 0, 0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(label="Total Posts", value=total_posts)
kpi2.metric(label="Total Engagements", value=f"{total_engagement:,}")
kpi3.metric(label="Avg. Engagement / Post", value=f"{avg_engagement_per_post:,}")
st.markdown("---")

# --- VISUALISASI DATA DENGAN TEMA HIJAU LEMBUT ---
st.subheader("Visualisasi Data Interaktif")

if not df_filtered.empty:
    # Mengatur palet warna hijau
    green_palette = sns.color_palette("YlGn", n_colors=10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Total Engagement per Platform")
        platform_engagement = df_filtered.groupby('Platform')['Engagements'].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        sns.barplot(x=platform_engagement.values, y=platform_engagement.index, palette=green_palette, ax=ax1, orient='h')
        ax1.set_xlabel('Total Engagements')
        ax1.set_ylabel('Platform')
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Distribusi Sentimen Postingan")
        sentiment_counts = df_filtered['Sentiment'].value_counts()
        # Warna hijau lembut untuk positif
        custom_colors = {'Positive': '#8FBC8F', 'Neutral': '#A9A9A9', 'Negative': '#F08080'}
        pie_colors = [custom_colors.get(s, 'gray') for s in sentiment_counts.index]
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=pie_colors)
        ax2.axis('equal')
        st.pyplot(fig2)

    st.markdown("#### Top 10 Influencer by Engagement")
    influencer_engagement = df_filtered.groupby('Influencer')['Engagements'].sum().sort_values(ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=influencer_engagement.values, y=influencer_engagement.index, palette=green_palette, orient='h', ax=ax3)
    ax3.set_xlabel('Total Engagements')
    ax3.set_ylabel('Influencer')
    st.pyplot(fig3)

else:
    st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih untuk ditampilkan pada grafik.")

st.markdown("---")

# --- BAGIAN AI ANALYST ASSISTANT ---
st.subheader("🤖 AI Analyst Assistant")
st.markdown("Dapatkan ringkasan analisis otomatis dari data yang telah Anda filter di atas.")

if st.button("Generate Insight"):
    if not api_key:
        st.error("Mohon masukkan Google AI API Key Anda di sidebar untuk menggunakan fitur ini.")
    elif df_filtered.empty:
        st.warning("Tidak ada data untuk dianalisis. Silakan sesuaikan filter Anda.")
    else:
        # ... (kode AI tetap sama) ...
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            data_summary = f"""Data Summary: Total Posts={total_posts}, Total Engagement={total_engagement}. Data Description: {df_filtered.describe(include='all').to_string()}"""
            prompt = f"""Anda adalah seorang Media Analyst profesional. Berdasarkan ringkasan data berikut, berikan insight utama dalam format poin-poin (bullet points). Fokus pada: 1. Performa keseluruhan. 2. Platform paling dominan. 3. Influencer paling efektif. 4. Saran strategis. Ringkasan Data: {data_summary}"""
            
            with st.spinner('AI (Gemini) sedang menganalisis data...'):
                response = model.generate_content(prompt)
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menghubungi Google AI: {e}")
