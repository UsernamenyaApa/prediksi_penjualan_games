import pickle
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu
import base64

# Set page configuration
st.set_page_config(page_title="Vizualize Dashboard", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
        .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load images
globe_img = get_img_as_base64("img/globe.png")

# Panggil fungsi dengan path gambar Anda
set_background('img/background.png')

# Add custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: rgba(30, 60, 114, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    # Add logo
    st.markdown(f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: none;
            }}
            .sidebar-logo {{
                margin: 25px auto 35px auto;
                width: 180px;
                display: block;
            }}
        </style>
        <img src="data:image/png;base64,{get_img_as_base64('img/logo.png')}" class="sidebar-logo">
    """, unsafe_allow_html=True)
    
    # Existing sidebar content
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "K-Means", "Linear Regression", "About"],
        icons=["grid", "diagram-3", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Main content
if selected == "Dashboard":
    st.markdown(
        f"""
        <style>
        .main-container {{
            text-align: center;
            color: white;
        }}
        .solution-text {{
            background: white;
            color: #FF5733;
            padding: 8px 20px;
            border-radius: 25px;
            display: inline-block;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .title-container {{
            font-size: 4.5rem;
            font-weight: bold;
            line-height: 1.2;
        }}
        .highlight {{
            background: #FF5733;
            padding: 5px 40px;
            border-radius: 100px;
            display: inline-block;
            margin: 0 10px;
        }}
        .subtitle {{
            font-size: 1.2rem;
            color: rgba(255,255,255,0.8);
        }}
        .globe-container {{
            margin-top: -20px;
            position: relative;
        }}
        .globe-image {{
            content: url("data:image/png;base64,{globe_img}");
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 20%, rgba(0,0,0,0) 100%);
            mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 10%, rgba(0,0,0,0) 100%);
        }}
        </style>
        <div class="main-container">
            <div class="solution-text">SOLUTION FOR DATA VISUALIZATION</div>
            <div class="title-container">
                Explore Your <span class="highlight">Data</span><br>
                In Your Way
            </div>
            <div class="subtitle">
                Gain unparalleled insights into your data with our robust analytics suite.
            </div>
            <div class="globe-container">
                <div class="globe-image"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Updated Data section with styled header and table
    st.markdown(
        """
        <style>
        .data-header {
            text-align: center;
            margin: 30px 0 40px 0;
            position: relative;
            z-index: 2;
        }
        .data-title {
            font-size: 3rem;
            font-weight: bold;
            line-height: 1.2;
        }
        .data-highlight {
            background: #FF5733;
            padding: 5px 40px;
            border-radius: 100px;
            display: inline-block;
            margin: 0 10px;
        }
        .data-subtitle {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.8);
            margin-top: 10px;
        }
        .dataframe {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            position: relative;
            z-index: 2;
            padding-bottom: 50px;
        }
        .dataframe th {
            background-color: rgba(255, 87, 51, 0.3);
            color: white !important;
            font-weight: bold;
            text-align: center !important;
        }
        .dataframe td {
            color: white !important;
            text-align: center !important;
        }
        .globe-container {
            z-index: 1;
        }
        </style>
        <div class="data-header">
            <div class="data-title">
                Video Games <span class="data-highlight">Dataset</span>
            </div>
            <div class="data-subtitle">
                Explore comprehensive video game sales data across different regions
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = pd.read_csv('vgsales.csv')
    
    # Add styling to the dataframe
    styled_df = df.style.set_properties(**{
        'background-color': 'rgba(255, 255, 255, 0.1)',
        'color': 'white',
        'border-color': 'transparent'
    })

    st.write(styled_df)

    # Menambahkan header dan penjelasan
    st.markdown(
        """
        <style>
        .explanation-header {
            text-align: center;
            margin: 40px 0 20px 0;
            position: relative;
            z-index: 2;
        }
        .explanation-header-title {
            font-size: 3rem;
            font-weight: bold;
            line-height: 1.2;
        }
        .explanation-highlight {
            background: #FF5733;
            padding: 5px 40px;
            border-radius: 100px;
            display: inline-block;
            margin: 0 10px;
        }
        .explanation-subtitle {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.8);
            margin-top: 10px;
            margin-bottom: 30px;
        }
        .explanation-container {
            display: flex;
            gap: 30px;
            margin-top: 20px;
            margin-bottom: 40px;
        }
        .explanation-card {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        .explanation-title {
            color: #FF5733;
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .explanation-text {
            color: white;
            font-size: 1rem;
            line-height: 1.6;
        }
        </style>
        <div class="explanation-header">
            <div class="explanation-header-title">
                Machine Learning <span class="explanation-highlight">Methods</span>
            </div>
            <div class="explanation-subtitle">
                Understanding the algorithms used in this analysis
            </div>
        </div>
        <div class="explanation-container">
            <div class="explanation-card">
                <div class="explanation-title">K-Means Clustering</div>
                <div class="explanation-text">
                    K-Means clustering adalah algoritma yang mengelompokkan data ke dalam K cluster berdasarkan kesamaan karakteristik. 
                    Dalam konteks dataset video game ini, K-Means digunakan untuk mengidentifikasi pola penjualan yang serupa di berbagai wilayah.
                    Algoritma ini membantu mengungkap segmen pasar dan strategi distribusi yang potensial berdasarkan performa penjualan regional.
                </div>
            </div>
            <div class="explanation-card">
                <div class="explanation-title">Linear Regression</div>
                <div class="explanation-text">
                    Linear Regression adalah metode statistik untuk memprediksi nilai berdasarkan hubungan linear antar variabel. 
                    Dalam analisis video game, model ini digunakan untuk memperkirakan penjualan global berdasarkan berbagai faktor seperti 
                    platform, genre, dan tahun rilis. Hal ini membantu publisher dalam memperkirakan potensi penjualan game baru.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif selected == "K-Means":
    st.title("K-Means")

    df = pd.read_csv('vgsales.csv')

    df = df.dropna(subset=['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

    # Drop data outlier
    numerical_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # K-Means
    cluster_data = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)

    # Finding elbow
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), inertia, 'o-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal Number of Clusters')
    ax.set_xticks(range(1, 11))
    ax.grid(True)

    st.pyplot(fig)

    elbow_df = pd.DataFrame({'Number of Clusters (K)': range(1, 11), 'Inertia': inertia})
    st.write(elbow_df)

    st.write('Select Number of Clusters:')
    clust = st.slider('Choose number of clusters:', 2, 10, 4, 1)

    def k_means(n_clust):
        kmeans = KMeans(n_clusters=n_clust, random_state=42)
        cluster = kmeans.fit_predict(scaled_features)
        cluster_data['Cluster'] = cluster

        # Calculate cluster centers and transform back to original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_definitions = pd.DataFrame(
            cluster_centers,
            columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
            index=[f'Cluster {i}' for i in range(len(cluster_centers))]
        )

        st.write("Cluster Centers:")
        st.write(cluster_definitions)

        # Cluster statistics
        cluster_stats = cluster_data.groupby('Cluster').mean()
        st.write("\nCluster Statistics:")
        st.write(cluster_stats)

        # PCA for visualization
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_data['Cluster'], cmap='viridis', s=50)
        ax.set_title('K-Means Clustering (with PCA)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')

        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        st.header('Cluster Plot')
        st.pyplot(fig)

        st.write(cluster_data)

    k_means(clust)

elif selected == "Linear Regression":
    st.title("Linear Regression")

    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('X_train_columns.pkl', 'rb') as columns_file:
        X_train_columns = pickle.load(columns_file)

    # List genre dan platform sesuai dataset
    genres = [
        "Genre_Adventure", "Genre_Fighting", "Genre_Misc", "Genre_Platform",
        "Genre_Puzzle", "Genre_Racing", "Genre_Role-Playing", "Genre_Shooter",
        "Genre_Simulation", "Genre_Sports", "Genre_Strategy"
    ]

    platforms = [
        "Platform_3DO", "Platform_3DS", "Platform_DC", "Platform_DS",
        "Platform_GB", "Platform_GBA", "Platform_GC", "Platform_GEN",
        "Platform_GG", "Platform_N64", "Platform_NES", "Platform_NG",
        "Platform_PC", "Platform_PCFX", "Platform_PS", "Platform_PS2",
        "Platform_PS3", "Platform_PS4", "Platform_PSP", "Platform_PSV",
        "Platform_SAT", "Platform_SCD", "Platform_SNES", "Platform_TG16",
        "Platform_WS", "Platform_Wii", "Platform_WiiU", "Platform_X360",
        "Platform_XB", "Platform_XOne"
    ]

    st.title("Prediksi Penjualan Game Global")

    # Input dari pengguna
    st.header("Input Data Game")
    name = st.text_input("Nama Game")
    selected_genres = st.multiselect("Genre Game", [g.replace("Genre_", "") for g in genres])
    selected_platform = st.selectbox("Platform Game", [p.replace("Platform_", "") for p in platforms])

    # Proses input menjadi format yang sesuai dengan model
    input_data = {}

    # Encode genres (one-hot encoding)
    for genre in genres:
        genre_name = genre.replace("Genre_", "")
        input_data[genre] = 1 if genre_name in selected_genres else 0

    # Encode platform (one-hot encoding)
    for platform in platforms:
        input_data[platform] = 1 if platform.replace("Platform_", "") == selected_platform else 0

    # Pastikan semua kolom yang diperlukan ada
    missing_cols = set(X_train_columns) - set(input_data.keys())
    for col in missing_cols:
        input_data[col] = 0

    # Buat dataframe input
    input_encoded = pd.DataFrame([input_data])

    # Tampilkan input yang telah di-encode
    st.subheader("Data Input yang Terencode")
    st.write(input_encoded)

    # Prediksi
    if st.button("Prediksi Penjualan"):
        prediction = model.predict(input_encoded)[0]
        st.success(f"Prediksi Penjualan Global: {prediction[0]:.2f} juta unit")
        
elif selected == "About":
    st.markdown("""
        <style>
        .about-header {
            text-align: center;
            color: white;
            padding: 2rem 0;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .team-container {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
        }
        .team-card {
            flex: 0 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 2rem;
            width: 250px;
            height: 350px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .team-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .member-image {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            margin: 0 auto 1rem auto;
            object-fit: cover;
            border: 3px solid #FF5733;
        }
        .member-name {
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .member-nim {
            color: #FF5733;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        <div class="about-header">Our Team</div>
        <div class="team-container">
    """, unsafe_allow_html=True)

    members = [
        {"name": "Kamalia Sekar Ramadhani", "nim": "1202220075", "image": "img/lia.png"},
        {"name": "Fitrotin Nadzilah", "nim": "1202223149", "image": "img/zila.png"},
        {"name": "Muhammad Raya Ramadhan", "nim": "1202223050", "image": "img/raya.png"},
        {"name": "Ignasius Jonathan Putra Perdana", "nim": "1202223361", "image": "img/johan.png"},
    ]


    cols = st.columns(4)

    for i, member in enumerate(members):
        with cols[i]:
            st.markdown(f"""
                <div class="team-card">
                    <img src="data:image/png;base64,{get_img_as_base64(member['image'])}" class="member-image">
                    <div class="member-name">{member['name']}</div>
                    <div class="member-nim">{member['nim']}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
