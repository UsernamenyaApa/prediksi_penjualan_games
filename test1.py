import pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Vizualize Dashboard", layout="wide")

# Sidebar menu
with st.sidebar:
    st.title("Vizualize")
    selected = option_menu(
        menu_title=None,  # Required
        options=["Dashboard", "K-Means", "Linear Regression", "About"],  # Required
        icons=["grid", "diagram-3", "graph-up", "info-circle"],  # Optional
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
    )

# Main content
if selected == "Dashboard":
    st.markdown(
        """
        <style>
        .main-container {
            text-align: center;
            color: white;
            background: linear-gradient(145deg, #1e1e1e, #2b2b2b);
            padding: 50px 0;
            border-radius: 10px;
        }
        .main-title {
            font-size: 3rem;
            font-weight: bold;
        }
        .main-subtitle {
            font-size: 1.5rem;
            margin-top: 20px;
        }
        .highlight {
            color: #f39c12;
        }
        .globe-container {
            margin-top: 50px;
        }
        .footer {
            margin-top: 50px;
            font-size: 0.9rem;
            color: #a0a0a0;
        }
        </style>
        <div class="main-container">
            <div class="main-title">Explore Your <span class="highlight">Data</span> In Your Way</div>
            <div class="main-subtitle">Gain unparalleled insights into your data with our robust analytics suite.</div>
            <div class="globe-container">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Globe_icon.svg/512px-Globe_icon.svg.png" alt="Globe" width="300">
            </div>
        </div>
        <div class="footer">SOLUTION FOR DATA VISUALIZATION</div>
        """,
        unsafe_allow_html=True,
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
    # Standarisasi
    cluster_data = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)

    # Mencari elbow
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data pada axes
    ax.plot(range(1, 11), inertia, 'o-')
    ax.set_xlabel('Jumlah Kluster (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Metode Elbow untuk Menentukan Jumlah Kluster Optimal')
    ax.set_xticks(range(1, 11))
    ax.grid(True)

    st.pyplot(fig)

    elbow_df = pd.DataFrame({'Jumlah Kluster (K)': range(1, 11), 'Inertia': inertia})
    st.write(elbow_df)

    st.write('Nilai Jumlah K / Kluster')
    clust = st.slider('Pilih jumlah kluster: ', 2, 10, 4, 1)

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

        st.write("Pusat Cluster:")
        st.write(cluster_definitions)

        # Cluster statistics (optional)
        cluster_stats = cluster_data.groupby('Cluster').mean()
        st.write("\nStatistik Cluster:")
        st.write(cluster_stats)

        # PCA for visualization (optional)
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)

        # Create a single subplot using plt.subplots()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot with color based on cluster and PCA components
        scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_data['Cluster'], cmap='viridis', s=50)
        ax.set_title('K-Means Clustering (dengan PCA)')
        ax.set_xlabel('Komponen Utama 1')
        ax.set_ylabel('Komponen Utama 2')

        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        # Display the plot in Streamlit
        st.header('Cluster Plot')
        st.pyplot(fig)

        # Display the cluster data (optional)
        st.write(cluster_data)

    k_means(clust)

elif selected == "Linear Regression":
    st.title("Linear Regression")

    # Load model dan kolom X_train (dengan penanganan error)
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('X_train_columns.pkl', 'rb') as f:
            X_train_columns = pickle.load(f)
    except FileNotFoundError:
        st.error("File model atau kolom tidak ditemukan. Pastikan 'best_model.pkl' dan 'X_train_columns.pkl' ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau kolom: {e}")
        st.stop()

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

    # Input Tahun
    year = st.slider("Pilih Tahun Rilis:", min_value=2000, max_value=2016, value=2010)

    # Create input_data with the necessary columns
    input_data = pd.DataFrame({
        'Year': [year],
        'Genre': ['Unknown'],  # Placeholder value
        'Platform': ['Unknown']  # Placeholder value
    })

    # Input untuk Genre (Bisa lebih dari satu)
    st.subheader("Pilih Genre (Bisa lebih dari satu):")
    genre_cols = st.columns(3)
    selected_genres = []
    for i, genre in enumerate(genres):
        with genre_cols[i % 3]:
            if st.checkbox(genre.replace("Genre_", ""), value=False):
                selected_genres.append(genre.replace("Genre_", ""))

    # Input untuk Platform (Hanya bisa memilih satu)
    st.subheader("Pilih Platform (Hanya satu pilihan):")
    platform_cols = st.columns(4)
    selected_platform = None

    for i, platform in enumerate(platforms):
        with platform_cols[i % 4]:
            if st.checkbox(platform.replace("Platform_", ""), value=False, key=platform):
                selected_platform = platform.replace("Platform_", "")

    # Update input_data with selected values
    if selected_genres:
        input_data['Genre'] = selected_genres[0] #ambil genre pertama saja
    else:
        input_data['Genre'] = 'Unknown'

    if selected_platform:
        input_data['Platform'] = selected_platform
    else:
        input_data['Platform'] = 'Unknown'

    st.subheader("Input yang Anda Pilih:")
    st.write(input_data)

    if st.button('Estimasi Global Sales'):
        input_encoded = pd.get_dummies(input_data, columns=['Genre', 'Platform'], drop_first=True)
        missing_cols = set(X_train_columns) - set(input_encoded.columns)
        for c in missing_cols:
            input_encoded[c] = 0
        input_encoded = input_encoded[X_train_columns]

        if input_encoded.ndim == 1:
            input_encoded = input_encoded.reshape(1, -1)

        try:
            predict = model.predict(input_encoded)
            st.write('Prediksi Global Sales dari data adalah: ', predict[0])
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

elif selected == "About":
    st.header('Anggota Kelompok')

    anggota = [
        {"name": "Kamalia Sekar Ramadhani", "nim": "1202220075", "image": "img/lia.jpg"},
        {"name": "Fitrotin Nadzilah", "nim": "1202223149", "image": "img/zila.jpg"},
        {"name": "Muhammad Raya Ramadhan", "nim": "1202223050", "image": "img/raya.jpg"},
        {"name": "Ignasius Jonathan Putra Perdana", "nim": "1202223361", "image": "img/johan.png"},
    ]

    cols = st.columns(len(anggota))

    for i, member in enumerate(anggota):
        with cols[i]:
            st.markdown(f"**{member['name']}**")
            st.write(f"NIM: {member['nim']}")
            st.image(member["image"], use_container_width=True, caption=f"{member['name']}")
