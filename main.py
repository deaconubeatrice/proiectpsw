import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

plt.style.use('ggplot')

st.set_page_config(page_title="Analiza activitatii unei companii de retail", layout="wide")
st.title("Analiza activitatii unei companii de retail")

# Incarcarea fisierului
uploaded_file = st.file_uploader("Incarcati fisierul CSV cu datele de retail", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 2.1 Incarcarea si explorarea initiala a datelor
    st.header("2.1. Incarcarea si explorarea initiala a datelor")

    st.subheader("Primele 5 inregistrari")
    st.dataframe(df.head())

    st.subheader("Dimensiunea setului de date")
    st.write(f"Numar linii: {df.shape[0]}, Numar coloane: {df.shape[1]}")

    st.subheader("Denumirile coloanelor")
    st.write(df.columns.tolist())

    # 2.2 Preprocesarea datelor si conversia variabilelor temporale
    st.header("2.2. Preprocesarea datelor si conversia variabilelor temporale")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    st.write("Primele 5 inregistrari dupa extragerea anului si lunii:")
    st.dataframe(df[['Date', 'Year', 'Month']].head())

    # 2.3 Tratarea valorilor lipsa si a valorilor extreme
    st.header("2.3. Tratarea valorilor lipsa si a valorilor extreme")

    st.subheader("Valori lipsa pe coloane")
    valori_lipsa = df.isnull().sum()
    st.dataframe(valori_lipsa)

    # Tratarea valorilor lipsa (in cazul in care exista)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    st.subheader("Boxplot pentru identificarea outlierilor")
    coloana_box = st.selectbox("Alegeti coloana numerica:",
                               ["Age", "Quantity", "Price per Unit", "Total Amount"])

    fig_box, ax_box = plt.subplots(figsize=(6, 3))
    ax_box.boxplot(df[coloana_box], vert=False)
    ax_box.set_xlabel(coloana_box)
    ax_box.set_title(f"Boxplot pentru {coloana_box}")
    st.pyplot(fig_box, use_container_width=False)

    # Eliminarea outlierilor cu z-score
    df_numeric = df.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(df_numeric))
    df_clean = df[(z_scores < 3).all(axis=1)]

    st.write(f"Numar inregistrari initial: {df.shape[0]}")
    st.write(f"Numar inregistrari dupa eliminarea outlierilor: {df_clean.shape[0]}")

    # 2.4 Codificarea variabilelor categorice
    st.header("2.4. Codificarea variabilelor categorice")

    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in ['Gender', 'Product Category']:
        df_encoded[col + '_encoded'] = label_encoder.fit_transform(df_encoded[col].astype(str))

    st.write("Primele 5 inregistrari dupa codificare:")
    st.dataframe(df_encoded[['Gender', 'Gender_encoded', 'Product Category', 'Product Category_encoded']].head())

    # 2.5 Scalarea variabilelor numerice
    st.header("2.5. Scalarea variabilelor numerice")

    coloane_numerice = ["Age", "Quantity", "Price per Unit", "Total Amount"]
    scaler = MinMaxScaler()

    df_scaled = df.copy()
    df_scaled[coloane_numerice] = scaler.fit_transform(df[coloane_numerice])

    st.write("Inainte de scalare:")
    st.dataframe(df[coloane_numerice].head())

    st.write("Dupa scalare (valori intre 0 si 1):")
    st.dataframe(df_scaled[coloane_numerice].head())

    # 2.6 Analiza statistica si agregarea datelor
    st.header("2.6. Analiza statistica si agregarea datelor")

    st.subheader("Vanzari totale pe categorie de produs")
    vanzari_categorie = df.groupby("Product Category")["Total Amount"].sum().reset_index()
    st.dataframe(vanzari_categorie)

    st.subheader("Valoarea medie a tranzactiilor pe gen")
    medie_gen = df.groupby("Gender")["Total Amount"].mean().reset_index()
    st.dataframe(medie_gen)

    st.subheader("Tabel pivot: vanzari pe categorie si gen")
    pivot_vanzari = pd.pivot_table(df,
                                   values='Total Amount',
                                   index='Product Category',
                                   columns='Gender',
                                   aggfunc='sum').reset_index()
    st.dataframe(pivot_vanzari)

    # 2.7 Vizualizarea datelor prin grafice
    st.header("2.7. Vizualizarea datelor prin grafice")

    st.subheader("Vanzari totale pe categorie - Bar chart")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=vanzari_categorie, x='Product Category', y='Total Amount',
                ax=ax1, palette='Blues_d')
    ax1.set_title("Vanzari totale pe categorie de produs")
    st.pyplot(fig1, use_container_width=False)

    st.subheader("Distributia varstei clientilor - Histograma")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(df['Age'], bins=15, kde=True, ax=ax2, color='steelblue')
    ax2.set_title("Distributia varstei clientilor")
    st.pyplot(fig2, use_container_width=False)

    # 2.8 Analiza valorii medii a tranzactiilor pe intervale de varsta
    st.header("2.8. Analiza valorii medii a tranzactiilor pe intervale de varsta")

    df['Age Group'] = pd.cut(df['Age'],
                             bins=[0, 25, 35, 50, 100],
                             labels=["18-25", "26-35", "36-50", "50+"])

    medie_varsta = df.groupby("Age Group", observed=True)["Total Amount"].mean().reset_index()
    st.dataframe(medie_varsta)

    fig_age, ax_age = plt.subplots(figsize=(7, 4))
    sns.barplot(data=medie_varsta, x='Age Group', y='Total Amount',
                ax=ax_age, palette='Greens_d')
    ax_age.set_title("Valoarea medie a tranzactiilor pe grupe de varsta")
    st.pyplot(fig_age, use_container_width=False)

    # 2.9 Segmentarea clientilor utilizand K-Means
    st.header("2.9. Segmentarea clientilor utilizand K-Means")

    # Folosim mai multe variabile, scalate
    variabile_cluster = ['Age', 'Quantity', 'Total Amount']
    X_cluster = df[variabile_cluster].copy()

    scaler_cluster = MinMaxScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

    st.subheader("Numar clienti pe clustere")
    segmente = df['Cluster'].value_counts().reset_index()
    segmente.columns = ['Cluster', 'Numar clienti']
    st.dataframe(segmente)

    st.subheader("Valoarea medie a tranzactiilor pe cluster")
    medie_cluster = df.groupby("Cluster")["Total Amount"].mean().reset_index()
    st.dataframe(medie_cluster)

    st.subheader("Vizualizare clustere - cantitate vs valoare")
    fig_scatter_cluster, ax_sc = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x='Quantity', y='Total Amount', hue='Cluster', palette='Set1', ax=ax_sc)
    ax_sc.set_title("Clustere K-Means: cantitate vs valoare tranzactie")
    st.pyplot(fig_scatter_cluster, use_container_width=False)

    # 2.10 Regresia liniara multipla cu statsmodels
    st.header("2.10. Regresia liniara multipla cu statsmodels")

    X_reg = df[['Age', 'Quantity', 'Price per Unit']]
    y_reg = df['Total Amount']
    X_reg = sm.add_constant(X_reg)

    model_ols = sm.OLS(y_reg, X_reg).fit()

    st.subheader("Coeficientii modelului")
    coef_df = pd.DataFrame({
        'Variabila': model_ols.params.index,
        'Coeficient': model_ols.params.values.round(2)
    })
    st.dataframe(coef_df)

    st.subheader("Performanta modelului")
    st.write(f"R² (cat din variatie explica modelul): {model_ols.rsquared:.3f}")

else:
    st.info("Fisierul retail_sales_dataset.csv nu este incarcat.")