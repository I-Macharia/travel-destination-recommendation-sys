import pandas as pd
import numpy as np
import pickle
import os
import glob
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ====================== CONFIG ======================
st.set_page_config(page_title="Africura", layout="wide")

# Background Images
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2019/09/20/23/47/sand-4492751_640.jpg");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_resource
def load_data():
    try:
        with open('Data/clean_df.pkl', 'rb') as f:
            clean_df = pickle.load(f)
        with open('Data/.tfidf_matrix2.pkl', 'rb') as f:
            tfidfv_matrix2 = pickle.load(f)
        with open('Data/.cosine_sim2.pkl', 'rb') as f:
            cosine_sim2 = pickle.load(f)
        with open('Data/.cosine_similarities.pkl', 'rb') as f:
            cosine_similarities = pickle.load(f)
        with open('Data/.indices.pkl', 'rb') as f:
            indices = pickle.load(f)
        return clean_df, cosine_sim2, cosine_similarities, indices
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

clean_df, cosine_sim2, cosine_similarities, indices = load_data()

# ====================== RECOMMENDATION ENGINE ======================
class RecommendationEngine:
    def __init__(self, clean_df, cosine_sim2, cosine_similarities):
        self.clean_df = clean_df
        self.cosine_sim2 = cosine_sim2
        self.cosine_similarities = cosine_similarities

    def recommend_place(self, name):
        indices_dict = {title: idx for idx, title in enumerate(self.clean_df['name'])}
        if name not in indices_dict:
            return pd.DataFrame({"Error": [f"Place '{name}' not found"]})
        
        idx = indices_dict[name]
        sim_scores = list(enumerate(self.cosine_sim2[idx] @ self.cosine_similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        
        rec_indices = [i[0] for i in sim_scores]
        return self.clean_df.iloc[rec_indices][
            ['name', 'country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ].astype({'LowerPrice': int, 'UpperPrice': int})

    def recommend_by_amenities(self, selected_amenities):
        # Simple filter + top recommendations
        mask = self.clean_df['combined_amenities'].str.contains('|'.join(selected_amenities), na=False)
        return self.clean_df[mask].head(10)[
            ['name', 'country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ]

    def recommend_subcategory(self, subcategory):
        indices_dict = {title: idx for idx, title in enumerate(self.clean_df['subcategories'])}
        if subcategory not in indices_dict:
            return pd.DataFrame({"Error": [f"Subcategory '{subcategory}' not found"]})
        
        idx = indices_dict[subcategory]
        sim_scores = list(enumerate(self.cosine_sim2[idx] @ self.cosine_similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        
        rec_indices = [i[0] for i in sim_scores]
        return self.clean_df.iloc[rec_indices][
            ['name', 'country', 'RankingType', 'subcategories', 'LowerPrice', 'UpperPrice']
        ]

    def recommend_country(self, country):
        return self.clean_df[self.clean_df['country'].str.lower() == country.lower()].head(10)

    def recommend_attraction(self, rating_threshold):
        return self.clean_df[self.clean_df['rating'] < rating_threshold][
            ['name', 'country', 'LowerPrice', 'UpperPrice', 'amenities', 'type']
        ].head(15)

# Initialize Engine
recommender = RecommendationEngine(clean_df, cosine_sim2, cosine_similarities)

# ====================== MAIN APP ======================
def main():
    st.sidebar.title("🌍 Africura")
    st.sidebar.subheader("African Travel Recommender")
    
    menu = ['About', 'Recomenders', 'Gallery']
    selection = st.sidebar.selectbox("Menu", menu)

    if selection == "About":
        st.markdown("<h1 style='color: #1E3A8A;'>Welcome to Africura</h1>", unsafe_allow_html=True)
        st.markdown("##### Your one-stop recommendation engine for unforgettable African destinations.")
        
        # Interactive Map
        fig = go.Figure(go.Scattermapbox(
            lat=clean_df['latitude'],
            lon=clean_df['longitude'],
            mode='markers',
            marker=dict(size=6, color=clean_df['rating'], colorscale='Viridis', opacity=0.8),
            text=clean_df['name'],
            hoverinfo='text'
        ))
        fig.update_layout(
            mapbox_style="stamen-terrain",
            mapbox=dict(center=dict(lat=8, lon=20), zoom=2),
            height=600,
            title="Explore African Destinations"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selection == "Recomenders":
        st.markdown("<h1 style='color: #1E3A8A;'>Africura Recommender</h1>", unsafe_allow_html=True)
        st.caption("Disclaimer: Prices are estimates only")

        tab1, tab2, tab3, tab4 = st.tabs(["By Place", "By Country", "By Subcategory", "By Rating"])

        with tab1:
            place_name = st.text_input("Enter Place Name")
            if st.button("Get Recommendations", key="place_btn"):
                if place_name:
                    recs = recommender.recommend_place(place_name)
                    st.dataframe(recs, use_container_width=True)

        with tab2:
            country_name = st.text_input("Enter Country Name")
            if st.button("Get Country Recommendations", key="country_btn"):
                if country_name:
                    recs = recommender.recommend_country(country_name)
                    st.dataframe(recs, use_container_width=True)

        with tab3:
            subcat = st.text_input("Enter Subcategory")
            if st.button("Get Subcategory Recommendations", key="subcat_btn"):
                if subcat:
                    recs = recommender.recommend_subcategory(subcat)
                    st.dataframe(recs, use_container_width=True)

        with tab4:
            rating_threshold = st.slider("Maximum Rating Threshold", 0.0, 5.0, 3.0, 0.1)
            if st.button("Get Attraction Recommendations", key="attraction_btn"):
                recs = recommender.recommend_attraction(rating_threshold)
                st.dataframe(recs, use_container_width=True)

        # Amenities Multi-select
        st.subheader("Recommend by Amenities")
        amenities_list = clean_df['combined_amenities'].str.split(', ').explode().unique()
        selected = st.multiselect("Select Amenities", options=sorted(amenities_list))
        if selected and st.button("Get Amenity-based Recommendations"):
            recs = recommender.recommend_by_amenities(selected)
            st.dataframe(recs, use_container_width=True)

    elif selection == "Gallery":
        st.markdown("<h1 style='color: #1E3A8A;'>Africura Gallery</h1>", unsafe_allow_html=True)
        
        image_urls = [
            "https://cdn.pixabay.com/photo/2018/03/17/10/05/wildlife-3233525_640.jpg",
            "https://cdn.pixabay.com/photo/2018/01/21/18/54/water-buffalo-3097317_640.jpg",
            "https://cdn.pixabay.com/photo/2018/03/03/10/02/architecture-3195322_640.jpg",
            "https://cdn.pixabay.com/photo/2017/07/14/04/50/cheetah-2502782_640.jpg",
            "https://cdn.pixabay.com/photo/2018/09/17/14/36/lion-3683994_640.jpg",
        ]
        
        cols = st.columns(3)
        for idx, url in enumerate(image_urls):
            with cols[idx % 3]:
                st.image(url, use_column_width=True)

if __name__ == "__main__":
    main()
