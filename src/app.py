import streamlit as st
import folium
import joblib
import geopandas as gpd
from streamlit_folium import st_folium
from shapely.geometry import Point
import plotly.figure_factory as ff
import yaml
import time
from get_satellite_features import *
from get_geo_features import *

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
bounds = config["bounds"]
input_file = config["input_data"]

# Load the trained model
@st.cache_resource()
def load_model():
    model_path = "./models/rf.pkl"
    model = joblib.load(model_path)
    return model

model = load_model()

# Load training data to determine UHI thresholds
@st.cache_resource()
def load_training_data():
    df = pd.read_csv(f"./{input_file}")  # Replace with actual training data file
    return df

training_data = load_training_data()

# Compute dynamic thresholds based on percentiles
low_threshold = np.percentile(training_data["UHI Index"], 33)
high_threshold = np.percentile(training_data["UHI Index"], 66)

# Function to extract data (Placeholder: Replace with actual function)
def extract_features(geo_df):
    """Extract features for a given GeoDataFrame location with progress tracking."""
 
    progress = st.progress(0)
    status_text = st.empty()

    status_text.write("Extracting weather features...")
    geo_df = get_weather_features(geo_df)
    progress.progress(33)

    status_text.write("Extracting building density features...")
    geo_df = get_building_features(geo_df)
    progress.progress(66)
    
    status_text.write("Extracting Sentinel2 and Landsat features...")
    geo_df = get_satellite_features(geo_df)
    progress.progress(100)
    
    status_text.write("Feature extraction complete!")
    geo_df.drop(columns=["geometry"], inplace=True)

    return geo_df

# Streamlit App
st.title("Urban Heat Island Index Predictor")
st.write("Select a location in NYC to predict the Urban Heat Island Index.")

if "map_initialized" not in st.session_state:
    st.session_state.map_initialized = folium.Map(
        max_bounds=True,
        location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], 
        min_lon=bounds[0],
        min_lat=bounds[1],
        max_lon=bounds[2],
        max_lat=bounds[3],
        zoom_start=13,
        # disable zoom
        zoom_control=False,
    )
    st.session_state.map_initialized.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    st.session_state.map_initialized.add_child(folium.LatLngPopup())

# Create map UI element
map_data = st_folium(st.session_state.map_initialized, width=700, height=500, key="map")

# Add click functionality
# m.add_child(folium.LatLngPopup())

# Display the map
# map_data = st_folium(m, width=700, height=500)

# Process user selection
if map_data and map_data.get("last_clicked"):
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    if not (bounds[1] <= lat <= bounds[3] and bounds[0] <= lon <= bounds[2]):
        st.error("Selected location is outside the prediction area. Please choose a point within NYC.")
    else:
        st.write(f"Selected Location: ({lat}, {lon})")
        # Convert to GeoDataFrame
        point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        
        # Extract features
        features = extract_features(point)
        
        # Predict
        prediction = model.predict(features.values)[0]
        
        # Display result
        st.subheader(f"Predicted Urban Heat Island Index: {prediction:.4f}")

        # Determine color classification
        if prediction < low_threshold:
            color = "green"
            category = "Low"
        elif prediction < high_threshold:
            color = "orange"
            category = "Moderate"
        else:
            color = "red"
            category = "High"
        
        # Create a histogram of training UHI data
        hist_data = [training_data["UHI Index"].tolist()]
        group_labels = ["NYC UHI Index Distribution"]
        
        fig = ff.create_distplot(hist_data, group_labels, show_rug=False, show_hist=False, show_curve=True)
        
        # Add vertical line for the predicted value
        fig.add_vline(x=prediction, line=dict(color="red", width=3), annotation_text="Predicted Value", annotation_position="top")
        
        # Display the histogram with the prediction marked
        st.markdown(f"#### Prediction Compared to NYC UHI Distribution: <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
        
