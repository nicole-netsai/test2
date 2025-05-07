import streamlit as st
import random
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import sqlite3
from sqlite3 import Error
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import matplotlib.pyplot as plt

# Database Setup (unchanged)
def create_connection():
    """Create a database connection"""
    conn = None
    try:
        conn = sqlite3.connect('parking.db')
        return conn
    except Error as e:
        st.error(f"Database connection error: {e}")
    return conn

def initialize_database():
    """Initialize database tables"""
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            # Create parking_lots table
            c.execute('''
                CREATE TABLE IF NOT EXISTS parking_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    rate TEXT NOT NULL,
                    location TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    special_info TEXT
                )
            ''')
            
            # Create parking_status table
            c.execute('''
                CREATE TABLE IF NOT EXISTS parking_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_id INTEGER NOT NULL,
                    occupied INTEGER NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    FOREIGN KEY (lot_id) REFERENCES parking_lots (id)
                )
            ''')
            
            # Create reservations table
            c.execute('''
                CREATE TABLE IF NOT EXISTS reservations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_id INTEGER NOT NULL,
                    permit_type TEXT NOT NULL,
                    license_plate TEXT NOT NULL,
                    arrival_time TEXT NOT NULL,
                    reservation_time TIMESTAMP NOT NULL,
                    user_id TEXT,
                    FOREIGN KEY (lot_id) REFERENCES parking_lots (id)
                )
            ''')
            conn.commit()
        except Error as e:
            st.error(f"Database initialization error: {e}")
        finally:
            conn.close()

# Initialize the database
initialize_database()

# Load the parking spot detection model (from the Jupyter notebook)
@st.cache_resource
def load_model():
    try:
        # Load the pre-trained model (in a real app, you would load your actual trained model)
        model = tf.keras.models.load_model('parking_model.h5')  # Replace with your model path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to classify parking spots (from the Jupyter notebook)
def classify_parking_spot(img_path, model):
    """Classify if a parking spot is occupied or empty"""
    try:
        img = image.load_img(img_path, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        score = predictions[0][0]
        
        # Assuming binary classification: 0=Empty, 1=Occupied
        return "Occupied" if score > 0.5 else "Empty", float(score)
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return "Error", 0.0

# App Configuration
st.set_page_config(
    page_title="University Smart Parking System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database Operations (unchanged)
def get_parking_lots():
    """Get all parking lots from database"""
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('''
                SELECT 
                    pl.id, pl.name, pl.capacity, pl.rate, pl.location, 
                    pl.latitude, pl.longitude, pl.special_info,
                    ps.occupied, ps.last_updated
                FROM parking_lots pl
                LEFT JOIN parking_status ps ON pl.id = ps.lot_id
                ORDER BY pl.name
            ''')
            rows = c.fetchall()
            return [{
                'id': row[0],
                'name': row[1],
                'capacity': row[2],
                'rate': row[3],
                'location': row[4],
                'coords': (row[5], row[6]),
                'special': row[7],
                'occupied': row[8] if row[8] is not None else 0,
                'last_updated': row[9] if row[9] is not None else datetime.now()
            } for row in rows]
        except Error as e:
            st.error(f"Error fetching parking lots: {e}")
            return []
        finally:
            conn.close()
    return []

def update_parking_status(lot_id, occupied):
    """Update parking status in database"""
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            # Check if record exists
            c.execute('SELECT id FROM parking_status WHERE lot_id = ?', (lot_id,))
            if c.fetchone():
                c.execute('''
                    UPDATE parking_status 
                    SET occupied = ?, last_updated = ?
                    WHERE lot_id = ?
                ''', (occupied, datetime.now(), lot_id))
            else:
                c.execute('''
                    INSERT INTO parking_status (lot_id, occupied, last_updated)
                    VALUES (?, ?, ?)
                ''', (lot_id, occupied, datetime.now()))
            conn.commit()
            return True
        except Error as e:
            st.error(f"Error updating parking status: {e}")
            return False
        finally:
            conn.close()
    return False

def add_reservation(lot_id, permit_type, license_plate, arrival_time, user_id="guest"):
    """Add a new reservation to database"""
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('''
                INSERT INTO reservations 
                (lot_id, permit_type, license_plate, arrival_time, reservation_time, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (lot_id, permit_type, license_plate, arrival_time, datetime.now(), user_id))
            
            # Update occupied count
            c.execute('''
                UPDATE parking_status 
                SET occupied = occupied + 1 
                WHERE lot_id = ?
            ''', (lot_id,))
            
            conn.commit()
            return True
        except Error as e:
            st.error(f"Error creating reservation: {e}")
            return False
        finally:
            conn.close()
    return False

# Initialize sample data if database is empty
def initialize_sample_data():
    """Insert sample data if database is empty"""
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM parking_lots')
            if c.fetchone()[0] == 0:
                sample_lots = [
                    ("Great Hall", 31, "0.70/hr", "Near Main Entrance", 37.7749, -122.4194, "Visitor Parking"),
                    ("Faculty of Science", 200, "1.00/hr", "MLT and SLT Buildings", 37.7755, -122.4180, "Students/Lecturers"),
                    ("Student Union Lot", 40, "1.00/hr", "Next to Student Center", 37.7735, -122.4210, "Student Permits"),
                    ("Athletics Field Parking", 150, "1.50/hr", "Near Sports Complex", 37.7760, -122.4200, "Event Parking")
                ]
                c.executemany('''
                    INSERT INTO parking_lots 
                    (name, capacity, rate, location, latitude, longitude, special_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', sample_lots)
                
                # Initialize random occupied counts
                c.execute('SELECT id FROM parking_lots')
                for lot_id in c.fetchall():
                    occupied = random.randint(int(lot_id[0]*10), int(lot_id[0]*20))
                    update_parking_status(lot_id[0], occupied)
                
                conn.commit()
        except Error as e:
            st.error(f"Error initializing sample data: {e}")
        finally:
            conn.close()

# Initialize sample data if needed
initialize_sample_data()

# UI Components (updated with parking spot detection)
def show_parking_map():
    """Interactive campus parking map"""
    parking_data = get_parking_lots()
    map_data = []
    for lot in parking_data:
        available = lot['capacity'] - lot['occupied']
        map_data.append({
            "Lot": lot['name'],
            "Latitude": lot['coords'][0],
            "Longitude": lot['coords'][1],
            "Available": available,
            "Capacity": lot['capacity'],
            "Rate": lot['rate'],
            "Status": "🟢 Good" if available > 20 else "🟡 Limited" if available > 0 else "🔴 Full"
        })
    
    df = pd.DataFrame(map_data)
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Lot",
        hover_data=["Available", "Capacity", "Rate", "Status"],
        color="Status",
        zoom=15,
        height=500,
        mapbox_style="open-street-map"
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

def parking_lot_card(lot):
    """UI card for each parking lot"""
    available = lot['capacity'] - lot['occupied']
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(lot['name'])
            st.caption(f"📍 {lot['location']}")
            st.caption(f"🎯 {lot['special']}")
        with col2:
            st.metric("Available", f"{available}/{lot['capacity']}")
        
        # Visual indicators
        progress_val = available/lot['capacity']
        if progress_val > 0.3:
            st.progress(progress_val, f"{int(progress_val*100)}% available")
        else:
            st.progress(progress_val, "Limited spaces!")
        
        # Action buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔍 View Details", key=f"view_{lot['id']}"):
                st.session_state.selected_lot = lot
                st.session_state.current_view = "detail"
                
                # Show detailed information
                with st.container():
                    st.subheader(f"Parking Lot: {lot['name']}")
                    
                    # Display status with color coding
                    status = "Vacant" if available > 0 else "Occupied"
                    status_color = "green" if status == "Vacant" else "red"
                    st.markdown(f"**Status:** <span style='color:{status_color}'>{status}</span>", 
                               unsafe_allow_html=True)
                    
                    # Display location details
                    st.write(f"**Location:** {lot['location']}")
                    
                    # Parking spot detection
                    st.subheader("Parking Spot Detection")
                    uploaded_file = st.file_uploader("Upload parking spot image", 
                                                   type=["jpg", "jpeg", "png"],
                                                   key=f"upload_{lot['id']}")
                    
                    if uploaded_file is not None:
                        # Save the uploaded file
                        image_path = f"temp_{lot['id']}.jpg"
                        with open(image_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Classify the parking spot
                        classification, confidence = classify_parking_spot(image_path, model)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(uploaded_file, caption="Uploaded Parking Spot", use_column_width=True)
                        with col2:
                            st.write("### Detection Results")
                            if classification == "Occupied":
                                st.error(f"🚗 Occupied ({confidence*100:.1f}% confidence)")
                            else:
                                st.success(f"🅿️ Empty ({confidence*100:.1f}% confidence)")
                        
                        # Clean up
                        os.remove(image_path)

        with btn_col2:
            if st.button("🗺️ Get Directions", key=f"dir_{lot['id']}"):
                # Generate Google Maps directions URL
                directions_url = (f"https://www.google.com/maps/dir/?api=1"
                                f"&origin=Current+Location"
                                f"&destination={lot['coords'][0]},{lot['coords'][1]}"
                                f"&travelmode=driving")
                
                # Open in new tab
                st.markdown(f"""
                <a href="{directions_url}" target="_blank">
                    <button style="
                        background-color: #4285F4;
                        color: white;
                        padding: 0.5em 1em;
                        border: none;
                        border-radius: 4px;
                        font-size: 1em;
                        cursor: pointer;
                    ">
                        Open in Google Maps
                    </button>
                </a>
                """, unsafe_allow_html=True)

# Main App
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .lot-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://cdn1.iconfinder.com/data/icons/higher-education/66/5-1024.png", width=150)
        st.title("Campus Parking")
        
        view_options = {
            "map": "🗺️ Parking Map",
            "list": "📋 All Parking Lots", 
            "reserve": "🅿️ Reserve Spot",
            "admin": "🔒 Admin Portal"
        }
        
        current_view = st.radio(
            "Navigation",
            options=list(view_options.values()),
            index=0,
            key="current_view"
        )
        
        st.markdown("---")
        st.caption(f"Last Updated: {datetime.now().strftime('%m/%d %I:%M %p')}")

    # Main Content Area
    parking_lots = get_parking_lots()
    
    if current_view == view_options["map"]:
        st.header("Campus Parking Map")
        show_parking_map()
        
    elif current_view == view_options["list"]:
        st.header("All Parking Facilities")
        
        search = st.text_input("Search parking lots", key="parking_search")
        
        filtered_lots = [
            lot for lot in parking_lots 
            if search.lower() in lot['name'].lower() or search.lower() in lot['location'].lower()
        ]
        
        for lot in filtered_lots:
            parking_lot_card(lot)

    elif current_view == view_options["reserve"]:
        st.header("Reserve Parking Spot")
        
        selected_lot = st.selectbox(
            "Select parking lot",
            options=parking_lots,
            format_func=lambda x: x['name'],
            index=0,
            key="reserve_lot"
        )
        
        available = selected_lot['capacity'] - selected_lot['occupied']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(selected_lot['name'])
            st.image("https://via.placeholder.com/500x300?text=Lot+"+selected_lot['name'].replace(" ", "+"), 
                    use_column_width=True)
        with col2:
            st.metric("Available Spots", available)
            st.caption(f"Total Capacity: {selected_lot['capacity']}")
            st.write(f"**Rate:** {selected_lot['rate']}")
            st.write(f"**Location:** {selected_lot['location']}")
            st.write(f"**Special:** {selected_lot['special']}")
            
            if available > 0:
                with st.form(key="reservation_form"):
                    permit_type = st.selectbox("Permit Type", ["Student", "Faculty", "Visitor", "Event"])
                    license_plate = st.text_input("Vehicle License Plate")
                    arrival_time = st.time_input("Estimated Arrival Time")
                    
                    if st.form_submit_button("Reserve Spot", type="primary"):
                        if add_reservation(
                            selected_lot['id'],
                            permit_type,
                            license_plate,
                            arrival_time.strftime("%H:%M")
                        ):
                            st.success("Reservation confirmed!")
                            time.sleep(1)
                            st.rerun()
            else:
                st.error("No available spots in this lot")

    elif current_view == view_options["admin"]:
        st.header("Admin Portal")
        
        # Simple auth
        password = st.text_input("Enter admin password", type="password")
        if password != "campus123":
            st.error("Incorrect password")
            st.stop()
        
        st.success("Admin access granted")
        
        tab1, tab2, tab3 = st.tabs(["Parking Status", "Analytics", "Spot Detection"])
        
        with tab1:
            st.subheader("Manage Parking Lots")
            
            for lot in parking_lots:
                with st.expander(lot['name']):
                    new_occupied = st.number_input(
                        "Occupied spots",
                        min_value=0,
                        max_value=lot['capacity'],
                        value=lot['occupied'],
                        key=f"admin_{lot['id']}"
                    )
                    
                    if st.button(f"Update {lot['name']}", key=f"update_{lot['id']}"):
                        if update_parking_status(lot['id'], new_occupied):
                            st.success("Updated successfully!")
                            time.sleep(0.5)
                            st.rerun()
        
        with tab2:
            st.subheader("Parking Analytics")
            
            # Data visualization
            data = []
            for lot in parking_lots:
                data.append({
                    "Lot": lot['name'],
                    "Capacity": lot['capacity'],
                    "Occupied": lot['occupied'],
                    "Utilization": lot['occupied'] / lot['capacity'] * 100
                })
            
            df = pd.DataFrame(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df, x="Lot", y=["Capacity", "Occupied"])
            with col2:
                st.metric("Total Campus Capacity", sum(lot['capacity'] for lot in parking_lots))
                st.metric("Current Utilization", 
                         f"{sum(lot['occupied'] for lot in parking_lots)/sum(lot['capacity'] for lot in parking_lots)*100:.1f}%")
        
        with tab3:
            st.subheader("Parking Spot Detection")
            
            # Upload image for classification
            uploaded_file = st.file_uploader("Upload parking spot image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display the uploaded image
                st.image(uploaded_file, caption="Uploaded Parking Spot", use_column_width=True)
                
                # Save the uploaded file temporarily
                image_path = "temp_upload.jpg"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Classify the parking spot
                classification, confidence = classify_parking_spot(image_path, model)
                
                # Display results
                st.write("### Detection Results")
                if classification == "Occupied":
                    st.error(f"🚗 Occupied ({confidence*100:.1f}% confidence)")
                else:
                    st.success(f"🅿️ Empty ({confidence*100:.1f}% confidence)")
                
                # Clean up
                os.remove(image_path)

    # Footer
    st.markdown("---")
    st.caption("© 2023 University Campus Parking System | v2.0 (with AI Detection)")

if __name__ == "__main__":
    main()
