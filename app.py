import streamlit as st
from google import genai
from PIL import Image
import json
import math
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LocalLens AI | Hyper-Local Discovery",
    page_icon="📍",
    layout="wide"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .shop-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- MOCK HYPER-LOCAL DATABASE (JAIPUR / SANGANER REGION) ---
LOCAL_SHOPS = [
    {
        "id": 1,
        "shop_name": "Sanganer Heritage Block Prints",
        "category": "Handicrafts & Textiles",
        "area": "Chippa Mohalla, Sanganer",
        "lat": 26.8192,
        "lon": 75.7885,
        "phone": "919829011111",
        "product": "Hand-Block Printed Indigo Cotton Bedsheet",
        "price": "₹1,250",
        "stock": "8 units in stock",
        "keywords": ["bedsheet", "cotton", "block print", "indigo", "blue", "floral", "textile", "fabric", "cloth"]
    },
    {
        "id": 2,
        "shop_name": "Jaipur Blue Pottery Art Studio",
        "category": "Artisanal Home Decor",
        "area": "Amer Road, Jaipur",
        "lat": 26.9535,
        "lon": 75.8412,
        "phone": "919829022222",
        "product": "Traditional Hand-Painted Blue Pottery Ceramic Vase",
        "price": "₹850",
        "stock": "5 units in stock",
        "keywords": ["pottery", "vase", "ceramic", "blue", "flower pot", "decor", "handicraft", "bowl", "jar"]
    },
    {
        "id": 3,
        "shop_name": "Sharma Brass & Plumbing Hardware",
        "category": "Specialized Hardware",
        "area": "Tonk Road, Pratap Nagar",
        "lat": 26.8045,
        "lon": 75.8091,
        "phone": "919829033333",
        "product": "Heavy-Duty 1/2 Inch Brass Gate Valve & Tap Spindle",
        "price": "₹320",
        "stock": "14 units in stock",
        "keywords": ["valve", "brass", "pipe", "tap", "plumbing", "hardware", "faucet", "leak", "metal", "fitting"]
    },
    {
        "id": 4,
        "shop_name": "Bassi Organic Farm Collective",
        "category": "Fresh Farm Produce",
        "area": "Durgapura Mandi Road",
        "lat": 26.8518,
        "lon": 75.7862,
        "phone": "919829044444",
        "product": "Raw Wildflower Honey & Cold-Pressed Mustard Oil",
        "price": "₹450",
        "stock": "Fresh harvest today",
        "keywords": ["honey", "organic", "mustard oil", "farm", "produce", "food", "natural", "jar", "bottle"]
    },
    {
        "id": 5,
        "shop_name": "Johari Artisanal Mojari House",
        "category": "Traditional Footwear",
        "area": "Johari Bazar, Jaipur",
        "lat": 26.9196,
        "lon": 75.8266,
        "phone": "919829055555",
        "product": "Hand-Embroidered Camel Leather Mojari Shoes",
        "price": "₹1,100",
        "stock": "All sizes available",
        "keywords": ["shoes", "mojari", "jutti", "leather", "footwear", "traditional", "embroidered", "sandal"]
    }
]

# --- HELPER: CALCULATE DISTANCE (HAVERSINE FORMULA) ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

# --- HELPER: AI VISION & NLP ANALYZER ---
def analyze_with_ai(api_key, text_query, uploaded_image):
    client = genai.Client(api_key=api_key)
    prompt = """
    You are a hyper-local shopping assistant. Analyze the user's text query and/or uploaded image.
    Identify the product they are looking for and return ONLY a valid JSON object with no markdown formatting:
    {
        "identified_item": "Short clear name of the item",
        "category": "General category",
        "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    }
    """
    contents = [prompt]
    if text_query:
        contents.append(f"User description: {text_query}")
    if uploaded_image:
        contents.append(uploaded_image)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    raw_text = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(raw_text)

# --- SIDEBAR: SETTINGS & USER LOCATION ---
with st.sidebar:
    st.title("⚙️ App Controls")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    st.markdown("---")
    st.subheader("📍 Your Current Location")
    user_lat = st.number_input("Latitude", value=26.8190, format="%.4f")
    user_lon = st.number_input("Longitude", value=75.7880, format="%.4f")
    st.caption("Default set to Sanganer, Jaipur.")
    st.markdown("---")
    st.info("💡 **Demo Tip:** Upload a photo of a brass tap, ceramic vase, shoe, or cloth—or type what you need!")

# --- MAIN HEADER ---
st.title("📍 LocalLens AI")
st.markdown("#### Discover real-time inventory from nearby independent artisans, farmers & local shops.")

# --- SEARCH & VISUAL UPLOAD SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    text_query = st.text_input(
        "What are you looking for today?",
        placeholder="e.g., 'Blue floral block print bedsheet' or 'brass valve for leaking tap'"
    )

with col2:
    uploaded_file = st.file_uploader("Or snap/upload a photo of the item", type=["jpg", "jpeg", "png", "webp"])

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Item for Visual Search", width=220)

search_clicked = st.button("🔍 Find in Nearby Shops", type="primary", use_container_width=True)

# --- SEARCH EXECUTION & RESULTS ---
if search_clicked:
    if not text_query and not image:
        st.warning("Please enter a description or upload a photo to search.")
    else:
        extracted_keywords = []
        identified_label = text_query

        # Step 1: Use Gemini AI if API Key is provided, else fallback to basic keyword split
        if api_key:
            with st.spinner("🤖 AI is analyzing your request & matching local inventory..."):
                try:
                    ai_result = analyze_with_ai(api_key, text_query, image)
                    identified_label = ai_result.get("identified_item", text_query)
                    extracted_keywords = [k.lower() for k in ai_result.get("keywords", [])]
                    st.success(f"**AI Identified:** {identified_label} | **Tags:** {', '.join(extracted_keywords)}")
                except Exception as e:
                    st.error(f"AI Error: {e}. Falling back to text matching.")
                    extracted_keywords = text_query.lower().split()
        else:
            st.info("Running in keyword mode (Add Gemini API Key in sidebar for full Image Recognition).")
            extracted_keywords = text_query.lower().split()

        # Step 2: Score & Rank Local Shops
        matched_shops = []
        for shop in LOCAL_SHOPS:
            score = sum(1 for kw in extracted_keywords if any(kw in skw for skw in shop["keywords"]))
            if score > 0:
                dist = calculate_distance(user_lat, user_lon, shop["lat"], shop["lon"])
                shop_copy = shop.copy()
                shop_copy["match_score"] = score
                shop_copy["distance_km"] = dist
                matched_shops.append(shop_copy)

        # Sort by match relevance, then proximity
        matched_shops.sort(key=lambda x: (-x["match_score"], x["distance_km"]))

        st.markdown("---")
        if not matched_shops:
            st.warning("No exact local matches found. Showing all verified neighborhood vendors nearby:")
            for shop in LOCAL_SHOPS:
                shop_copy = shop.copy()
                shop_copy["distance_km"] = calculate_distance(user_lat, user_lon, shop["lat"], shop["lon"])
                matched_shops.append(shop_copy)

        # Step 3: Split Layout - Shop Cards on Left, Map on Right
        res_col, map_col = st.columns([1.3, 1])

        with res_col:
            st.subheader(f"🏪 {len(matched_shops)} Local Vendors Found")
            for shop in matched_shops:
                whatsapp_msg = f"Hi {shop['shop_name']}, I found your '{shop['product']}' on LocalLens AI. Is it available today?"
                whatsapp_url = f"https://wa.me{shop['phone']}?text={whatsapp_msg.replace(' ', '%20')}"

                st.markdown(f"""
                <div class="shop-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#1b5e20;">{shop['shop_name']}</h4>
                        <span class="badge">📍 {shop['distance_km']} km away</span>
                    </div>
                    <p style="margin:5px 0; color:#555;"><b>Area:</b> {shop['area']} • <i>{shop['category']}</i></p>
                    <p style="margin:8px 0; font-size:1.05em;"><b>Matched Item:</b> {shop['product']} — <span style="color:#2e7d32; font-weight:bold;">{shop['price']}</span></p>
                    <p style="margin:5px 0; font-size:0.9em; color:#388e3c;">✅ {shop['stock']} (Saves ~2 days shipping vs online)</p>
                    <a href="{whatsapp_url}" target="_blank" style="display:inline-block; margin-top:8px; background-color:#25D366; color:white; padding:6px 14px; border-radius:6px; text-decoration:none; font-weight:bold;">
                        💬 Reserve via WhatsApp
                    </a>
                </div>
                """, unsafe_allow_html=True)

        with map_col:
            st.subheader("📍 Neighborhood Map")
            map_points = [{"lat": shop["lat"], "lon": shop["lon"], "shop": shop["shop_name"]} for shop in matched_shops]
            if map_points:
                map_df = pd.DataFrame(map_points)
                st.map(map_df[["lat", "lon"]])
            else:
                st.info("No vendors to plot on the map yet.")
