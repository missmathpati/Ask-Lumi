"""
Streamlit application for multimodal RAG-based product chatbot.
"""

import streamlit as st
import pandas as pd
from PIL import Image
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ModelLoader
from retrieval import ProductRetriever
from rag_pipeline import RAGPipeline


@st.cache_resource
def load_system(api_key):
    """Load models and initialize system (cached for performance)."""
    # Create a simple callback that just stores messages
    messages = []
    def status_callback(message):
        messages.append(message)
        # Also print for terminal visibility
        print(message)
    
    loader = ModelLoader(progress_callback=status_callback)
    loader.initialize_all()
    retriever = ProductRetriever(loader)
    rag = RAGPipeline(retriever, api_key=api_key)
    return loader, retriever, rag


def display_product_card(product_row, idx):
    """Display a product card with image and details."""
    # Use responsive columns - stack on mobile, side-by-side on desktop
    col1, col2 = st.columns([1, 2], gap="medium")
    
    with col1:
        image_path = product_row.get("image_path", None)
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                # Responsive image width - use None for responsive behavior
                st.image(img, width=None)
            except Exception:
                st.write("Image not available")
        else:
            st.write("Image not available")
    
    with col2:
        st.markdown(f"### {product_row.get('Product Name', 'N/A')}")
        st.markdown(f"**Brand:** {product_row.get('Brand Name', 'N/A')}")
        st.markdown(f"**Category:** {product_row.get('Category', 'N/A')}")
        st.markdown(f"**Price:** {product_row.get('Selling Price', 'N/A')}")
        similarity = product_row.get('similarity', 0)
        st.markdown(f"**Similarity Score:** {similarity:.3f}")


def main():
    st.set_page_config(
        page_title="Ask Lumi!",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add viewport meta tag for mobile responsiveness
    st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0">', unsafe_allow_html=True)
    
    # Custom CSS for chatbot styling with animations
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
        
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        @keyframes typing {
            0% { width: 0; }
            100% { width: 100%; }
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes fadeOut {
            from { opacity: 1; transform: scale(1); }
            to { opacity: 0; transform: scale(0.95); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes zoomIn {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }
        
        /* Splash screen fade out */
        .splash-fade-out {
            animation: fadeOut 0.5s ease-out forwards;
        }
        
        /* Main content slide in */
        .main-content {
            animation: slideIn 0.6s ease-out;
        }
        
        /* Background animated particles */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .particle {
            position: absolute;
            border-radius: 50%;
            opacity: 0.1;
            animation: float 15s infinite ease-in-out;
        }
        
        .particle:nth-child(1) { width: 80px; height: 80px; background: #667eea; top: 20%; left: 10%; animation-delay: 0s; }
        .particle:nth-child(2) { width: 60px; height: 60px; background: #f093fb; top: 60%; left: 80%; animation-delay: 2s; }
        .particle:nth-child(3) { width: 100px; height: 100px; background: #4facfe; top: 80%; left: 20%; animation-delay: 4s; }
        .particle:nth-child(4) { width: 50px; height: 50px; background: #667eea; top: 40%; left: 70%; animation-delay: 6s; }
        .particle:nth-child(5) { width: 70px; height: 70px; background: #f093fb; top: 10%; left: 50%; animation-delay: 8s; }
        .particle:nth-child(6) { width: 90px; height: 90px; background: #4facfe; top: 50%; left: 30%; animation-delay: 10s; }
        
        /* Main container with gradient background matching image */
        .main .block-container {
            background: linear-gradient(180deg, #e0f2fe 0%, #fce7f3 50%, #e9d5ff 100%);
            padding-top: 1rem;
            padding-bottom: 2rem;
            min-height: 100vh;
        }
        
        /* Page background */
        .stApp {
            background: linear-gradient(180deg, #e0f2fe 0%, #fce7f3 50%, #e9d5ff 100%);
        }
        
        /* Remove white block under header */
        .stApp > header {
            background-color: transparent;
        }
        
        .stApp [data-testid="stHeader"] {
            background-color: transparent;
        }
        
        /* Custom loading animation */
        .lumi-thinking {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 50%, rgba(79, 172, 254, 0.1) 100%);
            border-radius: 20px;
            margin: 1rem 0;
            animation: fadeIn 0.3s ease-out;
        }
        
        .thinking-dots {
            display: flex;
            gap: 8px;
        }
        
        .thinking-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .thinking-dot:nth-child(1) { animation-delay: 0s; }
        .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        
        .thinking-text {
            color: #667eea;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        /* Main title styling with animation */
        .main-title {
            font-size: 4rem;
            font-weight: 800;
            color: #2d3748;
            text-align: center;
            margin-bottom: 0.5rem;
            animation: fadeIn 1s ease-out;
        }
        
        /* Subtitle with fade-in */
        .subtitle {
            text-align: center;
            color: #4a5568;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            animation: fadeIn 1.5s ease-out;
        }
        
        /* Chatbot message bubble styling */
        .chat-bubble {
            padding: 1rem 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            max-width: 80%;
            animation: fadeIn 0.5s ease-out;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .chat-bubble-user {
            background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
            background-size: 200% 200%;
            animation: gradientShift 3s ease infinite;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        }
        
        .chat-bubble-bot {
            background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(240, 147, 251, 0.08) 100%);
            color: #2d3748;
            margin-right: auto;
            border-bottom-left-radius: 4px;
            border: 2px solid rgba(102, 126, 234, 0.2);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }
        
        /* Loading animation */
        .loading-dots {
            display: inline-block;
        }
        
        .loading-dots::after {
            content: '...';
            animation: typing 1.5s steps(3, end) infinite;
        }
        
        /* Card styling */
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border: 1px solid #e5e7eb;
        }
        
        /* Button styling with purple-pink-blue gradient */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
            background-size: 200% 200%;
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            animation: gradientShift 3s ease infinite;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            animation: pulse 1s ease infinite;
        }
        
        /* Input styling with friendly robot theme */
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            background: white;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        }
        
        .stTextInput > div > div > input:focus {
            border: 2px solid #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15), 0 4px 16px rgba(102, 126, 234, 0.2);
            outline: none;
            transform: translateY(-2px);
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #a0aec0;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 1rem 1.5rem;
            font-weight: 600;
        }
        
        /* Info boxes */
        .stInfo {
            background: #f7fafc;
            border-left: 4px solid #667eea;
            border-radius: 12px;
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Success message */
        .stSuccess {
            background: #f0fff4;
            border-left: 4px solid #48bb78;
            border-radius: 12px;
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: #ffffff;
        }
        
        /* Chat container with friendly robot theme */
        .chat-container {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240, 147, 251, 0.05) 50%, rgba(79, 172, 254, 0.05) 100%);
            border-radius: 25px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
            animation: fadeIn 0.8s ease-out;
            border: 2px solid rgba(102, 126, 234, 0.2);
            backdrop-filter: blur(10px);
        }
        
        /* Video container styling */
        .stVideo {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
            margin: 1rem 0;
        }
        
        /* Logo styling */
        .stImage img {
            border-radius: 20px;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
        }
        
        /* Product cards with gradient accents */
        .product-card-gradient {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid;
            border-image: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%) 1;
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #1f2937;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #4b5563;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #9ca3af;
            padding: 2rem 0;
            font-size: 0.9rem;
        }
        
        /* Mobile Responsive Styles */
        @media screen and (max-width: 768px) {
            /* Adjust main title for mobile */
            .main-title {
                font-size: 2.5rem !important;
                padding: 0 1rem;
            }
            
            .subtitle {
                font-size: 1rem !important;
                padding: 0 1rem;
            }
            
            /* Make chat bubbles full width on mobile */
            .chat-bubble {
                max-width: 95% !important;
                padding: 0.75rem 1rem !important;
                font-size: 0.9rem;
            }
            
            /* Adjust chat container */
            .chat-container {
                padding: 1rem !important;
                margin: 0.5rem 0 !important;
            }
            
            /* Make buttons touch-friendly */
            .stButton > button {
                padding: 0.9rem 1.5rem !important;
                font-size: 1rem !important;
                min-height: 44px; /* iOS touch target minimum */
            }
            
            /* Adjust input fields */
            .stTextInput > div > div > input {
                padding: 0.9rem 1rem !important;
                font-size: 1rem !important;
                min-height: 44px;
            }
            
            /* Stack columns on mobile */
            .main .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
            }
            
            /* Adjust logo size */
            .stImage img {
                max-width: 150px !important;
            }
            
            /* Product cards full width on mobile */
            .product-card {
                padding: 1rem !important;
            }
            
            /* Hide or adjust particles on mobile for performance */
            .particle {
                display: none;
            }
            
            /* Adjust sidebar */
            .css-1d391kg {
                width: 100% !important;
            }
            
            /* Make tabs more mobile-friendly */
            .stTabs [data-baseweb="tab"] {
                padding: 0.75rem 1rem !important;
                font-size: 0.9rem !important;
            }
            
            /* Adjust thinking animation */
            .lumi-thinking {
                padding: 1rem !important;
                font-size: 0.9rem !important;
            }
            
            /* Footer adjustments */
            .footer {
                padding: 1rem 0 !important;
                font-size: 0.8rem !important;
            }
            
            /* Video container on mobile */
            .stVideo {
                width: 100% !important;
            }
        }
        
        /* Tablet Responsive Styles */
        @media screen and (min-width: 769px) and (max-width: 1024px) {
            .main-title {
                font-size: 3rem !important;
            }
            
            .subtitle {
                font-size: 1.1rem !important;
            }
            
            .chat-bubble {
                max-width: 85% !important;
            }
            
            .stButton > button {
                padding: 0.8rem 2rem !important;
            }
            
            /* Reduce particle count on tablets */
            .particle:nth-child(4),
            .particle:nth-child(5),
            .particle:nth-child(6) {
                display: none;
            }
        }
        
        /* Ensure sidebar is collapsible on mobile */
        @media screen and (max-width: 768px) {
            [data-testid="stSidebar"] {
                width: 100% !important;
            }
            
            [data-testid="stSidebar"][aria-expanded="false"] {
                width: 0 !important;
            }
        }
        
        /* Touch-friendly expanders */
        @media screen and (max-width: 768px) {
            .streamlit-expanderHeader {
                padding: 1rem !important;
                font-size: 1rem !important;
                min-height: 48px;
                display: flex;
                align-items: center;
            }
        }
        
        /* Image responsiveness */
        @media screen and (max-width: 768px) {
            img {
                max-width: 100% !important;
                height: auto !important;
            }
            
            /* Product images in cards */
            .product-card img {
                width: 100% !important;
                max-width: 300px !important;
            }
            
            /* Streamlit image containers */
            .stImage img,
            [data-testid="stImage"] img {
                max-width: 100% !important;
                width: 100% !important;
                height: auto !important;
            }
        }
        
        /* General image responsiveness for all screen sizes */
        img {
            max-width: 100%;
            height: auto;
        }
        
        /* Streamlit image wrapper */
        .stImage,
        [data-testid="stImage"] {
            width: 100%;
        }
        
        .stImage img,
        [data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
        }
        
        /* Prevent horizontal scroll on mobile */
        .stApp {
            overflow-x: hidden;
        }
        
        /* Improve text readability on small screens */
        @media screen and (max-width: 768px) {
            p, li, span {
                line-height: 1.6 !important;
            }
            
            h1, h2, h3 {
                line-height: 1.3 !important;
            }
            
            /* Make sidebar collapsible on mobile */
            [data-testid="stSidebar"] {
                min-width: 0px !important;
            }
            
            /* Adjust API key input section */
            .stTextInput {
                width: 100% !important;
            }
            
            /* Make file uploader touch-friendly */
            .stFileUploader {
                width: 100% !important;
            }
            
            /* Adjust info boxes */
            .stInfo, .stSuccess, .stWarning, .stError {
                font-size: 0.9rem !important;
                padding: 0.75rem !important;
            }
            
            /* Better spacing for tabs */
            .stTabs [data-baseweb="tab-list"] {
                flex-wrap: wrap;
            }
        }
        
        /* Landscape mobile orientation */
        @media screen and (max-width: 768px) and (orientation: landscape) {
            .main-title {
                font-size: 2rem !important;
            }
            
            .splash-content {
                padding: 1rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Splash screen with Lumi Hi animation
    if 'show_splash' not in st.session_state:
        st.session_state.show_splash = True
    
    if st.session_state.show_splash:
        # Splash screen container with white background and decorative circles
        st.markdown("""
        <style>
            .stApp {
                background: white !important;
            }
            .main .block-container {
                background: white !important;
            }
            
            /* Decorative circles for splash screen */
            .splash-bg-circles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 0;
                overflow: hidden;
                pointer-events: none;
            }
            
            .splash-circle {
                position: absolute;
                border-radius: 50%;
                opacity: 0.15;
                animation: float 20s infinite ease-in-out;
            }
            
            .splash-circle-blue {
                background: #667eea;
                width: 200px;
                height: 200px;
                top: 10%;
                left: 15%;
                animation-delay: 0s;
            }
            
            .splash-circle-purple {
                background: #764ba2;
                width: 150px;
                height: 150px;
                top: 60%;
                right: 20%;
                animation-delay: 3s;
            }
            
            .splash-circle-pink {
                background: #f093fb;
                width: 180px;
                height: 180px;
                bottom: 15%;
                left: 10%;
                animation-delay: 6s;
            }
            
            .splash-circle-blue-small {
                background: #4facfe;
                width: 120px;
                height: 120px;
                top: 40%;
                right: 10%;
                animation-delay: 2s;
            }
            
            .splash-circle-purple-small {
                background: #a78bfa;
                width: 100px;
                height: 100px;
                top: 80%;
                left: 50%;
                animation-delay: 4s;
            }
            
            .splash-circle-pink-small {
                background: #ec4899;
                width: 130px;
                height: 130px;
                top: 11%;
                right: 40%;
                animation-delay: 5s;
            }
            
            .splash-content {
                position: relative;
                z-index: 1;
            }
            
            /* Mobile responsive splash screen */
            @media screen and (max-width: 768px) {
                .splash-circle {
                    width: 80px !important;
                    height: 80px !important;
                }
                
                .splash-circle-blue {
                    width: 100px !important;
                    height: 100px !important;
                }
                
                .splash-circle-purple,
                .splash-circle-pink {
                    width: 80px !important;
                    height: 80px !important;
                }
                
                .splash-circle-blue-small,
                .splash-circle-purple-small,
                .splash-circle-pink-small {
                    width: 60px !important;
                    height: 60px !important;
                }
                
                /* Hide some circles on very small screens */
                .splash-circle-purple-small,
                .splash-circle-pink-small {
                    display: none;
                }
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Add decorative circles
        st.markdown("""
        <div class="splash-bg-circles">
            <div class="splash-circle splash-circle-blue"></div>
            <div class="splash-circle splash-circle-purple"></div>
            <div class="splash-circle splash-circle-pink"></div>
            <div class="splash-circle splash-circle-blue-small"></div>
            <div class="splash-circle splash-circle-purple-small"></div>
            <div class="splash-circle splash-circle-pink-small"></div>
        </div>
        """, unsafe_allow_html=True)
        
        splash_col1, splash_col2, splash_col3 = st.columns([1, 2, 1])
        with splash_col2:
            st.markdown("""
            <div class="splash-content" style="text-align: center; padding: 4rem 2rem;">
            """, unsafe_allow_html=True)
            
            # Lumi Hi animation video - centered
            if os.path.exists("Lumi.mov"):
                import base64
                video_file = open("Lumi.mov", "rb")
                video_bytes = video_file.read()
                video_file.close()
                video_base64 = base64.b64encode(video_bytes).decode()
                
                st.markdown(f"""
                <div style="margin: 1rem auto; max-width: min(500px, 90vw); padding: 0 1rem;">
                    <video autoplay loop muted playsinline style="width: 100%; display: block; max-width: 100%; height: auto;">
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    </video>
                </div>
                """, unsafe_allow_html=True)
            
            # Centered text under video - responsive font size
            st.markdown("""
            <div style="margin-top: 2rem; text-align: center; padding: 0 1rem;">
                <p style="color: #2d3748; font-size: clamp(1rem, 4vw, 1.5rem); font-weight: 600; margin-bottom: 2rem; animation: fadeIn 1s ease-out;">
                    💬 Hi, I am Lumi, Your intelligent product discovery chatbot
                </p>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to continue with animation - responsive columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Get Started", use_container_width=True, key="start_button"):
                    # Add transition animation
                    st.markdown("""
                    <div class="splash-fade-out" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: white; z-index: 9999; pointer-events: none;"></div>
                    """, unsafe_allow_html=True)
                    st.session_state.show_splash = False
                    st.rerun()
        
        st.stop()  # Stop here until user clicks "Get Started"
    
    # Main content with slide-in animation
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Background animated particles (only after splash)
    st.markdown("""
    <div class="bg-animation">
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Header with just logo centered - responsive
    header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
    with header_col2:
        if os.path.exists("Lumi_logo.png"):
            # Responsive logo sizing
            st.image("Lumi_logo.png", width=200)
    
    # Check for OpenAI API key
    if 'openai_api_key' not in st.session_state or not st.session_state.openai_api_key:
        st.markdown("---")
        st.markdown("### 🔑 OpenAI API Key Required")
        st.markdown("""
        Please enter your OpenAI API key to use Lumi. Your key will be stored securely in this session 
        and will not be shared or saved permanently.
        """)
        
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key. You can get one from https://platform.openai.com/api-keys"
        )
        
        # Responsive button layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔓 Submit API Key", use_container_width=True):
                if api_key_input.strip():
                    st.session_state.openai_api_key = api_key_input.strip()
                    st.success("✅ API key saved! Loading system...")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key.")
        
        st.info("💡 **Note:** Your API key is only stored in your browser session and will be cleared when you close the app.")
        st.stop()
    
    # Initialize system with progress indicators
    cache_exists = os.path.exists("embeddings_cache.pkl")
    
    if not cache_exists:
        st.warning("⚠️ **First-time setup detected!** Computing embeddings will take 5-10 minutes. This only happens once - subsequent runs will be much faster!")
        st.info("💡 **Tip:** Check your terminal for detailed progress. The embeddings will be cached for future use.")
    
    # Show loading status
    with st.spinner("Loading system... Please wait. Check terminal for detailed progress."):
        try:
            loader, retriever, rag = load_system(api_key=st.session_state.openai_api_key)
            st.success("✓ **System loaded successfully!** Ready to use.")
        except Exception as e:
            st.error(f"❌ **Error loading system:** {str(e)}")
            st.exception(e)
            st.info("💡 **Troubleshooting:** Make sure all required files are present (amazon_processed.csv, clip_head.pth, images/)")
            # Allow user to reset API key if there's an error
            if st.button("🔄 Reset API Key"):
                st.session_state.openai_api_key = None
                st.rerun()
            st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        num_results = st.slider("Number of results to retrieve", 3, 10, 5)
        st.markdown("---")
        st.markdown("### 🔑 API Key")
        st.markdown(f"**Status:** {'✅ Set' if st.session_state.openai_api_key else '❌ Not Set'}")
        if st.button("🔄 Reset API Key", use_container_width=True):
            st.session_state.openai_api_key = None
            st.rerun()
        st.markdown("---")
        st.markdown("### 📚 About Ask Lumi")
        st.markdown("""
        **Ask Lumi** is an intelligent product discovery assistant that helps you find products using:
        
        🔍 **Text Search** - Ask questions in natural language
        
        🖼️ **Image Search** - Upload images to find similar products
        
        🎨 **Visual Discovery** - Request to see product images
        
        Powered by advanced **RAG (Retrieval-Augmented Generation)** technology combining:
        - CLIP vision-language model
        - FAISS similarity search
        - OpenAI GPT for natural responses
        """)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific in your queries for better results
        - Upload clear product images for image search
        - Use natural language - Lumi understands context!
        """)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Lumi", "🖼️ Product Search", "🎨 Show Me"])
    
    # Tab 1: Text Query - Chatbot style
    with tab1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Welcome message - no logo, just text
        st.markdown("### 💬 Chat with Lumi")
        st.markdown("Hi! I'm Lumi, your friendly product discovery assistant. Ask me anything about products!")
        
        st.markdown("---")
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-bubble chat-bubble-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble chat-bubble-bot"><strong>Lumi:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                    # Show products if they exist
                    if 'products' in msg and len(msg['products']) > 0:
                        for idx, product in enumerate(msg['products'], 1):
                            with st.expander(f"🔹 {product.get('Product Name', 'N/A')}", expanded=False):
                                display_product_card(product, idx)
        
        # Text input with chatbot styling
        text_query = st.text_input(
            "Type your message...",
            placeholder="e.g., 'Show me wireless bluetooth headphones under $100'",
            key="text_input",
            help="💡 Tip: Be specific about features, price range, or use case"
        )
        
        # Responsive button layout - stack on mobile
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            search_clicked = st.button("💬 Send", key="text_search", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if search_clicked:
            if text_query.strip():
                # Add user message to chat
                st.session_state.chat_history.append({'role': 'user', 'content': text_query})
                
                # Custom animated loading
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div class="lumi-thinking">
                    <div class="thinking-dots">
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                    </div>
                    <span class="thinking-text">💭 Lumi is thinking...</span>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    answer, retrieved = rag.answer_text_query_rag(text_query, k=num_results)
                    loading_placeholder.empty()
                    
                    # Convert retrieved products to list for storage
                    products_list = [retrieved.iloc[i].to_dict() for i in range(len(retrieved))]
                    
                    # Add bot response to chat
                    st.session_state.chat_history.append({
                        'role': 'bot', 
                        'content': answer,
                        'products': products_list
                    })
                    
                    st.rerun()
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a message!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Image Query
    with tab2:
        st.markdown("### 🖼️ Product Search")
        st.markdown("Upload a product image and ask Lumi about it. She'll find similar products and answer your questions!")
        
        # Image upload with better styling
        uploaded_file = st.file_uploader(
            "📤 Upload a product image",
            type=["png", "jpg", "jpeg"],
            key="image_upload",
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image in a styled container - responsive
            st.markdown("---")
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Your Uploaded Image", width=None)
            
            # Text query input (recommended for better results)
            st.markdown("---")
            st.markdown("#### 💬 Ask Lumi about this image")
            image_query = st.text_input(
                "Your question:",
                placeholder="e.g., 'What is the name of this product, and how do I use it?'",
                key="image_query",
                help="💡 Adding a specific question helps Lumi provide more targeted answers"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                search_clicked_img = st.button("🔍 Search", key="image_search", use_container_width=True)
            
            if search_clicked_img:
                # Custom animated loading
                loading_placeholder_img = st.empty()
                loading_placeholder_img.markdown("""
                <div class="lumi-thinking">
                    <div class="thinking-dots">
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                    </div>
                    <span class="thinking-text">🔍 Analyzing image...</span>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    answer, retrieved = rag.answer_image_query_rag(
                        image,
                        user_query=image_query if image_query.strip() else None,
                        k=num_results
                    )
                    loading_placeholder_img.empty()
                    
                    # Display answer in chatbot bubble
                    st.markdown(f'<div class="chat-bubble chat-bubble-bot"><strong>Lumi:</strong> {answer}</div>', unsafe_allow_html=True)
                    
                    # Display retrieved products
                    st.markdown("---")
                    st.markdown("### 📦 Similar Products Found")
                    for idx, (_, product) in enumerate(retrieved.iterrows(), 1):
                        with st.expander(f"🔹 Product {idx}: {product.get('Product Name', 'N/A')}", expanded=False):
                            display_product_card(product, idx)
                except Exception as e:
                    loading_placeholder_img.empty()
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("👆 **Upload an image above** to search for similar products and ask Lumi questions about it!")
    
    # Tab 3: Show Me Picture
    with tab3:
        st.markdown("### 🎨 Show Me Products")
        st.markdown("Ask Lumi to show you product images! Perfect for visual browsing and discovery.")
        
        # Text input for show me picture queries
        show_query = st.text_input(
            "What would you like to see?",
            placeholder="e.g., 'Show me a picture of hotwheels' or 'Give me soft toys with images under $50'",
            key="show_picture_input",
            help="💡 Ask naturally - Lumi will show you product images based on your request"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            show_clicked = st.button("🎨 Show Me", key="show_picture_search", use_container_width=True)
        
        if show_clicked:
            if show_query.strip():
                # Custom animated loading
                loading_placeholder_show = st.empty()
                loading_placeholder_show.markdown("""
                <div class="lumi-thinking">
                    <div class="thinking-dots">
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                    </div>
                    <span class="thinking-text">🎨 Finding products...</span>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Retrieve products using text query
                    answer, retrieved = rag.answer_text_query_rag(show_query, k=num_results)
                    loading_placeholder_show.empty()
                    
                    # Display answer in chatbot bubble
                    st.markdown(f'<div class="chat-bubble chat-bubble-bot"><strong>Lumi:</strong> {answer}</div>', unsafe_allow_html=True)
                    
                    # Display retrieved products with images prominently displayed
                    st.markdown("---")
                    st.markdown("### 🖼️ Product Images")
                    
                    # Show up to 5 products with images in a grid-like layout
                    num_to_show = min(num_results, 5)
                    for idx, (_, product) in enumerate(retrieved.head(num_to_show).iterrows(), 1):
                        # Create a card-like container - responsive padding
                        st.markdown(f'<div style="background: white; border-radius: 12px; padding: clamp(1rem, 3vw, 1.5rem); margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid #e5e7eb;">', unsafe_allow_html=True)
                        
                        # Responsive columns - stack on mobile
                        col1, col2 = st.columns([1, 2], gap="medium")
                        
                        with col1:
                            image_path = product.get("image_path", None)
                            if image_path and os.path.exists(image_path):
                                try:
                                    img = Image.open(image_path)
                                    st.image(img, width=None)
                                except Exception:
                                    st.write("Image not available")
                            else:
                                st.write("Image not available")
                        
                        with col2:
                            st.markdown(f"#### 🛍️ {product.get('Product Name', 'N/A')}")
                            st.markdown(f"**🏷️ Brand:** {product.get('Brand Name', 'N/A')}")
                            st.markdown(f"**📂 Category:** {product.get('Category', 'N/A')}")
                            st.markdown(f"**💰 Price:** {product.get('Selling Price', 'N/A')}")
                            similarity = product.get('similarity', 0)
                            st.markdown(f"**⭐ Match Score:** {similarity:.3f}")
                            
                            # Show product description snippet
                            desc = product.get('product_text', '')
                            if desc:
                                desc_snippet = desc[:200] + ("..." if len(desc) > 200 else "")
                                with st.expander("📋 View Details"):
                                    st.write(desc_snippet)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    loading_placeholder_show.empty()
                    st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p style="margin: 0.5rem 0;">✨ <strong>Ask Lumi</strong> - Your Intelligent Product Discovery Assistant</p>
            <p style="margin: 0.5rem 0; font-size: 0.85rem;">Powered by CLIP, FAISS, and OpenAI GPT</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Close main-content div
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
