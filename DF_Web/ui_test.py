"""
UI Components Test - Demo Page
Tests all UI components with standardized design
"""

import streamlit as st
from ui_components import NotebookUI

# Page config
st.set_page_config(
    page_title="UI Components Test",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize UI
ui = NotebookUI()
ui.apply_custom_css()

# Main content
st.markdown("<h1>DeepScan UI Components Test</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: var(--text-secondary); font-size: 16px; margin-bottom: 32px;'>Testing standardized modern flat design components</p>", unsafe_allow_html=True)

# Section 1: Buttons
st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.markdown("### 1. Button Styles")
st.markdown("Testing Primary and Secondary button styles with hover effects.")

col1, col2, col3 = st.columns(3)
with col1:
    st.button("Primary Button", type="primary", use_container_width=True)
with col2:
    st.button("Secondary Button", use_container_width=True)
with col3:
    st.button("Disabled Button", disabled=True, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Section 2: File Uploader
st.markdown("<div class='custom-card' style='margin-top: 24px;'>", unsafe_allow_html=True)
st.markdown("### 2. File Uploader")
st.markdown("Clean modern design with dashed border and hover effect.")

uploaded_file = st.file_uploader(
    "Drop your image here",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, PNG (Max 10MB)"
)

st.markdown("</div>", unsafe_allow_html=True)

# Section 3: Tabs
st.markdown("<div class='custom-card' style='margin-top: 24px;'>", unsafe_allow_html=True)
st.markdown("### 3. Tabs - Flat Design")
st.markdown("Modern tabs with bottom border indicator.")

tab1, tab2, tab3 = st.tabs(["Analysis", "History", "Settings"])

with tab1:
    st.markdown("**Analysis Tab Content**")
    st.markdown("This tab shows analysis results.")
    
with tab2:
    st.markdown("**History Tab Content**")
    st.markdown("View scan history and past results.")
    
with tab3:
    st.markdown("**Settings Tab Content**")
    st.markdown("Configure application settings.")

st.markdown("</div>", unsafe_allow_html=True)

# Section 4: Select & Slider
st.markdown("<div class='custom-card' style='margin-top: 24px;'>", unsafe_allow_html=True)
st.markdown("### 4. Select Box & Slider")

col1, col2 = st.columns(2)

with col1:
    st.selectbox(
        "Model Selection",
        ["CNN-Custom-V1", "EfficientNet-B0", "ResNet50", "VGG16"],
        index=0
    )

with col2:
    st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

st.markdown("</div>", unsafe_allow_html=True)

# Section 5: Cards
st.markdown("<div class='custom-card' style='margin-top: 24px;'>", unsafe_allow_html=True)
st.markdown("### 5. Custom Cards")
st.markdown("Standardized card system with consistent styling.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='custom-card' style='text-align: center;'>
        <div style='font-size: 48px; color: var(--accent-primary); margin-bottom: 8px;'>📊</div>
        <div style='font-size: 20px; font-weight: 600; color: var(--text-primary);'>150</div>
        <div style='font-size: 14px; color: var(--text-secondary);'>Total Scans</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='custom-card' style='text-align: center;'>
        <div style='font-size: 48px; color: var(--success); margin-bottom: 8px;'>✓</div>
        <div style='font-size: 20px; font-weight: 600; color: var(--text-primary);'>98.5%</div>
        <div style='font-size: 14px; color: var(--text-secondary);'>Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='custom-card' style='text-align: center;'>
        <div style='font-size: 48px; color: var(--danger); margin-bottom: 8px;'>⚠</div>
        <div style='font-size: 20px; font-weight: 600; color: var(--text-primary);'>12</div>
        <div style='font-size: 14px; color: var(--text-secondary);'>Fake Detected</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Section 6: Input Fields
st.markdown("<div class='custom-card' style='margin-top: 24px;'>", unsafe_allow_html=True)
st.markdown("### 6. Input Fields")

col1, col2 = st.columns(2)

with col1:
    st.text_input("Project Name", placeholder="Enter project name")
    st.number_input("Batch Size", min_value=1, max_value=100, value=32)

with col2:
    st.text_area("Description", placeholder="Enter description", height=85)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top: 48px; padding: 24px; text-align: center; color: var(--text-tertiary); border-top: 1px solid var(--bg-elevated);'>", unsafe_allow_html=True)
st.markdown("**UI Components Test** • Modern Flat Design • Dark Theme Foundation", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Design Tokens")
    st.markdown("""
    **Colors:**
    - Primary: #3B82F6 (Blue)
    - Success: #10B981 (Green)
    - Danger: #EF4444 (Red)
    
    **Backgrounds:**
    - Primary: #0F172A
    - Secondary: #1E293B
    - Elevated: #334155
    
    **Typography:**
    - H1: 32px/700
    - H2: 24px/600
    - Body: 16px/400
    - Label: 14px/400
    
    **Spacing:**
    - xs: 8px
    - sm: 12px
    - md: 24px
    - lg: 32px
    
    **Border Radius:**
    - sm: 8px
    - md: 12px
    - lg: 16px
    """)
