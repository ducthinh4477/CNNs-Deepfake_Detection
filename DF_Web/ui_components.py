"""
DeepScan UI Components Module
=============================
Modular UI components for the 3-Pane Notebook Layout.
Each component is small, reusable, and focused on a single responsibility.

Author: HCMUTE Senior Design Team
"""

import streamlit as st
from datetime import datetime
from styles import get_base_css


# ============================================================
# MODEL CONFIGURATION - Global constant for sync across panels
# ============================================================
MODEL_INFO = {
    'CNN Custom (CIFAKE)': {
        'accuracy': '99.2%',
        'accuracy_value': 99.2,
        'speed': '~0.5s',
        'color': '#10B981'
    },
    'EfficientNet-B0': {
        'accuracy': '98.8%',
        'accuracy_value': 98.8,
        'speed': '~1.2s',
        'color': '#3B82F6'
    },
    'ResNet50': {
        'accuracy': '98.5%',
        'accuracy_value': 98.5,
        'speed': '~2.0s',
        'color': '#3B82F6'
    }
}


class NotebookUI:
    """
    Enterprise SaaS UI Component Library for DeepScan
    Design System: Modern Dark / Deep Slate Professional
    """
    
    def __init__(self):
        if 'scan_history' not in st.session_state:
            st.session_state.scan_history = []
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False
    
    @staticmethod
    def apply_custom_css():
        """Apply Enterprise SaaS professional CSS styling"""
        st.markdown(get_base_css(), unsafe_allow_html=True)
    
    # ========================================================
    # METRIC CARDS
    # ========================================================
    
    @staticmethod
    def render_metric_card(value: str, label: str, color: str = None, tooltip: str = None):
        """
        Render a compact metric card with value and label
        
        Args:
            value: The metric value to display (e.g., "99.2%", "High", "3")
            label: The metric label (e.g., "Accuracy", "Risk Level")
            color: Optional color for the value (#10B981, #EF4444, etc.)
            tooltip: Optional tooltip text
        """
        value_color = f'color: {color};' if color else 'color: var(--text-primary);'
        tooltip_attr = f'title="{tooltip}"' if tooltip else ''
        
        st.markdown(f"""
            <div class="metric-card" {tooltip_attr}>
                <div class="metric-value" style="{value_color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # ADD SOURCE BUTTON (NotebookLM Style) - Z-Index Trick
    # ========================================================
    
    @staticmethod
    def render_add_source_btn():
        """
        Render NotebookLM-style Add Source button using z-index trick.
        The actual file_uploader is transparent and overlays the custom button.
        Returns the file uploader widget reference.
        """
        # Container with relative positioning for z-index trick
        st.markdown("""
            <div class="add-source-container">
                <div class="add-source-btn-visual">
                    <div class="add-source-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <line x1="12" y1="5" x2="12" y2="19"></line>
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                        </svg>
                    </div>
                    <div class="add-source-text">Add Source</div>
                    <div class="add-source-hint">Click or drag image here</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Real file uploader - styled to be transparent and overlay the custom button
        uploaded_file = st.file_uploader(
            "Upload image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility="collapsed",
            key="file_uploader_main"
        )
        
        return uploaded_file
    
    @staticmethod
    def render_file_loaded_badge(filename: str):
        """Render a success badge showing loaded file"""
        truncated_name = filename[:25] + '...' if len(filename) > 25 else filename
        st.markdown(f"""
            <div class="file-loaded-badge">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <div>
                    <div class="file-loaded-title">File loaded</div>
                    <div class="file-loaded-name">{truncated_name}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # MODEL SELECTOR
    # ========================================================
    
    @staticmethod
    def render_model_selector():
        """
        Render model selector dropdown with info card.
        Uses global MODEL_INFO constant for consistency across panels.
        Returns the selected model option.
        """
        st.markdown("""
            <div class="widget-label">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                    <rect x="3" y="3" width="7" height="7"/>
                    <rect x="14" y="3" width="7" height="7"/>
                    <rect x="14" y="14" width="7" height="7"/>
                    <rect x="3" y="14" width="7" height="7"/>
                </svg>
                AI Model
            </div>
        """, unsafe_allow_html=True)
        
        model_opt = st.selectbox(
            "Model",
            options=list(MODEL_INFO.keys()),
            index=0,
            label_visibility="collapsed"
        )
        
        # Use global MODEL_INFO constant
        info = MODEL_INFO.get(model_opt, MODEL_INFO['CNN Custom (CIFAKE)'])
        
        st.markdown(f"""
            <div class="model-info-card">
                <div class="model-info-row">
                    <span class="model-info-label">Accuracy</span>
                    <span class="model-info-value" style="color: {info['color']};">{info['accuracy']}</span>
                </div>
                <div class="model-info-row">
                    <span class="model-info-label">Speed</span>
                    <span class="model-info-value">{info['speed']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        return model_opt
    
    # ========================================================
    # CONFIDENCE THRESHOLD SLIDER
    # ========================================================
    
    @staticmethod
    def render_threshold_slider():
        """
        Render confidence threshold slider with visual feedback
        Returns the threshold value
        """
        st.markdown("""
            <div class="widget-label">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                    <line x1="4" y1="21" x2="4" y2="14"/>
                    <line x1="4" y1="10" x2="4" y2="3"/>
                    <line x1="12" y1="21" x2="12" y2="12"/>
                    <line x1="12" y1="8" x2="12" y2="3"/>
                    <line x1="20" y1="21" x2="20" y2="16"/>
                    <line x1="20" y1="12" x2="20" y2="3"/>
                </svg>
                Confidence Threshold
            </div>
        """, unsafe_allow_html=True)
        
        threshold = st.slider(
            "Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div class="threshold-display">
                <span class="threshold-value">{threshold:.0%}</span>
            </div>
        """, unsafe_allow_html=True)
        
        return threshold
    
    # ========================================================
    # RESULT BADGE
    # ========================================================
    
    @staticmethod
    def render_result_badge(is_real: bool, confidence: float):
        """
        Render a prominent result status badge
        
        Args:
            is_real: True if authentic, False if fabricated
            confidence: Confidence score (0-1)
        """
        status = "AUTHENTIC" if is_real else "FABRICATED"
        status_color = "#10B981" if is_real else "#EF4444"
        status_bg = "rgba(16, 185, 129, 0.1)" if is_real else "rgba(239, 68, 68, 0.1)"
        status_border = "rgba(16, 185, 129, 0.3)" if is_real else "rgba(239, 68, 68, 0.3)"
        subtitle = "Verified authentic content" if is_real else "Potential manipulation detected"
        
        st.markdown(f"""
            <div class="result-badge" style="background: {status_bg}; border: 1px solid {status_border}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 8px;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background: {status_color}; box-shadow: 0 0 8px {status_color};"></div>
                    <span style="font-size: 18px; font-weight: 700; color: {status_color}; text-transform: uppercase; letter-spacing: 0.1em;">
                        {status}
                    </span>
                </div>
                <div style="font-size: 12px; color: var(--text-secondary);">
                    {subtitle}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # GAUGE CHART
    # ========================================================
    
    @staticmethod
    def render_gauge_chart(value: float, label: str = "Confidence Score"):
        """
        Render a half-circle gauge chart
        
        Args:
            value: Value between 0-1
            label: Label text below gauge
        """
        percentage = value * 100
        angle = (percentage / 100) * 180
        
        if value >= 0.75:
            gauge_color = "#10B981"
        elif value < 0.5:
            gauge_color = "#EF4444"
        else:
            gauge_color = "#3B82F6"
        
        st.markdown(f"""
            <div class="section-card" style="padding: 16px;">
                <div class="section-card-title" style="margin-bottom: 12px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12 6 12 12 16 14"/>
                    </svg>
                    {label}
                </div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="position: relative; width: 140px; height: 70px; overflow: hidden;">
                        <div style="position: absolute; top: 0; left: 0; width: 140px; height: 140px; border-radius: 50%; background: var(--bg-elevated); clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%);"></div>
                        <div style="position: absolute; top: 0; left: 0; width: 140px; height: 140px; border-radius: 50%; background: conic-gradient({gauge_color} 0deg, {gauge_color} {angle}deg, transparent {angle}deg); clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%); filter: drop-shadow(0 0 10px {gauge_color}40);"></div>
                        <div style="position: absolute; top: 15px; left: 15px; width: 110px; height: 110px; border-radius: 50%; background: var(--bg-secondary); clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%);"></div>
                        <div style="position: absolute; top: 25px; left: 25px; width: 90px; height: 45px; background: var(--bg-primary); border-radius: 90px 90px 0 0; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; padding-bottom: 5px;">
                            <div style="font-size: 24px; font-weight: 800; color: {gauge_color}; text-shadow: 0 0 20px {gauge_color}50;">{percentage:.0f}%</div>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 11px; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.08em;">Model Confidence</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # SESSION STATS
    # ========================================================
    
    @staticmethod
    def render_session_stats(total: int, real_count: int, fake_count: int):
        """Render session statistics grid"""
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                <div class="section-card" style="text-align: center; padding: 12px 8px;">
                    <div style="font-size: 20px; font-weight: 700; color: var(--text-primary);">{total}</div>
                    <div style="font-size: 9px; color: var(--text-tertiary); text-transform: uppercase;">Total</div>
                </div>
                <div class="section-card" style="text-align: center; padding: 12px 8px;">
                    <div style="font-size: 20px; font-weight: 700; color: var(--success);">{real_count}</div>
                    <div style="font-size: 9px; color: var(--text-tertiary); text-transform: uppercase;">Real</div>
                </div>
                <div class="section-card" style="text-align: center; padding: 12px 8px;">
                    <div style="font-size: 20px; font-weight: 700; color: var(--danger);">{fake_count}</div>
                    <div style="font-size: 9px; color: var(--text-tertiary); text-transform: uppercase;">Fake</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # EMPTY STATES
    # ========================================================
    
    @staticmethod
    def render_empty_state():
        """Render empty state for main workspace when no image is loaded"""
        st.markdown("""
            <div style="text-align: center; padding: 80px 20px;">
                <div style="width: 80px; height: 80px; margin: 0 auto 24px; background: rgba(59, 130, 246, 0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                </div>
                <h3 style="font-size: 22px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">
                    No Image Selected
                </h3>
                <p style="font-size: 14px; color: var(--text-tertiary); max-width: 350px; margin: 0 auto 24px;">
                    Upload an image using the <strong style="color: var(--accent-primary);">+ Add Source</strong> button in the left panel to begin deepfake analysis
                </p>
                <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                    <div style="padding: 8px 16px; background: var(--bg-secondary); border-radius: 20px; font-size: 12px; color: var(--text-tertiary);">📷 JPG</div>
                    <div style="padding: 8px 16px; background: var(--bg-secondary); border-radius: 20px; font-size: 12px; color: var(--text-tertiary);">🖼️ PNG</div>
                    <div style="padding: 8px 16px; background: var(--bg-secondary); border-radius: 20px; font-size: 12px; color: var(--text-tertiary);">🌐 WEBP</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_report_empty_state():
        """Render empty state for report panel when no analysis done"""
        st.markdown("""
            <div class="section-card" style="text-align: center; padding: 32px 16px;">
                <div style="width: 56px; height: 56px; margin: 0 auto 16px; background: rgba(59, 130, 246, 0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                    </svg>
                </div>
                <div style="font-size: 14px; font-weight: 600; color: var(--text-primary); margin-bottom: 6px;">No Analysis Yet</div>
                <div style="font-size: 12px; color: var(--text-tertiary); line-height: 1.5;">
                    Upload an image and click<br/><strong style="color: var(--accent-primary);">Run Deep Analysis</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # PANEL HEADERS
    # ========================================================
    
    @staticmethod
    def render_panel_header(title: str, icon_svg: str = None):
        """Render a panel header with icon"""
        icon_html = icon_svg if icon_svg else ''
        st.markdown(f"""
            <div class="panel-title">
                {icon_html}
                {title}
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # TIPS SECTION
    # ========================================================
    
    @staticmethod
    def render_tips():
        """Render quick tips section"""
        st.markdown("""
            <div class="section-card-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="16" x2="12" y2="12"/>
                    <line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                Quick Tips
            </div>
            <div class="section-card" style="padding: 12px;">
                <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.7;">
                    <div style="display: flex; align-items: flex-start; gap: 6px; margin-bottom: 6px;">
                        <span style="color: var(--accent-primary);">•</span>
                        <span>Use high-resolution images for better accuracy</span>
                    </div>
                    <div style="display: flex; align-items: flex-start; gap: 6px; margin-bottom: 6px;">
                        <span style="color: var(--accent-primary);">•</span>
                        <span>Check heatmap for manipulation areas</span>
                    </div>
                    <div style="display: flex; align-items: flex-start; gap: 6px;">
                        <span style="color: var(--accent-primary);">•</span>
                        <span>Fourier analysis reveals frequency anomalies</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # FILE METADATA
    # ========================================================
    
    @staticmethod
    def render_file_metadata(filename: str, width: int, height: int, mode: str):
        """Render file metadata info"""
        st.markdown(f"""
            <div class="file-metadata">
                <div class="file-metadata-item">
                    <span class="file-metadata-label">📁</span>
                    <span class="file-metadata-value">{filename}</span>
                </div>
                <div class="file-metadata-item">
                    <span class="file-metadata-label">📐</span>
                    <span class="file-metadata-value">{width} × {height} px</span>
                </div>
                <div class="file-metadata-item">
                    <span class="file-metadata-label">🎨</span>
                    <span class="file-metadata-value">{mode}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # INTERPRETATION HINTS
    # ========================================================
    
    @staticmethod
    def render_heatmap_hint():
        """Render interpretation hint for heatmap analysis"""
        st.markdown("""
            <div style="margin-top: 12px; padding: 12px 16px; background: rgba(59, 130, 246, 0.1); border-radius: 10px; border-left: 4px solid var(--accent-primary);">
                <div style="font-size: 13px; font-weight: 600; color: var(--accent-primary); margin-bottom: 4px;">💡 How to interpret</div>
                <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
                    Red/yellow regions indicate areas where the AI model detected potential manipulation. 
                    Blue areas are considered authentic.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_fourier_magnitude_hint():
        """Render interpretation hint for Fourier magnitude spectrum"""
        st.markdown("""
            <div style="margin-top: 12px; padding: 12px 16px; background: rgba(139, 92, 246, 0.1); border-radius: 10px; border-left: 4px solid #8B5CF6;">
                <div style="font-size: 13px; font-weight: 600; color: #8B5CF6; margin-bottom: 4px;">📊 Frequency Domain Analysis</div>
                <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
                    The bright center represents low-frequency components (smooth areas). 
                    Patterns extending outward indicate high-frequency details and edges.
                    Anomalous patterns may suggest AI-generated artifacts.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_fourier_highpass_hint():
        """Render interpretation hint for high-pass filter"""
        st.markdown("""
            <div style="margin-top: 12px; padding: 12px 16px; background: rgba(249, 115, 22, 0.1); border-radius: 10px; border-left: 4px solid #F97316;">
                <div style="font-size: 13px; font-weight: 600; color: #F97316; margin-bottom: 4px;">🔍 High-Frequency Artifacts</div>
                <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
                    This view highlights high-frequency components by suppressing low frequencies.
                    Bright spots indicate sharp transitions and edges. Unnatural patterns may reveal manipulation.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================
    # ANALYSIS DETAILS EXPANDER
    # ========================================================
    
    @staticmethod
    def render_analysis_details(is_real: bool, model_name: str = "CNN-Custom-V1"):
        """Render analysis details in an expander"""
        status_color = "#10B981" if is_real else "#EF4444"
        assessment = "No artifacts detected" if is_real else "Artifacts found"
        
        with st.expander("📋 Analysis Details", expanded=False):
            st.markdown(f"""
                <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.8;">
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px 16px;">
                        <span style="color: var(--text-tertiary);">Model:</span>
                        <span style="color: var(--text-primary);">{model_name}</span>
                        <span style="color: var(--text-tertiary);">Processing:</span>
                        <span style="color: var(--text-primary);">~0.5s</span>
                        <span style="color: var(--text-tertiary);">Resolution:</span>
                        <span style="color: var(--text-primary);">224×224 normalized</span>
                        <span style="color: var(--text-tertiary);">Assessment:</span>
                        <span style="color: {status_color};">{assessment}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

