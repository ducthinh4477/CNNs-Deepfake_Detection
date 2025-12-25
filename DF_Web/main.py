"""
DeepScan - AI Deepfake Detection Application
=============================================
3-Pane Notebook Layout: Left Sidebar | Main Workspace | Right Report Panel

Author: HCMUTE Senior Design Team
"""

import streamlit as st

# CRITICAL: st.set_page_config MUST be the first Streamlit command
# This fixes the 10s white screen / scroll-to-render bug
st.set_page_config(
    layout="wide",
    page_title="DeepScan - AI Forensic Analysis",
    page_icon="🔬",
    initial_sidebar_state="collapsed"
)

from PIL import Image
import time
from datetime import datetime

from ai_logic import DeepfakeAI
from ui_components import NotebookUI, MODEL_INFO
from styles import get_notebook_layout_css


class DeepfakeApp:
    """
    DeepScan - Enterprise Deepfake Detection Application
    Layout: 3-Pane Notebook Style (Left Sidebar | Main Workspace | Right Report Panel)
    """
    
    def __init__(self):
        self.ui = NotebookUI()
        self.ai = DeepfakeAI(model_path="custom_cnn_cifake.pth")
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'left_sidebar_collapsed': False,
            'right_panel_collapsed': False,
            'is_processing': False,
            'scan_history': [],
            'results': None,
            'current_file': None,
            'model_opt': 'CNN Custom (CIFAKE)',
            'confidence_threshold': 0.5
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Main application entry point"""
        # Apply CSS (page_config already set at module level)
        self.ui.apply_custom_css()
        st.markdown(get_notebook_layout_css(), unsafe_allow_html=True)
        
        # Calculate column ratios based on collapse state
        col_ratios = self._get_column_ratios()
        
        # Create 3-column layout
        col_left, col_main, col_right = st.columns(col_ratios, gap="small")
        
        # Render each panel
        with col_left:
            uploaded_file = self._render_left_panel()
        
        with col_main:
            self._render_main_workspace(uploaded_file)
        
        with col_right:
            self._render_right_panel()
    
    def _get_column_ratios(self):
        """Calculate column ratios based on collapse state"""
        left_collapsed = st.session_state.left_sidebar_collapsed
        right_collapsed = st.session_state.right_panel_collapsed
        
        if left_collapsed and right_collapsed:
            return [0.5, 9, 0.5]
        elif left_collapsed:
            return [0.5, 6.5, 3]
        elif right_collapsed:
            return [2, 7.5, 0.5]
        else:
            return [2, 5, 3]
    
    # =========================================================
    # LEFT PANEL
    # =========================================================
    
    def _render_left_panel(self):
        """Render left sidebar panel"""
        if st.session_state.left_sidebar_collapsed:
            return self._render_collapsed_left_sidebar()
        else:
            return self._render_full_left_sidebar()
    
    def _render_collapsed_left_sidebar(self):
        """Render collapsed left sidebar with expand button"""
        st.markdown('<div class="collapsed-sidebar">', unsafe_allow_html=True)
        
        if st.button("▶", key="expand_left", help="Expand Sidebar"):
            st.session_state.left_sidebar_collapsed = False
            st.rerun()
        
        st.markdown("""
            <div style="margin-top: 16px;">
                <div class="icon-btn-vertical" title="Upload">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Return current file from session state
        return st.session_state.get('current_file')
    
    def _render_full_left_sidebar(self):
        """Render full left sidebar with all controls"""
        st.markdown('<div class="left-sidebar-panel">', unsafe_allow_html=True)
        
        # Header with collapse button
        col_title, col_btn = st.columns([4, 1])
        with col_title:
            st.markdown("""
                <div class="panel-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    DeepScan
                </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if st.button("◀", key="collapse_left", help="Collapse"):
                st.session_state.left_sidebar_collapsed = True
                st.rerun()
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Add Source Button
        uploaded_file = self.ui.render_add_source_btn()
        
        # Store uploaded file in session state
        if uploaded_file:
            st.session_state['current_file'] = uploaded_file
            self.ui.render_file_loaded_badge(uploaded_file.name)
        
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Model Selector
        model_opt = self.ui.render_model_selector()
        st.session_state['model_opt'] = model_opt
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Confidence Threshold Slider
        threshold = self.ui.render_threshold_slider()
        st.session_state['confidence_threshold'] = threshold
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return uploaded_file
    
    # =========================================================
    # MAIN WORKSPACE
    # =========================================================
    
    def _render_main_workspace(self, uploaded_file):
        """Render main workspace with image and analysis tabs"""
        st.markdown('<div class="main-workspace-panel">', unsafe_allow_html=True)
        
        # Workspace Header
        st.markdown("""
            <div class="workspace-header">
                <h1 class="workspace-title">Image Analysis</h1>
                <p class="workspace-subtitle">Deep learning powered forensic analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get file from session if not passed directly
        if not uploaded_file:
            uploaded_file = st.session_state.get('current_file')
        
        if uploaded_file:
            self._render_image_workspace(uploaded_file)
        else:
            self.ui.render_empty_state()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_image_workspace(self, uploaded_file):
        """Render workspace when image is loaded"""
        image = Image.open(uploaded_file).convert('RGB')
        filename = uploaded_file.name
        
        # 3 Tabs: Original, Heatmap, Fourier
        tab1, tab2, tab3 = st.tabs(["🖼️ Original", "🔥 Heatmap Analysis", "📊 Fourier Frequency"])
        
        # TAB 1: Original Image
        with tab1:
            st.markdown('<div class="image-viewport">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            self.ui.render_file_metadata(filename, image.size[0], image.size[1], image.mode)
        
        # TAB 2: Heatmap Analysis
        with tab2:
            st.markdown('<div class="image-viewport">', unsafe_allow_html=True)
            with st.spinner("Generating AI heatmap..."):
                heatmap_img = self.ai.generate_heatmap(image)
            st.image(heatmap_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            self.ui.render_heatmap_hint()
        
        # TAB 3: Fourier Frequency Analysis
        with tab3:
            fourier_view = st.radio(
                "View Mode",
                options=["Magnitude Spectrum", "High-Pass Filter"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown('<div class="image-viewport">', unsafe_allow_html=True)
            with st.spinner("Computing Fourier transform..."):
                if fourier_view == "Magnitude Spectrum":
                    fourier_img = self.ai.generate_fourier_analysis(image)
                else:
                    fourier_img = self.ai.generate_fourier_highpass(image)
            st.image(fourier_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if fourier_view == "Magnitude Spectrum":
                self.ui.render_fourier_magnitude_hint()
            else:
                self.ui.render_fourier_highpass_hint()
        
        # Analyze Button
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        analyze_btn = st.button(
            "🔬 Run Deep Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get('is_processing', False)
        )
        
        if analyze_btn:
            self._run_analysis(image, filename)
    
    def _run_analysis(self, image, filename):
        """Execute deepfake analysis with progress tracking"""
        st.session_state.is_processing = True
        
        # Progress display
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        steps = [
            ("Loading image...", 20),
            ("Preprocessing...", 40),
            ("Running CNN analysis...", 60),
            ("Generating predictions...", 80),
            ("Compiling results...", 100)
        ]
        
        for step_text, progress in steps:
            progress_text.markdown(f"""
                <div style="text-align: center; font-size: 13px; color: var(--text-secondary);">
                    {step_text}
                </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(progress)
            time.sleep(0.2)
        
        # Run actual prediction
        label, conf, _ = self.ai.predict(image)
        
        # Store results in session state
        st.session_state['results'] = (label, conf, filename)
        
        # Add to history
        st.session_state.scan_history.append({
            'filename': filename,
            'label': label,
            'confidence': conf,
            'time': datetime.now().strftime("%H:%M")
        })
        
        # Cleanup
        progress_text.empty()
        progress_bar.empty()
        st.session_state.is_processing = False
        
        # Rerun to update right panel
        st.rerun()
    
    # =========================================================
    # RIGHT PANEL
    # =========================================================
    
    def _render_right_panel(self):
        """Render right report panel"""
        if st.session_state.right_panel_collapsed:
            self._render_collapsed_right_panel()
        else:
            self._render_full_right_panel()
    
    def _render_collapsed_right_panel(self):
        """Render collapsed right panel with expand button"""
        st.markdown('<div class="collapsed-sidebar">', unsafe_allow_html=True)
        
        if st.button("◀", key="expand_right", help="Expand Report"):
            st.session_state.right_panel_collapsed = False
            st.rerun()
        
        # Show mini result indicator if results exist
        if st.session_state.get('results'):
            label, conf, _ = st.session_state['results']
            status_color = "#10B981" if label == 1 else "#EF4444"
            st.markdown(f"""
                <div style="margin-top: 16px; text-align: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{status_color}" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                        <polyline points="22 4 12 14.01 9 11.01"/>
                    </svg>
                    <div style="font-size: 10px; color: {status_color}; font-weight: 600; margin-top: 4px;">{conf:.0%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_full_right_panel(self):
        """Render full right report panel with scrollable content"""
        st.markdown('<div class="right-report-panel">', unsafe_allow_html=True)
        
        # Header with collapse button
        col_title, col_btn = st.columns([4, 1])
        with col_title:
            st.markdown("""
                <div class="panel-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                        <line x1="12" y1="20" x2="12" y2="10"/>
                        <line x1="18" y1="20" x2="18" y2="4"/>
                        <line x1="6" y1="20" x2="6" y2="16"/>
                    </svg>
                    Analysis Report
                </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if st.button("▶", key="collapse_right", help="Collapse"):
                st.session_state.right_panel_collapsed = True
                st.rerun()
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Results or Empty State
        if st.session_state.get('results'):
            self._render_analysis_results()
        else:
            self.ui.render_report_empty_state()
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Session Stats
        self._render_session_stats()
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Tips
        self.ui.render_tips()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_analysis_results(self):
        """Render analysis results with gauge, metrics, and details"""
        label, conf, filename = st.session_state['results']
        is_real = label == 1
        
        # Result Badge
        self.ui.render_result_badge(is_real, conf)
        
        # Gauge Chart
        self.ui.render_gauge_chart(conf)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Metrics Grid
        risk_level = "Low" if is_real else "High"
        risk_color = "#10B981" if is_real else "#EF4444"
        trust_score = "High" if conf >= 0.8 else "Medium" if conf >= 0.6 else "Low"
        trust_color = "#10B981" if conf >= 0.8 else "#F59E0B" if conf >= 0.6 else "#EF4444"
        
        col1, col2 = st.columns(2)
        with col1:
            self.ui.render_metric_card(risk_level, "Risk Level", risk_color)
        with col2:
            self.ui.render_metric_card(trust_score, "Trust Score", trust_color)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # File Info
        st.markdown(f"""
            <div class="section-card" style="padding: 12px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span style="font-size: 12px; font-weight: 600; color: var(--text-primary);">File Analyzed</span>
                </div>
                <div style="font-size: 11px; color: var(--text-tertiary); word-break: break-all;">{filename}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Analysis Details
        self.ui.render_analysis_details(is_real)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Action Buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
                st.session_state['results'] = None
                st.rerun()
        with col_b:
            if st.button("📤 Export", use_container_width=True, key="export_btn"):
                st.toast("📋 Report copied to clipboard!", icon="✅")
    
    def _render_session_stats(self):
        """Render session statistics section"""
        st.markdown("""
            <div class="section-card-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2">
                    <path d="M3 3v18h18"/>
                    <path d="m19 9-5 5-4-4-3 3"/>
                </svg>
                Session Stats
            </div>
        """, unsafe_allow_html=True)
        
        history = st.session_state.get('scan_history', [])
        total = len(history)
        fake_count = sum(1 for item in history if item.get('label') == 0)
        real_count = total - fake_count
        
        self.ui.render_session_stats(total, real_count, fake_count)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Model Accuracy Bar - SYNC with selected model
        current_model = st.session_state.get('model_opt', 'CNN Custom (CIFAKE)')
        model_data = MODEL_INFO.get(current_model, MODEL_INFO['CNN Custom (CIFAKE)'])
        accuracy_value = model_data['accuracy_value']
        accuracy_str = model_data['accuracy']
        accuracy_color = model_data['color']
        
        st.markdown(f"""
            <div class="section-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 11px; color: var(--text-tertiary);">Model Accuracy</span>
                    <span style="font-size: 13px; font-weight: 700; color: {accuracy_color};">{accuracy_str}</span>
                </div>
                <div style="height: 6px; background: var(--bg-elevated); border-radius: 3px; overflow: hidden;">
                    <div style="height: 100%; width: {accuracy_value}%; background: linear-gradient(90deg, {accuracy_color}, {accuracy_color}dd); border-radius: 3px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = DeepfakeApp()
    app.run()
