"""
DeepScan Styles Module
======================
Centralized CSS styling for the DeepScan Deepfake Detection application.
This module separates all CSS from Python logic for better maintainability.

Author: HCMUTE Senior Design Team
Design System: Modern Dark / Deep Slate Professional
"""


def get_base_css() -> str:
    """
    Return base CSS styling for the application.
    Includes: CSS variables, global styles, typography, components, and responsive design.
    
    Returns:
        str: Complete CSS wrapped in <style> tags
    """
    return """
    <style>
        /* ============================================
           FONTS - SF Pro inspired
        ============================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ============================================
           DARK THEME FOUNDATION - Professional Color System
        ============================================ */
        :root {
            /* ===== BACKGROUND COLORS ===== */
            --bg-primary: #0F172A;      /* Deep Blue/Grey - Main background */
            --bg-secondary: #1E293B;    /* Surface/Card/Sidebar */
            --bg-elevated: #334155;     /* Hover state/Input fields */
            
            /* ===== TEXT COLORS ===== */
            --text-primary: #FFFFFF;    /* Headings/Important text */
            --text-secondary: #E2E8F0;  /* Body content/Descriptions */
            --text-tertiary: #CBD5E1;   /* Muted text/Labels */
            
            /* ===== ACCENT COLORS ===== */
            --accent-primary: #3B82F6;  /* Blue - Primary actions */
            --accent-hover: #2563EB;    /* Blue darker - Hover state */
            
            /* ===== STATUS COLORS ===== */
            --success: #10B981;         /* Real/Authentic detected */
            --danger: #EF4444;          /* Fake/Manipulated detected */
            
            /* ===== SPACING SYSTEM ===== */
            --spacing-xs: 8px;
            --spacing-sm: 12px;
            --spacing-md: 24px;
            --spacing-lg: 32px;
            --spacing-xl: 48px;
            
            /* ===== BORDER RADIUS ===== */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            
            /* ===== TYPOGRAPHY SCALE ===== */
            --font-h1-size: 32px;
            --font-h1-weight: 700;
            --font-h1-line: 1.2;
            --font-h2-size: 24px;
            --font-h2-weight: 600;
            --font-h2-line: 1.3;
            --font-body-size: 16px;
            --font-body-weight: 400;
            --font-body-line: 1.6;
            --font-label-size: 14px;
            --font-label-weight: 400;
            --font-label-line: 1.5;
        }
        
        /* ============================================
           GLOBAL STYLES - Enterprise Foundation
           OPTIMIZED: min-height + overflow-y to fix rendering delay
        ============================================ */
        html, body {
            min-height: 100vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            background: #0F172A !important;
        }
        
        .main {
            background: #0F172A !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            min-height: 100vh !important;
            overflow-y: auto !important;
        }
        
        body {
            font-family: 'Inter', sans-serif !important;
            font-size: var(--font-body-size) !important;
            font-weight: var(--font-body-weight) !important;
            line-height: var(--font-body-line) !important;
            background: var(--bg-primary) !important;
        }
        
        .main > div,
        .main .block-container,
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"] {
            background: var(--bg-primary) !important;
            min-height: 100vh !important;
        }
        
        /* Removed heavy pseudo-element gradient - causes render delay */
        
        .block-container {
            padding: 0 !important;
            padding-top: 1rem !important;
            max-width: 100% !important;
            width: 100% !important;
            position: relative;
            z-index: 1;
            background: var(--bg-primary) !important;
            min-height: 100vh !important;
            overflow-y: auto !important;
        }
        
        .main .stMainBlockContainer {
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        [data-testid="stAppViewContainer"] > section {
            padding: 0 !important;
        }
        
        section[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"],
        div[data-testid="column"],
        .element-container {
            background: transparent !important;
        }
        
        /* ============================================
           TYPOGRAPHY HIERARCHY
        ============================================ */
        h1 {
            font-family: 'Inter', sans-serif !important;
            font-size: var(--font-h1-size) !important;
            font-weight: var(--font-h1-weight) !important;
            line-height: var(--font-h1-line) !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.025em;
        }
        
        h2 {
            font-family: 'Inter', sans-serif !important;
            font-size: var(--font-h2-size) !important;
            font-weight: var(--font-h2-weight) !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.025em;
        }
        
        h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.025em;
        }
        
        p, span, div, label {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* ============================================
           SIDEBAR - Enterprise Navigation
        ============================================ */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--bg-elevated) !important;
        }
        
        [data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(180deg, rgba(88, 166, 255, 0.03) 0%, transparent 30%);
            pointer-events: none;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding: var(--spacing-md) var(--spacing-sm) !important;
            background: transparent !important;
        }
        
        /* ============================================
           SIDEBAR SECTIONS & LOGO
        ============================================ */
        .sidebar-section {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-elevated);
            border-radius: var(--radius-md);
            padding: var(--spacing-sm) var(--spacing-md);
            margin-bottom: var(--spacing-sm);
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .sidebar-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        
        .sidebar-section:hover {
            border-color: var(--accent-primary);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(59, 130, 246, 0.1);
        }
        
        .sidebar-section:hover::before {
            opacity: 1;
        }
        
        .sidebar-section-header {
            display: flex;
            align-items: center;
            gap: var(--spacing-xs);
            margin-bottom: var(--spacing-sm);
            padding-bottom: var(--spacing-xs);
            border-bottom: 1px solid var(--bg-elevated);
        }
        
        .sidebar-section-icon {
            width: 24px;
            height: 24px;
            border-radius: var(--radius-sm);
            background: rgba(59, 130, 246, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-primary);
            flex-shrink: 0;
        }
        
        .sidebar-section-title {
            font-size: var(--font-label-size);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-primary);
        }
        
        .sidebar-logo {
            padding: var(--spacing-xs) 0 var(--spacing-md) 0;
            margin-bottom: var(--spacing-md);
            border-bottom: 1px solid var(--bg-elevated);
            position: relative;
        }
        
        .sidebar-logo::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 60px;
            height: 2px;
            background: var(--accent-primary);
            border-radius: 999px;
        }
        
        .sidebar-logo-text {
            font-size: 32px;
            font-weight: 700;
            color: #3B82F6;
            letter-spacing: -0.03em;
        }
        
        .sidebar-logo-subtitle {
            font-size: 14px;
            color: #CBD5E1;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.25rem;
        }
        
        /* ============================================
           CARD SYSTEM
        ============================================ */
        .content-card,
        .custom-card {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-elevated);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
        }
        
        .content-card:hover,
        .custom-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-sm);
            padding-bottom: var(--spacing-xs);
            border-bottom: 1px solid var(--bg-elevated);
        }
        
        .card-title {
            font-size: var(--font-label-size);
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            display: flex;
            align-items: center;
            gap: var(--spacing-xs);
        }
        
        /* ============================================
           METRIC CARDS
        ============================================ */
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            position: relative;
            cursor: default;
            transition: all 0.2s ease;
            overflow: hidden;
        }
        
        .metric-card:hover {
            border-color: var(--accent-primary);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(59, 130, 246, 0.1);
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1;
            margin-bottom: 0.375rem;
        }
        
        .metric-value.success {
            color: var(--success);
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        }
        
        .metric-value.danger {
            color: var(--danger);
            text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        }
        
        .metric-label {
            font-size: 14px;
            font-weight: 600;
            color: #CBD5E1;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        /* ============================================
           GAUGE CHART
        ============================================ */
        .gauge-container {
            position: relative;
            width: 100%;
            padding: 1rem 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .gauge-chart {
            position: relative;
            width: 140px;
            height: 70px;
            overflow: hidden;
        }
        
        .gauge-background {
            position: absolute;
            top: 0;
            left: 0;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: conic-gradient(var(--bg-elevated) 0deg, var(--bg-elevated) 360deg);
            clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%);
        }
        
        .gauge-fill {
            position: absolute;
            top: 0;
            left: 0;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%);
            transition: all 1s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .gauge-fill.success {
            filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.4));
        }
        
        .gauge-fill.danger {
            filter: drop-shadow(0 0 10px rgba(239, 68, 68, 0.4));
        }
        
        .gauge-inner {
            position: absolute;
            top: 15px;
            left: 15px;
            width: 110px;
            height: 110px;
            border-radius: 50%;
            background: var(--bg-secondary);
            clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%);
        }
        
        .gauge-center {
            position: absolute;
            top: 25px;
            left: 25px;
            width: 90px;
            height: 45px;
            background: var(--bg-elevated);
            border-radius: 90px 90px 0 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            padding-bottom: 5px;
        }
        
        .gauge-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1;
        }
        
        .gauge-value.success {
            color: var(--success);
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        }
        
        .gauge-value.danger {
            color: var(--danger);
            text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        }
        
        .gauge-label {
            font-size: 0.6rem;
            font-weight: 600;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.75rem;
        }
        
        /* ============================================
           STATUS BADGES
        ============================================ */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            transition: all 0.2s ease;
        }
        
        .status-badge.authentic {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .status-badge.fabricated {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.625rem;
            position: relative;
        }
        
        .status-indicator.authentic {
            background: var(--success);
            box-shadow: 0 0 6px var(--success);
        }
        
        .status-indicator.fabricated {
            background: var(--danger);
            box-shadow: 0 0 6px var(--danger);
        }
        
        /* ============================================
           PROGRESS BAR
        ============================================ */
        .progress-container {
            margin: 1rem 0;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .progress-label {
            font-size: 0.65rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .progress-value {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .progress-track {
            height: 6px;
            background: var(--bg-elevated);
            border-radius: 999px;
            overflow: hidden;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .progress-fill.success {
            background: linear-gradient(90deg, var(--success), #10B981);
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        
        .progress-fill.danger {
            background: linear-gradient(90deg, var(--danger), #EF4444);
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }
        
        /* ============================================
           PROCESSING STATE
        ============================================ */
        .processing-container {
            background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--bg-secondary) 100%);
            border: 1px solid var(--bg-elevated);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .processing-spinner {
            width: 48px;
            height: 48px;
            border: 3px solid var(--bg-elevated);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1.25rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .processing-text {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .processing-subtext {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.375rem;
        }
        
        /* ============================================
           EMPTY STATE - Optimized (removed heavy animation)
        ============================================ */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: var(--bg-secondary);
            border: 1px solid var(--bg-elevated);
            border-radius: 16px;
            position: relative;
        }
        
        /* Removed pulse-glow animation - caused render delay */
        
        .empty-state-title-large {
            font-size: 2rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 1rem;
            position: relative;
            letter-spacing: -0.02em;
        }
        
        .empty-state-description-enhanced {
            font-size: 1rem;
            color: var(--text-secondary);
            max-width: 400px;
            margin: 0 auto 2rem;
            position: relative;
            line-height: 1.6;
        }
        
        /* ============================================
           BUTTON STYLES
        ============================================ */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: var(--font-label-size) !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            transition: all 200ms ease !important;
            background: transparent !important;
            border: 1px solid var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            cursor: pointer !important;
        }
        
        .stButton > button:hover {
            background: var(--bg-elevated) !important;
            border-color: var(--text-tertiary) !important;
            transform: scale(1.02) !important;
            filter: brightness(1.1) !important;
        }
        
        .stButton > button:active {
            transform: scale(0.98) !important;
        }
        
        .stButton > button:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
        }
        
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: var(--accent-primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }
        
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {
            filter: brightness(1.1) !important;
            transform: scale(1.02) !important;
        }
        
        /* ============================================
           FILE UPLOADER
        ============================================ */
        [data-testid="stFileUploader"] {
            background: #0F172A !important;
            border: 2px dashed var(--bg-elevated) !important;
            border-radius: 8px !important;
            transition: all 200ms ease !important;
            padding: 24px !important;
            text-align: center !important;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent-primary) !important;
            background: #1E293B !important;
        }
        
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] [role="button"] {
            background: var(--accent-primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            transition: all 200ms ease !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] [role="button"]:hover {
            filter: brightness(1.1) !important;
            transform: scale(1.02) !important;
        }
        
        [data-testid="stFileUploader"] section {
            padding: 24px !important;
        }
        
        [data-testid="stFileUploader"] svg {
            width: 48px !important;
            height: 48px !important;
            color: var(--accent-primary) !important;
        }
        
        [data-testid="stFileUploader"] small {
            color: var(--text-primary) !important;
            font-size: 14px !important;
            display: block !important;
            margin-top: 8px !important;
        }
        
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span {
            color: var(--text-primary) !important;
        }
        
        /* ============================================
           TABS
        ============================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0 !important;
            background: var(--bg-primary) !important;
            border-bottom: 1px solid var(--bg-elevated) !important;
            padding: 0 !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            padding: 12px 24px !important;
            background: transparent !important;
            color: var(--text-tertiary) !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            transition: all 200ms ease !important;
            border-radius: 0 !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary) !important;
            background: transparent !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
            background: transparent !important;
            border-bottom: 2px solid var(--accent-primary) !important;
            font-weight: 600 !important;
        }
        
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 24px !important;
        }
        
        /* ============================================
           SELECT BOX & SLIDER
        ============================================ */
        [data-testid="stSelectbox"] > div > div {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
            transition: all 200ms ease !important;
        }
        
        [data-testid="stSelectbox"] > div > div:hover {
            border-color: var(--accent-primary) !important;
        }
        
        [data-testid="stSelectbox"] > div > div:focus-within {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        [data-testid="stSelectbox"] label {
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: var(--font-label-size) !important;
        }
        
        [data-baseweb="popover"] {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--bg-elevated) !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        }
        
        [data-baseweb="menu"] ul {
            background: #1E293B !important;
        }
        
        [data-baseweb="menu"] li {
            background: #1E293B !important;
            color: #F8FAFC !important;
        }
        
        [data-baseweb="menu"] li:hover {
            background: #334155 !important;
        }
        
        .stSlider {
            padding-bottom: var(--spacing-lg) !important;
        }
        
        .stSlider > div > div > div {
            background: var(--accent-primary) !important;
        }
        
        .stSlider label {
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
        }
        
        .stSlider [role="slider"] {
            background: var(--accent-primary) !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2), 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
            width: 20px !important;
            height: 20px !important;
            transition: all 200ms ease !important;
        }
        
        .stSlider [role="slider"]:hover {
            box-shadow: 0 0 0 6px rgba(59, 130, 246, 0.3), 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
            transform: scale(1.15) !important;
        }
        
        .stSlider > div > div > div > div {
            background: var(--bg-elevated) !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        
        .stSlider > div > div > div > div > div {
            background: var(--accent-primary) !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        
        /* ============================================
           ALERTS & MESSAGES
        ============================================ */
        .stAlert {
            border-radius: 16px !important;
            border: 1px solid var(--bg-elevated) !important;
            background: var(--bg-elevated) !important;
        }
        
        .info-banner {
            background: var(--bg-elevated);
            border: 1px solid var(--bg-elevated);
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            margin-top: var(--spacing-md);
        }
        
        .warning-banner {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: var(--radius-md);
            padding: var(--spacing-sm) var(--spacing-md);
            margin-top: var(--spacing-sm);
            font-size: 14px;
            color: #3B82F6;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        
        /* ============================================
           HISTORY ITEMS
        ============================================ */
        .history-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--bg-primary);
            border: 1px solid transparent;
            border-radius: var(--radius-md);
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .history-item:hover {
            background: var(--bg-elevated);
            border-color: var(--bg-elevated);
            transform: translateX(4px);
        }
        
        .history-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        
        .history-status.authentic {
            background: var(--success);
            box-shadow: 0 0 6px var(--success);
        }
        
        .history-status.fabricated {
            background: var(--danger);
            box-shadow: 0 0 6px var(--danger);
        }
        
        .history-name {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .history-time {
            font-size: 0.6rem;
            color: var(--text-tertiary);
            margin-top: 0.125rem;
        }
        
        /* ============================================
           SCROLLBAR - Global Custom Scrollbar
        ============================================ */
        
        /* Apply to ALL elements globally */
        *, *::before, *::after {
            scrollbar-width: thin;
            scrollbar-color: #475569 #1E293B;
        }
        
        /* Webkit browsers (Chrome, Safari, Edge) */
        *::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        *::-webkit-scrollbar-track {
            background: #1E293B;
            border-radius: 5px;
        }
        
        *::-webkit-scrollbar-thumb {
            background: #475569;
            border-radius: 5px;
            border: 2px solid #1E293B;
        }
        
        *::-webkit-scrollbar-thumb:hover {
            background: #64748B;
        }
        
        /* Main container scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1E293B;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #475569;
            border-radius: 5px;
            border: 2px solid #1E293B;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #64748B;
        }
        
        /* ============================================
           HIDE STREAMLIT BRANDING
        ============================================ */
        #MainMenu {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: #0F172A !important;
        }
        
        /* ============================================
           DARK BACKGROUND ENFORCEMENT
        ============================================ */
        [data-testid="stApp"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stImageContainer"],
        [data-testid="stSpinner"] {
            background: transparent !important;
        }
        
        * {
            scrollbar-color: #475569 #1E293B !important;
        }
        
        input[type="text"],
        input[type="number"],
        textarea {
            background: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
        }
        
        .stApp > header,
        .stApp > div {
            background: #0F172A !important;
        }
        
        /* ============================================
           RESPONSIVE DESIGN
        ============================================ */
        @media (max-width: 1024px) {
            .block-container {
                padding: 1rem 1.5rem !important;
            }
            
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            
            [data-testid="column"] {
                width: 100% !important;
                flex: none !important;
            }
            
            .gauge-chart {
                width: 120px;
                height: 60px;
            }
        }
        
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem !important;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
            
            .gauge-chart {
                width: 100px;
                height: 50px;
            }
            
            .empty-state-title-large {
                font-size: 1.5rem;
            }
            
            .stButton > button {
                padding: 0.75rem 1rem !important;
                font-size: 0.9rem !important;
            }
        }
        
        @media (max-width: 480px) {
            .sidebar-logo-text {
                font-size: 1rem;
            }
            
            .empty-state {
                padding: 2rem 1rem;
            }
            
            .processing-container {
                padding: 2rem 1rem;
            }
        }
    </style>
    """


def get_notebook_layout_css() -> str:
    """
    Return CSS for 3-Pane Notebook Layout.
    This layout provides: Left Sidebar | Main Workspace | Right Report Panel
    
    Returns:
        str: Layout-specific CSS wrapped in <style> tags
    """
    return """
    <style>
        /* ============================================
           3-PANE NOTEBOOK LAYOUT - Native App Style
        ============================================ */
        
        /* Remove ALL default Streamlit padding */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Hide default sidebar completely */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Full height columns */
        [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            align-items: stretch !important;
        }
        
        [data-testid="column"] {
            padding: 0 !important;
        }
        
        /* ============================================
           LEFT SIDEBAR PANEL
        ============================================ */
        .left-sidebar-panel {
            background: var(--bg-secondary);
            border-right: 1px solid var(--bg-elevated);
            height: calc(100vh - 0px);
            padding: 20px;
            overflow-y: auto;
            position: sticky;
            top: 0;
        }
        
        .left-sidebar-panel::-webkit-scrollbar {
            width: 4px;
        }
        
        .left-sidebar-panel::-webkit-scrollbar-thumb {
            background: var(--bg-elevated);
            border-radius: 4px;
        }
        
        .left-sidebar-collapsed {
            background: var(--bg-secondary);
            border-right: 1px solid var(--bg-elevated);
            height: calc(100vh - 0px);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 16px 8px;
        }
        
        /* ============================================
           ADD SOURCE BUTTON - Z-Index Trick for Click
        ============================================ */
        .add-source-container {
            position: relative;
            margin-bottom: 0;
        }
        
        .add-source-btn-visual {
            background: rgba(59, 130, 246, 0.1);
            border: 2px dashed rgba(59, 130, 246, 0.4);
            border-radius: 16px;
            padding: 24px 16px;
            text-align: center;
            pointer-events: none;
        }
        
        .add-source-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: rgba(59, 130, 246, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px;
        }
        
        .add-source-icon svg {
            color: var(--accent-primary);
        }
        
        .add-source-text {
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-primary);
            margin-bottom: 4px;
        }
        
        .add-source-hint {
            font-size: 11px;
            color: var(--text-tertiary);
        }
        
        /* Z-INDEX TRICK: Make file uploader overlay the custom button */
        .left-sidebar-panel [data-testid="stFileUploader"] {
            position: relative !important;
            margin-top: -120px !important;
            z-index: 10 !important;
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        
        .left-sidebar-panel [data-testid="stFileUploader"] > div {
            background: transparent !important;
            border: none !important;
        }
        
        .left-sidebar-panel [data-testid="stFileUploader"] section {
            background: transparent !important;
            border: 2px dashed transparent !important;
            border-radius: 16px !important;
            padding: 24px 16px !important;
            min-height: 110px !important;
            cursor: pointer !important;
            transition: all 300ms ease !important;
        }
        
        .left-sidebar-panel [data-testid="stFileUploader"] section:hover {
            background: rgba(59, 130, 246, 0.08) !important;
            border-color: rgba(59, 130, 246, 0.5) !important;
        }
        
        /* Hide the default uploader text/icons - we show our custom one underneath */
        .left-sidebar-panel [data-testid="stFileUploader"] section > div:first-child {
            opacity: 0 !important;
        }
        
        .left-sidebar-panel [data-testid="stFileUploader"] section span,
        .left-sidebar-panel [data-testid="stFileUploader"] section small,
        .left-sidebar-panel [data-testid="stFileUploader"] section svg {
            opacity: 0 !important;
        }
        
        /* Show uploaded file info when file is selected */
        .left-sidebar-panel [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            opacity: 1 !important;
            background: rgba(34, 197, 94, 0.1) !important;
            border: 1px solid rgba(34, 197, 94, 0.3) !important;
            border-radius: 8px !important;
            margin-top: 8px !important;
        }
        
        /* Widget styling */
        .widget-group {
            background: var(--bg-primary);
            border: 1px solid var(--bg-elevated);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }
        
        .widget-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        /* ============================================
           MAIN WORKSPACE PANEL
        ============================================ */
        .main-workspace-panel {
            background: var(--bg-primary);
            min-height: calc(100vh - 0px);
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
        }
        
        /* Workspace Header */
        .workspace-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--bg-elevated);
        }
        
        .workspace-title {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
        }
        
        .workspace-subtitle {
            font-size: 13px;
            color: var(--text-tertiary);
            margin-top: 4px;
        }
        
        /* View Options Pills/Tabs */
        .view-options {
            display: flex;
            gap: 4px;
            background: var(--bg-secondary);
            padding: 4px;
            border-radius: 10px;
        }
        
        .view-pill {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-tertiary);
            cursor: pointer;
            transition: all 200ms ease;
            border: none;
            background: transparent;
        }
        
        .view-pill:hover {
            color: var(--text-secondary);
            background: var(--bg-elevated);
        }
        
        .view-pill.active {
            background: var(--accent-primary);
            color: white;
        }
        
        /* ============================================
           IMAGE VIEWPORT - CRITICAL STYLING
        ============================================ */
        .image-viewport-container {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--bg-elevated);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        
        .image-viewport {
            flex: 1;
            background: var(--bg-primary);
            border: 1px solid var(--bg-elevated);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            max-height: 60vh;
            min-height: 300px;
        }
        
        .image-viewport img {
            max-width: 100% !important;
            max-height: 60vh !important;
            height: auto !important;
            width: auto !important;
            object-fit: contain !important;
        }
        
        /* Force Streamlit images to respect viewport */
        .image-viewport [data-testid="stImage"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            max-height: 60vh !important;
        }
        
        .image-viewport [data-testid="stImage"] img {
            max-height: 55vh !important;
            object-fit: contain !important;
        }
        
        /* ============================================
           RUN ANALYSIS BUTTON - Horizontal Bar
        ============================================ */
        .analyze-btn-container {
            margin-top: 20px;
            padding: 0 20%;
        }
        
        .analyze-btn-container button {
            width: 100% !important;
            background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 16px 32px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            color: white !important;
            cursor: pointer !important;
            transition: all 300ms ease !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        }
        
        .analyze-btn-container button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        }
        
        .analyze-btn-container button:active {
            transform: translateY(0) !important;
        }
        
        /* ============================================
           RIGHT REPORT PANEL - Scrollable
        ============================================ */
        .right-report-panel {
            background: var(--bg-secondary);
            border-left: 1px solid var(--bg-elevated);
            height: calc(100vh - 0px);
            padding: 20px;
            overflow-y: auto !important;
            position: sticky;
            top: 0;
        }
        
        .right-report-panel::-webkit-scrollbar {
            width: 4px;
        }
        
        .right-report-panel::-webkit-scrollbar-thumb {
            background: var(--bg-elevated);
            border-radius: 4px;
        }
        
        .right-panel-collapsed {
            background: var(--bg-secondary);
            border-left: 1px solid var(--bg-elevated);
            height: calc(100vh - 0px);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 16px 8px;
        }
        
        /* ============================================
           PANEL HEADERS
        ============================================ */
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 12px;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--bg-elevated);
        }
        
        .panel-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* ============================================
           SECTION CARDS
        ============================================ */
        .section-card {
            background: var(--bg-primary);
            border: 1px solid var(--bg-elevated);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }
        
        .section-card-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        /* ============================================
           ICON BUTTONS
        ============================================ */
        .icon-btn-vertical {
            background: transparent;
            border: 1px solid var(--bg-elevated);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            cursor: pointer;
            color: var(--text-tertiary);
            transition: all 200ms ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .icon-btn-vertical:hover {
            background: var(--bg-elevated);
            color: var(--accent-primary);
            border-color: var(--accent-primary);
        }
        
        /* ============================================
           FILE METADATA
        ============================================ */
        .file-metadata {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            padding: 12px;
            background: var(--bg-primary);
            border-radius: 8px;
            margin-top: 12px;
            font-size: 13px;
        }
        
        .file-metadata-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .file-metadata-label {
            color: var(--text-tertiary);
        }
        
        .file-metadata-value {
            color: var(--text-primary);
            font-weight: 600;
        }
        
        /* ============================================
           TABS OVERRIDE FOR MAIN WORKSPACE
        ============================================ */
        .main-workspace-panel [data-testid="stTabs"] {
            background: transparent;
        }
        
        .main-workspace-panel [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: var(--bg-secondary) !important;
            padding: 4px !important;
            border-radius: 10px !important;
            gap: 4px !important;
            border: none !important;
        }
        
        .main-workspace-panel [data-testid="stTabs"] button[data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: var(--text-tertiary) !important;
            border: none !important;
        }
        
        .main-workspace-panel [data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--accent-primary) !important;
            color: white !important;
        }
        
        .main-workspace-panel [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        
        .main-workspace-panel [data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }
        
        /* ============================================
           RESPONSIVE
        ============================================ */
        @media (max-width: 1200px) {
            .left-sidebar-panel,
            .right-report-panel {
                padding: 12px;
            }
            
            .main-workspace-panel {
                padding: 16px;
            }
            
            .analyze-btn-container {
                padding: 0 10%;
            }
        }
    </style>
    """


def get_all_css() -> str:
    """
    Return all CSS combined: base styles + notebook layout.
    Use this for convenience when you need all styles at once.
    
    Returns:
        str: Combined CSS from get_base_css() and get_notebook_layout_css()
    """
    return get_base_css() + get_notebook_layout_css()
