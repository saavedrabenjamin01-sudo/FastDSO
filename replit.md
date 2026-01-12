# PredDist - Distribution Prediction Portal

## Overview
PredDist is a Flask-based web application designed to optimize product distribution. It enables users to analyze sales data, manage inventory across stores and distribution centers, and generate precise purchase and distribution forecasts. The system aims to provide actionable insights for inventory optimization, sales trend analysis, and efficient supply chain management.

## User Preferences
I want the agent to use clear and concise language.
I prefer iterative development with regular updates.
Please ask for confirmation before making any major changes to the codebase or architectural decisions.
I expect detailed explanations for complex logic or significant modifications.
Focus on delivering robust and maintainable code.
Do not make changes to the folder `Excel tipo/`.

## System Architecture

### Technology Stack
- **Backend**: Python 3.11, Flask 3.0.3
- **Database**: SQLite (app.db)
- **ORM**: Flask-SQLAlchemy
- **Authentication**: Flask-Login
- **Data Processing**: Pandas, OpenPyXL

### Core Architectural Patterns
- **Modular Monolith**: Application is structured into logical modules (e.g., Sales, Stock, Predictions, Forecast) within a single Flask application.
- **Background Job Processing**: Utilizes `ThreadPoolExecutor` for asynchronous processing of large file uploads and other compute-intensive tasks, preventing timeouts and enhancing user experience.
- **Role-Based Access Control (RBAC)**: Fine-grained permission system managing user access to features and data based on assigned roles (Admin, Management, CategoryManager, WarehouseOps, Viewer).
- **Simulation Mode**: Allows users to perform "what-if" analysis and test parameters without persisting results to the database, utilizing Flask session for temporary data storage.
- **Audit Trail**: Comprehensive logging of user actions and system events for accountability and debugging.

### UI/UX Decisions
- **Dashboard**: Modern card-based layout with KPIs, side-by-side tables, and a topbar with global search.
- **Responsive Design**: Templates are designed for a professional and intuitive user experience with paginated tables, search, filter chips, and improved exports.
- **Unified Processing Modal**: Animated SVG icons and real-time progress bars for background operations.
- **Branding**: Login page features corporate branding elements.
- **Interactive Visualizations**: Plotly charts for advanced forecasting modules.

### Key Features
- **Data Management**: Upload and manage sales data, store stock, and distribution center stock with background processing.
- **Distribution Predictions**: Generate product distribution suggestions based on moving averages and historical data.
- **Advanced Forecasting (Forecast Compra V2)**: Comprehensive purchase forecasting incorporating lead time, safety stock, coverage, and various demand methods.
- **Inventory Optimization**:
    - **Store-to-Store Rebalancing**: Suggests optimal stock transfers between stores based on Weeks of Cover (WOC) and sales velocity.
    - **Stock-out Replenishment (BREAK_REPLENISH)**: Automatically identifies and suggests replenishment for out-of-stock SKU-Store pairs with historical demand.
    - **Slow Stock & Smart Reallocation**: Extended dead/slow-moving inventory manager with configurable thresholds. Features include:
        - Configurable parameters: HISTORY_WINDOW_WEEKS (12), RECENT_WINDOW_WEEKS (4), DEAD_DAYS_STORE (60), DEAD_DAYS_GLOBAL (90), DEAD_PURCHASE_DAYS (120), SLOW_RATE_THRESHOLD (0.3), MIN_WOC (1.5), MAX_WOC (6.0)
        - Store-level classification: DEAD_STORE, SLOW_STORE, LOW_STOCK, HEALTHY_STORE
        - CD-level classification: DEAD_CD (incl. stale purchases), SLOW_CD (uses CD-only WOC)
        - KPI cards: Dead Store, Slow Store, Dead CD, Healthy Store, Transfers
        - Coverage weeks (WOC) calculation: store-level and CD-only for accurate immobilization detection
        - Filter controls (SKU search, store dropdown, classification filter)
        - Smart transfer suggestions from DEAD/SLOW donors to active receivers
        - 4-tab layout: Tiendas, Global, Transferencias, CD
- **Run Management**: Centralized "Runs Center" for managing, activating, comparing, and tracking the status of various data processing runs.
- **User & Access Management**: Admin users can manage other users and their roles, with permissions governing feature access.
- **Reporting & Exports**: Export predictions, forecasts, and analysis results to Excel.
- **Store Health Index**: Diagnostic scoring module (0-100) per store with:
    - Weighted health formula: Fill Rate (30%), Stockout Rate (30%), Overstock Rate (20%), Sales Velocity (20%)
    - Status badges: Green (≥80 Saludable), Yellow (50-79 Atención), Red (<50 Crítico)
    - Edge case handling: Redistributes velocity weight when no sales data chain-wide
    - KPI summary cards, paginated table with sorting, weights configuration form, Excel export
    - Access via "Salud Tiendas" link under Operaciones (requires store_health:view permission)
- **In-app Alerts Module**: Proactive alerts based on stock and sales velocity (read-only, computed dynamically):
    - Alert types: PROJECTED_STOCKOUT (WOC < MIN_WOC), OVERSTOCK (WOC > MAX_WOC), SILENT_SKU (no sales for DEAD_DAYS)
    - Severity levels: HIGH (critical threshold breached), MEDIUM (warning level), LOW (minor concern)
    - Dashboard integration: Top 10 alerts panel with highest severity first
    - Dedicated /alerts page with filters: type, severity, store, SKU search
    - KPI cards: High/Medium/Low counts, Quiebres, Sobrestock, Sin movimiento
    - Suggested actions: Reponer, Redistribuir, Revisar, Liquidar
    - Access via "Alertas" link under Operaciones (requires alerts:view permission)

## External Dependencies
- **Database**: SQLite (for `app.db`)
- **Python Libraries**:
    - Flask (web framework)
    - Flask-SQLAlchemy (ORM)
    - Flask-Login (user authentication)
    - Pandas (data manipulation and analysis)
    - OpenPyXL (read/write Excel files)
    - Plotly (interactive charts)