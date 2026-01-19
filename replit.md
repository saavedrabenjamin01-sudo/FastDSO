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
    - **Store-to-Store Rebalancing**: Consolidated single-page module with two modes:
        - **Auto Suggestions**: System analyzes all stores and generates optimal transfers based on WOC and sales velocity
        - **Assisted Manual Plan**: User provides SKU list only (no quantities), system calculates optimal transfer amounts
        - Shared parameters: weeks_window, WOC min/target/max, retain_woc, stock_floor, min_transfer_qty
        - Tabbed UI with [Sugerencias Automáticas] [Plan Manual Asistido] toggle
    - **Assisted Manual Plan Features**:
        - Destination store required, donor store optional (system finds best donor)
        - SKU input via CSV/Excel file or paste list (one SKU per line)
        - Calculation logic: need_units = max(ceil(WOC_target * sales_rate_dest) - stock_dest, 0)
        - Status indicators: OK, NO-NEED (sufficient WOC), NO-DONOR, NO-SALES
        - Save plan to database or export to Excel
        - Routes: /rebalancing/manual/calculate, /save, /export-calculated
    - **Stock-out Replenishment (BREAK_REPLENISH)**: Automatically identifies and suggests replenishment for out-of-stock SKU-Store pairs with historical demand.
    - **Slow Stock & Smart Reallocation**: Extended dead/slow-moving inventory manager with configurable thresholds. Features include:
        - Configurable parameters: HISTORY_WINDOW_WEEKS (12), RECENT_WINDOW_WEEKS (4), DEAD_DAYS_STORE (60), DEAD_DAYS_GLOBAL (90), DEAD_PURCHASE_DAYS (120), SLOW_RATE_THRESHOLD (0.3), MIN_WOC (1.5), MAX_WOC (6.0)
        - Store-level classification: DEAD_STORE, SLOW_STORE, LOW_STOCK, HEALTHY_STORE
        - CD-level classification: DEAD_CD (incl. stale purchases), SLOW_CD (uses CD-only WOC)
        - KPI cards: Dead Store, Slow Store, Dead CD, Healthy Store, Transfers
        - Coverage weeks (WOC) calculation: store-level and CD-only for accurate immobilization detection
        - Filter controls (SKU search, store dropdown, classification filter, flagged filter)
        - Smart transfer suggestions from DEAD/SLOW donors to active receivers
        - 4-tab layout: Tiendas, Global, Transferencias, CD
        - **Product Flagging Workflow**: Mark products for slow stock management:
            - Product model extensions: eligible_for_distribution, risk_score, risk_reason
            - Risk score (0-100): score_sale (max 100, days_since_last_sale/120), score_stock (max 40, log10 scale), score_purchase (max 30, days_since_last_purchase/180)
            - Risk bands: HIGH (≥70), MEDIUM (40-69), LOW (<40)
            - Routes: /slow_stock/flag, /slow_stock/unflag, /slow_stock/flag_bulk
            - Category dashboard integration: Per-row and bulk flag actions in "Sin movimiento" drilldown
            - Store Health exclusion: Flagged products excluded from scope calculation
            - Audit logging: All flag/unflag operations logged with risk score and state change
- **Run Management**: Centralized "Runs Center" for managing, activating, comparing, and tracking the status of various data processing runs.
- **User & Access Management**: Admin users can manage other users and their roles, with permissions governing feature access.
- **Reporting & Exports**: Export predictions, forecasts, and analysis results to Excel.
- **Store Health Index**: Diagnostic scoring module (0-100) per store with:
    - Weighted health formula: Fill Rate (30%), Stockout Rate (30%), Overstock Rate (20%), Sales Velocity (20%)
    - Status badges: Green (≥80 Saludable), Yellow (50-79 Atención), Red (<50 Crítico)
    - Edge case handling: Redistributes velocity weight when no sales data chain-wide
    - KPI summary cards, paginated table with sorting, weights configuration form, Excel export
    - Access via "Salud Tiendas" link under Operaciones (requires store_health:view permission)
- **In-app Alerts Module**: Proactive alerts based on Macro Sales (SalesWeeklyAgg) and lifecycle tables (read-only, computed dynamically):
    - Alert types: 
        - PROJECTED_STOCKOUT (WOC < MIN_WOC): Store has stock but low coverage
        - OVERSTOCK (WOC > MAX_WOC): Stock exceeds WOC threshold
        - SILENT_SKU (no sales for DEAD_DAYS): No movement using SkuLifecycle.last_sale_date_global
        - BROKEN_STOCK (NEW): Store has zero stock but recent demand within 45 days using SkuStoreLifecycle
    - Data sources: SalesWeeklyAgg for sales rates, SkuLifecycle/SkuStoreLifecycle for last sale dates
    - Severity levels: HIGH (critical threshold breached), MEDIUM (warning level), LOW (minor concern)
    - Dashboard integration: Top 10 alerts panel with highest severity first
    - Dedicated /alerts page with filters: type, severity, store, SKU search
    - KPI cards: High/Medium/Low counts, Quiebres, Sobrestock, Sin movimiento, Quiebre reciente
    - Suggested actions: Reponer, Redistribuir, Revisar, Liquidar
    - Access via "Alertas" link under Operaciones (requires alerts:view permission)
- **Explainability Layer**: Transparent explanation system for distribution and forecast suggestions:
    - Distribution explanations: Shows SMA3 calculation, store stock adjustment, and suggested quantity derivation
    - Forecast explanations: Details avg_last4, demand projection, lead time, safety stock, and purchase calculation
    - API endpoints: `/api/explain/distribution`, `/api/explain/forecast`
    - Dashboard integration: "Why?" button (?) on each prediction row with modal showing bullet-point explanation
    - Forecast V2 integration: "Why?" button on purchase suggested KPI with detailed calculation breakdown
    - Helper functions: `explain_distribution_suggestion()`, `explain_forecast_purchase()` for generating explanations
- **Macro Sales Layer**: Full catalog visibility for accurate health metrics and cold-start distribution:
    - `SalesWeeklyAgg` model: Aggregated weekly sales (product_id, store_id, week_start, units, category)
    - Indexes: (product_id, store_id, week_start), (store_id, week_start), (category, store_id, week_start)
    - Route: `/sales_macro_upload` for uploading full catalog sales
    - Background processing with bulk inserts and weekly aggregation
    - Load modes: `replace_range` (default, replaces weeks), `append_range` (adds to existing)
    - Automatic Product.category population from uploads
    - Distribution generation uses macro layer when available for demand estimation
    - **Single Source of Truth**: SalesWeeklyAgg is the primary data source for:
        - Alerts module (sales rates, WOC calculations)
        - Store Health Index (sales velocity metrics)
        - Forecast V2 (demand estimation)
        - Store-to-Store Redistribution (sales rates for WOC calculations)
- **SKU Lifecycle Layer**: Tracks last sale dates for faster alert computation:
    - `SkuLifecycle` model: Global last sale date per SKU (last_sale_date_global)
    - `SkuStoreLifecycle` model: Store-level last sale date per SKU-Store pair (last_sale_date_store)
    - Automatically populated during Macro Sales upload
    - Enables BROKEN_STOCK alert detection without querying raw sales tables
- **Category-Based Cold Start**: Distribution suggestions for new SKUs without sales history:
    - Parameters: min_fill=2, target_WOC_new=1.0, eligible_store_top_n=10, category_window_days=90
    - Eligibility: SKU must have category set in Product.category
    - Store ranking: Top 10 stores by category weekly sales rate from SalesWeeklyAgg
    - Suggested qty: max(min_fill, ceil(target_WOC × cat_sales_rate)) - store_stock
    - CD allocation: Prioritizes higher category sales rate stores
    - Model tag: "COLD_START_CATEGORY ({category})"
- **Store Health SKU Scope**: Configurable scope for relevant SKU set in health calculations:
    - Modes: `core` (sales in 90 days OR in runs OR has store stock), `runs` (last 5 distribution runs), `full` (all SKUs)
    - Transparency: Shows SKUs in scope, excluded count, exclusion reasons
    - Uses SalesWeeklyAgg when available for scope calculations
- **Dashboard por Categoría**: Category-level executive dashboard with:
    - Route: `/dashboard_category` for category-filtered analytics
    - Filters: Category selector, time window (7-180 days), optional store filter
    - Optimized queries: Uses SalesWeeklyAgg.category index directly (no product_ids materialization)
    - Stock queries: Join Product table for category filter on StockCD/StockSnapshot
    - 4 KPI cards: Unidades vendidas, Demanda semanal promedio, Stock CD, Stock tiendas
    - Health summary: Bar showing healthy vs dead/slow products proportion with tooltip explaining "Sin movimiento"
    - Charts: Top 10 tiendas por ventas (bar), Distribución de stock CD vs tiendas (pie)
    - Top 10 SKUs accordion with expandable details
    - Access via "Dashboard por categoría" link under Ventas y Compras
    - **Sin movimiento drilldown**: Explainable and actionable SKU list
        - Definition: SKUs with stock (CD or stores) but no sales in selected window
        - Tooltip on health bar explaining the concept
        - "Ver SKUs sin movimiento" button links to paginated drilldown list
        - Route: `/dashboard_category/no_movement` with pagination (10 per page)
        - Table columns: SKU, Producto, Categoría, Stock CD, Stock Tiendas, Stock Total, Última venta
        - Export endpoint: `/dashboard_category/no_movement/export` generates Excel file
        - Optimized queries: Uses SalesWeeklyAgg.category index, no product_ids materialization

## External Dependencies
- **Database**: SQLite (for `app.db`)
- **Python Libraries**:
    - Flask (web framework)
    - Flask-SQLAlchemy (ORM)
    - Flask-Login (user authentication)
    - Pandas (data manipulation and analysis)
    - OpenPyXL (read/write Excel files)
    - Plotly (interactive charts)