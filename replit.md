# PredDist - Distribution Prediction Portal

## Overview
PredDist is a Flask-based web application designed to optimize product distribution and supply chain management. It provides tools for analyzing sales data, managing inventory across stores and distribution centers, and generating precise purchase and distribution forecasts. The system aims to deliver actionable insights for inventory optimization, sales trend analysis, and efficient supply chain operations.

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
- **Modular Monolith**: Application structured into logical modules within a single Flask application.
- **Background Job Processing**: Uses `ThreadPoolExecutor` for asynchronous handling of intensive tasks like file uploads.
- **Role-Based Access Control (RBAC)**: Manages user permissions based on assigned roles (Admin, Management, CategoryManager, WarehouseOps, Viewer).
- **Simulation Mode**: Allows "what-if" analysis without database persistence, using Flask session for temporary data.
- **Audit Trail**: Logs user actions and system events.

### UI/UX Decisions
- **Dashboard**: Card-based layout with KPIs, tables, and global search.
- **Responsive Design**: Professional and intuitive templates with pagination, search, filters, and improved exports.
- **Unified Processing Modal**: Features animated SVG icons, real-time progress bars, and success/error result states for background operations. After job completion, shows result summary with "Continue" and "Load Another" action buttons.
- **Upload Result Feedback**: Processing modal transforms into success (green checkmark with bounce animation) or error (red X with shake animation) state, displaying job-specific result messages (e.g., record counts, weeks processed).
- **Branding**: Login page incorporates corporate branding.
- **Interactive Visualizations**: Plotly charts are used for advanced forecasting.

### System Design and Features
- **Data Management**: Tools for uploading and managing sales, store stock, and distribution center stock data.
- **Distribution Predictions**: Generates product distribution suggestions based on moving averages and historical data.
    - **Stockout-Aware Backoff**: For all SMA modes (SMA1/SMA2/SMA3), if a store has 0 sales in the primary window AND is currently stocked out (stock=0), the system falls back to a 12-week window rate if proven demand exists. This prevents high-selling stores from receiving 0 units just because they were stocked out during the primary window. Reason code: BACKOFF_STOCKOUT.
- **Advanced Forecasting (Forecast Compra V2)**: Lifecycle-aware purchase forecasting using SalesWeeklyAgg as single source of truth:
    - **Centralized Decision Engine**: `forecast_decision_engine()` with 4-step pipeline: Risk Evaluation → Demand Estimation → Coverage Calculation → Business Rules
    - **Decision Paths (A-I)**: Deterministic decision outcomes - A: Overstock Block, B: Slow Stock Blocked, C: Dead SKU Blocked, D: Breakage BUY, E: Breakage Redistribute, F: Adequate Coverage, G: Projected Stockout BUY, H: Deficit BUY, I: Review
    - **Precedence Rules**: Breakage and projected stockout alerts take precedence over overstock blocks
    - **Lifecycle Classification**: SKUs classified as ACTIVE (30d sales), SLOW (31-90d), DEAD (90d+), or NEW (no history)
    - **Model Selection**: ACTIVE uses user-selected SMA, SLOW uses 12-week SMA with 0.5 penalty, DEAD blocks purchases, NEW uses category cold start
    - **Alerts Integration**: PROJECTED_STOCKOUT/BROKEN_STOCK force buy_now=true, OVERSTOCK blocks purchase (unless breakage exists)
    - **Slow Stock Integration**: Reduces purchase qty by 50% for flagged SKUs
    - **Explainability**: Each result includes recommendation (BUY/NO_BUY/REDISTRIBUTE/BLOCKED), reason_code, explanation (1 sentence), decision_path for internal logging
    - **Single SKU Export**: /forecast/export route generates Excel with complete SKU forecast details
    - **Batch Forecast Mode**: Two-tab interface (Single/Batch) with file upload (CSV/XLSX) or text paste, max 500 SKUs per run, stored in ForecastBatchRun and ForecastBatchItem models, with paginated results table, KPI cards (BUY_NOW/REVIEW/DO_NOT_BUY/ERROR counts), explain modal, and batch export
    - **SKU-Aware Navigation**: Export, Alerts, and Slow Stock buttons pass SKU as query param for contextual filtering
    - **Decision Consistency**: Server-side validation ensures purchase_qty <= 0 cannot have BUY_NOW decision across all API routes (single, batch, category)
    - **Error Handling**: Missing SKUs in batch mode show status=ERROR with reason_code=SKU_NOT_FOUND, included in results and exports
    - **Explanation Pills**: Visual CSS pills replace bracketed labels ([MACRO], [CRÍTICO], [COMPRAR], etc.) for clearer decision context
    - **Normalized Alerts**: All forecast routes use compute_alerts_normalized() with BREAKAGE type for consistency
    - **Category Forecast Mode**: Third tab "Forecast por Categoría" for executive-level category purchase decisions with aggregated demand using SalesWeeklyAgg, KPI cards (weekly demand, horizon demand, total stock, deficit), decision badge (BUY_NOW/REVIEW/DO_NOT_BUY), action split showing redistribute potential vs purchase required, and contextual cross-links to dashboard_category, slow_stock, alerts, and rebalancing
- **Inventory Optimization**:
    - **Store-to-Store Rebalancing**: A consolidated module offering auto-suggestions based on WOC and sales velocity, and an assisted manual plan where the system calculates optimal transfer amounts for user-provided SKUs.
    - **Stock-out Replenishment (BREAK_REPLENISH)**: Identifies and suggests replenishment for out-of-stock SKU-Store pairs with historical demand.
    - **Slow Stock & Smart Reallocation**: Manages dead and slow-moving inventory with configurable thresholds, offering store-level and CD-level classification, KPI cards, and smart transfer suggestions. Includes a product flagging workflow to manage risk. Features category filter dropdown, page size selector (10/25/50), improved table styling with sticky headers and zebra striping, and product name ellipsis with tooltips.
- **FastPlanner (Warehouse Execution)**: A Kanban-based module for warehouse operations to manage distribution plan execution:
    - **Kanban Board**: 5-column workflow (APPROVED → IN_PROGRESS → PACKED → DISPATCHED → CLOSED)
    - **Plan Cards**: Display folio, urgency (LOW/MEDIUM/URGENT), SKU/unit/store counts, assigned operator
    - **Status Transitions**: Take (claim plan), Pack, Dispatch, Close actions with activity logging
    - **Plan Details**: Line-by-line view with pagination, commercial/warehouse notes, activity timeline
    - **Picking Export**: Excel export with SKU, product, store, quantity preserving leading zeros
    - **Permissions**: `planner:view` for viewing, `planner:operate` for status changes (Admin/WarehouseOps only)
    - **Models**: DistributionPlan, DistributionPlanLine, PlanActivityLog
- **Run Approval Workflow**: Multi-step authorization process for distribution runs:
    - **Workflow Statuses**: DRAFT → PENDING_APPROVAL → APPROVED/REJECTED
    - **Submit for Authorization**: CategoryManager can submit with urgency level (LOW/MEDIUM/URGENT) and observation note
    - **Approval Queue**: Management sees pending runs in Runs Center with approve/reject actions
    - **Plan Creation**: Approved runs create frozen DistributionPlan with lines copied from Prediction rows
    - **Rejection Workflow**: Rejected runs show comment and can be resubmitted after edits
    - **Notifications**: Sidebar badge shows pending count for managers, dashboard alert banner with link
    - **Permissions**: `runs:submit` for Admin/CategoryManager, `runs:approve` for Admin/Management
- **Run Management**: A "Runs Center" for managing, activating, comparing, and tracking data processing runs.
- **User & Access Management**: Administrators can manage users and their roles, controlling feature access.
- **Reporting & Exports**: Enables exporting predictions, forecasts, and analysis results to Excel.
- **Store Health Index**: A diagnostic scoring module (0-100) per store, based on weighted metrics like Fill Rate, Stockout Rate, Overstock Rate, and Sales Velocity, with status badges and configuration options.
- **In-app Alerts Module**: Proactive alerts for conditions such as projected stockouts, overstock, silent SKUs (no sales), and broken stock, based on aggregated sales and lifecycle data. Features category filter dropdown for filtering alerts by product category with preserved pagination.
- **Explainability Layer**: Provides transparent explanations for distribution and forecast suggestions, detailing calculations and decision logic, accessible via "Why?" buttons and debug fields.
- **Macro Sales Layer**: Offers full catalog visibility through the `SalesWeeklyAgg` model for aggregated weekly sales, serving as a single source of truth for various modules including alerts, store health, and forecasting.
- **SKU Lifecycle Layer**: Tracks global and store-level last sale dates (`SkuLifecycle`, `SkuStoreLifecycle`) for faster alert computation and demand analysis.
- **Category-Based Cold Start**: Provides distribution suggestions for new SKUs without sales history, using category sales data to rank eligible stores and determine quantities.
- **Store Health SKU Scope**: Configurable scope for relevant SKU sets in health calculations (e.g., `core`, `runs`, `full`).
- **Dashboard por Categoría**: An executive dashboard providing category-level analytics with filters, KPI cards, charts, and a "Sin movimiento" drilldown for SKUs with stock but no recent sales.

## External Dependencies
- **Database**: SQLite (for `app.db`)
- **Python Libraries**:
    - Flask
    - Flask-SQLAlchemy
    - Flask-Login
    - Pandas
    - OpenPyXL
    - Plotly