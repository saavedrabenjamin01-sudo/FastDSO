# PredDist - Distribution Prediction Portal

## Overview
PredDist is a Flask-based web application designed to optimize product distribution and supply chain management. It provides tools for analyzing sales data, managing inventory across stores and distribution centers, and generating precise purchase and distribution forecasts. The system aims to deliver actionable insights for inventory optimization, sales trend analysis, and efficient supply chain operations, ultimately enhancing supply chain efficiency and profitability.

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
- **Background Job Processing**: Uses `ThreadPoolExecutor` for asynchronous handling of intensive tasks.
- **Role-Based Access Control (RBAC)**: Manages user permissions based on assigned roles.
- **Simulation Mode**: Allows "what-if" analysis without database persistence.
- **Audit Trail**: Logs user actions and system events.

### UI/UX Decisions
- **Dashboard**: Card-based layout with KPIs, tables, and global search.
- **Responsive Design**: Professional and intuitive templates with pagination, search, filters, and improved exports.
- **Unified Processing Modal**: Features animated SVG icons, real-time progress bars, and success/error result states for background operations.
- **Branding**: Login page incorporates corporate branding.
- **Interactive Visualizations**: Plotly charts are used for advanced forecasting.

### System Design and Features
- **Data Management**: Tools for uploading and managing sales, store stock, and distribution center stock data.
- **Distribution Predictions**: Generates product distribution suggestions based on moving averages and historical data, including stockout-aware backoff mechanisms.
- **Advanced Forecasting (Forecast Compra V2)**: Lifecycle-aware purchase forecasting using `SalesWeeklyAgg` with a centralized decision engine and deterministic decision paths (A-I). Features batch and category forecasting modes, and integrated alerts for projected stockouts and overstock.
- **Inventory Optimization**:
    - **Store-to-Store Rebalancing**: Consolidated module for auto-suggestions based on WOC and sales velocity with category-aware demand blending. Supports various destination modes and manual assisted modes. Strict donor conservation: `donor_available[(store_id, product_id)]` dict tracks remaining availability across all assignments (waterfall). Retain threshold: `keep = max(ceil(retain_woc * rate), stock_floor)`, `available_to_give = max(0, on_hand - keep)`. Validation pass aggregates by (donor, SKU) and hard-caps any overflow. UI shows donor_on_hand and remaining_after_plan. Test scenarios at `/admin/rebalancing_test_scenarios`. **SKU Traceability**: Every SKU in scope lands in exactly one bucket—Suggestions, Sin Necesidad (no_need), or Excluidos (excluded with reason code). Reason codes: HELD_SKU, NO_PRODUCT_MATCH, NO_STORE_DATA, ZERO_DEMAND, TOP_N_TRUNCATION, MAX_ROWS_LIMIT. UI has tabbed view with search-by-SKU and scope stats bar. Regression test at `/admin/rebalancing_trace_test`.
    - **Stock-out Replenishment (BREAK_REPLENISH)**: Identifies and suggests replenishment for out-of-stock SKU-Store pairs.
    - **Slow Stock & Smart Reallocation**: Manages dead and slow-moving inventory with configurable thresholds and smart transfer suggestions, including a product flagging workflow.
- **FastPlanner (Warehouse Execution)**: A Kanban-based module for warehouse operations with a 5-column workflow (APPROVED → IN_PROGRESS → PACKED → DISPATCHED → CLOSED), priority blocking, exception handling, and a Problem SKUs (Hold System) to block items with fulfillment issues.
- **Run Approval Workflow**: Multi-step authorization process for distribution runs (DRAFT → PENDING_APPROVAL → APPROVED/REJECTED) with urgency levels and notifications.
- **Run Management**: A "Runs Center" for managing, activating, comparing, and tracking data processing runs.
- **User & Access Management**: Administrators can manage users and their roles.
- **Reporting & Exports**: Enables exporting predictions, forecasts, and analysis results to Excel.
- **Store Health Index**: A diagnostic scoring module (0-100) per store using demand-weighted metrics (Fill Rate, Break Rate, Overstock Rate, No Movement Rate) with configurable time windows and category filters, providing suggested actions.
- **In-app Alerts Module**: Proactive alerts for conditions such as projected stockouts, overstock, silent SKUs, and broken stock.
- **Explainability Layer**: Provides transparent explanations for distribution and forecast suggestions.
- **Macro Sales Layer**: Offers full catalog visibility through the `SalesWeeklyAgg` model for aggregated weekly sales.
- **Data Sourcing Consistency**: All analysis modules (Rebalancing, Slow Stock, Store Health) use `get_analysis_end_date()` to derive the analysis window from `max(SalesWeeklyAgg.week_start)` instead of `date.today()`, ensuring alignment with loaded data.
- **Event-Sourced Inventory Ledger**: Operational stock tracking via `InventoryBaseline` + `InventoryEvent` models. Stock = snapshot baseline + event deltas. Event types: CD_RECEIPT, CD_ADJUSTMENT, STORE_SALE, STORE_RECEIPT, PLANNER_INCIDENT_TO_HOLD, HOLD_RELEASE_TO_CD. All stock reads across all modules use `get_cd_operational_stock()` / `get_store_operational_stock()` helpers (single + bulk variants) with fallback to snapshots when no events exist. `ProductHold` tracks items blocked from fulfillment. Events include batch_id, ref_type, ref_id for full traceability. Admin debug route at `/admin/operational_stock_debug?sku=XXX` compares snapshot vs operational stock per SKU.
- **Admin Diagnostics**: `/admin/diagnostics` page displays data source statistics (counts, date ranges) for MacroSales, StockSnapshot, StockCD, and Inventory Ledger tables.
- **SKU Lifecycle Layer**: Tracks global and store-level last sale dates for faster alert computation.
- **Category-Based Cold Start**: Provides distribution suggestions for new SKUs using category sales data. Eligibility requires only "no macro sales in active window" (store stock does not block). Multi-level fallback: category weights → relaxed filters → overall top stores → equal distribution. Per-store debug table output behind `app.debug` or `COLD_START_DEBUG` env var. The upload flow always proceeds to generation (no dead-end). A `/generate_from_pending` CTA route handles pending SKUs from session. Category is always updated from request files (normalized whitespace).
- **Demand-Capped Allocation**: All allocations are capped by projected demand, not CD stock availability. Sales-based SKUs: per-store need = `max(0, weekly_demand * horizon - store_stock)`, total_alloc_cap = `min(sum(per-store need), cd_stock)`. Cold-start SKUs: estimated_weekly_sku = `max(1, round(category_weekly_mean * NEW_SKU_SHARE))` where `NEW_SKU_SHARE=0.03`, total_need = `estimated_weekly_sku * horizon_weeks`, hard-capped by `ABSOLUTE_CAP_PER_SKU=50`. Per-store caps: `ceil(alloc_cap * weight * 1.2)`. Debug: `[ALLOC_CAP]` log per SKU with mode, demand, caps, final_alloc.
- **Proportional Category Allocation**: Both cold start and sales-based SMA allocations use store category weights (Laplace-smoothed, cached per category from SalesWeeklyAgg). Cold start uses Largest Remainder Method for integer allocation proportional to store category sales share, with TOP_K_GUARANTEE for top 5 stores. Sales-based SMA applies a clamped category weight factor (0.6-1.6x) relative to median weight, enabling high-performing category stores to receive more replenishment. Reason codes: COLD_START_PROPORTIONAL, BREAK_PROPORTIONAL_ADJUST, FORCE_MIN_ALLOC, TOP_K_GUARANTEE.
- **Dashboard por Categoría**: An executive dashboard for category-level analytics.

## External Dependencies
- **Database**: SQLite (for `app.db`)
- **Python Libraries**:
    - Flask
    - Flask-SQLAlchemy
    - Flask-Login
    - Pandas
    - OpenPyXL
    - Plotly