# PredDist - Distribution Prediction Portal

## Overview
PredDist is a Flask-based web application designed to optimize product distribution and supply chain management. It analyzes sales data, manages inventory across stores and distribution centers, and generates precise purchase and distribution forecasts. The system aims to deliver actionable insights for inventory optimization, sales trend analysis, and efficient supply chain operations, ultimately enhancing supply chain efficiency and profitability.

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
- **Modular Monolith**: Application structured into logical modules.
- **Background Job Processing**: Uses `ThreadPoolExecutor` for asynchronous tasks.
- **Role-Based Access Control (RBAC)**: Manages user permissions.
- **Simulation Mode**: Allows "what-if" analysis without database persistence.
- **Audit Trail**: Logs user actions and system events.

### UI/UX Decisions
- **Dashboard**: Card-based layout with KPIs, tables, and global search.
- **Responsive Design**: Professional templates with pagination, search, and filters.
- **Unified Processing Modal**: Features animated SVG icons, real-time progress bars, and result states for background operations.
- **Branding**: Login page incorporates corporate branding.
- **Interactive Visualizations**: Plotly charts for forecasting.

### System Design and Features
- **Data Management**: Tools for uploading and managing sales, store stock, and distribution center stock data.
- **Distribution Predictions**: Generates product distribution suggestions based on moving averages and historical data, including stockout-aware backoff.
- **Advanced Forecasting (Forecast Compra V2)**: Lifecycle-aware purchase forecasting using `SalesWeeklyAgg` with a centralized decision engine and deterministic decision paths. Includes batch and category forecasting modes, and integrated alerts for projected stockouts and overstock.
- **Inventory Optimization**:
    - **Store-to-Store Rebalancing**: Consolidated module for auto-suggestions based on WOC and sales velocity with category-aware demand blending. Supports various destination and manual assisted modes. Includes strict donor conservation and SKU traceability with reason codes.
    - **Stock-out Replenishment (BREAK_REPLENISH)**: Identifies and suggests replenishment for out-of-stock SKU-Store pairs.
    - **Slow Stock & Smart Reallocation**: Manages dead and slow-moving inventory with configurable thresholds and smart transfer suggestions, including a product flagging workflow.
- **FastPlanner (Warehouse Execution)**: Kanban-based module for warehouse operations with a 5-column workflow, priority blocking, exception handling, and a Problem SKUs (Hold System).
- **Run Approval Workflow**: Multi-step authorization process for distribution runs (DRAFT → PENDING_APPROVAL → APPROVED/REJECTED) with urgency levels and notifications.
- **Run Management**: A "Runs Center" for managing, activating, comparing, and tracking data processing runs.
- **User & Access Management**: Administrators can manage users and their roles.
- **Reporting & Exports**: Exports predictions, forecasts, and analysis results to Excel.
- **Store Health Index**: A diagnostic scoring module (0-100) per store using demand-weighted metrics (Fill Rate, Break Rate, Overstock Rate, No Movement Rate) with configurable time windows and category filters, providing suggested actions.
- **In-app Alerts Module**: Proactive alerts for conditions such as projected stockouts, overstock, silent SKUs, and broken stock.
- **Explainability Layer**: Provides transparent explanations for distribution and forecast suggestions.
- **Macro Sales Layer**: Offers full catalog visibility through the `SalesWeeklyAgg` model for aggregated weekly sales.
- **SKU Normalization**: Canonical `normalize_sku` helper ensures consistent SKU strings across all modules.
- **Centralized CD Stock Map**: `get_cd_stock_map` is the single source of truth for CD stock across all modules.
- **Data Sourcing Consistency**: All analysis modules use `get_analysis_end_date()` to derive the analysis window, ensuring alignment with loaded data.
- **Event-Sourced Inventory Ledger**: Operational stock tracking via `InventoryBaseline` + `InventoryEvent` models for full traceability.
- **Admin Diagnostics**: `/admin/diagnostics` page displays data source statistics.
- **SKU Lifecycle Layer**: Tracks global and store-level last sale dates for faster alert computation.
- **Category-Based Cold Start**: Provides distribution suggestions for new SKUs using category sales data with multi-level fallback and proportional allocation.
- **Demand-Capped Allocation**: All allocations are capped by projected demand, not CD stock availability.
- **Proportional Category Allocation**: Uses store category weights for both cold start and sales-based SMA allocations, enabling high-performing category stores to receive more replenishment.
- **Dashboard por Categoría**: An executive dashboard for category-level analytics.
- **Stock Query Module**: Rich list+detail view for inventory consultation with paginated product table, search, filters, status badges, and action links.
- **AI Copilot (GPT V1 Cross FastDSO)**: Read-only AI analysis layer for distribution runs and forecasts. Features executive narratives, anomaly detection, and suggested parameter tweaks. Utilizes a dual-provider architecture (OpenAI and Ollama) with a strict JSON contract for output.
- **FastWMS (Warehouse Management Module)**: Integrated WMS module for location-level inventory tracking within the CD. Models: `WmsWarehouse`, `WmsLocation`, `WmsInventory` (with `updated_at`), `WmsMovement`, `WmsMoveRun`, `WmsMoveLine`, `WmsInventoryEvent`. Default records: warehouse MAIN, location BULK-DEFAULT. Stock CD upload accepts WMS columns; `replace_all` preserves movement history (optional `wms_reset` checkbox). Sidebar: "Inventory by Location" and "Moves". Permission: `wms:view`.
- **FastWMS Inventory Page**: Full-featured inventory view with filter bar (search, warehouse, location, stock status, min units/pallets), 4 KPI cards, sortable table (avail desc/asc, units desc, sku asc) with sticky header and zebra rows, 25-row pagination, row actions (expand detail, prefill Move, disabled Reserve), inline detail panel with metrics and quick links, proper empty states, info callout. Export: `GET /wms/inventory/export` returns filtered .xlsx. CSS: `wms-table`, `wms-detail-panel`, `wms-info-callout` in style.css.

## External Dependencies
- **Database**: SQLite (for `app.db`)
- **Python Libraries**:
    - Flask
    - Flask-SQLAlchemy
    - Flask-Login
    - Pandas
    - OpenPyXL
    - Plotly
- **AI Services**:
    - OpenAI (via Replit AI Integrations)
    - Ollama (local llama3.1:8b or qwen2.5:3b)