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
- **FastWMS (Warehouse Management Module)**: Integrated WMS module for location-level inventory tracking within the CD. Models: `WmsWarehouse`, `WmsLocation` (with `rack`, `level`, `pallet_position` fields; `location_code` derived as `RACK-LEVEL-PALLET`), `WmsInventory` (with `updated_at`), `WmsMovement`, `WmsMoveRun`, `WmsMoveLine`, `WmsInventoryEvent`. Default records: warehouse MAIN, location BULK-DEFAULT. Stock CD upload accepts WMS columns including `rack`, `level`, `pallet_position` (or legacy `location_code`); `replace_all` preserves movement history (optional `wms_reset` checkbox). Helpers: `build_location_code(rack, level, pallet_position)` and `_parse_location_code(code)`. Startup-safe ALTER TABLE migration parses existing location_code into rack/level/pallet_position. Sidebar: "Inventory by Location" and "Moves". Permission: `wms:view`.
- **FastWMS Inventory Page**: Full-featured inventory view with filter bar (search, warehouse, rack, location code, stock status, min units/pallets), 4 KPI cards, sortable (rack_asc default, avail desc/asc, units desc, ubicación A-Z) with sticky header and zebra rows, row actions (expand detail, prefill Move, disabled Reserve), proper empty states, info callout. Location cards show "Rack A | Nivel 02 | Pallet 03 (code)" when structured; falls back to location_code for legacy data. Export: `GET /wms/inventory/export` returns filtered .xlsx with Rack/Nivel/Pallet/Ubicación columns. `_group_inventory_by_location()` carries rack/level/pallet_position per group; default sort rack_asc. Location moves dropdown shows "Rack A | Nivel 02 | Pallet 03 (TYPE)" when structured.
- **FastWMS Pick Waves V1**: Auto-created when a distribution plan is approved and sent to FastPlanner. Models: `WmsPickWave`, `WmsPickTask` (status: OPEN/IN_PROGRESS/DONE/SHORT), `WmsPickReservation` (with task_id, location_code, qty_picked, status: OPEN/PARTIAL/PICKED/SHORT), `WmsPickIssue` (operator issue log: reason_code, missing_qty, operator_note), `WmsOperatorKpiDaily`. **Reservation priority**: PICK locations first, then BULK, then within type: available_units ASC → rack ASC → level ASC → pallet_position ASC → location_code ASC (empties small locations first). `wms_create_pick_wave_for_plan()` consolidates SKU demand, applies strict priority, sets wave READY_TO_ASSIGN or NEEDS_REVIEW. Workflow: READY_TO_ASSIGN → ASSIGNED → IN_PROGRESS → PICKED → CLOSED (+ NEEDS_REVIEW). Priority gating: URGENT waves block non-urgent. **JSON API endpoints**: `POST /api/wms/waves/<id>/start` (ASSIGNED→IN_PROGRESS), `POST /api/wms/tasks/<id>/pick` (incremental pick, allocates picked qty across reservations proportionally, decrements on_hand+allocated), `POST /api/wms/tasks/<id>/report-issue` (missing_qty + reason_code + operator_note, releases allocated_units without decrementing on_hand, creates quality flag event for DAMAGED/QUALITY_HOLD/MIXED_SKU), `POST /api/wms/waves/<id>/finish` (requires all tasks DONE or SHORT). **Wave detail page**: Task table with expandable per-location reservation breakdown (status, qty_picked), inline issue log, per-task "Registrar pick" and "Reportar incidencia" buttons (only in IN_PROGRESS), JS pick modal and issue modal, NEEDS_REVIEW badge. Allowed reason codes: NO_STOCK, DAMAGED, NOT_FOUND, LOCATION_BLOCKED, MIXED_SKU, COUNT_MISMATCH, QUALITY_HOLD, OTHER. Sidebar: "WMS Waves", "Mi Picking", "WMS KPIs".
- **Distribución Manual**: User-defined distribution module (`/distribution/manual`). Users upload CSV/XLSX with `sku`, `quantity`, `store` (required) + optional `product_name`, `category`, `note`. No allocation logic — each row specifies exact destination. Store validation with exact + normalized matching; unknown stores block upload with error list. (sku, store) duplicates are aggregated (quantities summed). Zero-qty rows ignored, negatives skipped. Two-step flow: parse → preview (KPIs: distinct SKUs, stores, units, lines) → create folio. Creates `DistributionPlan` with `source='MANUAL'`, auto-generates WMS pick wave. Product stubs created for unknown SKUs; existing products enriched with name/category. Folio format: `MAN-YYYYMMDD-HHMMSS`. `DistributionPlan.source` column (AUTO/MANUAL) + `closed_by_user_id`, `closed_at`, `exceptions_file_name`, `exceptions_count` columns via ALTER TABLE migrations. Sidebar: "Distribución manual" under Operaciones. Planner kanban/detail show MANUAL badge.

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