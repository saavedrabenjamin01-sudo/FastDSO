# PredDist - Distribution Prediction Portal

## Overview
PredDist is a Flask-based web application designed to optimize product distribution and supply chain management. It analyzes sales data, manages inventory, and generates precise purchase and distribution forecasts. The system aims to deliver actionable insights for inventory optimization, sales trend analysis, and efficient supply chain operations, ultimately enhancing supply chain efficiency and profitability.

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
- **Event-Sourced Inventory Ledger**: Operational stock tracking for full traceability.
- **SKU Normalization**: Canonical `normalize_sku` helper for consistent SKU strings.

### UI/UX Decisions
- **Dashboard**: Card-based layout with KPIs, tables, and global search.
- **Responsive Design**: Professional templates with pagination, search, and filters.
- **Unified Processing Modal**: Features animated SVG icons, real-time progress bars, and result states.
- **Branding**: Login page incorporates corporate branding.
- **Interactive Visualizations**: Plotly charts for forecasting.

### System Design and Features
- **Data Management**: Tools for uploading and managing sales, store stock, and distribution center stock data.
- **Distribution Predictions**: Generates product distribution suggestions based on moving averages and historical data, including stockout-aware backoff, and store-to-store rebalancing.
- **Advanced Forecasting (Forecast Compra V2)**: Lifecycle-aware purchase forecasting with batch and category modes, and alerts for projected stockouts and overstock.
- **Inventory Optimization**: Includes slow stock management, smart reallocation, and stock-out replenishment.
- **FastPlanner (Warehouse Execution)**: Kanban-based module for warehouse operations with a 5-column workflow, priority blocking, and exception handling.
- **Run Approval Workflow**: Multi-step authorization process for distribution runs.
- **Run Management**: A "Runs Center" for managing, activating, comparing, and tracking data processing runs.
- **User & Access Management**: Administrators can manage users and their roles.
- **Reporting & Exports**: Exports predictions, forecasts, and analysis results to Excel.
- **Store Health Index**: A diagnostic scoring module (0-100) per store using demand-weighted metrics.
- **In-app Alerts Module**: Proactive alerts for conditions such as projected stockouts, overstock, silent SKUs, and broken stock.
- **Explainability Layer**: Provides transparent explanations for distribution and forecast suggestions.
- **Macro Sales Layer**: Offers full catalog visibility through the `SalesWeeklyAgg` model.
- **Category-Based Cold Start**: Provides distribution suggestions for new SKUs using category sales data with multi-level fallback and proportional allocation.
- **Demand-Capped Allocation**: All allocations are capped by projected demand.
- **Proportional Category Allocation**: Uses store category weights for both cold start and sales-based allocations.
- **Stock Query Module**: Rich list+detail view for inventory consultation.
- **AI Copilot (GPT V1 Cross FastDSO)**: Read-only AI analysis layer for distribution runs and forecasts, providing executive narratives, anomaly detection, and suggested parameter tweaks.
- **FastWMS (Warehouse Management Module)**: Integrated WMS module for location-level inventory tracking within the Distribution Center, supporting pick waves, mobile picking, and stock buckets (`MAIN`, `WEB`).
- **Distribution Review Flow**: Post-approval review and adjustment of distribution plans before WMS integration.
- **Distribución Manual**: User-defined distribution module for manual CSV/XLSX uploads.
- **RBAC Admin UI**: Comprehensive roles management interface with permission matrix.

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
    - OpenAI
    - Ollama