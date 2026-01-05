# PredDist - Distribution Prediction Portal

## Overview
PredDist is a Flask-based web application for managing product distribution predictions. It helps users analyze sales data, manage inventory across stores and distribution centers, and generate purchase forecasts.

## Current State
- **Status**: Working
- **Default credentials**: admin / admin
- **Port**: 5000

## Project Architecture

### Technology Stack
- **Backend**: Python 3.11, Flask 3.0.3
- **Database**: SQLite (app.db)
- **ORM**: Flask-SQLAlchemy
- **Authentication**: Flask-Login
- **Data Processing**: Pandas, OpenPyXL

### Directory Structure
```
/
├── app.py              # Main Flask application
├── templates/          # Jinja2 HTML templates
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── upload_stock.html
│   ├── upload_stock_cd.html
│   ├── stock_query.html
│   ├── purchase_forecast_v2.html
│   └── purchase_projection.html
├── static/
│   ├── css/style.css
│   └── img/           # Logo and favicon
├── Excel tipo/         # Sample Excel files
├── requirements.txt
└── .env.example
```

### Database Models
- **User**: Authentication (username, password_hash, role, is_active)
- **Product**: SKU and product name
- **Store**: Store names
- **Run**: Versioned run history (run_id, run_type, user_id, folio, responsable, categoria, store_filter, notes, status, rows_count, predictions_count)
- **DistributionRecord**: Sales history (product, store, quantity, date, run_id)
- **StockCD**: Distribution center stock snapshots
- **StockSnapshot**: Store stock snapshots
- **Prediction**: Generated distribution predictions (linked to run_id)
- **ForecastResult**: Stored forecast V2 results (linked to run_id)
- **AuditLog**: Audit trail (user_id, username_snapshot, role_snapshot, action, entity_type, entity_id, run_id, status, message, metadata_json, ip_address, user_agent)
- **RebalanceRun**: Store-to-store rebalancing run metadata (run_id, created_at, created_by_user_id, params_json)
- **RebalanceSuggestion**: Individual transfer suggestions (run_id, product_id, from_store_id, to_store_id, qty, sales_rate_to, woc_from, woc_to, score, reason)
- **Job**: Background job tracking (id, job_type, status, progress, message, payload_json, result_json, user_id, created_at, updated_at)

### Key Features
1. **Dashboard**: Overview of predictions, KPIs, and stock status
2. **Sales Upload**: Import sales data from CSV/Excel (background processing)
3. **Stock Management**: Upload store and CD stock levels (background processing)
4. **Predictions**: Generate distribution suggestions using moving averages
5. **Forecast Compra (V2)**: Advanced purchase forecasting with lead time, safety stock, coverage, demand methods
6. **Runs History**: View all runs (sales uploads, distribution, forecast) with versioning
7. **Export**: Download predictions and forecasts as Excel files
8. **Admin Users**: User management with RBAC (Admin only)
9. **Store-to-Store Rebalancing**: Transfer stock between stores based on WOC (weeks of cover) and sales velocity
10. **Simulation Mode**: Run calculations without saving to database for what-if analysis
11. **Background Job Processing**: Large file uploads processed asynchronously with progress tracking
12. **Stock-out Replenishment (BREAK_REPLENISH)**: Automatic suggestions for out-of-stock SKU-Store pairs with historical demand

## Simulation Mode

The simulation mode allows users to run distribution and rebalancing calculations without saving results to the database. This is useful for:
- Testing different parameters before committing
- What-if analysis scenarios
- Previewing impacts without affecting actual data

### How to Use
1. Check the "Simular (no guardar)" checkbox before generating predictions or rebalancing
2. Results are displayed with a yellow warning banner indicating simulation mode
3. Simulated results can be exported with a watermark column
4. Use "Limpiar simulación" to clear simulation data
5. Active simulations are shown on the dashboard

### Technical Details
- Results are stored in Flask session with 30-minute expiry
- Session-based storage means results are per-user and temporary
- Exports include "SIMULACIÓN: SÍ - NO GUARDADO" column with yellow highlight

## Role-Based Access Control (RBAC)

### Roles
| Role | Description |
|------|-------------|
| **Admin** | Full access to all features including user management and reset operations |
| **Management** | View-only + exports + runs history (no uploads, no reset) |
| **CategoryManager** | Sales upload + distribution generate/export + forecast V2 + runs view |
| **WarehouseOps** | Stock uploads (store + CD) + stock query + distribution export (no forecast) |
| **Viewer** | Dashboard + stock query view only |

### Permissions
- `dashboard:view` - View dashboard
- `sales:upload` - Upload sales data
- `stock_store:upload` - Upload store stock
- `stock_cd:upload` - Upload CD stock
- `stock:query` - Query stock levels
- `distribution:generate` - Generate distribution predictions
- `distribution:export` - Export distributions
- `forecast_v2:view` - View forecast V2
- `forecast_v2:run` - Run and export forecasts
- `runs:view` - View runs history
- `admin:users` - Manage users
- `admin:reset` - Reset data operations
- `audit:view` - View audit trail (Admin, Management)
- `rebalancing:view` - View rebalancing suggestions (Admin, Management, CategoryManager, WarehouseOps)
- `rebalancing:run` - Generate rebalancing suggestions (Admin, CategoryManager, WarehouseOps)

### Test Users
- `admin / admin` - Admin role
- `management / test123` - Management role
- `categorymanager / test123` - CategoryManager role
- `warehouseops / test123` - WarehouseOps role
- `viewer / test123` - Viewer role

## Running the Application
The application runs on port 5000 with the "Start Flask App" workflow.

## Background Job System

The application uses a background job system (ThreadPoolExecutor) to process large file uploads asynchronously, preventing Cloudflare 524 timeouts.

### How It Works
1. When user uploads a file, the app saves it immediately and creates a Job record
2. User is redirected to job status page with real-time progress updates (1.5s polling)
3. Background thread processes the file with progress updates
4. When complete, user can continue to their destination

### Job Types
- `upload_stock_cd`: Stock CD file upload
- `upload_sales`: Sales data upload with automatic prediction generation

### Technical Details
- ThreadPoolExecutor with 4 workers
- Isolated SQLAlchemy sessions for thread safety (scoped_session)
- Batch insert operations (1000 rows per batch)
- Progress tracking: 0-100%
- Status: queued → running → done/error
- Files stored in `instance/uploads/` and deleted after processing

### Routes
- `/jobs/<id>`: JSON API for polling job status
- `/jobs/<id>/view`: HTML page with auto-polling progress bar

## Stock-out Replenishment (BREAK_REPLENISH)

The stock-out replenishment layer is a post-processing step that automatically identifies SKU-Store pairs that are out of stock but have historical demand, and suggests replenishment quantities.

### Qualification Criteria
A SKU-Store pair qualifies for BREAK_REPLENISH if ALL of these are true:
- Store stock snapshot <= 0 (out of stock)
- Recent sales (last 1 week) = 0 (confirms stock-out)
- Historical sales (last 8 weeks) > 0 (proves there was demand)
- CD stock exists for that SKU
- The pair did NOT already receive a base forecast suggestion

### Configuration Constants
```python
STOCKOUT_RECENT_WEEKS = 1      # Weeks to check for "no recent sales"
STOCKOUT_HIST_WEEKS = 8        # Weeks of historical data to compute average
STOCKOUT_TARGET_WOC = 1.0      # Target weeks of cover for replenishment
STOCKOUT_MAX_QTY = 3           # Maximum qty per store for stock-out replenishment
STOCKOUT_DEBUG = False         # Enable debug logging
```

### How It Works
1. Runs as post-processing after base distribution predictions
2. Identifies stock-out candidates using store stock and sales history
3. Computes replenishment qty: `min(round(hist_avg * target_woc), max_qty)`
4. Prioritizes stores with higher historical demand
5. Applies CD stock limiting (base forecast gets priority)
6. Marks predictions with "| BREAK_REPLENISH" in model_name

## Recent Changes
- January 5, 2026: Added Stock-out Replenishment layer (BREAK_REPLENISH) for out-of-stock SKU-Store pairs
- January 5, 2026: Optimized rebalancing module with indexed lookups and bulk inserts
- December 31, 2025: Implemented Background Job System with ThreadPoolExecutor (4 workers)
- December 31, 2025: Added Job model for tracking background tasks
- December 31, 2025: Refactored Stock CD upload to use background jobs with bulk operations
- December 31, 2025: Refactored Sales upload to use background jobs with bulk operations
- December 31, 2025: Created job status page template with auto-polling progress bar
- December 31, 2025: Added isolated session handling for thread-safe database operations
- December 30, 2025: Refreshed login page with corporate branding (rounded card, logo, blue gradient button, no sidebar)
- December 30, 2025: Created base_auth.html for auth pages (minimal layout without sidebar)
- December 30, 2025: Added Store-to-Store Rebalancing module (V1) with RebalanceRun and RebalanceSuggestion models
- December 30, 2025: Added /rebalancing route with configurable WOC thresholds, KPI cards, and results table
- December 30, 2025: Added /export_rebalancing Excel export for rebalancing suggestions
- December 30, 2025: Added rebalancing:view and rebalancing:run permissions
- December 30, 2025: Added GET /api/forecast_v2 endpoint with history, forecast, confidence bands, and KPIs (sku required, store/horizon_weeks/history_weeks/lead_time_weeks/safety_pct optional)
- December 30, 2025: Added Plotly interactive charts to Forecast Compra V2 with line chart (sales history) and bar chart (demand vs stock vs suggested)
- December 30, 2025: Added API endpoint /api/forecast_v2/chart_data for chart data with SKU filtering
- December 30, 2025: Added KPI cards (total suggested, demand/week, stock CD, SKUs with purchase)
- December 30, 2025: Implemented full Audit Trail system with AuditLog model, log_audit helper, and /audit UI page
- December 30, 2025: Added audit logging to: sales upload, stock uploads, resets, distribution runs, exports
- December 30, 2025: Extended Run model with user_id, rows_count, predictions_count tracking
- December 30, 2025: Added audit:view permission for Admin and Management roles
- December 29, 2025: Implemented Role-Based Access Control (RBAC) with 5 roles and 12 permissions
- December 29, 2025: Added Admin Users management page with role/password management
- December 29, 2025: Updated sidebar to hide menu items based on user permissions
- December 29, 2025: Added professional UI/UX with paginated tables, search, filter chips, and improved exports
- December 29, 2025: Implemented Runs History system with versioned tracking for sales, distribution, and forecast runs
- December 29, 2025: Removed legacy purchase_forecast module, consolidated to Forecast Compra V2
- December 29, 2025: Initial Replit environment setup, configured to bind to 0.0.0.0:5000
