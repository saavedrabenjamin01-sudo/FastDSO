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

### Key Features
1. **Dashboard**: Overview of predictions, KPIs, and stock status
2. **Sales Upload**: Import sales data from CSV/Excel
3. **Stock Management**: Upload store and CD stock levels
4. **Predictions**: Generate distribution suggestions using moving averages
5. **Forecast Compra (V2)**: Advanced purchase forecasting with lead time, safety stock, coverage, demand methods
6. **Runs History**: View all runs (sales uploads, distribution, forecast) with versioning
7. **Export**: Download predictions and forecasts as Excel files
8. **Admin Users**: User management with RBAC (Admin only)
9. **Store-to-Store Rebalancing**: Transfer stock between stores based on WOC (weeks of cover) and sales velocity
10. **Simulation Mode**: Run calculations without saving to database for what-if analysis

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

## Recent Changes
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
