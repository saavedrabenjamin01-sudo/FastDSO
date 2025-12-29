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
│   ├── purchase_forecast.html
│   └── purchase_projection.html
├── static/
│   ├── css/style.css
│   └── img/           # Logo and favicon
├── Excel tipo/         # Sample Excel files
├── requirements.txt
└── .env.example
```

### Database Models
- **User**: Authentication (username, password_hash)
- **Product**: SKU and product name
- **Store**: Store names
- **DistributionRecord**: Sales history (product, store, quantity, date)
- **StockCD**: Distribution center stock snapshots
- **StockSnapshot**: Store stock snapshots
- **Prediction**: Generated distribution predictions

### Key Features
1. **Dashboard**: Overview of predictions, KPIs, and stock status
2. **Sales Upload**: Import sales data from CSV/Excel
3. **Stock Management**: Upload store and CD stock levels
4. **Predictions**: Generate distribution suggestions using moving averages
5. **Purchase Forecast**: Project purchase needs based on sales trends
6. **Export**: Download predictions as Excel files

## Running the Application
The application runs on port 5000 with the "Start Flask App" workflow.

## Recent Changes
- December 29, 2025: Initial Replit environment setup, configured to bind to 0.0.0.0:5000
