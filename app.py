import os
import io
import datetime as dt
from datetime import datetime, timedelta, date
from io import BytesIO
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func, or_
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required, logout_user, current_user, UserMixin
)

# ------------------ RBAC Configuration ------------------
ROLES = ['Admin', 'Management', 'CategoryManager', 'WarehouseOps', 'Viewer']

ROLE_PERMISSIONS = {
    'Admin': [
        'dashboard:view', 'sales:upload', 'stock_store:upload', 'stock_cd:upload',
        'stock:query', 'distribution:generate', 'distribution:export',
        'forecast_v2:view', 'forecast_v2:run', 'runs:view', 'admin:users', 'admin:reset', 'audit:view',
        'rebalancing:view', 'rebalancing:run'
    ],
    'Management': [
        'dashboard:view', 'stock:query', 'distribution:export',
        'forecast_v2:view', 'runs:view', 'audit:view', 'rebalancing:view'
    ],
    'CategoryManager': [
        'dashboard:view', 'sales:upload', 'distribution:generate', 'distribution:export',
        'forecast_v2:view', 'forecast_v2:run', 'runs:view', 'rebalancing:view', 'rebalancing:run'
    ],
    'WarehouseOps': [
        'dashboard:view', 'stock_store:upload', 'stock_cd:upload', 'stock:query',
        'distribution:export', 'rebalancing:view', 'rebalancing:run'
    ],
    'Viewer': [
        'dashboard:view', 'stock:query'
    ]
}
MIN_WEEKS = 3  # mínimo de semanas de historia requeridas por SKU–Tienda


# ------------------ Pagination Helper ------------------
class Pagination:
    def __init__(self, page, per_page, total, items):
        self.page = page
        self.per_page = per_page
        self.total = total
        self.items = items
        self.pages = max(1, (total + per_page - 1) // per_page)
    
    @property
    def has_prev(self):
        return self.page > 1
    
    @property
    def has_next(self):
        return self.page < self.pages
    
    @property
    def prev_num(self):
        return self.page - 1 if self.has_prev else None
    
    @property
    def next_num(self):
        return self.page + 1 if self.has_next else None
    
    def iter_pages(self, left_edge=1, left_current=2, right_current=2, right_edge=1):
        last = 0
        for num in range(1, self.pages + 1):
            if num <= left_edge or \
               (self.page - left_current <= num <= self.page + right_current) or \
               num > self.pages - right_edge:
                if last + 1 != num:
                    yield None
                yield num
                last = num


# ------------------ Config ------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)  # ✅ primero se crea app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')

db_path = os.path.join(BASE_DIR, 'app.db')  # ✅ app.db junto a app.py

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'SQLALCHEMY_DATABASE_URI',
    f'sqlite:///{db_path}'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)  # ✅ si en tu proyecto aún no existe db más abajo

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ------------------ Modelos ------------------


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='Viewer')
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def has_permission(self, permission):
        permissions = ROLE_PERMISSIONS.get(self.role, [])
        return permission in permissions

    def get_permissions(self):
        return ROLE_PERMISSIONS.get(self.role, [])


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sku = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)


class Store(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)


class DistributionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey(
        'product.id'), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    event_date = db.Column(db.Date, nullable=False)
    run_id = db.Column(db.String(36), nullable=True, index=True)

    product = db.relationship('Product')
    store = db.relationship('Store')

class StockCD(db.Model):
    __tablename__ = 'stock_cd'
    id = db.Column(db.Integer, primary_key=True)
    as_of_date = db.Column(db.Date, nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

    product = db.relationship('Product')

class StockSnapshot(db.Model):
    __tablename__ = 'stock_snapshot'
    id = db.Column(db.Integer, primary_key=True)
    as_of_date = db.Column(db.Date, nullable=False)  # fecha del snapshot
    product_id = db.Column(db.Integer, db.ForeignKey(
        'product.id'), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

    product = db.relationship('Product')
    store = db.relationship('Store')


class Run(db.Model):
    __tablename__ = 'run'
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    run_type = db.Column(db.String(50), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    folio = db.Column(db.String(100), nullable=True)
    responsable = db.Column(db.String(100), nullable=True)
    categoria = db.Column(db.String(100), nullable=True)
    store_filter = db.Column(db.String(255), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='completed', nullable=False)
    mode = db.Column(db.String(50), nullable=True)
    rows_count = db.Column(db.Integer, nullable=True)
    predictions_count = db.Column(db.Integer, nullable=True)

    user = db.relationship('User', backref='runs')

    def label(self):
        parts = []
        if self.folio:
            parts.append(f"Folio {self.folio}")
        if self.responsable:
            parts.append(f"Resp {self.responsable}")
        date_str = self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else ''
        if parts:
            return f"{' | '.join(parts)} — {date_str}"
        return date_str or f"Run {self.run_id[:8]}"

PredictionRun = Run


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey(
        'product.id'), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=False)
    target_period_start = db.Column(db.Date, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    model_name = db.Column(db.String(255), default='SMA_3w')
    run_id = db.Column(db.String(36), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    product = db.relationship('Product')
    store = db.relationship('Store')


class ForecastResult(db.Model):
    __tablename__ = 'forecast_result'
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(36), nullable=False, index=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    sku = db.Column(db.String(64), nullable=False)
    name = db.Column(db.String(255), nullable=True)
    demand_per_week = db.Column(db.Float, nullable=False)
    available_cd = db.Column(db.Integer, nullable=False)
    required_units = db.Column(db.Float, nullable=False)
    suggested = db.Column(db.Integer, nullable=False)
    campaign_tag = db.Column(db.String(100), nullable=True)

    product = db.relationship('Product')


class AuditLog(db.Model):
    """Audit trail for all significant actions in the system."""
    __tablename__ = 'audit_log'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    username_snapshot = db.Column(db.String(80), nullable=True)
    role_snapshot = db.Column(db.String(50), nullable=True)
    action = db.Column(db.String(100), nullable=False, index=True)
    entity_type = db.Column(db.String(100), nullable=True)
    entity_id = db.Column(db.String(100), nullable=True)
    run_id = db.Column(db.String(36), nullable=True, index=True)
    status = db.Column(db.String(20), nullable=False, default='success')
    message = db.Column(db.String(500), nullable=True)
    metadata_json = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)

    user = db.relationship('User', backref='audit_logs')


class RebalanceRun(db.Model):
    """Run metadata for store-to-store rebalancing suggestions."""
    __tablename__ = 'rebalance_run'
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    params_json = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref='rebalance_runs')

    def get_params(self):
        if self.params_json:
            try:
                return json.loads(self.params_json)
            except:
                return {}
        return {}


class RebalanceSuggestion(db.Model):
    """Individual rebalancing suggestion (transfer from one store to another)."""
    __tablename__ = 'rebalance_suggestion'
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(36), db.ForeignKey('rebalance_run.run_id'), nullable=False, index=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    from_store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=False)
    to_store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=False)
    qty = db.Column(db.Integer, nullable=False)
    sales_rate_to = db.Column(db.Float, nullable=True)
    woc_from = db.Column(db.Float, nullable=True)
    woc_to = db.Column(db.Float, nullable=True)
    score = db.Column(db.Float, nullable=True)
    reason = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    product = db.relationship('Product')
    from_store = db.relationship('Store', foreign_keys=[from_store_id])
    to_store = db.relationship('Store', foreign_keys=[to_store_id])
    run = db.relationship('RebalanceRun', backref='suggestions')


# ------------------ Login loader ------------------


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from datetime import date  # seguramente ya lo tienes más arriba

@app.context_processor
def inject_globals():
    """
    Variables globales disponibles en todos los templates Jinja.
    """
    return {
        'date': date,
        'ROLES': ROLES,
        'ROLE_PERMISSIONS': ROLE_PERMISSIONS
    }


def require_permission(permission):
    """Decorator to require a specific permission to access a route."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if not current_user.has_permission(permission):
                flash(f'No tienes permiso para acceder a esta sección.', 'warning')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ------------------ Audit Trail Helper ------------------
import json

def log_audit(action, status="success", message="", entity_type=None, entity_id=None, run_id=None, metadata=None):
    """
    Log an audit trail entry. Safe to call - won't crash if logging fails.
    
    Args:
        action: Action identifier (e.g., "sales.upload", "distribution.run")
        status: "success" or "fail"
        message: Human-readable description
        entity_type: Type of entity affected (e.g., "DistributionRecord", "StockCD")
        entity_id: ID of specific entity (optional)
        run_id: UUID of related run (optional)
        metadata: Dict with additional context (will be JSON-serialized)
    """
    try:
        user_id = None
        username_snapshot = None
        role_snapshot = None
        ip_address = None
        user_agent = None
        
        if current_user and current_user.is_authenticated:
            user_id = current_user.id
            username_snapshot = current_user.username
            role_snapshot = current_user.role
        
        if request:
            ip_address = request.remote_addr
            user_agent = request.headers.get('User-Agent', '')[:500]
        
        metadata_json = None
        if metadata:
            try:
                metadata_json = json.dumps(metadata, default=str, ensure_ascii=False)
            except Exception:
                metadata_json = str(metadata)
        
        audit_entry = AuditLog(
            user_id=user_id,
            username_snapshot=username_snapshot,
            role_snapshot=role_snapshot,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id else None,
            run_id=run_id,
            status=status,
            message=message[:500] if message else None,
            metadata_json=metadata_json,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.session.add(audit_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to write audit log: {e}")


# ------------------ Utilidades ------------------


def next_monday(ref: datetime | None = None):
    ref = ref or datetime.utcnow()
    days_ahead = (7 - ref.weekday()) % 7
    days_ahead = 7 if days_ahead == 0 else days_ahead
    return (ref + timedelta(days=days_ahead)).date()

def has_any_stock_loaded() -> bool:
    """True si existe al menos un snapshot de stock en la tabla."""
    return db.session.query(StockSnapshot.id).first() is not None


# ------------------ Simulation Mode Helpers ------------------
from flask import session

SIMULATION_EXPIRY_MINUTES = 30

def store_simulation_results(sim_type, results, kpis=None, cd_remaining=None, meta=None):
    """
    Store simulation results in Flask session.
    sim_type: 'distribution', 'forecast', 'rebalancing'
    """
    if 'simulations' not in session:
        session['simulations'] = {}
    
    session['simulations'][sim_type] = {
        'results': results,
        'kpis': kpis or {},
        'cd_remaining': cd_remaining or [],
        'meta': meta or {},
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': current_user.id if current_user.is_authenticated else None
    }
    session.modified = True


def get_simulation_results(sim_type):
    """
    Retrieve simulation results from Flask session.
    Returns None if expired or not found.
    """
    if 'simulations' not in session:
        return None
    
    sim_data = session['simulations'].get(sim_type)
    if not sim_data:
        return None
    
    timestamp = datetime.fromisoformat(sim_data['timestamp'])
    if datetime.utcnow() - timestamp > timedelta(minutes=SIMULATION_EXPIRY_MINUTES):
        clear_simulation(sim_type)
        return None
    
    return sim_data


def clear_simulation(sim_type=None):
    """Clear simulation data from session."""
    if 'simulations' not in session:
        return
    
    if sim_type:
        session['simulations'].pop(sim_type, None)
    else:
        session['simulations'] = {}
    session.modified = True


def has_active_simulation(sim_type=None):
    """Check if there's an active simulation."""
    if sim_type:
        return get_simulation_results(sim_type) is not None
    return bool(session.get('simulations', {}))


# Predicción: promedio móvil 3 semanas por SKU-Tienda

from datetime import date
from collections import defaultdict

def generate_predictions(
    mode: str = "sma3_min3",
    meta: dict | None = None,
    df: pd.DataFrame | None = None,
    sales_run_id: str | None = None,
    simulate: bool = False
):
    from uuid import uuid4
    run_id = str(uuid4())
    
    meta = meta or {}
    
    if not simulate:
        prediction_run = Run(
            run_id=run_id,
            run_type='distribution',
            folio=meta.get("folio"),
            responsable=meta.get("responsable"),
            categoria=meta.get("categoria"),
            notes=meta.get("fecha_doc"),
            mode=mode,
            status='completed'
        )
        db.session.add(prediction_run)
        db.session.flush()

    # 1) Origen de datos: df pasado o histórico completo
    if df is None:
        rows = DistributionRecord.query.all()
        if not rows:
            db.session.commit()
            return run_id, 0
        data = [{
            'sku': r.product.sku,
            'store': r.store.name,
            'quantity': r.quantity,
            'date': pd.to_datetime(r.event_date)
        } for r in rows]
        df = pd.DataFrame(data)
    else:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

    # Normalizar columnas clave (alineado con /upload)
    df['sku'] = df['sku'].astype(str).str.strip()
    df['store'] = df['store'].astype(str).str.strip()
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    # Normalizar semana (lunes como inicio)
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')

    # 2) Parámetros según modo seleccionado
    if mode == "sma3_min3":
        win = 3
        min_weeks = 3
        use_stock = True
        model_tag = "Promedio móvil 3 semanas (ajustado por stock)"
    elif mode == "sma2_min2":
        win = 2
        min_weeks = 2
        use_stock = True
        model_tag = "Promedio móvil 2 semanas (ajustado por stock)"
    elif mode == "sma1_no_min":
        win = 1
        min_weeks = 1   # mínimo 1 semana
        use_stock = True
        model_tag = "Última semana (mínimo 1 semana)"
    elif mode == "sma3_ignore_stock":
        win = 3
        min_weeks = 3
        use_stock = False
        model_tag = "Promedio móvil 3 semanas (sin ajuste de stock)"
    else:
        win = 3
        min_weeks = 3
        use_stock = True
        model_tag = "Promedio móvil 3 semanas (ajustado por stock)"

    # 2.b) Aplicar metadata al nombre del modelo
    if meta:
        extra_bits = []
        if meta.get("folio"):
            extra_bits.append(f"Folio: {meta['folio']}")
        if meta.get("responsable"):
            extra_bits.append(f"Resp: {meta['responsable']}")
        if meta.get("categoria"):
            extra_bits.append(f"Cat: {meta['categoria']}")
        if meta.get("fecha_doc"):
            extra_bits.append(f"Fecha doc: {meta['fecha_doc']}")
        if extra_bits:
            model_tag = f"{model_tag} — " + " | ".join(extra_bits)

    # 3) Generar sugerencias base por SKU–Tienda
    raw_preds = []

    for (sku, store), gdf in df.groupby(['sku', 'store']):
        weekly = (
            gdf.groupby('week_start', as_index=False)['quantity']
             .sum()
             .sort_values('week_start')
        )

        # Exigir mínimo de semanas (blindado a >=1)
        effective_min_weeks = max(min_weeks, 1)
        if weekly.shape[0] < effective_min_weeks:
            continue

        base_mean = float(weekly.tail(win)['quantity'].mean())

        # Normalizar claves para buscar en BD
        sku_key = str(sku).strip()
        store_key = str(store).strip()

        # Buscar producto y tienda; si no existen, saltar
        product = Product.query.filter_by(sku=sku_key).first()
        if not product:
            continue

        store_ent = Store.query.filter_by(name=store_key).first()
        if not store_ent:
            continue

        # Ajuste por stock tienda
        if use_stock:
            latest_stock = (
                db.session.query(StockSnapshot)
                .filter(
                    StockSnapshot.product_id == product.id,
                    StockSnapshot.store_id == store_ent.id
                )
                .order_by(StockSnapshot.as_of_date.desc())
                .first()
            )
            stock_qty = latest_stock.quantity if latest_stock else 0
        else:
            stock_qty = 0

        suggested = max(int(round(base_mean)) - stock_qty, 0)

        raw_preds.append({
            "sku": sku_key,
            "store": store_key,
            "product_id": product.id,
            "store_id": store_ent.id,
            "suggested": suggested,
            "model_name": model_tag,
        })

    # 4) Aplicar límite por stock CD (priorizar tiendas con más demanda)

    # Usamos SIEMPRE la última fecha de snapshot disponible
    snapshot_date = db.session.query(db.func.max(StockCD.as_of_date)).scalar()

    if snapshot_date:
        cd_stock_rows = StockCD.query.filter_by(as_of_date=snapshot_date).all()
        cd_stock = {row.product_id: row.quantity for row in cd_stock_rows}
    else:
        cd_stock_rows = []
        cd_stock = {}

    per_product = defaultdict(list)
    for r in raw_preds:
        per_product[r["product_id"]].append(r)

    final_preds = []

    for product_id, items in per_product.items():
        # Ordenar de mayor a menor sugerido
        items.sort(key=lambda x: x["suggested"], reverse=True)

        available = cd_stock.get(product_id, None)
        if available is None:
            # Sin stock CD cargado → dejar lo sugerido tal cual
            final_preds.extend(items)
            continue

        for idx, it in enumerate(items):
            want = it["suggested"]
            give = min(want, available)
            it["suggested"] = give
            available -= give
            final_preds.append(it)

            if available <= 0:
                # Lo que queda en la lista se queda en 0
                for it2 in items[idx+1:]:
                    it2["suggested"] = 0
                    final_preds.append(it2)
                break

    # 5) Guardar predicciones y registrar asignaciones
    target_week = next_monday()
    preds_bulk = []
    assigned_by_product = defaultdict(int)

    for it in final_preds:
        assigned_qty = int(it["suggested"])
        assigned_by_product[it["product_id"]] += assigned_qty

    if simulate:
        sim_results = []
        product_cache = {p.id: p for p in Product.query.all()}
        store_cache = {s.id: s for s in Store.query.all()}
        
        for it in final_preds:
            prod = product_cache.get(it["product_id"])
            store_obj = store_cache.get(it["store_id"])
            sim_results.append({
                'sku': it['sku'],
                'product_name': prod.name if prod else '',
                'store': it['store'],
                'quantity': int(it['suggested']),
                'model_name': it.get('model_name', ''),
                'target_week': target_week.isoformat()
            })
        
        cd_remaining = []
        if snapshot_date:
            for cd_row in cd_stock_rows:
                assigned = assigned_by_product.get(cd_row.product_id, 0)
                remaining = max(cd_row.quantity - assigned, 0)
                prod = product_cache.get(cd_row.product_id)
                cd_remaining.append({
                    'sku': prod.sku if prod else '',
                    'product_name': prod.name if prod else '',
                    'original': cd_row.quantity,
                    'assigned': assigned,
                    'remaining': remaining
                })
        
        kpis = {
            'total_units': sum(r['quantity'] for r in sim_results),
            'num_skus': len(set(r['sku'] for r in sim_results)),
            'num_stores': len(set(r['store'] for r in sim_results)),
            'num_predictions': len(sim_results)
        }
        
        return run_id, len(final_preds), sim_results, cd_remaining, kpis

    for it in final_preds:
        assigned_qty = int(it["suggested"])

        existing = Prediction.query.filter_by(
            product_id=it["product_id"],
            store_id=it["store_id"],
            target_period_start=target_week,
            run_id=run_id
        ).first()

        if existing:
            existing.quantity = assigned_qty
            existing.model_name = it["model_name"]
        else:
            preds_bulk.append(Prediction(
                product_id=it["product_id"],
                store_id=it["store_id"],
                target_period_start=target_week,
                quantity=assigned_qty,
                model_name=it["model_name"],
                run_id=run_id
            ))

    if preds_bulk:
        db.session.add_all(preds_bulk)

    # 6) Descontar del stock CD lo que realmente asignamos
    if snapshot_date:
        for product_id, assigned_total in assigned_by_product.items():
            cd_row = StockCD.query.filter_by(as_of_date=snapshot_date, product_id=product_id).first()
            if cd_row:
                cd_row.quantity = max(cd_row.quantity - assigned_total, 0)

    db.session.commit()

    try:
        g.latest_run_id = run_id
    except Exception:
        pass

    return run_id, len(final_preds), None, None, None

from flask import g
from datetime import date

@app.context_processor
def inject_sidebar_counts():
    """
    Inyecta sidebar_counts en TODAS las plantillas que extienden base.html,
    así evitamos errores de 'sidebar_counts is undefined'.
    """
    try:
        # Total registros de ventas cargadas
        sales_count = DistributionRecord.query.count()

        # SKUs distintos con ventas
        sku_sales = db.session.query(DistributionRecord.product_id).distinct().count()

        # SKUs con stock en tiendas
        store_stock_skus = db.session.query(StockSnapshot.product_id).distinct().count()

        # SKUs con stock en CD (solo > 0)
        cd_stock_skus = (
            db.session.query(StockCD.product_id)
            .filter(StockCD.quantity > 0)
            .distinct()
            .count()
        )

        # Predicciones de la última semana sugerida
        latest_week = db.session.query(db.func.max(Prediction.target_period_start)).scalar()
        if latest_week:
            predictions_count = (
                Prediction.query
                .filter(Prediction.target_period_start == latest_week)
                .count()
            )
        else:
            predictions_count = 0

        counts = {
            "dashboard": sales_count,        # o lo que quieras mostrar en el badge
            "upload": sku_sales,
            "stock_store": store_stock_skus,
            "stock_cd": cd_stock_skus,
            "predictions": predictions_count,
        }

    except Exception:
        # En caso de que algo falle (por migraciones, DB vacía, etc.)
        counts = {
            "dashboard": 0,
            "upload": 0,
            "stock_store": 0,
            "stock_cd": 0,
            "predictions": 0,
        }

    return dict(sidebar_counts=counts)

# ------------------ Rutas ------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    # si ya está logeado, directo al dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = (request.form.get('username') or "").strip().lower()
        password = request.form.get('password') or ""

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            if not user.is_active:
                flash('Tu cuenta está desactivada. Contacta al administrador.', 'danger')
                return render_template('login.html')
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Credenciales inválidas.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
def index():
    # raíz redirige al login o dashboard según estado
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

from datetime import date, timedelta
from sqlalchemy import func

from datetime import date, timedelta
from sqlalchemy import func

from datetime import date, timedelta
from sqlalchemy import func

from datetime import date, timedelta
from collections import defaultdict

from datetime import date, timedelta
from sqlalchemy import func

from datetime import date, timedelta
from sqlalchemy import func

from datetime import date
from sqlalchemy import func, desc

@app.route('/dashboard', methods=['GET'])
@login_required
@require_permission('dashboard:view')
def dashboard():
    # --- Filtros globales ---
    store_filter = (request.args.get('store') or '').strip()
    folio_filter = (request.args.get('folio') or '').strip()
    responsable_filter = (request.args.get('responsable') or '').strip()
    run_id_filter = (request.args.get('run_id') or '').strip()

    # --- Pagination params per table ---
    sales_page = request.args.get('sales_page', 1, type=int)
    sales_size = request.args.get('sales_size', 10, type=int)
    sales_search = (request.args.get('sales_search') or '').strip()

    pred_page = request.args.get('pred_page', 1, type=int)
    pred_size = request.args.get('pred_size', 10, type=int)
    pred_search = (request.args.get('pred_search') or '').strip()

    cd_page = request.args.get('cd_page', 1, type=int)
    cd_size = request.args.get('cd_size', 10, type=int)
    cd_search = (request.args.get('cd_search') or '').strip()

    # Validate page sizes
    sales_size = sales_size if sales_size in [10, 25, 50] else 10
    pred_size = pred_size if pred_size in [10, 25, 50] else 10
    cd_size = cd_size if cd_size in [10, 25, 50] else 10

    # --- Fechas / helpers ---
    today = date.today()

    # --- Load available runs (limit 50, newest first, all run types) ---
    runs = (
        PredictionRun.query
        .filter(PredictionRun.run_id.isnot(None))
        .filter(PredictionRun.run_id != "")
        .order_by(PredictionRun.created_at.desc())
        .limit(50)
        .all()
    )

    # --- Determine selected_run_id (validate it exists) ---
    valid_run_ids = {r.run_id for r in runs}
    if run_id_filter and run_id_filter in valid_run_ids:
        selected_run_id = run_id_filter
    elif runs:
        selected_run_id = runs[0].run_id
    else:
        selected_run_id = None

    # --- Última semana de predicción (del run seleccionado) ---
    if selected_run_id:
        latest_week = (
            db.session.query(func.max(Prediction.target_period_start))
            .filter(Prediction.run_id == selected_run_id)
            .scalar()
        )
    else:
        latest_week = None

    # --- Query base de predicciones + joins ---
    pred_q = (
        db.session.query(Prediction, Product, Store)
        .join(Product, Prediction.product_id == Product.id)
        .join(Store, Prediction.store_id == Store.id)
    )

    # Always filter by selected run_id
    if selected_run_id:
        pred_q = pred_q.filter(Prediction.run_id == selected_run_id)

    # Filtro tienda
    if store_filter:
        pred_q = pred_q.filter(Store.name == store_filter)

    # Filtros folio / responsable (filter within the selected run)
    if folio_filter:
        pred_q = pred_q.filter(Prediction.model_name.ilike(f"%Folio:%{folio_filter}%"))
    if responsable_filter:
        pred_q = pred_q.filter(Prediction.model_name.ilike(f"%Resp:%{responsable_filter}%"))

    # Search filter for predictions
    if pred_search:
        pred_q = pred_q.filter(
            or_(
                Product.sku.ilike(f"%{pred_search}%"),
                Product.name.ilike(f"%{pred_search}%"),
                Store.name.ilike(f"%{pred_search}%")
            )
        )

    pred_total = pred_q.count()
    predictions = (
        pred_q
        .order_by(Product.sku.asc(), Store.name.asc())
        .offset((pred_page - 1) * pred_size)
        .limit(pred_size)
        .all()
    )
    pred_pagination = Pagination(pred_page, pred_size, pred_total, predictions)

    # --- KPI (from full filtered set, not just current page) ---
    kpi_q = (
        db.session.query(
            func.coalesce(func.sum(Prediction.quantity), 0),
            func.count(func.distinct(Prediction.product_id)),
            func.count(func.distinct(Prediction.store_id))
        )
        .filter(Prediction.run_id == selected_run_id) if selected_run_id else None
    )
    if kpi_q and store_filter:
        kpi_q = kpi_q.join(Store, Prediction.store_id == Store.id).filter(Store.name == store_filter)
    kpi_result = kpi_q.first() if kpi_q else (0, 0, 0)
    kpi_units_suggested = int(kpi_result[0] or 0) if kpi_result else 0
    kpi_skus_distintos = int(kpi_result[1] or 0) if kpi_result else 0
    kpi_tiendas_alcanzadas = int(kpi_result[2] or 0) if kpi_result else 0

    # Stock CD hoy (total)
    kpi_stock_cd_total = (
        db.session.query(func.coalesce(func.sum(StockCD.quantity), 0))
        .filter(StockCD.as_of_date == today)
        .scalar()
    )

    # --- Top ventas (with pagination) ---
    sales_q = (
        db.session.query(
            Product.sku.label("sku"),
            Product.name.label("product"),
            Store.name.label("store"),
            func.coalesce(func.sum(DistributionRecord.quantity), 0).label("units")
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .join(Store, DistributionRecord.store_id == Store.id)
    )
    if store_filter:
        sales_q = sales_q.filter(Store.name == store_filter)

    if sales_search:
        sales_q = sales_q.filter(
            or_(
                Product.sku.ilike(f"%{sales_search}%"),
                Product.name.ilike(f"%{sales_search}%")
            )
        )

    sales_q = sales_q.group_by(Product.sku, Product.name, Store.name)
    sales_count_q = db.session.query(func.count()).select_from(sales_q.subquery())
    sales_total = sales_count_q.scalar() or 0

    top_sales = (
        sales_q
        .order_by(desc("units"))
        .offset((sales_page - 1) * sales_size)
        .limit(sales_size)
        .all()
    )
    sales_pagination = Pagination(sales_page, sales_size, sales_total, top_sales)

    # --- Remanente CD (with pagination) ---
    stock_cd_filtered = []
    cd_pagination = Pagination(1, cd_size, 0, [])
    if selected_run_id:
        pred_product_ids = (
            db.session.query(Prediction.product_id)
            .filter(Prediction.run_id == selected_run_id)
            .distinct()
            .subquery()
        )

        cd_q = (
            db.session.query(StockCD, Product)
            .join(Product, StockCD.product_id == Product.id)
            .filter(StockCD.as_of_date == today)
            .filter(StockCD.product_id.in_(pred_product_ids))
        )

        if cd_search:
            cd_q = cd_q.filter(
                or_(
                    Product.sku.ilike(f"%{cd_search}%"),
                    Product.name.ilike(f"%{cd_search}%")
                )
            )

        cd_total = cd_q.count()
        stock_cd_filtered = (
            cd_q
            .order_by(Product.sku.asc())
            .offset((cd_page - 1) * cd_size)
            .limit(cd_size)
            .all()
        )
        cd_pagination = Pagination(cd_page, cd_size, cd_total, stock_cd_filtered)

    # --- Lista de tiendas para el select ---
    stores = Store.query.order_by(Store.name.asc()).all()

    return render_template(
        "dashboard.html",
        stores=stores,
        selected_store=store_filter,
        folio=folio_filter,
        responsable=responsable_filter,

        latest_week=latest_week,

        predictions=predictions,
        pred_pagination=pred_pagination,
        pred_search=pred_search,
        pred_size=pred_size,

        top_sales=top_sales,
        sales_pagination=sales_pagination,
        sales_search=sales_search,
        sales_size=sales_size,

        stock_cd_filtered=stock_cd_filtered,
        cd_pagination=cd_pagination,
        cd_search=cd_search,
        cd_size=cd_size,

        kpi_units_suggested=kpi_units_suggested,
        kpi_skus_distintos=kpi_skus_distintos,
        kpi_tiendas_alcanzadas=kpi_tiendas_alcanzadas,
        kpi_stock_cd_total=int(kpi_stock_cd_total or 0),

        # (opcional) por si después quieres mostrar “corrida actual”
        runs=runs,
        selected_run_id=selected_run_id,
    )

@app.route('/purchase_forecast', methods=['GET', 'POST'])
@login_required
@require_permission('forecast_v2:view')
def purchase_forecast():
    """Legacy route - redirect to V2."""
    return redirect(url_for('purchase_forecast_v2'))

@app.route('/purchase_forecast_v2', methods=['GET'])
@login_required
@require_permission('forecast_v2:view')
def purchase_forecast_v2():
    """
    Purchase Forecast V2 - Interactive forecast page with Plotly charts.
    Uses /api/forecast_v2 for data.
    """
    stores = Store.query.order_by(Store.name.asc()).all()
    products = Product.query.order_by(Product.sku.asc()).limit(500).all()

    return render_template(
        'purchase_forecast_v2.html',
        stores=stores,
        products=products,
    )


@app.route('/purchase_forecast_v2_legacy', methods=['GET', 'POST'])
@login_required
@require_permission('forecast_v2:view')
def purchase_forecast_v2_legacy():
    """
    Legacy Purchase Forecast V2 (batch mode):
    - Lead time (days)
    - Safety stock (weeks)
    - Coverage horizon (weeks)
    - Demand method: sma3, sma2, last_week
    - Optional store filter and campaign tag
    """
    today = date.today()
    stores = Store.query.order_by(Store.name.asc()).all()

    if request.method == 'POST':
        try:
            lead_time_days = int(request.form.get('lead_time_days', 14))
        except ValueError:
            lead_time_days = 14
        try:
            coverage_weeks = float(request.form.get('coverage_weeks', 4))
        except ValueError:
            coverage_weeks = 4.0
        try:
            safety_weeks = float(request.form.get('safety_weeks', 1))
        except ValueError:
            safety_weeks = 1.0
        try:
            min_weeks_history = int(request.form.get('min_weeks_history', 1))
        except ValueError:
            min_weeks_history = 1

        demand_method = request.form.get('demand_method', 'sma3').strip()
        store_filter = request.form.get('store_filter', '').strip()
        campaign_tag = request.form.get('campaign_tag', '').strip()
    else:
        lead_time_days = 14
        coverage_weeks = 4.0
        safety_weeks = 1.0
        min_weeks_history = 1
        demand_method = 'sma3'
        store_filter = ''
        campaign_tag = ''

    lead_time_days = max(lead_time_days, 0)
    coverage_weeks = max(coverage_weeks, 0.5)
    safety_weeks = max(safety_weeks, 0)
    min_weeks_history = max(min_weeks_history, 1)

    if demand_method == 'sma3':
        lookback_days = 21
        weeks_divisor = 3.0
    elif demand_method == 'sma2':
        lookback_days = 14
        weeks_divisor = 2.0
    else:
        lookback_days = 7
        weeks_divisor = 1.0

    cutoff = today - timedelta(days=lookback_days)
    lead_time_weeks = lead_time_days / 7.0
    total_weeks_needed = coverage_weeks + safety_weeks + lead_time_weeks

    week_expr = db.func.strftime('%Y-%W', DistributionRecord.event_date)
    sales_q = (
        db.session.query(
            Product.id.label('product_id'),
            Product.sku.label('sku'),
            Product.name.label('name'),
            db.func.sum(DistributionRecord.quantity).label('qty_period'),
            db.func.count(db.func.distinct(week_expr)).label('weeks_with_sales')
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .join(Store, DistributionRecord.store_id == Store.id)
        .filter(DistributionRecord.event_date >= cutoff)
    )

    if store_filter:
        sales_q = sales_q.filter(Store.name == store_filter)

    sales_rows = (
        sales_q
        .group_by(Product.id, Product.sku, Product.name)
        .having(db.func.count(db.func.distinct(week_expr)) >= min_weeks_history)
        .all()
    )

    latest_cd_date = db.session.query(db.func.max(StockCD.as_of_date)).scalar() or today
    cd_rows = (
        db.session.query(
            StockCD.product_id,
            db.func.sum(StockCD.quantity).label('cd_qty')
        )
        .filter(StockCD.as_of_date == latest_cd_date)
        .group_by(StockCD.product_id)
        .all()
    )
    cd_stock_map = {r.product_id: int(r.cd_qty) for r in cd_rows}

    result = []
    for r in sales_rows:
        demand_per_week = float(r.qty_period) / weeks_divisor if weeks_divisor > 0 else 0.0
        required_units = demand_per_week * total_weeks_needed
        available_units = cd_stock_map.get(r.product_id, 0)
        suggested = max(int(round(required_units)) - available_units, 0)

        result.append({
            "sku": r.sku,
            "name": r.name,
            "demand_per_week": round(demand_per_week, 2),
            "available_cd": available_units,
            "required_units": round(required_units, 2),
            "suggested": suggested,
            "campaign_tag": campaign_tag,
        })

    result.sort(key=lambda x: x["suggested"], reverse=True)

    has_cd_stock = bool(cd_rows)
    has_sales = bool(sales_rows)

    forecast_run_id = None
    if request.method == 'POST' and result:
        from uuid import uuid4
        forecast_run_id = str(uuid4())
        forecast_run = Run(
            run_id=forecast_run_id,
            run_type='forecast_v2',
            folio=campaign_tag or None,
            store_filter=store_filter or None,
            notes=f"method={demand_method}, lead={lead_time_days}d, coverage={coverage_weeks}w, safety={safety_weeks}w",
            status='completed',
            mode=demand_method
        )
        db.session.add(forecast_run)

        for r in result:
            product = Product.query.filter_by(sku=str(r['sku'])).first()
            if product:
                fr = ForecastResult(
                    run_id=forecast_run_id,
                    product_id=product.id,
                    sku=str(r['sku']),
                    name=r['name'],
                    demand_per_week=r['demand_per_week'],
                    available_cd=r['available_cd'],
                    required_units=r['required_units'],
                    suggested=r['suggested'],
                    campaign_tag=campaign_tag or None
                )
                db.session.add(fr)

        db.session.commit()

    result_limited = result[:50]

    return render_template(
        'purchase_forecast_v2.html',
        rows=result_limited,
        total_skus=len(result),
        stores=stores,
        lead_time_days=lead_time_days,
        coverage_weeks=coverage_weeks,
        safety_weeks=safety_weeks,
        min_weeks_history=min_weeks_history,
        demand_method=demand_method,
        store_filter=store_filter,
        campaign_tag=campaign_tag,
        today=today,
        cd_snapshot_date=latest_cd_date,
        has_cd_stock=has_cd_stock,
        has_sales=has_sales,
        forecast_run_id=forecast_run_id,
    )

@app.route('/export_purchase_forecast_v2', methods=['POST'])
@login_required
@require_permission('forecast_v2:run')
def export_purchase_forecast_v2():
    """Export Purchase Forecast V2 results to Excel."""
    from datetime import datetime
    today = date.today()

    try:
        lead_time_days = int(request.form.get('lead_time_days', 14))
    except ValueError:
        lead_time_days = 14
    try:
        coverage_weeks = float(request.form.get('coverage_weeks', 4))
    except ValueError:
        coverage_weeks = 4.0
    try:
        safety_weeks = float(request.form.get('safety_weeks', 1))
    except ValueError:
        safety_weeks = 1.0
    try:
        min_weeks_history = int(request.form.get('min_weeks_history', 1))
    except ValueError:
        min_weeks_history = 1

    demand_method = request.form.get('demand_method', 'sma3').strip()
    store_filter = request.form.get('store_filter', '').strip()
    campaign_tag = request.form.get('campaign_tag', '').strip()

    lead_time_days = max(lead_time_days, 0)
    coverage_weeks = max(coverage_weeks, 0.5)
    safety_weeks = max(safety_weeks, 0)
    min_weeks_history = max(min_weeks_history, 1)

    if demand_method == 'sma3':
        lookback_days = 21
        weeks_divisor = 3.0
    elif demand_method == 'sma2':
        lookback_days = 14
        weeks_divisor = 2.0
    else:
        lookback_days = 7
        weeks_divisor = 1.0

    cutoff = today - timedelta(days=lookback_days)
    lead_time_weeks = lead_time_days / 7.0
    total_weeks_needed = coverage_weeks + safety_weeks + lead_time_weeks

    week_expr = db.func.strftime('%Y-%W', DistributionRecord.event_date)
    sales_q = (
        db.session.query(
            Product.id.label('product_id'),
            Product.sku.label('sku'),
            Product.name.label('name'),
            db.func.sum(DistributionRecord.quantity).label('qty_period'),
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .join(Store, DistributionRecord.store_id == Store.id)
        .filter(DistributionRecord.event_date >= cutoff)
    )

    if store_filter:
        sales_q = sales_q.filter(Store.name == store_filter)

    sales_rows = (
        sales_q
        .group_by(Product.id, Product.sku, Product.name)
        .having(db.func.count(db.func.distinct(week_expr)) >= min_weeks_history)
        .all()
    )

    latest_cd_date = db.session.query(db.func.max(StockCD.as_of_date)).scalar() or today
    cd_rows = (
        db.session.query(
            StockCD.product_id,
            db.func.sum(StockCD.quantity).label('cd_qty')
        )
        .filter(StockCD.as_of_date == latest_cd_date)
        .group_by(StockCD.product_id)
        .all()
    )
    cd_stock_map = {r.product_id: int(r.cd_qty) for r in cd_rows}

    rows = []
    for r in sales_rows:
        demand_per_week = float(r.qty_period) / weeks_divisor if weeks_divisor > 0 else 0.0
        required_units = demand_per_week * total_weeks_needed
        available_units = cd_stock_map.get(r.product_id, 0)
        suggested = max(int(round(required_units)) - available_units, 0)

        rows.append({
            "SKU": str(r.sku),
            "Producto": r.name,
            "Demanda/Semana": round(demand_per_week, 2),
            "Stock CD": available_units,
            "Requerido": round(required_units, 2),
            "Sugerido Compra": suggested,
            "Campana": campaign_tag or "",
            "Metodo": demand_method,
            "Lead Time Dias": lead_time_days,
            "Cobertura Sem": coverage_weeks,
            "Safety Sem": safety_weeks,
        })

    rows.sort(key=lambda x: x["Sugerido Compra"], reverse=True)

    if not rows:
        rows.append({
            "SKU": "",
            "Producto": "Sin datos",
            "Demanda/Semana": 0,
            "Stock CD": 0,
            "Requerido": 0,
            "Sugerido Compra": 0,
            "Campana": campaign_tag or "",
            "Metodo": demand_method,
            "Lead Time Dias": lead_time_days,
            "Cobertura Sem": coverage_weeks,
            "Safety Sem": safety_weeks,
        })

    df = pd.DataFrame(rows)
    df["SKU"] = df["SKU"].astype(str)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Forecast Compra V2")
        ws = writer.sheets["Forecast Compra V2"]
        for cell in ws["A"]:
            cell.number_format = "@"

    output.seek(0)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    tag_part = f"_{campaign_tag.replace(' ', '_')}" if campaign_tag else ""
    fname = f"forecast_compra_v2_{ts}{tag_part}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route('/api/forecast_v2/chart_data', methods=['GET'])
@login_required
@require_permission('forecast_v2:view')
def api_forecast_v2_chart_data():
    """
    API endpoint returning forecast + sales history data as JSON for Plotly charts.
    Query params: sku (optional), store_filter (optional), demand_method, lead_time_days, coverage_weeks, safety_weeks
    """
    from flask import jsonify
    today = date.today()

    sku_filter = request.args.get('sku', '').strip()
    store_filter = request.args.get('store_filter', '').strip()
    demand_method = request.args.get('demand_method', 'sma3').strip()

    try:
        lead_time_days = int(request.args.get('lead_time_days', 14))
    except ValueError:
        lead_time_days = 14
    try:
        coverage_weeks = float(request.args.get('coverage_weeks', 4))
    except ValueError:
        coverage_weeks = 4.0
    try:
        safety_weeks = float(request.args.get('safety_weeks', 1))
    except ValueError:
        safety_weeks = 1.0

    lead_time_days = max(lead_time_days, 0)
    coverage_weeks = max(coverage_weeks, 0.5)
    safety_weeks = max(safety_weeks, 0)

    if demand_method == 'sma3':
        lookback_days = 21
        weeks_divisor = 3.0
    elif demand_method == 'sma2':
        lookback_days = 14
        weeks_divisor = 2.0
    else:
        lookback_days = 7
        weeks_divisor = 1.0

    lead_time_weeks = lead_time_days / 7.0
    total_weeks_needed = coverage_weeks + safety_weeks + lead_time_weeks

    history_weeks = 12
    history_cutoff = today - timedelta(days=history_weeks * 7)

    week_expr = db.func.strftime('%Y-%W', DistributionRecord.event_date)

    history_q = (
        db.session.query(
            Product.sku.label('sku'),
            Product.name.label('name'),
            week_expr.label('week'),
            db.func.sum(DistributionRecord.quantity).label('qty')
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .join(Store, DistributionRecord.store_id == Store.id)
        .filter(DistributionRecord.event_date >= history_cutoff)
    )

    if sku_filter:
        history_q = history_q.filter(Product.sku == sku_filter)
    if store_filter:
        history_q = history_q.filter(Store.name == store_filter)

    history_rows = (
        history_q
        .group_by(Product.sku, Product.name, week_expr)
        .order_by(Product.sku, week_expr)
        .all()
    )

    history_by_sku = {}
    for r in history_rows:
        sku_str = str(r.sku)
        if sku_str not in history_by_sku:
            history_by_sku[sku_str] = {'name': r.name, 'weeks': [], 'quantities': []}
        history_by_sku[sku_str]['weeks'].append(r.week)
        history_by_sku[sku_str]['quantities'].append(int(r.qty))

    cutoff = today - timedelta(days=lookback_days)
    sales_q = (
        db.session.query(
            Product.id.label('product_id'),
            Product.sku.label('sku'),
            Product.name.label('name'),
            db.func.sum(DistributionRecord.quantity).label('qty_period'),
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .join(Store, DistributionRecord.store_id == Store.id)
        .filter(DistributionRecord.event_date >= cutoff)
    )

    if sku_filter:
        sales_q = sales_q.filter(Product.sku == sku_filter)
    if store_filter:
        sales_q = sales_q.filter(Store.name == store_filter)

    sales_rows = (
        sales_q
        .group_by(Product.id, Product.sku, Product.name)
        .all()
    )

    latest_cd_date = db.session.query(db.func.max(StockCD.as_of_date)).scalar() or today
    cd_rows = (
        db.session.query(
            StockCD.product_id,
            db.func.sum(StockCD.quantity).label('cd_qty')
        )
        .filter(StockCD.as_of_date == latest_cd_date)
        .group_by(StockCD.product_id)
        .all()
    )
    cd_stock_map = {r.product_id: int(r.cd_qty) for r in cd_rows}

    forecast_data = []
    for r in sales_rows:
        demand_per_week = float(r.qty_period) / weeks_divisor if weeks_divisor > 0 else 0.0
        required_units = demand_per_week * total_weeks_needed
        available_units = cd_stock_map.get(r.product_id, 0)
        suggested = max(int(round(required_units)) - available_units, 0)

        sku_str = str(r.sku)
        hist = history_by_sku.get(sku_str, {'weeks': [], 'quantities': []})

        forecast_data.append({
            "sku": sku_str,
            "name": r.name,
            "demand_per_week": round(demand_per_week, 2),
            "available_cd": available_units,
            "required_units": round(required_units, 2),
            "suggested": suggested,
            "history_weeks": hist['weeks'],
            "history_quantities": hist['quantities'],
        })

    forecast_data.sort(key=lambda x: x["suggested"], reverse=True)

    all_skus = [{"sku": str(p.sku), "name": p.name} for p in Product.query.order_by(Product.sku).all()]

    total_suggested = sum(f['suggested'] for f in forecast_data)
    total_demand = sum(f['demand_per_week'] for f in forecast_data)
    total_stock_cd = sum(f['available_cd'] for f in forecast_data)
    skus_with_purchase = sum(1 for f in forecast_data if f['suggested'] > 0)

    if sku_filter:
        forecast_response = [f for f in forecast_data if f['sku'] == sku_filter]
        if not forecast_response:
            hist = history_by_sku.get(sku_filter, {'weeks': [], 'quantities': []})
            product = Product.query.filter_by(sku=sku_filter).first()
            if product:
                forecast_response = [{
                    "sku": sku_filter,
                    "name": product.name,
                    "demand_per_week": 0,
                    "available_cd": cd_stock_map.get(product.id, 0),
                    "required_units": 0,
                    "suggested": 0,
                    "history_weeks": hist['weeks'],
                    "history_quantities": hist['quantities'],
                }]
    else:
        forecast_response = forecast_data[:100]

    return jsonify({
        "forecast": forecast_response,
        "all_skus": all_skus,
        "kpis": {
            "total_suggested": total_suggested,
            "total_demand_per_week": round(total_demand, 2),
            "total_stock_cd": total_stock_cd,
            "skus_with_purchase": skus_with_purchase,
            "total_skus": len(forecast_data),
        },
        "params": {
            "demand_method": demand_method,
            "lead_time_days": lead_time_days,
            "coverage_weeks": coverage_weeks,
            "safety_weeks": safety_weeks,
            "store_filter": store_filter,
            "sku_filter": sku_filter,
        }
    })


@app.route('/api/forecast_v2', methods=['GET'])
@login_required
@require_permission('forecast_v2:view')
def api_forecast_v2():
    """
    Detailed forecast API for a specific SKU.
    Query params:
      - sku (required): SKU code
      - store (optional): store name, empty = all stores
      - horizon_weeks (default 8): forecast horizon
      - history_weeks (default 12): weeks of history to return
      - lead_time_weeks (default 4): lead time in weeks
      - safety_pct (default 0.10): safety stock percentage
    """
    import math
    from flask import jsonify
    today = date.today()

    sku_param = request.args.get('sku', '').strip()
    if not sku_param:
        return jsonify({"error": "sku parameter is required"}), 400

    store_param = request.args.get('store', '').strip()

    try:
        horizon_weeks = int(request.args.get('horizon_weeks', 8))
    except ValueError:
        horizon_weeks = 8
    try:
        history_weeks = int(request.args.get('history_weeks', 12))
    except ValueError:
        history_weeks = 12
    try:
        lead_time_weeks = float(request.args.get('lead_time_weeks', 4))
    except ValueError:
        lead_time_weeks = 4.0
    try:
        safety_pct = float(request.args.get('safety_pct', 0.10))
    except ValueError:
        safety_pct = 0.10

    horizon_weeks = max(horizon_weeks, 1)
    history_weeks = max(history_weeks, 1)
    lead_time_weeks = max(lead_time_weeks, 0)
    safety_pct = max(safety_pct, 0)

    product = Product.query.filter_by(sku=sku_param).first()
    if not product:
        log_audit(
            action='forecast_v2.run',
            status='fail',
            message=f"SKU not found: {sku_param}",
            metadata={'sku': sku_param, 'store': store_param or 'ALL'}
        )
        return jsonify({"error": f"SKU '{sku_param}' not found"}), 404

    history_cutoff = today - timedelta(days=history_weeks * 7)

    monday_expr = db.func.date(DistributionRecord.event_date, 'weekday 0', '-6 days')
    history_q = (
        db.session.query(
            monday_expr.label('week_start'),
            db.func.sum(DistributionRecord.quantity).label('units')
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .join(Store, DistributionRecord.store_id == Store.id)
        .filter(DistributionRecord.product_id == product.id)
        .filter(DistributionRecord.event_date >= history_cutoff)
    )

    if store_param:
        history_q = history_q.filter(Store.name == store_param)

    history_rows = (
        history_q
        .group_by(monday_expr)
        .order_by(db.text('week_start'))
        .all()
    )

    history_data = [{"week": str(r.week_start), "units": int(r.units)} for r in history_rows]

    if len(history_data) >= 4:
        last4_units = [h['units'] for h in history_data[-4:]]
        avg_last4 = sum(last4_units) / 4.0
    elif history_data:
        avg_last4 = sum(h['units'] for h in history_data) / len(history_data)
    else:
        avg_last4 = 0

    if not history_data:
        log_audit(
            action='forecast_v2.run',
            status='success',
            message=f"No history data for SKU {sku_param}",
            metadata={
                'sku': sku_param,
                'store': store_param or 'ALL',
                'horizon_weeks': horizon_weeks,
                'lead_time_weeks': lead_time_weeks,
                'safety_pct': safety_pct,
                'kpis': {'avg_last4': 0, 'forecast_total': 0, 'stock_cd': 0, 'suggested_purchase': 0, 'weeks_of_cover': 0}
            }
        )
        return jsonify({
            "sku": str(sku_param),
            "store": store_param if store_param else "ALL",
            "history": [],
            "forecast": [],
            "bands": {"lower": [], "upper": []},
            "kpis": {
                "avg_last4": 0,
                "forecast_total": 0,
                "stock_cd": 0,
                "suggested_purchase": 0,
                "weeks_of_cover": 0
            }
        })

    forecast_data = []
    current_week = today - timedelta(days=today.weekday())
    for i in range(horizon_weeks):
        week_start = current_week + timedelta(weeks=i)
        forecast_units = round(avg_last4)
        forecast_data.append({"week": week_start.strftime('%Y-%m-%d'), "units": forecast_units})

    if len(history_data) >= 4:
        last4_units = [h['units'] for h in history_data[-4:]]
        std_dev = (sum((x - avg_last4) ** 2 for x in last4_units) / 4) ** 0.5
    else:
        std_dev = avg_last4 * 0.2

    bands_lower = [max(0, round(avg_last4 - 1.5 * std_dev)) for _ in range(horizon_weeks)]
    bands_upper = [round(avg_last4 + 1.5 * std_dev) for _ in range(horizon_weeks)]

    latest_cd_date = db.session.query(db.func.max(StockCD.as_of_date)).scalar() or today
    cd_stock_row = (
        db.session.query(db.func.sum(StockCD.quantity).label('qty'))
        .filter(StockCD.product_id == product.id)
        .filter(StockCD.as_of_date == latest_cd_date)
        .first()
    )
    stock_cd = int(cd_stock_row.qty) if cd_stock_row and cd_stock_row.qty else 0

    forecast_total = sum(f['units'] for f in forecast_data)
    demand_over_lead_time = avg_last4 * lead_time_weeks
    safety_units = math.ceil(demand_over_lead_time * safety_pct)
    suggested_purchase = max(0, math.ceil(demand_over_lead_time + safety_units - stock_cd))

    weeks_of_cover = round(stock_cd / max(avg_last4, 1), 1)

    kpis = {
        "avg_last4": round(avg_last4, 2),
        "forecast_total": forecast_total,
        "stock_cd": stock_cd,
        "suggested_purchase": suggested_purchase,
        "weeks_of_cover": weeks_of_cover
    }

    log_audit(
        action='forecast_v2.run',
        status='success',
        message=f"Forecast for SKU {sku_param}",
        metadata={
            'sku': sku_param,
            'store': store_param or 'ALL',
            'horizon_weeks': horizon_weeks,
            'lead_time_weeks': lead_time_weeks,
            'safety_pct': safety_pct,
            'kpis': kpis
        }
    )

    return jsonify({
        "sku": str(sku_param),
        "store": store_param if store_param else "ALL",
        "history": history_data,
        "forecast": forecast_data,
        "bands": {
            "lower": bands_lower,
            "upper": bands_upper
        },
        "kpis": kpis
    })


@app.route('/export_cd_remanente', methods=['GET'])
@login_required
@require_permission('distribution:export')
def export_cd_remanente():
    """Exporta a Excel el remanente de CD solo de los SKUs usados en una corrida específica."""
    today = date.today()
    
    run_id = request.args.get("run_id", "").strip()
    if not run_id:
        latest_run = (
            PredictionRun.query
            .filter(PredictionRun.run_id.isnot(None))
            .filter(PredictionRun.run_id != "")
            .order_by(PredictionRun.created_at.desc())
            .first()
        )
        run_id = latest_run.run_id if latest_run else None

    if not run_id:
        df_empty = pd.DataFrame([{
            "SKU": "",
            "Producto": "",
            "Stock CD disponible": "",
            "Corrida": "",
            "Fecha": today.strftime("%Y-%m-%d"),
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_empty.to_excel(writer, index=False, sheet_name="Remanente")
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=f"remanente_cd_{today.strftime('%Y%m%d')}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    preds = (
        db.session.query(Prediction.product_id)
        .filter(Prediction.run_id == run_id)
        .distinct()
        .all()
    )
    product_ids = [p.product_id for p in preds]

    cd_rows = (
        db.session.query(StockCD, Product)
        .join(Product, StockCD.product_id == Product.id)
        .filter(
            StockCD.as_of_date == today,
            StockCD.product_id.in_(product_ids)
        )
        .order_by(Product.sku.asc())
        .all()
    )

    data = []
    for cd, prod in cd_rows:
        data.append({
            "SKU": prod.sku,
            "Producto": prod.name,
            "Stock CD disponible": cd.quantity,
            "Corrida": run_id[:8],
            "Fecha": today.strftime("%Y-%m-%d"),
        })

    if not data:
        data.append({
            "SKU": "",
            "Producto": "",
            "Stock CD disponible": "",
            "Corrida": run_id[:8] if run_id else "",
            "Fecha": today.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Remanente")
    output.seek(0)

    run_id_short = run_id[:8] if run_id else "norun"
    return send_file(
        output,
        as_attachment=True,
        download_name=f"FastDSO_CD_Remainder_{run_id_short}_{today.strftime('%Y%m%d')}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
@require_permission('sales:upload')
def upload():
    # ¿hay stock de tienda cargado?
    has_stock = db.session.query(StockSnapshot.id).first() is not None
    require_stock_confirm = not has_stock  # si NO hay stock, pedimos confirmación

    if request.method == 'POST':
        # si no hay stock previo, exigir el checkbox
        if require_stock_confirm and not request.form.get('confirm_no_stock'):
            flash('No hay stock de tienda cargado. Confirma que quieres continuar.', 'warning')
            return redirect(url_for('upload'))

        file = request.files.get('file')
        if not file:
            flash('Sube un archivo CSV o Excel.', 'warning')
            return redirect(url_for('upload'))

        filename = file.filename.lower()
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file, dtype=str)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file, dtype=str)
            else:
                flash('Formato no soportado. Usa .csv o .xlsx', 'danger')
                return redirect(url_for('upload'))
        except Exception as e:
            flash(f'Error leyendo archivo: {e}', 'danger')
            return redirect(url_for('upload'))

        # normalizar columnas
        df.columns = [str(c).strip().lower() for c in df.columns]
        needed = {'sku', 'product_name', 'store', 'quantity', 'date'}
        if not needed.issubset(set(df.columns)):
            flash('El archivo debe tener las columnas: sku, product_name, store, quantity, date', 'danger')
            return redirect(url_for('upload'))

        # leer parámetros del formulario primero
        analysis_mode = request.form.get('analysis_mode', 'sma3_min3')
        meta = {
            "folio": request.form.get('folio', '').strip() or None,
            "responsable": request.form.get('responsable', '').strip() or None,
            "categoria": request.form.get('categoria', '').strip() or None,
            "fecha_doc": request.form.get('fecha_doc', '').strip() or None,
        }

        # Create sales_upload run
        from uuid import uuid4
        sales_run_id = str(uuid4())
        user_id = current_user.id if current_user.is_authenticated else None
        sales_run = Run(
            run_id=sales_run_id,
            run_type='sales_upload',
            user_id=user_id,
            folio=meta.get("folio"),
            responsable=meta.get("responsable"),
            categoria=meta.get("categoria"),
            notes=meta.get("fecha_doc"),
            status='completed',
            rows_count=len(df)
        )
        db.session.add(sales_run)
        db.session.flush()

        created = 0
        for _, row in df.iterrows():
            sku = str(row['sku']).strip()
            pname = str(row['product_name']).strip()
            store_name = str(row['store']).strip()
            qty = int(row['quantity'])
            event_date = pd.to_datetime(row['date']).date()

            # upsert product
            product = Product.query.filter_by(sku=sku).first()
            if not product:
                product = Product(sku=sku, name=pname)
                db.session.add(product)
                db.session.flush()

            # upsert store
            store = Store.query.filter_by(name=store_name).first()
            if not store:
                store = Store(name=store_name)
                db.session.add(store)
                db.session.flush()

            # guardar distribución histórica con run_id
            dist = DistributionRecord(
                product_id=product.id,
                store_id=store.id,
                quantity=qty,
                event_date=event_date,
                run_id=sales_run_id
            )
            db.session.add(dist)
            created += 1

        db.session.commit()
        
        distinct_skus = df['sku'].nunique() if 'sku' in df.columns else 0
        distinct_stores = df['store'].nunique() if 'store' in df.columns else 0
        log_audit(
            action="sales.upload",
            message=f"Cargados {created} registros de ventas",
            entity_type="DistributionRecord",
            run_id=sales_run_id,
            metadata={
                "filename": filename,
                "rows_count": created,
                "distinct_skus": distinct_skus,
                "distinct_stores": distinct_stores,
                "mode": analysis_mode,
                "folio": meta.get("folio"),
                "responsable": meta.get("responsable"),
                "categoria": meta.get("categoria")
            }
        )

        run_id, n_preds = generate_predictions(mode=analysis_mode, meta=meta, df=df, sales_run_id=sales_run_id)

        if n_preds == 0:
            flash(
                'Carga exitosa de ventas, pero no se generaron predicciones. '
                'Revisa que cada SKU/Tienda tenga al menos el número mínimo de semanas '
                'de venta requerido por el método seleccionado.',
                'warning'
            )
        else:
            flash(
                f'Carga exitosa: {created} registros. '
                f'Predicciones generadas / actualizadas: {n_preds}',
                'success'
            )

        return redirect(url_for('dashboard'))

    # GET
    return render_template('upload.html', require_stock_confirm=require_stock_confirm)

@app.route('/stock', methods=['GET', 'POST'])
@login_required
@require_permission('stock_store:upload')
def upload_stock():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            flash('Sube un archivo CSV o Excel', 'warning')
            return redirect(url_for('upload_stock'))

        filename = f.filename.lower()
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(f, dtype=str)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(f, dtype=str)
            else:
                flash('Formato no soportado. Usa .csv o .xlsx', 'danger')
                return redirect(url_for('upload_stock'))
        except Exception as e:
            flash(f'Error leyendo archivo de stock: {e}', 'danger')
            return redirect(url_for('upload_stock'))

        # normalizar encabezados
        df.columns = [str(c).strip() for c in df.columns]
        cols_lower = {c.lower(): c for c in df.columns}

        sku_col = cols_lower.get('sku') or cols_lower.get('codigo')
        prod_col = cols_lower.get('producto') or cols_lower.get('product') or cols_lower.get('product_name') or cols_lower.get('nombre')

        if not sku_col:
            flash('El archivo debe tener columna "SKU" o "Codigo".', 'danger')
            return redirect(url_for('upload_stock'))

        # columnas de tiendas = todo lo que no es SKU ni Producto
        store_cols = [c for c in df.columns if c not in (sku_col, prod_col)]
        if not store_cols:
            flash('No se encontraron columnas de tiendas.', 'danger')
            return redirect(url_for('upload_stock'))

        # limpieza básica
        df[sku_col] = df[sku_col].astype('string').str.strip()
        if prod_col:
            df[prod_col] = df[prod_col].astype('string').str.strip()

        # pasar columnas de tiendas a int
        for sc in store_cols:
            df[sc] = pd.to_numeric(df[sc], errors='coerce').fillna(0).astype(int)

        today = date.today()
        created = 0
        updated = 0

        # 🔥 CACHE para no golpear la BD por cada celda
        # 1) cache de productos existentes por SKU
        existing_products = {
            p.sku: p for p in Product.query.all()
        }

        # 2) cache de tiendas existentes por nombre
        existing_stores = {
            s.name: s for s in Store.query.all()
        }

        # 3) cache de snapshots del día por (product_id, store_id)
        existing_snapshots = {
            (ss.product_id, ss.store_id): ss
            for ss in StockSnapshot.query.filter_by(as_of_date=today).all()
        }

        # ahora iteramos el dataframe
        for _, row in df.iterrows():
            sku = row[sku_col]
            if not sku:
                continue

            pname = row[prod_col] if prod_col and pd.notna(row[prod_col]) else sku

            # producto: usar cache o crear
            prod = existing_products.get(sku)
            if not prod:
                prod = Product(sku=sku, name=str(pname))
                db.session.add(prod)
                db.session.flush()  # para tener prod.id
                existing_products[sku] = prod  # meterlo al cache

            # por cada tienda
            for sc in store_cols:
                store_name = sc  # el nombre de la columna es el nombre de la tienda
                qty = int(row[sc])

                # tienda: usar cache o crear
                store = existing_stores.get(store_name)
                if not store:
                    store = Store(name=store_name)
                    db.session.add(store)
                    db.session.flush()
                    existing_stores[store_name] = store

                key = (prod.id, store.id)
                snapshot = existing_snapshots.get(key)

                if snapshot:
                    # actualizar
                    snapshot.quantity = qty
                    updated += 1
                else:
                    # crear
                    ss = StockSnapshot(
                        as_of_date=today,
                        product_id=prod.id,
                        store_id=store.id,
                        quantity=qty
                    )
                    db.session.add(ss)
                    existing_snapshots[key] = ss
                    created += 1

        db.session.commit()
        log_audit(
            action="stock_store.upload",
            message=f"Stock tiendas cargado: {created} nuevos, {updated} actualizados",
            entity_type="StockSnapshot",
            metadata={
                "filename": filename,
                "created": created,
                "updated": updated,
                "distinct_skus": len(df),
                "store_count": len(store_cols),
                "as_of_date": str(today)
            }
        )
        flash(f'Stock cargado. Nuevos: {created}. Actualizados: {updated}.', 'success')
        return redirect(url_for('dashboard'))

    # GET
    return render_template('upload_stock.html')

from datetime import date
from sqlalchemy import func
import io

@app.route('/purchase_projection', methods=['GET', 'POST'])
@login_required
@require_permission('forecast_v2:view')
def purchase_projection():
    # Parámetros por defecto
    lead_time_w = 2
    safety_pct = 0

    if request.method == 'POST':
        try:
            lead_time_w = max(int(request.form.get('lead_time_w', 2)), 0)
        except:
            lead_time_w = 2
        try:
            safety_pct = max(float(request.form.get('safety_pct', 0)), 0.0)
        except:
            safety_pct = 0.0

    # Última semana objetivo con predicciones
    latest_week = db.session.query(func.max(Prediction.target_period_start)).scalar()
    preview = []
    if not latest_week:
        return render_template('purchase_projection.html',
                               latest_week=None, preview=preview,
                               lead_time_w=lead_time_w, safety_pct=safety_pct)

    # Demanda total por SKU (sumando todas las tiendas) de la última corrida
    preds = (
        db.session.query(Product.id, Product.sku, Product.name, func.sum(Prediction.quantity).label('demand'))
        .join(Product, Prediction.product_id == Product.id)
        .filter(Prediction.target_period_start == latest_week)
        .group_by(Product.id, Product.sku, Product.name)
        .order_by(Product.sku.asc())
        .all()
    )

    # Stock CD de hoy
    today = date.today()
    cd_rows = (
        db.session.query(StockCD.product_id, StockCD.quantity)
        .filter(StockCD.as_of_date == today)
        .all()
    )
    cd_map = {pid: qty for pid, qty in cd_rows}

    # Cálculo:
    # necesidad = max(demanda * (lead_time + 1) * (1 + safety) - stock_cd, 0)
    result = []
    for pid, sku, name, demand in preds:
        demand = int(demand or 0)
        stock_cd = int(cd_map.get(pid, 0))
        mult = (lead_time_w + 1)  # demanda de horizonte: esta semana + LT semanas
        safety_factor = 1.0 + (safety_pct / 100.0)
        need = max(int(round(demand * mult * safety_factor)) - stock_cd, 0)

        result.append({
            "product_id": pid,
            "SKU": str(sku),
            "Producto": name,
            "Demanda base (últ. dist.)": demand,
            "Stock CD hoy": stock_cd,
            "Lead time (sem)": lead_time_w,
            "Colchón (%)": safety_pct,
            "Necesidad Neta": need,
        })

    # Preview top 20 con necesidad > 0
    result_sorted = sorted(result, key=lambda x: x["Necesidad Neta"], reverse=True)
    preview = [r for r in result_sorted if r["Necesidad Neta"] > 0][:20]

    return render_template('purchase_projection.html',
                           latest_week=latest_week,
                           preview=preview,
                           lead_time_w=lead_time_w,
                           safety_pct=safety_pct)

@app.route('/export_purchase_projection', methods=['POST'])
@login_required
@require_permission('forecast_v2:run')
def export_purchase_projection():
    try:
        lead_time_w = max(int(request.form.get('lead_time_w', 2)), 0)
    except:
        lead_time_w = 2
    try:
        safety_pct = max(float(request.form.get('safety_pct', 0)), 0.0)
    except:
        safety_pct = 0.0

    latest_week = db.session.query(func.max(Prediction.target_period_start)).scalar()
    if not latest_week:
        flash('No hay distribución reciente para proyectar.', 'warning')
        return redirect(url_for('purchase_projection'))

    preds = (
        db.session.query(Product.id, Product.sku, Product.name, func.sum(Prediction.quantity).label('demand'))
        .join(Product, Prediction.product_id == Product.id)
        .filter(Prediction.target_period_start == latest_week)
        .group_by(Product.id, Product.sku, Product.name)
        .order_by(Product.sku.asc())
        .all()
    )

    today = date.today()
    cd_rows = (
        db.session.query(StockCD.product_id, StockCD.quantity)
        .filter(StockCD.as_of_date == today)
        .all()
    )
    cd_map = {pid: qty for pid, qty in cd_rows}

    rows = []
    for pid, sku, name, demand in preds:
        demand = int(demand or 0)
        stock_cd = int(cd_map.get(pid, 0))
        mult = (lead_time_w + 1)
        safety_factor = 1.0 + (safety_pct / 100.0)
        need = max(int(round(demand * mult * safety_factor)) - stock_cd, 0)

        rows.append({
            "SKU": str(sku),
            "Producto": name,
            "Demanda base (últ. dist.)": demand,
            "Stock CD hoy": stock_cd,
            "Lead time (sem)": lead_time_w,
            "Colchón (%)": safety_pct,
            "Necesidad Neta": need,
            "Semana objetivo": latest_week.strftime("%Y-%m-%d"),
            "Fecha cálculo": today.strftime("%Y-%m-%d"),
        })

    # Excel (SKU como texto)
    import pandas as pd
    output = io.BytesIO()
    df = pd.DataFrame(rows)
    df["SKU"] = df["SKU"].astype(str)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Proyeccion")
        ws = writer.sheets["Proyeccion"]
        for cell in ws["A"]:
            cell.number_format = "@"
    output.seek(0)

    fname = f"proyeccion_compra_{today.strftime('%Y%m%d')}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ======================================================
# RESET DE DATOS
# ======================================================

@app.route('/reset_sales', methods=['POST'])
@login_required
@require_permission('admin:reset')
def reset_sales():
    count = db.session.query(DistributionRecord).count()
    db.session.query(DistributionRecord).delete()
    db.session.commit()
    log_audit(
        action="reset.sales",
        message=f"Eliminados {count} registros de ventas",
        entity_type="DistributionRecord",
        metadata={"deleted_count": count}
    )
    flash('✅ Se eliminaron todos los registros de ventas cargadas.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/reset_store_stock', methods=['POST'])
@login_required
@require_permission('admin:reset')
def reset_store_stock():
    count = db.session.query(StockSnapshot).count()
    db.session.query(StockSnapshot).delete()
    db.session.commit()
    log_audit(
        action="reset.stock_store",
        message=f"Eliminados {count} snapshots de stock tiendas",
        entity_type="StockSnapshot",
        metadata={"deleted_count": count}
    )
    flash('✅ Se eliminó todo el stock de tiendas.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/reset_predictions', methods=['POST'])
@login_required
@require_permission('admin:reset')
def reset_predictions():
    count = db.session.query(Prediction).count()
    db.session.query(Prediction).delete()
    db.session.commit()
    log_audit(
        action="reset.predictions",
        message=f"Eliminadas {count} predicciones",
        entity_type="Prediction",
        metadata={"deleted_count": count}
    )
    flash('✅ Se eliminaron todas las distribuciones sugeridas.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/reset_stock_cd', methods=['POST'])
@login_required
@require_permission('admin:reset')
def reset_stock_cd():
    count = StockCD.query.count()
    StockCD.query.delete()
    db.session.commit()
    log_audit(
        action="reset.stock_cd",
        message=f"Eliminados {count} registros stock CD",
        entity_type="StockCD",
        metadata={"deleted_count": count}
    )
    flash('Stock CD completamente reiniciado.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/stock_cd', methods=['GET', 'POST'])
@login_required
@require_permission('stock_cd:upload')
def upload_stock_cd():
    from datetime import date

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash('Sube un archivo CSV o Excel.', 'warning')
            return redirect(url_for('upload_stock_cd'))

        modo = request.form.get('modo', 'replace')  # 'replace' o 'add'
        snapshot_date = date.today()

        filename = file.filename.lower()
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file, dtype=str)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file, dtype=str)
            else:
                flash('Formato no soportado. Usa .csv o .xlsx', 'danger')
                return redirect(url_for('upload_stock_cd'))
        except Exception as e:
            flash(f'Error leyendo archivo: {e}', 'danger')
            return redirect(url_for('upload_stock_cd'))

        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'sku' not in df.columns or 'quantity' not in df.columns:
            flash('El archivo debe tener columnas sku y quantity.', 'danger')
            return redirect(url_for('upload_stock_cd'))

        df['sku'] = df['sku'].astype(str).str.strip()
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

        # Detectar columna de nombre del producto si existe
        name_col = None
        for col in ['product_name', 'producto', 'name', 'nombre']:
            if col in df.columns:
                name_col = col
                break

        # modo replace → borrar snapshot de ESA fecha antes de cargar
        if modo == 'replace':
            StockCD.query.filter_by(as_of_date=snapshot_date).delete()

        created = 0
        updated = 0

        for _, row in df.iterrows():
            sku = row['sku']
            qty = int(row['quantity'])

            # Obtener nombre del producto si está disponible
            pname = None
            if name_col and pd.notna(row.get(name_col)):
                pname = str(row[name_col]).strip()

            product = Product.query.filter_by(sku=sku).first()
            if not product:
                # Crear producto con nombre real si está disponible
                product = Product(sku=sku, name=pname if pname else f"SKU {sku}")
                db.session.add(product)
                db.session.flush()
            elif pname and product.name.startswith("SKU "):
                # Actualizar nombre si teníamos un placeholder
                product.name = pname

            stock_row = StockCD.query.filter_by(
                as_of_date=snapshot_date,
                product_id=product.id
            ).first()

            if stock_row:
                if modo == 'add':
                    stock_row.quantity += qty
                else:  # replace
                    stock_row.quantity = qty
                updated += 1
            else:
                db.session.add(StockCD(
                    as_of_date=snapshot_date,
                    product_id=product.id,
                    quantity=qty
                ))
                created += 1

        db.session.commit()
        log_audit(
            action="stock_cd.upload",
            message=f"Stock CD cargado: {created} nuevos, {updated} actualizados",
            entity_type="StockCD",
            metadata={
                "filename": filename,
                "created": created,
                "updated": updated,
                "distinct_skus": len(df),
                "mode": modo,
                "snapshot_date": str(snapshot_date)
            }
        )
        flash(f'Stock CD cargado. Nuevos: {created}, Actualizados: {updated}', 'success')
        return redirect(url_for('dashboard'))

    return render_template('upload_stock_cd.html')

@app.route('/stock_query', methods=['GET'])
@login_required
@require_permission('stock:query')
def stock_query():
    sku = (request.args.get("sku") or "").strip()
    scope = (request.args.get("scope") or "cd").strip()  # cd | tiendas
    store_name = (request.args.get("store") or "").strip()

    result = None
    stores = Store.query.order_by(Store.name.asc()).all()

    if sku:
        # SKU SIEMPRE como texto
        product = Product.query.filter_by(sku=sku).first()

        if product:
            if scope == "cd":
                # Tomamos el registro más reciente (no solo hoy)
                cd_row = (StockCD.query
                          .filter_by(product_id=product.id)
                          .order_by(StockCD.as_of_date.desc())
                          .first())
                result = {
                    "sku": product.sku,
                    "product": product.name,
                    "scope": "cd",
                    "as_of_date": cd_row.as_of_date if cd_row else None,
                    "quantity": int(cd_row.quantity) if cd_row else 0
                }

            else:
                # Stock de tiendas: último snapshot por tienda
                q = (db.session.query(StockSnapshot, Store)
                     .join(Store, StockSnapshot.store_id == Store.id)
                     .filter(StockSnapshot.product_id == product.id))

                if store_name:
                    q = q.filter(Store.name == store_name)

                rows = (q.order_by(StockSnapshot.as_of_date.desc()).all())

                # quedarnos con el más reciente por tienda
                latest_by_store = {}
                for snap, st in rows:
                    if st.id not in latest_by_store:
                        latest_by_store[st.id] = {
                            "store": st.name,
                            "as_of_date": snap.as_of_date,
                            "quantity": int(snap.quantity)
                        }

                result = {
                    "sku": product.sku,
                    "product": product.name,
                    "scope": "tiendas",
                    "stores": list(latest_by_store.values())
                }

        else:
            result = {"error": f"No existe producto con SKU: {sku}"}

    return render_template(
        "stock_query.html",
        stores=stores,
        selected_scope=scope,
        selected_store=store_name,
        sku=sku,
        result=result
    )

@app.route('/export_predictions', methods=['GET'])
@login_required
@require_permission('distribution:export')
def export_predictions():
    from io import BytesIO
    import pandas as pd

    run_id = request.args.get("run_id", "").strip()
    if not run_id:
        latest_run = (
            PredictionRun.query
            .filter(PredictionRun.run_id.isnot(None))
            .filter(PredictionRun.run_id != "")
            .order_by(PredictionRun.created_at.desc())
            .first()
        )
        run_id = latest_run.run_id if latest_run else None

    if not run_id:
        flash('No hay predicciones para exportar.', 'warning')
        return redirect(url_for('dashboard'))

    preds = (
        db.session.query(Prediction, Product, Store)
        .join(Product, Prediction.product_id == Product.id)
        .join(Store, Prediction.store_id == Store.id)
        .filter(Prediction.run_id == run_id)
        .order_by(Product.sku.asc(), Store.name.asc())
        .all()
    )

    rows = []
    for p, prod, st in preds:
        folio = ""
        responsable = ""
        categoria = ""
        fecha_doc = ""

        parts = (p.model_name or "").split("—", 1)
        base_model = parts[0].strip()
        meta_part = parts[1].strip() if len(parts) > 1 else ""

        if meta_part:
            meta_items = [m.strip() for m in meta_part.split("|")]
            for item in meta_items:
                low = item.lower()
                if low.startswith("folio:"):
                    folio = item.split(":", 1)[1].strip()
                elif low.startswith("resp:"):
                    responsable = item.split(":", 1)[1].strip()
                elif low.startswith("cat:"):
                    categoria = item.split(":", 1)[1].strip()
                elif low.startswith("fecha doc:"):
                    fecha_doc = item.split(":", 1)[1].strip()

        rows.append({
            "SKU": str(prod.sku),
            "Producto": prod.name,
            "Tienda": st.name,
            "Cantidad sugerida": p.quantity,
            "Modelo": base_model,
            "Folio": folio,
            "Responsable": responsable,
            "Categoría": categoria,
            "Fecha documento": fecha_doc,
            "Corrida": run_id[:8],
        })

    if not rows:
        flash('No hay filas para exportar.', 'warning')
        return redirect(url_for('dashboard'))

    df = pd.DataFrame(rows)
    df["SKU"] = df["SKU"].astype(str)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Distribucion")
        ws = writer.sheets["Distribucion"]
        for cell in ws["A"]:
            cell.number_format = "@"

    output.seek(0)
    today = date.today()
    fname = f"FastDSO_Distribution_{run_id[:8]}_{today.strftime('%Y%m%d')}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@app.route('/runs', methods=['GET'])
@login_required
@require_permission('runs:view')
def runs():
    """Display history of all runs with pagination, search, and filter chips."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = (request.args.get('search') or '').strip()
    run_type_filter = (request.args.get('run_type') or '').strip()

    per_page = per_page if per_page in [10, 25, 50] else 10

    runs_q = Run.query

    if run_type_filter:
        runs_q = runs_q.filter(Run.run_type == run_type_filter)

    if search:
        runs_q = runs_q.filter(
            or_(
                Run.run_id.ilike(f"%{search}%"),
                Run.folio.ilike(f"%{search}%"),
                Run.responsable.ilike(f"%{search}%"),
                Run.categoria.ilike(f"%{search}%"),
                Run.run_type.ilike(f"%{search}%")
            )
        )

    total = runs_q.count()
    runs_list = (
        runs_q
        .order_by(Run.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    pagination = Pagination(page, per_page, total, runs_list)

    return render_template(
        'runs.html',
        runs=runs_list,
        pagination=pagination,
        search=search,
        per_page=per_page,
        run_type_filter=run_type_filter
    )


@app.route('/run/<run_id>', methods=['GET'])
@login_required
@require_permission('runs:view')
def view_run(run_id):
    """View details of a specific run."""
    run = Run.query.filter_by(run_id=run_id).first_or_404()

    if run.run_type == 'distribution':
        predictions = (
            db.session.query(Prediction, Product, Store)
            .join(Product, Prediction.product_id == Product.id)
            .join(Store, Prediction.store_id == Store.id)
            .filter(Prediction.run_id == run_id)
            .order_by(Prediction.quantity.desc())
            .limit(100)
            .all()
        )
        return render_template('run_detail.html', run=run, predictions=predictions)

    elif run.run_type == 'forecast_v2':
        forecast_results = (
            ForecastResult.query
            .filter_by(run_id=run_id)
            .order_by(ForecastResult.suggested.desc())
            .limit(100)
            .all()
        )
        return render_template('run_detail.html', run=run, forecast_results=forecast_results)

    elif run.run_type == 'sales_upload':
        sales_summary = (
            db.session.query(
                Product.sku,
                Product.name,
                db.func.sum(DistributionRecord.quantity).label('total_qty')
            )
            .join(Product, DistributionRecord.product_id == Product.id)
            .filter(DistributionRecord.run_id == run_id)
            .group_by(Product.id, Product.sku, Product.name)
            .order_by(db.func.sum(DistributionRecord.quantity).desc())
            .limit(50)
            .all()
        )
        return render_template('run_detail.html', run=run, sales_summary=sales_summary)

    return render_template('run_detail.html', run=run)


# ======================================================
# ADMIN USER MANAGEMENT
# ======================================================

@app.route('/admin/users', methods=['GET'])
@login_required
@require_permission('admin:users')
def admin_users():
    """Admin page to manage users."""
    users = User.query.order_by(User.username.asc()).all()
    return render_template('admin_users.html', users=users, roles=ROLES)


@app.route('/admin/users/create', methods=['POST'])
@login_required
@require_permission('admin:users')
def admin_create_user():
    """Create a new user."""
    username = (request.form.get('username') or '').strip().lower()
    password = request.form.get('password') or ''
    role = request.form.get('role') or 'Viewer'
    
    if not username or not password:
        flash('Usuario y contraseña son requeridos.', 'danger')
        return redirect(url_for('admin_users'))
    
    if role not in ROLES:
        flash('Rol inválido.', 'danger')
        return redirect(url_for('admin_users'))
    
    existing = User.query.filter_by(username=username).first()
    if existing:
        flash(f'El usuario "{username}" ya existe.', 'danger')
        return redirect(url_for('admin_users'))
    
    new_user = User(
        username=username,
        password_hash=generate_password_hash(password),
        role=role,
        is_active=True
    )
    db.session.add(new_user)
    db.session.commit()
    
    flash(f'Usuario "{username}" creado con rol {role}.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/update', methods=['POST'])
@login_required
@require_permission('admin:users')
def admin_update_user(user_id):
    """Update user role and active status."""
    user = User.query.get_or_404(user_id)
    
    if user.username == 'admin' and current_user.id != user.id:
        flash('No puedes modificar al usuario admin.', 'danger')
        return redirect(url_for('admin_users'))
    
    role = request.form.get('role')
    is_active = request.form.get('is_active') == 'on'
    
    if role and role in ROLES:
        user.role = role
    
    if user.username != 'admin':
        user.is_active = is_active
    
    db.session.commit()
    flash(f'Usuario "{user.username}" actualizado.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/reset-password', methods=['POST'])
@login_required
@require_permission('admin:users')
def admin_reset_password(user_id):
    """Reset user password."""
    user = User.query.get_or_404(user_id)
    new_password = request.form.get('new_password') or ''
    
    if not new_password:
        flash('La nueva contraseña es requerida.', 'danger')
        return redirect(url_for('admin_users'))
    
    user.password_hash = generate_password_hash(new_password)
    db.session.commit()
    
    flash(f'Contraseña de "{user.username}" actualizada.', 'success')
    return redirect(url_for('admin_users'))


# ======================================================
# AUDIT TRAIL
# ======================================================

@app.route('/audit', methods=['GET'])
@login_required
@require_permission('audit:view')
def audit_view():
    """View audit trail with filters and pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    
    action_filter = request.args.get('action', '').strip()
    user_filter = request.args.get('user', '').strip()
    status_filter = request.args.get('status', '').strip()
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    
    query = AuditLog.query
    
    if action_filter:
        query = query.filter(AuditLog.action.ilike(f'%{action_filter}%'))
    if user_filter:
        query = query.filter(AuditLog.username_snapshot.ilike(f'%{user_filter}%'))
    if status_filter:
        query = query.filter(AuditLog.status == status_filter)
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(AuditLog.created_at >= from_date)
        except:
            pass
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(AuditLog.created_at < to_date)
        except:
            pass
    
    total = query.count()
    logs = query.order_by(AuditLog.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()
    
    pagination = Pagination(page, per_page, total, logs)
    
    unique_actions = db.session.query(AuditLog.action).distinct().order_by(AuditLog.action).all()
    unique_actions = [a[0] for a in unique_actions]
    
    return render_template('audit.html', 
                           logs=logs, 
                           pagination=pagination,
                           unique_actions=unique_actions,
                           filters={
                               'action': action_filter,
                               'user': user_filter,
                               'status': status_filter,
                               'date_from': date_from,
                               'date_to': date_to
                           })


# ======================================================
# REBALANCING (Store-to-Store) ROUTES
# ======================================================
import uuid
import math

def compute_rebalancing_suggestions(
    weeks_window=4,
    target_woc_min=1.5,
    target_woc_target=2.5,
    target_woc_max=6.0,
    retain_woc=4.0,
    stock_floor=1,
    min_transfer_qty=2,
    store_filter=None
):
    """
    Compute store-to-store rebalancing suggestions.
    Returns list of dicts: {product_id, sku, name, from_store_id, from_store, to_store_id, to_store, qty, sales_rate_to, woc_from, woc_to, score, reason}
    """
    today = date.today()
    cutoff = today - timedelta(days=weeks_window * 7)
    
    latest_stock_date = db.session.query(db.func.max(StockSnapshot.as_of_date)).scalar()
    if not latest_stock_date:
        return []
    
    stock_map = {}
    stock_q = (
        db.session.query(
            StockSnapshot.product_id,
            StockSnapshot.store_id,
            StockSnapshot.quantity
        )
        .filter(StockSnapshot.as_of_date == latest_stock_date)
        .all()
    )
    for pid, sid, qty in stock_q:
        stock_map[(pid, sid)] = qty
    
    week_expr = db.func.strftime('%Y-%W', DistributionRecord.event_date)
    sales_q = (
        db.session.query(
            DistributionRecord.product_id,
            DistributionRecord.store_id,
            week_expr.label('week'),
            db.func.sum(DistributionRecord.quantity).label('weekly_qty')
        )
        .filter(DistributionRecord.event_date >= cutoff)
        .group_by(DistributionRecord.product_id, DistributionRecord.store_id, week_expr)
        .all()
    )
    
    weekly_sales = defaultdict(lambda: defaultdict(list))
    for pid, sid, week, qty in sales_q:
        weekly_sales[pid][(pid, sid)].append(qty)
    
    sales_rate_map = {}
    for pid in weekly_sales:
        for key, weeks_list in weekly_sales[pid].items():
            avg_rate = sum(weeks_list) / max(len(weeks_list), 1)
            sales_rate_map[key] = avg_rate
    
    product_info = {p.id: (p.sku, p.name) for p in Product.query.all()}
    store_info = {s.id: s.name for s in Store.query.all()}
    
    all_products = set(pid for (pid, sid) in stock_map.keys()) | set(pid for (pid, sid) in sales_rate_map.keys())
    
    suggestions = []
    
    for pid in all_products:
        stores_for_pid = set()
        for (p, s) in stock_map.keys():
            if p == pid:
                stores_for_pid.add(s)
        for (p, s) in sales_rate_map.keys():
            if p == pid:
                stores_for_pid.add(s)
        
        store_data = []
        for sid in stores_for_pid:
            stock = stock_map.get((pid, sid), 0)
            rate = sales_rate_map.get((pid, sid), 0)
            woc = stock / max(rate, 0.01)
            store_data.append({
                'store_id': sid,
                'stock': stock,
                'rate': rate,
                'woc': woc
            })
        
        receivers = []
        for sd in store_data:
            if sd['rate'] > 0 and sd['woc'] < target_woc_min:
                need = max(math.ceil(target_woc_target * sd['rate']) - sd['stock'], 0)
                if need > 0:
                    receivers.append({
                        **sd,
                        'need': need,
                        'priority': (sd['rate'], -sd['woc'])
                    })
        receivers.sort(key=lambda x: x['priority'], reverse=True)
        
        donors = []
        for sd in store_data:
            if sd['woc'] > target_woc_max:
                keep_units = max(math.ceil(retain_woc * sd['rate']), stock_floor)
                give_units = max(sd['stock'] - keep_units, 0)
                if give_units > 0:
                    donors.append({
                        **sd,
                        'give': give_units
                    })
        donors.sort(key=lambda x: x['give'], reverse=True)
        
        for receiver in receivers:
            if store_filter and store_info.get(receiver['store_id']) != store_filter:
                continue
            
            remaining_need = receiver['need']
            for donor in donors:
                if remaining_need <= 0:
                    break
                if donor['give'] <= 0:
                    continue
                if donor['store_id'] == receiver['store_id']:
                    continue
                
                transfer = min(remaining_need, donor['give'])
                
                is_extreme = receiver['woc'] < 0.5
                if transfer < min_transfer_qty and not is_extreme:
                    continue
                
                score = receiver['rate'] * 10 + (target_woc_min - receiver['woc']) * 5
                
                reason = f"WOC {receiver['woc']:.1f} < {target_woc_min}, rate {receiver['rate']:.1f}/wk"
                
                suggestions.append({
                    'product_id': pid,
                    'sku': product_info.get(pid, ('?', '?'))[0],
                    'name': product_info.get(pid, ('?', '?'))[1],
                    'from_store_id': donor['store_id'],
                    'from_store': store_info.get(donor['store_id'], '?'),
                    'to_store_id': receiver['store_id'],
                    'to_store': store_info.get(receiver['store_id'], '?'),
                    'qty': transfer,
                    'sales_rate_to': receiver['rate'],
                    'woc_from': donor['woc'],
                    'woc_to': receiver['woc'],
                    'score': score,
                    'reason': reason
                })
                
                remaining_need -= transfer
                donor['give'] -= transfer
    
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    return suggestions


@app.route('/rebalancing', methods=['GET', 'POST'])
@login_required
@require_permission('rebalancing:view')
def rebalancing():
    """Store-to-store rebalancing suggestions."""
    stores = Store.query.order_by(Store.name.asc()).all()
    
    weeks_window = 4
    target_woc_min = 1.5
    target_woc_target = 2.5
    target_woc_max = 6.0
    retain_woc = 4.0
    stock_floor = 1
    min_transfer_qty = 2
    store_filter = ''
    
    suggestions = []
    run_info = None
    kpis = {'total_units': 0, 'num_moves': 0, 'num_skus': 0, 'num_receivers': 0}
    
    if request.method == 'POST':
        if not current_user.has_permission('rebalancing:run'):
            flash('No tienes permiso para ejecutar rebalanceos.', 'warning')
            return redirect(url_for('rebalancing'))
        
        try:
            weeks_window = int(request.form.get('weeks_window', 4))
        except:
            weeks_window = 4
        try:
            target_woc_min = float(request.form.get('target_woc_min', 1.5))
        except:
            target_woc_min = 1.5
        try:
            target_woc_target = float(request.form.get('target_woc_target', 2.5))
        except:
            target_woc_target = 2.5
        try:
            target_woc_max = float(request.form.get('target_woc_max', 6.0))
        except:
            target_woc_max = 6.0
        try:
            retain_woc = float(request.form.get('retain_woc', 4.0))
        except:
            retain_woc = 4.0
        try:
            stock_floor = int(request.form.get('stock_floor', 1))
        except:
            stock_floor = 1
        try:
            min_transfer_qty = int(request.form.get('min_transfer_qty', 2))
        except:
            min_transfer_qty = 2
        
        store_filter = request.form.get('store_filter', '').strip()
        simulate = request.form.get('simulate') == '1'
        
        suggestions = compute_rebalancing_suggestions(
            weeks_window=weeks_window,
            target_woc_min=target_woc_min,
            target_woc_target=target_woc_target,
            target_woc_max=target_woc_max,
            retain_woc=retain_woc,
            stock_floor=stock_floor,
            min_transfer_qty=min_transfer_qty,
            store_filter=store_filter or None
        )
        
        run_id = str(uuid.uuid4())
        params = {
            'weeks_window': weeks_window,
            'target_woc_min': target_woc_min,
            'target_woc_target': target_woc_target,
            'target_woc_max': target_woc_max,
            'retain_woc': retain_woc,
            'stock_floor': stock_floor,
            'min_transfer_qty': min_transfer_qty,
            'store_filter': store_filter or None
        }
        
        kpis['total_units'] = sum(s['qty'] for s in suggestions)
        kpis['num_moves'] = len(suggestions)
        kpis['num_skus'] = len(set(s['product_id'] for s in suggestions))
        kpis['num_receivers'] = len(set(s['to_store_id'] for s in suggestions))
        
        if simulate:
            store_simulation_results('rebalancing', suggestions, kpis, meta=params)
            run_info = {
                'run_id': 'SIMULATION',
                'created_at': datetime.utcnow(),
                'params': params,
                'is_simulation': True
            }
            flash(f'SIMULACIÓN: {len(suggestions)} sugerencias calculadas (no guardadas).', 'info')
        else:
            rebalance_run = RebalanceRun(
                run_id=run_id,
                created_by_user_id=current_user.id if current_user.is_authenticated else None,
                params_json=json.dumps(params)
            )
            db.session.add(rebalance_run)
            
            for s in suggestions:
                sugg = RebalanceSuggestion(
                    run_id=run_id,
                    product_id=s['product_id'],
                    from_store_id=s['from_store_id'],
                    to_store_id=s['to_store_id'],
                    qty=s['qty'],
                    sales_rate_to=s['sales_rate_to'],
                    woc_from=s['woc_from'],
                    woc_to=s['woc_to'],
                    score=s['score'],
                    reason=s['reason']
                )
                db.session.add(sugg)
            
            db.session.commit()
            
            log_audit(
                action='rebalancing.run',
                status='success',
                message=f'Generated {len(suggestions)} rebalancing suggestions',
                entity_type='RebalanceRun',
                run_id=run_id,
                metadata={
                    'params': params,
                    'num_suggestions': len(suggestions)
                }
            )
            
            run_info = {
                'run_id': run_id,
                'created_at': datetime.utcnow(),
                'params': params
            }
            
            flash(f'Se generaron {len(suggestions)} sugerencias de redistribución.', 'success')
    
    else:
        latest_run = RebalanceRun.query.order_by(RebalanceRun.created_at.desc()).first()
        if latest_run:
            run_info = {
                'run_id': latest_run.run_id,
                'created_at': latest_run.created_at,
                'params': latest_run.get_params()
            }
            params = run_info['params']
            weeks_window = params.get('weeks_window', 4)
            target_woc_min = params.get('target_woc_min', 1.5)
            target_woc_target = params.get('target_woc_target', 2.5)
            target_woc_max = params.get('target_woc_max', 6.0)
            retain_woc = params.get('retain_woc', 4.0)
            stock_floor = params.get('stock_floor', 1)
            min_transfer_qty = params.get('min_transfer_qty', 2)
            store_filter = params.get('store_filter', '') or ''
            
            db_suggestions = RebalanceSuggestion.query.filter_by(run_id=latest_run.run_id).all()
            for s in db_suggestions:
                suggestions.append({
                    'product_id': s.product_id,
                    'sku': s.product.sku if s.product else '?',
                    'name': s.product.name if s.product else '?',
                    'from_store_id': s.from_store_id,
                    'from_store': s.from_store.name if s.from_store else '?',
                    'to_store_id': s.to_store_id,
                    'to_store': s.to_store.name if s.to_store else '?',
                    'qty': s.qty,
                    'sales_rate_to': s.sales_rate_to,
                    'woc_from': s.woc_from,
                    'woc_to': s.woc_to,
                    'score': s.score,
                    'reason': s.reason
                })
            
            kpis['total_units'] = sum(s['qty'] for s in suggestions)
            kpis['num_moves'] = len(suggestions)
            kpis['num_skus'] = len(set(s['product_id'] for s in suggestions))
            kpis['num_receivers'] = len(set(s['to_store_id'] for s in suggestions))
    
    suggestions_limited = suggestions[:50]
    
    return render_template(
        'rebalancing.html',
        stores=stores,
        suggestions=suggestions_limited,
        total_suggestions=len(suggestions),
        kpis=kpis,
        run_info=run_info,
        params={
            'weeks_window': weeks_window,
            'target_woc_min': target_woc_min,
            'target_woc_target': target_woc_target,
            'target_woc_max': target_woc_max,
            'retain_woc': retain_woc,
            'stock_floor': stock_floor,
            'min_transfer_qty': min_transfer_qty,
            'store_filter': store_filter
        }
    )


@app.route('/export_rebalancing', methods=['GET'])
@login_required
@require_permission('rebalancing:view')
def export_rebalancing():
    """Export rebalancing suggestions to Excel."""
    run_id = request.args.get('run_id', '').strip()
    
    if not run_id:
        latest_run = RebalanceRun.query.order_by(RebalanceRun.created_at.desc()).first()
        if latest_run:
            run_id = latest_run.run_id
        else:
            flash('No hay corridas de redistribución disponibles.', 'warning')
            return redirect(url_for('rebalancing'))
    
    run = RebalanceRun.query.filter_by(run_id=run_id).first()
    if not run:
        flash('Corrida no encontrada.', 'danger')
        return redirect(url_for('rebalancing'))
    
    suggestions = RebalanceSuggestion.query.filter_by(run_id=run_id).order_by(RebalanceSuggestion.score.desc()).all()
    
    rows = []
    for s in suggestions:
        rows.append({
            'SKU': s.product.sku if s.product else '',
            'Producto': s.product.name if s.product else '',
            'Tienda Origen': s.from_store.name if s.from_store else '',
            'Tienda Destino': s.to_store.name if s.to_store else '',
            'Cantidad': s.qty,
            'Venta/Sem Destino': round(s.sales_rate_to, 2) if s.sales_rate_to else 0,
            'WOC Origen': round(s.woc_from, 2) if s.woc_from else 0,
            'WOC Destino': round(s.woc_to, 2) if s.woc_to else 0,
            'Score': round(s.score, 2) if s.score else 0,
            'Razón': s.reason or ''
        })
    
    df = pd.DataFrame(rows)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Redistribución')
    output.seek(0)
    
    filename = f"redistribucion_{run_id[:8]}_{date.today().isoformat()}.xlsx"
    
    log_audit(
        action='rebalancing.export',
        status='success',
        message=f'Exported {len(rows)} rebalancing suggestions',
        entity_type='RebalanceRun',
        run_id=run_id,
        metadata={'rows': len(rows)}
    )
    
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/export_simulation/<sim_type>', methods=['GET'])
@login_required
def export_simulation(sim_type):
    """Export simulation results to Excel with watermark."""
    sim_data = get_simulation_results(sim_type)
    
    if not sim_data:
        flash('No hay simulación activa para exportar.', 'warning')
        return redirect(url_for('dashboard'))
    
    results = sim_data.get('results', [])
    kpis = sim_data.get('kpis', {})
    meta = sim_data.get('meta', {})
    
    if sim_type == 'rebalancing':
        rows = []
        for s in results:
            rows.append({
                'SIMULACIÓN': 'SÍ - NO GUARDADO',
                'SKU': s.get('sku', ''),
                'Producto': s.get('name', ''),
                'Tienda Origen': s.get('from_store', ''),
                'Tienda Destino': s.get('to_store', ''),
                'Cantidad': s.get('qty', 0),
                'Venta/Sem Destino': round(s.get('sales_rate_to', 0), 2),
                'WOC Origen': round(s.get('woc_from', 0), 2),
                'WOC Destino': round(s.get('woc_to', 0), 2),
                'Score': round(s.get('score', 0), 2),
                'Razón': s.get('reason', '')
            })
        filename = f"SIMULACION_redistribucion_{date.today().isoformat()}.xlsx"
    elif sim_type == 'distribution':
        rows = []
        for r in results:
            rows.append({
                'SIMULACIÓN': 'SÍ - NO GUARDADO',
                'SKU': r.get('sku', ''),
                'Producto': r.get('product_name', ''),
                'Tienda': r.get('store', ''),
                'Cantidad Sugerida': r.get('quantity', 0),
                'Modelo': r.get('model_name', ''),
                'Semana Objetivo': r.get('target_week', '')
            })
        filename = f"SIMULACION_distribucion_{date.today().isoformat()}.xlsx"
    else:
        flash('Tipo de simulación no reconocido.', 'warning')
        return redirect(url_for('dashboard'))
    
    df = pd.DataFrame(rows)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Simulación')
        
        workbook = writer.book
        worksheet = writer.sheets['Simulación']
        
        from openpyxl.styles import PatternFill, Font
        yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
        for row in worksheet.iter_rows(min_row=2, max_row=len(rows)+1, min_col=1, max_col=1):
            for cell in row:
                cell.fill = yellow_fill
                cell.font = Font(bold=True, color='FF0000')
    
    output.seek(0)
    
    log_audit(
        action=f'simulation.export.{sim_type}',
        status='success',
        message=f'Exported simulation with {len(rows)} rows',
        metadata={'sim_type': sim_type, 'rows': len(rows)}
    )
    
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/clear_simulation/<sim_type>', methods=['GET'])
@login_required
def clear_simulation_route(sim_type):
    """Clear simulation data and redirect."""
    clear_simulation(sim_type)
    redirect_to = request.args.get('redirect', 'dashboard')
    flash('Simulación limpiada.', 'info')
    return redirect(url_for(redirect_to))


def init_database():
    """Initialize database with safe schema migration for RBAC and Audit columns."""
    from sqlalchemy import inspect, text
    
    db.create_all()
    
    inspector = inspect(db.engine)
    
    user_columns = [col['name'] for col in inspector.get_columns('user')]
    
    if 'role' not in user_columns:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(50) DEFAULT 'Admin'"))
            conn.commit()
        print("✅ Added 'role' column to user table")
    
    if 'is_active' not in user_columns:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE user ADD COLUMN is_active BOOLEAN DEFAULT 1"))
            conn.commit()
        print("✅ Added 'is_active' column to user table")
    
    if 'run' in inspector.get_table_names():
        run_columns = [col['name'] for col in inspector.get_columns('run')]
        
        if 'user_id' not in run_columns:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE run ADD COLUMN user_id INTEGER"))
                conn.commit()
            print("✅ Added 'user_id' column to run table")
        
        if 'rows_count' not in run_columns:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE run ADD COLUMN rows_count INTEGER"))
                conn.commit()
            print("✅ Added 'rows_count' column to run table")
        
        if 'predictions_count' not in run_columns:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE run ADD COLUMN predictions_count INTEGER"))
                conn.commit()
            print("✅ Added 'predictions_count' column to run table")
    
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(
            username="admin",
            password_hash=generate_password_hash("admin"),
            role='Admin',
            is_active=True
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Usuario admin creado (admin / admin)")
    elif not admin.role or admin.role not in ROLES:
        admin.role = 'Admin'
        admin.is_active = True
        db.session.commit()
        print("✅ Usuario admin actualizado con rol Admin")


if __name__ == "__main__":
    with app.app_context():
        init_database()

    app.run(host='0.0.0.0', port=5000, debug=True)