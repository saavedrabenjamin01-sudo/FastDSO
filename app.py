import os
import io
import datetime as dt
from datetime import datetime, timedelta, date
from io import BytesIO
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required, logout_user, current_user, UserMixin
)
MIN_WEEKS = 3  # mínimo de semanas de historia requeridas por SKU–Tienda


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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


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


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey(
        'product.id'), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=False)
    target_period_start = db.Column(db.Date, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    model_name = db.Column(db.String(255), default='SMA_3w')
    # 🔑 NUEVO — NO EXISTÍA ANTES
    run_id = db.Column(db.String(36), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    product = db.relationship('Product')
    store = db.relationship('Store')

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
        'date': date
    }


# ------------------ Utilidades ------------------


def next_monday(ref: datetime | None = None):
    ref = ref or datetime.utcnow()
    days_ahead = (7 - ref.weekday()) % 7
    days_ahead = 7 if days_ahead == 0 else days_ahead
    return (ref + timedelta(days=days_ahead)).date()

def has_any_stock_loaded() -> bool:
    """True si existe al menos un snapshot de stock en la tabla."""
    return db.session.query(StockSnapshot.id).first() is not None

# Predicción: promedio móvil 3 semanas por SKU-Tienda


from datetime import date
from collections import defaultdict

from collections import defaultdict
from datetime import date

def generate_predictions(

    mode: str = "sma3_min3",
    meta: dict | None = None,
    df: pd.DataFrame | None = None
):
    # NEW: run_id único por corrida
    from uuid import uuid4
    run_id = str(uuid4())

    # 1) Origen de datos: df pasado o histórico completo
    if df is None:
        rows = DistributionRecord.query.all()
        if not rows:
            return 0
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
                run_id=run_id  # NEW
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

    # NEW (opcional): guardar el último run_id en el contexto del request
    try:
        g.latest_run_id = run_id
    except Exception:
        pass

    # Devolvemos el TOTAL de predicciones trabajadas (nuevas + actualizadas)
    return len(final_preds)

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
def dashboard():
    # --- Filtros ---
    store_filter = (request.args.get('store') or '').strip()
    folio_filter = (request.args.get('folio') or '').strip()
    responsable_filter = (request.args.get('responsable') or '').strip()

    # --- Fechas / helpers ---
    today = date.today()

    # --- Última semana de predicción ---
    latest_week = db.session.query(func.max(Prediction.target_period_start)).scalar()

    # --- Último run_id (corrida más reciente) ---
    # Regla:
    # - Si NO hay folio/responsable => mostramos solo la última corrida (run_id)
    # - Si hay folio o responsable => permitimos histórico (no forzamos run_id)
    latest_run_id = (
        db.session.query(Prediction.run_id)
        .order_by(Prediction.id.desc())
        .limit(1)
        .scalar()
    )

    force_latest_run = (not folio_filter and not responsable_filter)

    # --- Query base de predicciones + joins ---
    pred_q = (
        db.session.query(Prediction, Product, Store)
        .join(Product, Prediction.product_id == Product.id)
        .join(Store, Prediction.store_id == Store.id)
    )

    if latest_run_id:
        pred_q = pred_q.filter(Prediction.run_id == latest_run_id)

    # Filtrar por semana (si existe)
    if latest_week:
        pred_q = pred_q.filter(Prediction.target_period_start == latest_week)

    # Filtrar por última corrida (solo si no estamos buscando histórico)
    if force_latest_run and latest_run_id:
        pred_q = pred_q.filter(Prediction.run_id == latest_run_id)

    # Filtro tienda
    if store_filter:
        pred_q = pred_q.filter(Store.name == store_filter)

    # Filtros folio / responsable (van en model_name)
    # (Si se usan, NO forzamos run_id; así puedes ver corridas anteriores)
    if folio_filter:
        pred_q = pred_q.filter(Prediction.model_name.ilike(f"%Folio:%{folio_filter}%"))
    if responsable_filter:
        pred_q = pred_q.filter(Prediction.model_name.ilike(f"%Resp:%{responsable_filter}%"))

    predictions = (
        pred_q
        .order_by(Product.sku.asc(), Store.name.asc())
        .limit(50)
        .all()
    )

    # --- KPI (solo sobre el set filtrado) ---
    kpi_units_suggested = sum(int(p.quantity or 0) for p, _, _ in predictions)
    kpi_skus_distintos = len(set(prod.id for _, prod, _ in predictions))
    kpi_tiendas_alcanzadas = len(set(st.id for _, _, st in predictions))

    # Stock CD hoy (total)
    kpi_stock_cd_total = (
        db.session.query(func.coalesce(func.sum(StockCD.quantity), 0))
        .filter(StockCD.as_of_date == today)
        .scalar()
    )

    # --- Top 10 ventas (resumen) ---
    # Usa DistributionRecord (ventas) y filtra por tienda si corresponde.
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

    top_sales = (
        sales_q
        .group_by(Product.sku, Product.name, Store.name)
        .order_by(desc("units"))
        .limit(10)
        .all()
    )

    # --- Remanente CD: SOLO SKUs de la última distribución (filtrada) ---
    stock_cd_filtered = []
    if predictions:
        product_ids = list({prod.id for _, prod, _ in predictions})
        stock_cd_filtered = (
            db.session.query(StockCD, Product)
            .join(Product, StockCD.product_id == Product.id)
            .filter(StockCD.as_of_date == today)
            .filter(StockCD.product_id.in_(product_ids))
            .order_by(Product.sku.asc())
            .limit(50)
            .all()
        )

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
        top_sales=top_sales,
        stock_cd_filtered=stock_cd_filtered,

        kpi_units_suggested=kpi_units_suggested,
        kpi_skus_distintos=kpi_skus_distintos,
        kpi_tiendas_alcanzadas=kpi_tiendas_alcanzadas,
        kpi_stock_cd_total=int(kpi_stock_cd_total or 0),

        # (opcional) por si después quieres mostrar “corrida actual”
        latest_run_id=latest_run_id,
    )

@app.route('/purchase_forecast', methods=['GET', 'POST'])
@login_required
def purchase_forecast():
    """
    Proyección de compras por SKU en base a:
      - Ventas de los últimos N días (default 30)
      - Promedio semanal
      - Stock CD disponible (snapshot de hoy)
      => Sugerencia de compra para cubrir X semanas (default 4)
    """
    # 1) leer parámetros del formulario
    if request.method == 'POST':
        try:
            horizon_days = int(request.form.get('horizon_days', 30))
        except ValueError:
            horizon_days = 30

        try:
            target_weeks = float(request.form.get('target_weeks', 4))
        except ValueError:
            target_weeks = 4.0
    else:
        # valores por defecto (cuando entras por primera vez)
        horizon_days = 30
        target_weeks = 4.0

    # blindajes básicos
    horizon_days = max(horizon_days, 7)       # al menos 1 semana de ventana
    target_weeks = max(target_weeks, 0.5)     # al menos media semana

    today = date.today()
    cutoff = today - timedelta(days=horizon_days)

    # 2) ventas por SKU en la ventana seleccionada
    sales_rows = (
        db.session.query(
            Product.id.label('product_id'),
            Product.sku.label('sku'),
            Product.name.label('name'),
            db.func.sum(DistributionRecord.quantity).label('qty_period')
        )
        .join(Product, DistributionRecord.product_id == Product.id)
        .filter(DistributionRecord.event_date >= cutoff)
        .group_by(Product.id, Product.sku, Product.name)
        .all()
    )

    if not sales_rows:
        return render_template(
            'purchase_forecast.html',
            rows=[],
            horizon_days=horizon_days,
            target_weeks=target_weeks,
            today=today,
        )

    # 3) stock CD de hoy por producto
    cd_rows = (
        db.session.query(
            StockCD.product_id,
            db.func.sum(StockCD.quantity).label('cd_qty')
        )
        .filter(StockCD.as_of_date == today)
        .group_by(StockCD.product_id)
        .all()
    )
    cd_stock_map = {r.product_id: r.cd_qty for r in cd_rows}

    # 4) armar tabla de proyección
    days_to_weeks = horizon_days / 7.0
    result = []

    for r in sales_rows:
        avg_week = float(r.qty_period) / days_to_weeks if days_to_weeks > 0 else 0.0
        cd_qty = int(cd_stock_map.get(r.product_id, 0))

        # demanda esperada para horizonte objetivo
        demand_target = avg_week * target_weeks

        # sugerencia de compra (no negativa)
        suggested = max(int(round(demand_target - cd_qty)), 0)

        result.append({
            "sku": r.sku,
            "name": r.name,
            "qty_period": int(r.qty_period),
            "avg_week": round(avg_week, 2),
            "cd_qty": cd_qty,
            "suggested": suggested,
        })

    # ordenar por mayor sugerido de compra
    result.sort(key=lambda x: x["suggested"], reverse=True)

    return render_template(
        'purchase_forecast.html',
        rows=result,
        horizon_days=horizon_days,
        target_weeks=target_weeks,
        today=today,
    )

@app.route('/export_cd_remanente', methods=['GET'])
@login_required
def export_cd_remanente():
    """Exporta a Excel el remanente de CD solo de los SKUs usados en la última distribución."""
    today = date.today()
    latest_week = db.session.query(func.max(Prediction.target_period_start)).scalar()

    if not latest_week:
        # no hay predicciones, devolvemos excel vacío
        df_empty = pd.DataFrame([{
            "SKU": "",
            "Producto": "",
            "Stock CD disponible": "",
            "Semana objetivo": "",
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

    # obtener los productos usados en la última predicción
    preds = (
        db.session.query(Prediction.product_id)
        .filter(Prediction.target_period_start == latest_week)
        .distinct()
        .all()
    )
    product_ids = [p.product_id for p in preds]

    # obtener el stock CD de esos productos
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
            "Semana objetivo": latest_week,
            "Fecha": today.strftime("%Y-%m-%d"),
        })

    if not data:
        data.append({
            "SKU": "",
            "Producto": "",
            "Stock CD disponible": "",
            "Semana objetivo": latest_week,
            "Fecha": today.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Remanente")
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"remanente_cd_{today.strftime('%Y%m%d')}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
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
                df = pd.read_csv(file, dtype={'sku': str})
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file, dtype={'sku': str})
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

            # guardar distribución histórica (ventas / movimientos)
            dist = DistributionRecord(
                product_id=product.id,
                store_id=store.id,
                quantity=qty,
                event_date=event_date
            )
            db.session.add(dist)
            created += 1

        db.session.commit()

        # leer parámetros del formulario
        analysis_mode = request.form.get('analysis_mode', 'sma3_min3')

        meta = {
            "folio": request.form.get('folio', '').strip() or None,
            "responsable": request.form.get('responsable', '').strip() or None,
            "categoria": request.form.get('categoria', '').strip() or None,
            "fecha_doc": request.form.get('fecha_doc', '').strip() or None,
        }

        n_preds = generate_predictions(mode=analysis_mode, meta=meta, df=df)

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
def upload_stock():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            flash('Sube un archivo CSV o Excel', 'warning')
            return redirect(url_for('upload_stock'))

        filename = f.filename.lower()
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(f, dtype={'SKU': 'string', 'sku': 'string'})
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(f, dtype={'SKU': 'string', 'sku': 'string'})
            else:
                flash('Formato no soportado. Usa .csv o .xlsx', 'danger')
                return redirect(url_for('upload_stock'))
        except Exception as e:
            flash(f'Error leyendo archivo de stock: {e}', 'danger')
            return redirect(url_for('upload_stock'))

        # normalizar encabezados
        df.columns = [str(c).strip() for c in df.columns]
        cols_lower = {c.lower(): c for c in df.columns}

        sku_col = cols_lower.get('sku')
        prod_col = cols_lower.get('producto') or cols_lower.get('product') or cols_lower.get('product_name')

        if not sku_col:
            flash('El archivo debe tener columna "SKU".', 'danger')
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
        flash(f'Stock cargado. Nuevos: {created}. Actualizados: {updated}.', 'success')
        return redirect(url_for('dashboard'))

    # GET
    return render_template('upload_stock.html')

from datetime import date
from sqlalchemy import func
import io

@app.route('/purchase_projection', methods=['GET', 'POST'])
@login_required
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
def reset_sales():
    db.session.query(DistributionRecord).delete()
    db.session.commit()
    flash('✅ Se eliminaron todos los registros de ventas cargadas.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/reset_store_stock', methods=['POST'])
@login_required
def reset_store_stock():
    db.session.query(StockSnapshot).delete()
    db.session.commit()
    flash('✅ Se eliminó todo el stock de tiendas.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/reset_predictions', methods=['POST'])
@login_required
def reset_predictions():
    db.session.query(Prediction).delete()
    db.session.commit()
    flash('✅ Se eliminaron todas las distribuciones sugeridas.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/reset_stock_cd', methods=['POST'])
@login_required
def reset_stock_cd():
    StockCD.query.delete()
    db.session.commit()
    flash('Stock CD completamente reiniciado.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/stock_cd', methods=['GET', 'POST'])
@login_required
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
                df = pd.read_csv(file, dtype={'sku': str})
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file, dtype={'sku': str})
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

        # modo replace → borrar snapshot de ESA fecha antes de cargar
        if modo == 'replace':
            StockCD.query.filter_by(as_of_date=snapshot_date).delete()

        created = 0
        updated = 0

        for _, row in df.iterrows():
            sku = row['sku']
            qty = int(row['quantity'])

            product = Product.query.filter_by(sku=sku).first()
            if not product:
                # opcional: crear producto si no existe
                product = Product(sku=sku, name=f"SKU {sku}")
                db.session.add(product)
                db.session.flush()

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
        flash(f'Stock CD cargado. Nuevos: {created}, Actualizados: {updated}', 'success')
        return redirect(url_for('dashboard'))

    return render_template('upload_stock_cd.html')

@app.route('/stock_query', methods=['GET'])
@login_required
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
def export_predictions():
    from io import BytesIO
    import pandas as pd

    latest_week = db.session.query(db.func.max(Prediction.target_period_start)).scalar()
    if not latest_week:
        flash('No hay predicciones para exportar.', 'warning')
        return redirect(url_for('dashboard'))

    preds = (
        db.session.query(Prediction, Product, Store)
        .join(Product, Prediction.product_id == Product.id)
        .join(Store, Prediction.store_id == Store.id)
        .filter(Prediction.target_period_start == latest_week)
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
            "Semana objetivo": latest_week.strftime("%Y-%m-%d"),
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
        # Columna A (SKU) como texto
        for cell in ws["A"]:
            cell.number_format = "@"

    output.seek(0)
    fname = f"distribucion_sugerida_{latest_week.strftime('%Y%m%d')}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        # Crear admin si no existe (Flask 3 friendly)
        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin = User(
                username="admin",
                password_hash=generate_password_hash("admin")
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Usuario admin creado (admin / admin)")

    app.run(host='0.0.0.0', port=5000, debug=True)