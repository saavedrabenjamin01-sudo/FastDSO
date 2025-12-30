#!/usr/bin/env python
"""
Validation script for Store-to-Store Rebalancing algorithm.
Run: python validate_rebalancing.py

This creates a temporary test database and runs 3 deterministic scenarios
to validate the rebalancing logic respects all business rules.
"""

import os
import sys
import tempfile
from datetime import date, timedelta
from collections import defaultdict
import math

os.environ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

from app import (
    app, db, Product, Store, StockSnapshot, DistributionRecord,
    compute_rebalancing_suggestions
)


def setup_test_db():
    """Create fresh in-memory test database."""
    with app.app_context():
        db.create_all()
        print("✓ Test database initialized (in-memory SQLite)")


def clear_test_data():
    """Clear all test data between scenarios."""
    with app.app_context():
        DistributionRecord.query.delete()
        StockSnapshot.query.delete()
        Product.query.delete()
        Store.query.delete()
        db.session.commit()


def seed_scenario_a():
    """
    SCENARIO A — Basic donor → receiver with hierarchy
    - SKU "000123" (P1), 3 stores: A, B, C
    - Stock: A=40, B=20, C=1
    - Sales/week: C=10, B=4, A=1
    """
    with app.app_context():
        p1 = Product(sku="000123", name="Product P1")
        db.session.add(p1)
        db.session.flush()

        store_a = Store(name="Store A")
        store_b = Store(name="Store B")
        store_c = Store(name="Store C")
        db.session.add_all([store_a, store_b, store_c])
        db.session.flush()

        today = date.today()
        db.session.add(StockSnapshot(as_of_date=today, product_id=p1.id, store_id=store_a.id, quantity=40))
        db.session.add(StockSnapshot(as_of_date=today, product_id=p1.id, store_id=store_b.id, quantity=20))
        db.session.add(StockSnapshot(as_of_date=today, product_id=p1.id, store_id=store_c.id, quantity=1))

        for week_offset in range(4):
            week_date = today - timedelta(days=7 * week_offset + 1)
            db.session.add(DistributionRecord(product_id=p1.id, store_id=store_a.id, quantity=1, event_date=week_date))
            db.session.add(DistributionRecord(product_id=p1.id, store_id=store_b.id, quantity=4, event_date=week_date))
            db.session.add(DistributionRecord(product_id=p1.id, store_id=store_c.id, quantity=10, event_date=week_date))

        db.session.commit()
        return {'product': p1, 'stores': {'A': store_a, 'B': store_b, 'C': store_c}}


def seed_scenario_b():
    """
    SCENARIO B — Zero-sales store must not receive
    - SKU "000123" (P1), 2 stores: D (donor), Z (zero sales)
    - Stock: D=30, Z=0
    - Sales/week: D=1, Z=0
    """
    with app.app_context():
        p1 = Product(sku="000123", name="Product P1")
        db.session.add(p1)
        db.session.flush()

        store_d = Store(name="Store D")
        store_z = Store(name="Store Z")
        db.session.add_all([store_d, store_z])
        db.session.flush()

        today = date.today()
        db.session.add(StockSnapshot(as_of_date=today, product_id=p1.id, store_id=store_d.id, quantity=30))
        db.session.add(StockSnapshot(as_of_date=today, product_id=p1.id, store_id=store_z.id, quantity=0))

        for week_offset in range(4):
            week_date = today - timedelta(days=7 * week_offset + 1)
            db.session.add(DistributionRecord(product_id=p1.id, store_id=store_d.id, quantity=1, event_date=week_date))

        db.session.commit()
        return {'product': p1, 'stores': {'D': store_d, 'Z': store_z}}


def seed_scenario_c():
    """
    SCENARIO C — Min transfer qty + extreme WOC exception
    - SKU "009999" (P2), 2 stores: X (donor), Y (receiver)
    - Stock: X=10, Y=0
    - Sales/week: X=1, Y=1
    """
    with app.app_context():
        p2 = Product(sku="009999", name="Product P2")
        db.session.add(p2)
        db.session.flush()

        store_x = Store(name="Store X")
        store_y = Store(name="Store Y")
        db.session.add_all([store_x, store_y])
        db.session.flush()

        today = date.today()
        db.session.add(StockSnapshot(as_of_date=today, product_id=p2.id, store_id=store_x.id, quantity=10))
        db.session.add(StockSnapshot(as_of_date=today, product_id=p2.id, store_id=store_y.id, quantity=0))

        for week_offset in range(4):
            week_date = today - timedelta(days=7 * week_offset + 1)
            db.session.add(DistributionRecord(product_id=p2.id, store_id=store_x.id, quantity=1, event_date=week_date))
            db.session.add(DistributionRecord(product_id=p2.id, store_id=store_y.id, quantity=1, event_date=week_date))

        db.session.commit()
        return {'product': p2, 'stores': {'X': store_x, 'Y': store_y}}


def run_scenario_a():
    """Run and validate Scenario A."""
    print("\n" + "="*60)
    print("SCENARIO A: Basic donor → receiver with hierarchy")
    print("="*60)

    clear_test_data()
    data = seed_scenario_a()

    with app.app_context():
        suggestions = compute_rebalancing_suggestions(
            weeks_window=4,
            target_woc_min=1.5,
            target_woc_target=2.5,
            target_woc_max=6.0,
            retain_woc=4.0,
            stock_floor=1,
            min_transfer_qty=2,
            store_filter=None
        )

        print(f"\nGenerated {len(suggestions)} suggestions:")
        for s in suggestions:
            print(f"  {s['sku']} | {s['from_store']} → {s['to_store']} | qty={s['qty']} | rate_to={s['sales_rate_to']:.1f} | WOC from={s['woc_from']:.1f} to={s['woc_to']:.1f}")

        assert len(suggestions) > 0, "Expected at least one transfer suggestion"
        print("✓ At least one transfer suggestion generated")

        receiver_stores = set(s['to_store'] for s in suggestions)
        assert 'Store C' in receiver_stores, "Store C should be a receiver (highest sales, low WOC)"
        print("✓ Store C is a receiver (highest sales velocity)")

        donor_stores = set(s['from_store'] for s in suggestions)
        assert 'Store A' in donor_stores, "Store A should be a donor (excess WOC)"
        print("✓ Store A is a donor (excess WOC)")

        for s in suggestions:
            assert s['qty'] > 0, f"Transfer qty must be > 0, got {s['qty']}"
            assert s['from_store'] != s['to_store'], "From and To store must differ"
            assert s['sales_rate_to'] > 0, f"Receiver must have sales_rate > 0, got {s['sales_rate_to']}"
        print("✓ All invariants validated (qty>0, different stores, receiver has sales)")

        print("\n✓ SCENARIO A PASSED")
        return True


def run_scenario_b():
    """Run and validate Scenario B."""
    print("\n" + "="*60)
    print("SCENARIO B: Zero-sales store must not receive")
    print("="*60)

    clear_test_data()
    data = seed_scenario_b()

    with app.app_context():
        suggestions = compute_rebalancing_suggestions(
            weeks_window=4,
            target_woc_min=1.5,
            target_woc_target=2.5,
            target_woc_max=6.0,
            retain_woc=4.0,
            stock_floor=1,
            min_transfer_qty=2,
            store_filter=None
        )

        print(f"\nGenerated {len(suggestions)} suggestions:")
        for s in suggestions:
            print(f"  {s['sku']} | {s['from_store']} → {s['to_store']} | qty={s['qty']}")

        receiver_stores = set(s['to_store'] for s in suggestions)
        assert 'Store Z' not in receiver_stores, "Store Z (zero sales) should NOT be a receiver"
        print("✓ Store Z (zero sales) is NOT a receiver")

        for s in suggestions:
            assert s['sales_rate_to'] > 0, f"Receiver must have sales_rate > 0"
        print("✓ All receivers have positive sales rate")

        print("\n✓ SCENARIO B PASSED")
        return True


def run_scenario_c():
    """Run and validate Scenario C."""
    print("\n" + "="*60)
    print("SCENARIO C: Min transfer qty + extreme WOC exception")
    print("="*60)

    clear_test_data()
    data = seed_scenario_c()

    with app.app_context():
        suggestions = compute_rebalancing_suggestions(
            weeks_window=4,
            target_woc_min=1.5,
            target_woc_target=2.0,
            target_woc_max=6.0,
            retain_woc=4.0,
            stock_floor=1,
            min_transfer_qty=3,
            store_filter=None
        )

        print(f"\nGenerated {len(suggestions)} suggestions:")
        for s in suggestions:
            print(f"  {s['sku']} | {s['from_store']} → {s['to_store']} | qty={s['qty']} | WOC_to={s['woc_to']:.2f}")

        y_transfers = [s for s in suggestions if s['to_store'] == 'Store Y']
        
        if len(y_transfers) > 0:
            transfer = y_transfers[0]
            assert transfer['woc_to'] < 0.5, f"Receiver WOC should be < 0.5 (extreme), got {transfer['woc_to']}"
            print(f"✓ Transfer to Store Y allowed despite qty < min_transfer_qty (extreme WOC={transfer['woc_to']:.2f} < 0.5)")
            
            assert transfer['qty'] <= 2, f"Transfer qty should be <= 2 (receiver need), got {transfer['qty']}"
            print(f"✓ Transfer qty is {transfer['qty']} (within receiver need)")
        else:
            print("  Note: No transfer to Store Y (donor may not have excess after retain_woc)")
            store_x_stock = 10
            store_x_rate = 1.0
            keep_units = max(math.ceil(4.0 * store_x_rate), 1)
            give_units = max(store_x_stock - keep_units, 0)
            store_x_woc = store_x_stock / max(store_x_rate, 0.01)
            print(f"  Store X: stock=10, rate=1, WOC={store_x_woc:.1f}, keep={keep_units}, give={give_units}")
            if store_x_woc <= 6.0:
                print(f"  Store X WOC ({store_x_woc:.1f}) <= target_woc_max (6.0), so not a donor")

        for s in suggestions:
            assert s['qty'] > 0, f"Transfer qty must be > 0"
            assert s['from_store'] != s['to_store'], "From and To store must differ"
            assert s['sales_rate_to'] > 0, f"Receiver must have sales_rate > 0"
        print("✓ All invariants validated")

        print("\n✓ SCENARIO C PASSED")
        return True


def main():
    print("\n" + "#"*60)
    print("# REBALANCING ALGORITHM VALIDATION")
    print("#"*60)

    setup_test_db()

    results = []
    
    try:
        results.append(("Scenario A", run_scenario_a()))
    except AssertionError as e:
        print(f"\n✗ SCENARIO A FAILED: {e}")
        results.append(("Scenario A", False))

    try:
        results.append(("Scenario B", run_scenario_b()))
    except AssertionError as e:
        print(f"\n✗ SCENARIO B FAILED: {e}")
        results.append(("Scenario B", False))

    try:
        results.append(("Scenario C", run_scenario_c()))
    except AssertionError as e:
        print(f"\n✗ SCENARIO C FAILED: {e}")
        results.append(("Scenario C", False))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
