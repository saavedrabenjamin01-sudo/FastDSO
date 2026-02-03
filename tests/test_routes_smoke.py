"""
Smoke tests for critical routes to catch routing errors (BuildError).
Run with: python tests/test_routes_smoke.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def test_routes():
    """Test critical routes for BuildError."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.test_client() as client:
        routes_to_test = [
            ('/store_health', 'Store Health'),
            ('/alerts', 'Alerts'),
            ('/', 'Dashboard'),
            ('/rebalancing', 'Rebalancing'),
        ]
        
        all_passed = True
        for route, name in routes_to_test:
            try:
                response = client.get(route, follow_redirects=True)
                if response.status_code in (200, 302):
                    print(f"[PASS] {name} ({route}) - Status: {response.status_code}")
                else:
                    print(f"[FAIL] {name} ({route}) - Unexpected status: {response.status_code}")
                    all_passed = False
            except Exception as e:
                print(f"[FAIL] {name} ({route}) - Error: {e}")
                all_passed = False
        
        return all_passed

if __name__ == '__main__':
    print("Running route smoke tests...")
    print("-" * 50)
    success = test_routes()
    print("-" * 50)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
