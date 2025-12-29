import requests
import time

BASE_URL = "http://127.0.0.1:5000"
session = requests.Session()

def login():
    print("1. Logging in as admin...")
    resp = session.post(f"{BASE_URL}/login", data={
        "username": "admin",
        "password": "admin"
    }, allow_redirects=False)
    if resp.status_code in [200, 302]:
        print("   Login successful!")
        return True
    else:
        print(f"   Login failed: {resp.status_code}")
        return False

def upload_sales():
    print("\n2. Uploading sales data (ventas_test.csv)...")
    with open("test_data/ventas_test.csv", "rb") as f:
        resp = session.post(f"{BASE_URL}/upload", 
                           files={"file": f}, 
                           data={
                               "confirm_no_stock": "1",
                               "analysis_mode": "sma3_min3"
                           },
                           allow_redirects=True)
    if "success" in resp.text.lower() or "predicciones" in resp.text.lower() or resp.status_code == 200:
        print("   Sales data uploaded!")
        return True
    else:
        print(f"   Upload may have issues. Status: {resp.status_code}")
        return True

def upload_stock_tiendas():
    print("\n3. Uploading store stock (stock_tiendas_test.csv)...")
    with open("test_data/stock_tiendas_test.csv", "rb") as f:
        resp = session.post(f"{BASE_URL}/stock", files={"file": f}, allow_redirects=True)
    if resp.status_code == 200:
        print("   Store stock uploaded!")
        return True
    else:
        print(f"   Upload may have issues. Status: {resp.status_code}")
        return True

def upload_stock_cd():
    print("\n4. Uploading CD stock (stock_cd_test.csv)...")
    with open("test_data/stock_cd_test.csv", "rb") as f:
        resp = session.post(f"{BASE_URL}/stock_cd", 
                           files={"file": f}, 
                           data={"modo": "replace"},
                           allow_redirects=True)
    if resp.status_code == 200:
        print("   CD stock uploaded!")
        return True
    else:
        print(f"   Upload may have issues. Status: {resp.status_code}")
        return True

def generate_predictions():
    print("\n5. Predictions are generated automatically during sales upload.")
    return True

def check_dashboard():
    print("\n6. Checking dashboard...")
    resp = session.get(f"{BASE_URL}/dashboard")
    if resp.status_code == 200:
        content = resp.text
        if "PlayStation" in content or "Xbox" in content or "Spider-Man" in content:
            print("   Dashboard shows product names correctly!")
        elif "SKU CONS" in content or "SKU GAME" in content:
            print("   WARNING: Dashboard still shows SKU placeholders instead of names")
        else:
            print("   Dashboard loaded (check manually for data)")
        
        if "Tienda Centro" in content or "Tienda Norte" in content:
            print("   Dashboard shows store data!")
        
        return True
    else:
        print(f"   Dashboard error: {resp.status_code}")
        return False

def main():
    print("=" * 50)
    print("FastDSO Test Suite")
    print("=" * 50)
    
    if not login():
        return
    
    time.sleep(0.5)
    upload_sales()
    
    time.sleep(0.5)
    upload_stock_tiendas()
    
    time.sleep(0.5)
    upload_stock_cd()
    
    time.sleep(0.5)
    generate_predictions()
    
    time.sleep(0.5)
    check_dashboard()
    
    print("\n" + "=" * 50)
    print("Test completed! Check the web UI for results.")
    print("=" * 50)

if __name__ == "__main__":
    main()
