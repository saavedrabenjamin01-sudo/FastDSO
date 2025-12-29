#!/usr/bin/env python3
"""
Reset the database for PredDist.
This script deletes the existing database and recreates all tables.
Use this when making schema changes that require a fresh start.
"""
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Deleted: {DB_PATH}")
else:
    print(f"Database not found: {DB_PATH}")

print("Starting app to recreate tables...")
os.system("python -c \"from app import app, db; app.app_context().push(); db.create_all(); print('Tables created successfully')\"")
print("Done! Database reset complete.")
