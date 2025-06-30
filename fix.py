import pickle

DB_FILE = "class_db.pkl"

with open(DB_FILE, "rb") as f:
    db = pickle.load(f)

if "pan" in db:
    del db["pan"]
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)
    print("✅ Removed incorrect 'pan' class from the database.")
else:
    print("⚠️ 'pan' not found in database.")
