from fastapi import FastAPI

app = FastAPI()

# Product database (temporary list)
products = [
    {"id": 1, "name": "Wireless Mouse", "price": 599, "category": "Electronics", "in_stock": True},
    {"id": 2, "name": "Notebook", "price": 99, "category": "Stationery", "in_stock": True},
    {"id": 3, "name": "Pen Set", "price": 49, "category": "Stationery", "in_stock": True},
    {"id": 4, "name": "USB Cable", "price": 199, "category": "Electronics", "in_stock": False},

    # Added products (Task 1)
    {"id": 5, "name": "Laptop Stand", "price": 1299, "category": "Electronics", "in_stock": True},
    {"id": 6, "name": "Mechanical Keyboard", "price": 2499, "category": "Electronics", "in_stock": True},
    {"id": 7, "name": "Webcam", "price": 1899, "category": "Electronics", "in_stock": False},
]


# Home route
@app.get("/")
def home():
    return {"message": "Welcome to My E-commerce Store API"}


# Q1 — Get all products
@app.get("/products")
def get_products():
    return {
        "products": products,
        "total": len(products)
    }


# Q2 — Filter by category
@app.get("/products/category/{category_name}")
def get_products_by_category(category_name: str):
    filtered = [
        p for p in products
        if p["category"].lower() == category_name.lower()
    ]

    if not filtered:
        return {"error": "No products found in this category"}

    return {
        "category": category_name,
        "products": filtered
    }


# Q3 — Only in-stock products
@app.get("/products/instock")
def get_instock_products():
    instock = [p for p in products if p["in_stock"]]

    return {
        "in_stock_products": instock,
        "count": len(instock)
    }


# Q4 — Store summary
@app.get("/store/summary")
def store_summary():
    total = len(products)
    in_stock = sum(1 for p in products if p["in_stock"])
    out_of_stock = total - in_stock
    categories = list(set(p["category"] for p in products))

    return {
        "store_name": "My E-commerce Store",
        "total_products": total,
        "in_stock": in_stock,
        "out_of_stock": out_of_stock,
        "categories": categories
    }


# Q5 — Search products by name
@app.get("/products/search/{keyword}")
def search_products(keyword: str):
    results = [
        p for p in products
        if keyword.lower() in p["name"].lower()
    ]

    if not results:
        return {"message": "No products matched your search"}

    return {
        "matches": results,
        "count": len(results)
    }


# ⭐ Bonus — Cheapest & most expensive products
@app.get("/products/deals")
def product_deals():
    cheapest = min(products, key=lambda p: p["price"])
    expensive = max(products, key=lambda p: p["price"])

    return {
        "best_deal": cheapest,
        "premium_pick": expensive
    }