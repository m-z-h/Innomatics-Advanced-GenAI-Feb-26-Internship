from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI()

class CustomerFeedback(BaseModel):
    customer_name: str = Field(min_length=2)
    product_id: int = Field(gt=0)
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=300)

class OrderItem(BaseModel):
    product_id: int = Field(gt=0)
    quantity: int = Field(ge=1, le=50)

class BulkOrder(BaseModel):
    company_name: str = Field(min_length=2)
    contact_email: str = Field(min_length=5)
    items: List[OrderItem] = Field(min_items=1)

# Product database (temporary list)
products = [
    {"id": 1, "name": "Wireless Mouse", "price": 499, "category": "Electronics", "in_stock": True},
    {"id": 2, "name": "Notebook", "price": 99, "category": "Stationery", "in_stock": True},
    {"id": 3, "name": "Pen Set", "price": 49, "category": "Stationery", "in_stock": True},
    {"id": 4, "name": "USB Hub", "price": 799, "category": "Electronics", "in_stock": False},

    # Added products (Task 1)
    {"id": 5, "name": "Laptop Stand", "price": 1299, "category": "Electronics", "in_stock": True},
    {"id": 6, "name": "Mechanical Keyboard", "price": 2499, "category": "Electronics", "in_stock": True},
    {"id": 7, "name": "Webcam", "price": 1899, "category": "Electronics", "in_stock": False},
]


feedback = []
orders = []
order_id_counter = 1

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


# Task 1: Filter products by query parameters
@app.get("/products/filter")
def filter_products(category: Optional[str] = None, min_price: Optional[int] = None, max_price: Optional[int] = None):
    filtered = products[:]
    if category:
        filtered = [p for p in filtered if p["category"].lower() == category.lower()]
    if min_price is not None:
        filtered = [p for p in filtered if p["price"] >= min_price]
    if max_price is not None:
        filtered = [p for p in filtered if p["price"] <= max_price]
    return {"products": filtered, "count": len(filtered)}


# Task 2: Get only the price of a product
@app.get("/products/{product_id}/price")
def get_product_price(product_id: int):
    product = next((p for p in products if p["id"] == product_id), None)
    if not product:
        return {"error": "Product not found"}
    return {"name": product["name"], "price": product["price"]}


# Task 3: Accept customer feedback
@app.post("/feedback")
def submit_feedback(f: CustomerFeedback):
    feedback.append(f.dict())
    return {"message": "Feedback submitted successfully", "feedback": f.dict(), "total_feedback": len(feedback)}


# Task 4: Build a product summary dashboard
@app.get("/products/summary")
def product_summary():
    total = len(products)
    in_stock = len([p for p in products if p["in_stock"]])
    out_of_stock = total - in_stock
    categories = list(set(p["category"] for p in products))
    cheapest = min(products, key=lambda p: p["price"])
    expensive = max(products, key=lambda p: p["price"])
    return {
        "total_products": total,
        "in_stock_count": in_stock,
        "out_of_stock_count": out_of_stock,
        "most_expensive": {"name": expensive["name"], "price": expensive["price"]},
        "cheapest": {"name": cheapest["name"], "price": cheapest["price"]},
        "categories": categories
    }


# Task 5: Validate & place a bulk order
@app.post("/orders/bulk")
def place_bulk_order(order: BulkOrder):
    confirmed = []
    failed = []
    grand_total = 0
    for item in order.items:
        product = next((p for p in products if p["id"] == item.product_id), None)
        if not product:
            failed.append({"product_id": item.product_id, "reason": "Product not found"})
            continue
        if not product["in_stock"]:
            failed.append({"product_id": item.product_id, "reason": f"{product['name']} is out of stock"})
            continue
        subtotal = product["price"] * item.quantity
        confirmed.append({"product": product["name"], "qty": item.quantity, "subtotal": subtotal})
        grand_total += subtotal
    return {"company": order.company_name, "confirmed": confirmed, "failed": failed, "grand_total": grand_total}


# Bonus: Order status tracker
@app.post("/orders")
def place_order(order: BulkOrder):
    global order_id_counter
    order_dict = order.dict()
    order_dict["id"] = order_id_counter
    order_dict["status"] = "pending"
    orders.append(order_dict)
    order_id_counter += 1
    return {"order_id": order_dict["id"], "status": "pending", "message": "Order placed successfully"}


@app.get("/orders/{order_id}")
def get_order(order_id: int):
    order = next((o for o in orders if o["id"] == order_id), None)
    if not order:
        return {"error": "Order not found"}
    return order


@app.patch("/orders/{order_id}/confirm")
def confirm_order(order_id: int):
    order = next((o for o in orders if o["id"] == order_id), None)
    if not order:
        return {"error": "Order not found"}
    if order["status"] != "pending":
        return {"error": "Order is not pending"}
    order["status"] = "confirmed"
    return {"message": "Order confirmed", "order_id": order_id, "status": "confirmed"}