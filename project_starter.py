import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from openai import OpenAI
from smolagents import tool, OpenAIServerModel, ToolCallingAgent, CodeAgent

load_dotenv()

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]


# Generates a randomized initial inventory subset from the global paper_supplies catalog list
def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


# Sets up the SQLite database by defining tables, parsing historical and request CSV files, and seeding initial records
def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


# Records a new transaction into the transactions table. This tracks either sales to customers or restocking orders from suppliers
def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

# Computes and retrieves a complete snapshot of available items and their net stock quantities
def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


# Computes the net stock level for a single item as of a given date
def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


# Computes when a supplier restock order will be delivered based on the requested order quantity and a starting date
def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


# Computes the current cash balance as of a specified date
def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0



# Generates a complete financial report for the company as of a specific date
def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }



# Searches and retrieves historical quotes that match any of the provided search terms
def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]



# Set up and load env parameters and instantiate the model.
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key
)



# Tools for inventory agent
@tool
def inventory_check_tool(paper_type: str, quantity: int, as_of_date: str) -> dict:
    """Check inventory availability for a paper type and determine whether the order can be fulfilled immediately or if a reorder is needed.
    Args:
        paper_type: The exact name of the paper product to check.
        quantity: The number of units requested by the customer.
        as_of_date: The date of the request in YYYY-MM-DD format.
    """
    stock_df = get_stock_level(paper_type, as_of_date)
    current_stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    inventory_df = pd.read_sql(
        "SELECT * FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": paper_type},
    )

    if inventory_df.empty:
        return {
            "item_name": paper_type,
            "requested_quantity": quantity,
            "exists_in_inventory_catalog": False,
            "current_stock": 0,
            "can_fulfill_now": False,
            "needs_reorder": True,
            "message": f"{paper_type} is not currently stocked in the inventory catalog.",
        }

    min_stock_level = int(inventory_df["min_stock_level"].iloc[0])
    unit_price = float(inventory_df["unit_price"].iloc[0])

    remaining_after_order = current_stock - quantity
    can_fulfill_now = current_stock >= quantity
    needs_reorder = remaining_after_order < min_stock_level

    return {
        "item_name": paper_type,
        "requested_quantity": quantity,
        "exists_in_inventory_catalog": True,
        "current_stock": current_stock,
        "min_stock_level": min_stock_level,
        "unit_price": unit_price,
        "can_fulfill_now": can_fulfill_now,
        "remaining_after_order": remaining_after_order,
        "needs_reorder": needs_reorder,
        "message": (
            "Enough stock available for immediate fulfillment."
            if can_fulfill_now
            else "Not enough stock available for immediate fulfillment."
        ),
    }


# Tools for quoting agent
@tool
def quote_generation_tool(customer_id: str, paper_type: str, quantity: int, as_of_date: str) -> dict:
    """Generate a customer quote for a paper order, including bulk discounts and recent quote history.
    Args:
        customer_id: The identifier for the customer requesting the quote.
        paper_type: The exact name of the paper product.
        quantity: The number of units of paper requested.
        as_of_date: The date of the request in YYYY-MM-DD format.
    """
    inventory_result = inventory_check_tool(paper_type, quantity, as_of_date)

    if not inventory_result["exists_in_inventory_catalog"]:
        return {
            "customer_id": customer_id,
            "item_name": paper_type,
            "requested_quantity": quantity,
            "quote_available": False,
            "message": f"Cannot generate quote because {paper_type} is not in the inventory catalog.",
        }

    unit_price = inventory_result["unit_price"]
    base_total = unit_price * quantity

    discount_rate = 0.0
    if quantity >= 1000:
        discount_rate = 0.15
    elif quantity >= 500:
        discount_rate = 0.10
    elif quantity >= 100:
        discount_rate = 0.05

    discount_amount = base_total * discount_rate
    final_total = base_total - discount_amount

    history = search_quote_history([paper_type], limit=3)

    return {
        "customer_id": customer_id,
        "item_name": paper_type,
        "requested_quantity": quantity,
        "unit_price": unit_price,
        "base_total": round(base_total, 2),
        "discount_rate": discount_rate,
        "discount_amount": round(discount_amount, 2),
        "final_total": round(final_total, 2),
        "can_fulfill_now": inventory_result["can_fulfill_now"],
        "needs_reorder": inventory_result["needs_reorder"],
        "quote_history": history,
        "quote_available": True,
        "message": f"Quote generated for {quantity} units of {paper_type}.",
    }


# Tools for ordering agent
@tool
def supplier_timeline_tool(paper_type: str, quantity_needed: int, as_of_date: str) -> dict:
    """Estimate when a supplier can deliver additional stock for a paper type.
    Args:
        paper_type: The name of the paper product to restock.
        quantity_needed: The amount of stock being ordered from the supplier.
        as_of_date: The date the restock order is placed in YYYY-MM-DD format.
    """
    estimated_delivery_date = get_supplier_delivery_date(as_of_date, quantity_needed)

    return {
        "item_name": paper_type,
        "quantity_needed": quantity_needed,
        "estimated_delivery_date": estimated_delivery_date,
        "message": f"Supplier can deliver {quantity_needed} units of {paper_type} by {estimated_delivery_date}.",
    }


@tool
def fulfill_order_tool(customer_id: str, paper_type: str, quantity: int, as_of_date: str) -> dict:
    """Fulfill a customer order if stock is available, execute sales transaction, and automatically place restocking orders with the supplier when inventory is low or insufficient.
        Args:
        customer_id: The identifier for the customer placing the order.
        paper_type: The exact name of the paper product to be purchased.
        quantity: The number of units the customer wants to buy.
        as_of_date: The date the order is executed in YYYY-MM-DD format.
        """
    # 1. Fetch current stock and catalog details
    inventory_result = inventory_check_tool(paper_type, quantity, as_of_date)

    if not inventory_result["exists_in_inventory_catalog"]:
        return {
            "customer_id": customer_id,
            "item_name": paper_type,
            "requested_quantity": quantity,
            "fulfilled": False,
            "message": f"Order cannot be fulfilled because {paper_type} is not in the inventory catalog.",
        }

    # Retrieve unit price and safety level
    unit_price = inventory_result["unit_price"]
    min_stock_level = inventory_result["min_stock_level"]

    # ==========================================
    # SCENARIO A: Insufficient Stock (Failed Sale -> Emergency Restock)
    # ==========================================
    if not inventory_result["can_fulfill_now"]:
        shortage = quantity - inventory_result["current_stock"]
        # Determine restocking quantity (shortage plus safety stock buffer)
        restock_quantity = shortage + (min_stock_level * 2)
        restock_cost = restock_quantity * unit_price
        
        # Check cash balance before purchasing
        cash = get_cash_balance(as_of_date)
        restock_info = {"restock_executed": False}

        if cash >= restock_cost:
            # Execute supplier order
            restock_tx = create_transaction(
                item_name=paper_type,
                transaction_type="stock_orders",
                quantity=restock_quantity,
                price=restock_cost,
                date=as_of_date,
            )
            supplier_info = supplier_timeline_tool(paper_type, restock_quantity, as_of_date)
            restock_info = {
                "restock_executed": True,
                "restock_quantity": restock_quantity,
                "restock_cost": round(restock_cost, 2),
                "restock_transaction_id": restock_tx,
                "supplier_timeline": supplier_info,
            }
        else:
            restock_info["reason"] = f"Insufficient cash to restock. Need: ${restock_cost:.2f}, Have: ${cash:.2f}"

        return {
            "customer_id": customer_id,
            "item_name": paper_type,
            "requested_quantity": quantity,
            "fulfilled": False,
            "shortage": shortage,
            "restock_info": restock_info,
            "message": "Insufficient stock to fulfill order immediately. A supplier order has been placed/attempted.",
        }

    # ==========================================
    # SCENARIO B: Successful Sale (Fulfill -> Proactive Restock if needed)
    # ==========================================
    quote_result = quote_generation_tool(customer_id, paper_type, quantity, as_of_date)
    
    # Record the sale transaction
    sale_tx = create_transaction(
        item_name=paper_type,
        transaction_type="sales",
        quantity=quantity,
        price=quote_result["final_total"],
        date=as_of_date,
    )

    # If the sale causes inventory to drop below the safety limit, trigger a proactive restock
    reorder_info = None
    if inventory_result["needs_reorder"]:
        reorder_quantity = max(quantity, min_stock_level * 2)
        reorder_cost = reorder_quantity * unit_price
        
        # Verify company cash
        cash = get_cash_balance(as_of_date)
        if cash >= reorder_cost:
            # Execute supplier order
            restock_tx = create_transaction(
                item_name=paper_type,
                transaction_type="stock_orders",
                quantity=reorder_quantity,
                price=reorder_cost,
                date=as_of_date,
            )
            supplier_info = supplier_timeline_tool(paper_type, reorder_quantity, as_of_date)
            
            reorder_info = {
                "restock_executed": True,
                "reorder_quantity": reorder_quantity,
                "reorder_cost": round(reorder_cost, 2),
                "restock_transaction_id": restock_tx,
                "supplier_timeline": supplier_info,
            }
        else:
            reorder_info = {
                "restock_executed": False,
                "reason": f"Insufficient cash to restock safety levels. Need: ${reorder_cost:.2f}, Have: ${cash:.2f}"
            }

    return {
        "customer_id": customer_id,
        "item_name": paper_type,
        "requested_quantity": quantity,
        "fulfilled": True,
        "transaction_id": sale_tx,
        "sale_price": quote_result["final_total"],
        "reorder_info": reorder_info,
        "message": f"Order fulfilled successfully for {quantity} units of {paper_type}.",
    }


# Set up your agents and create an orchestration agent that will manage them.
class InventoryAgent(ToolCallingAgent):
    def __init__(self, model):
        super().__init__(
            tools=[inventory_check_tool],
            model=model,
            name="inventory_agent",
            description="Checks item stock level, safety minimums, and flags if restocks are needed."
        )

    def check_stock(self, paper_type: str, quantity: int, request_date: str) -> str:
        prompt = f"""
        You are the Inventory Agent. 
        Please check if we can fulfill an order for {quantity} units of '{paper_type}' as of {request_date}.
        Use the `inventory_check_tool` tool to query the database.
        State the stock status, whether it is sufficient, and if restocking is needed.
        """
        return self.run(prompt)


class QuoteAgent(ToolCallingAgent):
    def __init__(self, model):
        super().__init__(
            tools=[quote_generation_tool],
            model=model,
            name="quote_agent",
            description="Generates customer quotes and applies appropriate tiered bulk discounts."
        )

    def generate_quote(self, customer_id: str, paper_type: str, quantity: int, request_date: str) -> str:
        prompt = f"""
        You are the Quote Specialist.
        Generate a quote for customer '{customer_id}' ordering {quantity} units of '{paper_type}' as of {request_date}.
        Use the `quote_generation_tool` to calculate standard price, discounts, and retrieve history.
        Summarize the quote details and explain the discount applied.
        """
        return self.run(prompt)


class FulfillmentAgent(ToolCallingAgent):
    def __init__(self, model):
        super().__init__(
            tools=[fulfill_order_tool, supplier_timeline_tool],
            model=model,
            name="fulfillment_agent",
            description="Logs sales transactions to customers and handles restocking orders to suppliers."
        )

    def process_fulfillment(self, customer_id: str, paper_type: str, quantity: int, request_date: str) -> str:
        prompt = f"""
        You are the Fulfillment Agent.
        Fulfill the order for customer '{customer_id}' requesting {quantity} units of '{paper_type}' as of {request_date}.
        Use `fulfill_order_tool` to execute the transaction in the database.
        If a restock is recommended or if stock is insufficient, make sure you coordinate with `supplier_timeline_tool` to get the timeline.
        Summarize the transaction ID, purchase status, and delivery date.
        """
        return self.run(prompt)


class OrchestratorAgent(ToolCallingAgent):
    """Orchestrator that coordinates the Munder Difflin paper order workflow."""
    
    def __init__(self, model):
        self.model = model
        
        # 1. Initialize the specialized sub-agents
        self.inventory_agent = InventoryAgent(model)
        self.quote_agent = QuoteAgent(model)
        self.fulfillment_agent = FulfillmentAgent(model)

        # 2. Create coordination routing tools
        @tool
        def check_stock_level(paper_type: str, quantity: int, request_date: str) -> str:
            """Check the database to see if we have enough stock of a paper type.
            
            Args:
                paper_type: The exact name of the paper product
                quantity: The number of units requested
                request_date: The date of the request (YYYY-MM-DD)
                
            Returns:
                A text report from the Inventory Agent indicating stock levels and restocking needs
            """
            return self.inventory_agent.check_stock(paper_type, quantity, request_date)

        @tool
        def create_customer_quote(customer_id: str, paper_type: str, quantity: int, request_date: str) -> str:
            """Generate a pricing quote for a customer order, including bulk discounts.
            
            Args:
                customer_id: The ID of the customer requesting the quote
                paper_type: The exact name of the paper product
                quantity: The quantity of paper requested
                request_date: The date of the request (YYYY-MM-DD)
                
            Returns:
                A quote summary with base price, discounts, and history from the Quote Agent
            """
            return self.quote_agent.generate_quote(customer_id, paper_type, quantity, request_date)

        @tool
        def execute_fulfillment(customer_id: str, paper_type: str, quantity: int, request_date: str) -> str:
            """Process order fulfillment, log sales transactions, and handle supplier restocking if stock is low.
            
            Args:
                customer_id: The ID of the customer placing the order
                paper_type: The exact name of the paper product
                quantity: The quantity of paper requested
                request_date: The date of the request (YYYY-MM-DD)
                
            Returns:
                A fulfillment report with transaction IDs, totals, and delivery dates from the Fulfillment Agent
            """
            return self.fulfillment_agent.process_fulfillment(customer_id, paper_type, quantity, request_date)

        # 3. Call the parent constructor with the coordination tools
        super().__init__(
            tools=[check_stock_level, create_customer_quote, execute_fulfillment],
            model=model,
            name="orchestrator",
            description="Coordinates specialized agents for inventory, quotes, and fulfillment.",
        )
        
        def process_request(self, customer_request: str) -> str:
            """
            Process a customer request through the coordinated agent workflow.
            """
            prompt = f"""
            You are the Chief Orchestrator. 
            A customer has sent a request: "{customer_request}".
            
            Your objective is to coordinate the workflow to handle this request using your available tools.
            
            CRITICAL workflow rules:
            1. Extract the paper product name, quantity, customer ID, and request date from the request details.
            2. Call `check_stock_level` first to check stock availability as of the requested date.
            3. If stock is sufficient:
            - Call `create_customer_quote` to get the pricing.
            - Call `execute_fulfillment` to log the sales transaction.
            - Once `execute_fulfillment` is run, immediately return your final answer summarizing the successful sale. Do not loop.
            4. If stock is insufficient:
            - Call `execute_fulfillment` EXACTLY ONCE to execute the emergency supplier restocking order.
            - Immediately after calling `execute_fulfillment`, use the `final_answer` tool to inform the customer of the stockout, shortage, and delivery date.
            - DO NOT check stock again or call any other tool after placing the restock order, as the new stock will only arrive in the future.
            5. Provide a final, comprehensive response to the customer summarizing all transaction details.
            """
            return self.run(prompt)



# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    inventory_sub = InventoryAgent(model=model)

    quote_sub = QuoteAgent(model=model)

    fulfillment_sub = FulfillmentAgent(model=model)

    orchestrator_agent = OrchestratorAgent(model=model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE THE MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = orchestrator_agent.run(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
