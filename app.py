import streamlit as st
import pandas as pd
from datetime import datetime
from orchestrator import (
    db_engine,
    init_database,
    generate_financial_report,
    OrchestratorAgent,
    model
)

# Set page config for a premium, wide-screen dashboard layout
st.set_page_config(
    page_title="Munder Difflin Operations Portal",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look and feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 1. State Initialization
if "initialized" not in st.session_state:
    init_database(db_engine)
    st.session_state.orchestrator = OrchestratorAgent(model=model)

    st.session_state.last_response = None # Store response here

    st.session_state.initialized = True

st.markdown("<h1 class='main-header'>📄 Munder Difflin Operations Portal</h1>", unsafe_allow_html=True)
st.write("Welcome to the Multi-Agent Paper Logistics & Fulfillment Dashboard.")

# Fetch current state metrics
current_date_str = datetime.now().strftime("%Y-%m-%d")
report = generate_financial_report(current_date_str)

# 2. Main Dashboard Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Available Cash Balance", f"${report['cash_balance']:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Inventory Valuation", f"${report['inventory_value']:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Assets", f"${report['total_assets']:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# 3. Sidebar inputs
st.sidebar.header("📥 Submit Customer Request")
customer_id = st.sidebar.text_input("Customer ID", value="Cust_101")
request_date = st.sidebar.date_input("Simulation Date", value=datetime.today())
customer_request_text = st.sidebar.text_area(
    "Customer Request Prompt",
    placeholder="e.g., We need 500 sheets of colored paper for our event."
)

submit_btn = st.sidebar.button("Submit Order to Agents")


# 4. Display the last response if it exists
if st.session_state.last_response:
    st.subheader("Agent Communication and Outcome")
    st.info(st.session_state.last_response)

# 5. Process Request via Multi-Agent System
if submit_btn and customer_request_text:
    formatted_date = request_date.strftime("%Y-%m-%d")
    prompt_with_date = f"{customer_request_text} (Date of request: {formatted_date}) (Customer ID: {customer_id})"
    
    with st.spinner("Orchestrator coordinating sub-agents..."):
        try:
            # Delegate to OrchestratorAgent
            response = st.session_state.orchestrator.process_request(prompt_with_date)

            # Save the response to session_state so it survives the rerun
            st.session_state.last_response = response
            
            # Display results
            st.success("Order Processed Successfully!")
            st.subheader("🤖 Agent Communication & Outcome")
            st.info(response)
            
            # Force metrics refresh
            st.rerun()
            
        except Exception as e:
            st.error(f"Execution Error: {e}")

# 5. Inventory & Transaction Logging Tables
tab1, tab2 = st.tabs(["Warehouse Inventory", "Transaction Ledger"])

with tab1:
    st.subheader("Current Stock Levels & Valuation")
    inventory_summary_df = pd.DataFrame(report['inventory_summary'])
    if not inventory_summary_df.empty:
        st.dataframe(
            inventory_summary_df.rename(columns={
                "item_name": "Item Name",
                "stock": "Quantity in Stock",
                "unit_price": "Unit Price ($)",
                "value": "Total Valuation ($)"
            }),
            use_container_width=True
        )
    else:
        st.warning("No inventory records found.")

with tab2:
    st.subheader("Database Transaction Ledger")
    try:
        transactions_df = pd.read_sql(
            "SELECT * FROM transactions ORDER BY transaction_date DESC, id DESC", 
            db_engine
        )
        if not transactions_df.empty:
            st.dataframe(
                transactions_df.rename(columns={
                    "id": "Tx ID",
                    "item_name": "Paper Type",
                    "transaction_type": "Transaction Type",
                    "units": "Quantity",
                    "price": "Total Price ($)",
                    "transaction_date": "Execution Date"
                }),
                use_container_width=True
            )
        else:
            st.info("No transactions logged in ledger.")
    except Exception as e:
        st.error(f"Error loading ledger: {e}")
