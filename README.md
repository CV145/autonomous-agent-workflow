# Munder Difflin Multi-Agent System: Coding Notebook & Reference

This repository is a structured reference implementation of a **Multi-Agent System** designed using the Hugging Face `smolagents` library. It automates core operations for the fictional Munder Difflin Paper Company—handling inventory checks, customer quote generation (with tiered bulk discounts), and logistics/fulfillment transactions.

---

## 🏗️ System Architecture

The project implements a **Hub-and-Spoke Router Pattern**. The Orchestrator acts as the centralized coordinator, accepting customer queries and invoking routing tools that delegate specific tasks to specialized sub-agents.

---
# Core Concepts Learned

1. The Subclassed Agent Pattern
Instead of instantiating generic agents via configurations, this project subclassed ToolCallingAgent to encapsulate domain-specific parameters, names, descriptions, and custom execution methods.

2. Method-Based Prompting & Execution
Rather than overriding system prompts in the constructor (which can conflict with the framework's internal ReAct instructions), we implemented a pattern where agents are passed their tools and base configuration at startup, and then use specific class methods to run runtime prompts:

Sub-agents run prompts like self.run(f"Check stock for {quantity} of {paper_type}...").
This preserves the default ReAct prompting structure needed to parse tool calls correctly while applying specialized instructions per turn.

3. The Router Tool Pattern (Orchestration)

Sub-agents are instantiated inside the Orchestrator's constructor (__init__).
Inner functions decorated with @tool are defined inside the Orchestrator's constructor, which capture self via closure.
These inner tools delegate tasks directly to the sub-agent methods.
The Orchestrator calls super().__init__(tools=[routing_tools], ...) to expose these routing paths to the LLM.

4. Strict Docstring Schema Validation
smolagents uses function type hints and docstrings to auto-generate the JSON tool schemas passed to the LLM. Under the hood, the library performs strict validation:

Every argument in the python signature must be documented in the docstring under an Args: block.
Failure to do so raises a DocstringParsingException.

---
# Setup

1. Prerequisites
Ensure you have Python 3.8+ installed.

2. Virtual Environment Setup
It is critical to install dependencies in the local virtual environment (.venv) rather than the global environment to prevent import conflicts:

bash
# Create the virtual environment (if not already present)
python -m venv .venv
# Activate the environment (Mac/Linux)
source .venv/bin/activate
# Install required packages
pip install -r requirements.txt
pip install smolagents

3. Configure Credentials
Create a .env file in the root directory:

env
OPENAI_API_KEY=your_vocareum_api_key_here

4. Run the Test Bench
Execute the script using the environment's python bin:

bash
./.venv/bin/python project_starter.py

---
# Code Snippets

A. Subclassed Agent with Method Execution
python
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
        
B. Nested Orchestrator Routing Tools
python
class OrchestratorAgent(ToolCallingAgent):
    def __init__(self, model):
        self.model = model
        
        # Sub-agents initialized locally
        self.inventory_agent = InventoryAgent(model)
        self.quote_agent = QuoteAgent(model)
        self.fulfillment_agent = FulfillmentAgent(model)
        # Routing tool defined as an inner function
        @tool
        def check_stock_level(paper_type: str, quantity: int, request_date: str) -> str:
            """Check the database to see if we have enough stock of a paper type.
            
            Args:
                paper_type: The exact name of the paper product
                quantity: The number of units requested
                request_date: The date of the request (YYYY-MM-DD)
            """
            return self.inventory_agent.check_stock(paper_type, quantity, request_date)
        super().__init__(
            tools=[check_stock_level, ...],
            model=model,
            name="orchestrator",
            description="Coordinates specialized agents for order workflows."
        )