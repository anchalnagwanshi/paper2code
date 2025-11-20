import logging
from langgraph.graph import StateGraph
from langgraph.graph import END
from schema import GraphState
import nodes
import config

logger = logging.getLogger(__name__)
# --- 1. Routers / Control Flow Logic ---

def decide_after_researcher(state: GraphState):
    """Decides the next step after the researcher node."""
    if state.get("error"):
        logger.error("Router: Error in Researcher. Halting.")
        return "end"
    logger.info("Router: Researcher OK. Proceeding to MLflow setup.")
    return "setup_tracking"

def decide_after_tracking(state: GraphState):
    """Decides the next step after MLflow setup."""
    if state.get("error"):
        logger.error("Router: Error in MLflow Setup. Halting.")
        return "end"
    logger.info("Router: MLflow OK. Proceeding to Coder.")
    return "coder"

def decide_after_qa(state: GraphState):
    """Decides the next step after the QA node (retry or end)."""
    if state.get("error"):
        if state["retries"] >= config.MAX_RETRIES:
            logger.error(f"Router: Max retries ({config.MAX_RETRIES}) exceeded. Halting.")
            return "end"
        logger.warning("Router: QA Failed. Retrying with Coder.")
        return "coder" # Loop back to coder
    logger.info("Router: QA Success. Ending.")
    return "end"

# --- 2. Graph Assembly ---

def create_graph():
    """Assembles and compiles the LangGraph application."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("researcher", nodes.researcher_node)
    workflow.add_node("setup_tracking", nodes.setup_tracking_node)
    workflow.add_node("coder", nodes.coder_node)
    workflow.add_node("qa", nodes.qa_node)

    # Set entry point
    workflow.set_entry_point("researcher")

    # Add conditional edges
    workflow.add_conditional_edges(
        "researcher",
        decide_after_researcher,
        {"setup_tracking": "setup_tracking", "end": END}
    )
    workflow.add_conditional_edges(
        "setup_tracking",
        decide_after_tracking,
        {"coder": "coder", "end": END}
    )
    workflow.add_conditional_edges(
        "qa",
        decide_after_qa,
        {"coder": "coder", "end": END}
    )
    
    # Add normal edges
    workflow.add_edge("coder", "qa")

    # Compile and return
    logger.info("Graph compiled.")
    return workflow.compile()

# Create a single compiled app instance to be imported
app = create_graph()