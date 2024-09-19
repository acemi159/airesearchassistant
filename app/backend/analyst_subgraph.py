from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from analyst_schema import Analyst, Perspectives, GenerateAnalystsState
from llm_wrapper import llm

analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of analysts:
{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme.
"""

def create_analysts(state: GenerateAnalystsState):
    
    """ Create analysts """

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)
    
    # System message
    system_message = analyst_instructions.format(topic=topic, 
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)

    # Generate question
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")]
    )
    
    # Write the list of analysis to state
    return {"analysts": analysts}


def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass


def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # Check if there a human feedback
    human_analyst_feedback = state.get("human_analyst_feedback", "")
    if human_analyst_feedback:
        return "create_analysts"

    # If no human feedback, then end
    return END


# Add nodes and edges
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    should_continue,
    ["create_analysts", END]
)

# Compile the graph
memory = MemorySaver()
graph = builder.compile(
    interrupt_before=["human_feedback"],
    checkpointer=memory,
)

# Save the graph view
graph.get_graph(xray=1).draw_mermaid_png(output_file_path="app/backend/analyst_subgraph.png")

if __name__ == "__main__":
     # Input
    max_analysts = 3 
    topic = "The benefits of adopting LangGraph as an agent framework"
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream({"topic":topic,"max_analysts":max_analysts,}, thread, stream_mode="values"):
        # Review
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)  