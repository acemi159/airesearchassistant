from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from research_schema import ResearchGraphState
from llm_wrapper import llm
from interview_subgraph import interview_builder
from analyst_subgraph import create_analysts, human_feedback


def initiate_all_interviews(state: ResearchGraphState):
    """ This is the 'map' step where we run each interview sub-graph using Send API """

    # Check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback", "")
    if human_analyst_feedback:
        # Return to create analysts
        return "create_analysts"

    # Otherwise, kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(
                content=f"So you said you were writing an article on {topic}?"
            )]
        })
                for analyst in state["analysts"]]
        
report_writer_instructions = """You are a technical writer creating a report on this overall topic:

{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task:

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Use no sub-headings.
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the '## Sources' header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from:

{context}
"""

def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke(
        [SystemMessage(content=system_message)] +
        [HumanMessage(content=f"Write a report based upon these memos.")]
    )
    
    return {"content": report}


intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

Your job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}
"""

def write_introduction(state: ResearchGraphState):
    # Full set of sections
    
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Write the introduction
    system_message = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    introduction = llm.invoke(
        [SystemMessage(content=system_message)] +
        [HumanMessage(content=f"Write an introduction for this report.")]
    )
    
    return {"introduction": introduction.content}

    
def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    
    sections = state["sections"]
    topic = state["topic"]
    
    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Write the conclusion
    system_message = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke(
        [SystemMessage(content=system_message)] +
        [HumanMessage(content=f"Write a conclusion for this report.")]
    )
    
    return {"conclusion": conclusion.content}
    
    
def finalize_report(state: ResearchGraphState):
    """ This is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None
        
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n---\n\n## Sources\n\n" + sources
    
    return {"final_report": final_report}


# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_introduction", "write_conclusion", "write_report"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile 
memory = MemorySaver()
graph = builder.compile(checkpointer=memory,
                        interrupt_before=["human_feedback"])

# Save graph image
graph.get_graph(xray=True).draw_mermaid_png(output_file_path="./app/backend/research_graph.png")


# Example flow
if __name__ == "__main__":
    
    max_analysts = 3
    topic = "The benefits of adopting LangGraph as an agent framework"
    thread = {"configurable": {"thread_id": 1}}

    # Run the graph until it first iterrupts
    for event in graph.stream({"topic": topic,
                               "max_analysts": max_analysts},
                              thread,
                              stream_mode="values"):
        
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)

    # Now we are in human feedback node which requires human input
    # Here we provide a feedback to change the output of "create_analysts" node
    graph.update_state(thread, {"human_analyst_feedback": "Add in the CEO of gen ai native startup"}, as_node="human_feedback")

    # Run the graph again to utilize the human feedback adn get the new list of analysts with respect to the feedback we provided
    for event in graph.stream(thread,
                              stream_mode="values"):
        
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)
                
    
    # Now we are in human feedback node which requires human input
    # Here we accept the result of "create_analysts" node
    graph.update_state(thread,
                       {"human_analyst_feedback": None},
                       as_node="human_feedback")
    
    # Since we are happy about the list of analysts, now we can let the graph generate the full report
    for event in graph.stream(None, thread, stream_mode="updates"):
        node_name = next(iter(event.keys()))
        print(f"--Node--:\n\t{node_name}")
        
    
    # Get the report 
    final_state = graph.get_state(thread)
    report = final_state["final_report"]
    print(report)