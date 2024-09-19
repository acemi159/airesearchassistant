import operator
from typing import List, Annotated
from typing_extensions import TypedDict

from analyst_schema import Analyst

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Maximum number of analysts to use
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst]
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for final report
    content: str # Content for final report
    conclusion: str # Conclusion for final report
    final_report: str # Final report
    
