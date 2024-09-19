import operator
from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from analyst_schema import Analyst


class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API
    
    
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrival.")