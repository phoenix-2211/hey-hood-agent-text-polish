from __future__ import annotations

import os
import sys

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.workflow import Edge, Workflow, node
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.abspath(os.path.join(current_dir, "..", "shared"))
if not os.path.exists(os.path.join(shared_dir, "firestore_client.py")):
    shared_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "shared"))
sys.path.append(shared_dir)
from security_checkpoint import security_checkpoint  # noqa: E402


class PolishedText(BaseModel):
    polished_text: str = Field(description="The corrected professional text")
    language_detected: str = Field(description="Language detected: en, ta, or ta-en")
    changes_made: list = Field(description="List of changes made")


polish_agent = LlmAgent(
    name="text_polisher",
    model="gemini-flash-latest",
    instruction="""You are a text editor for a civic
community app in India called Hey Hood.

The user has written text in casual language with
possible spelling errors. Your job:
1. Fix spelling mistakes
2. Fix grammar
3. Format into clear professional sentences
4. Keep the original meaning intact
5. Keep it natural — do NOT over-formalize
6. Handle Tamil-English (Tanglish) naturally
7. Keep Tamil words if they add authenticity

Return only the corrected text, language detected,
and a brief list of changes made.
Do not add commentary.""",
    output_key="polished_result",
    output_schema=PolishedText,
)


@node
def return_polished(ctx: Context, node_input):
    result = ctx.state.get("polished_result", {})
    yield Event(output=result)


root_agent = Workflow(
    name="text_polish_workflow",
    edges=[
        ("START", security_checkpoint, polish_agent, return_polished),
        Edge(
            from_node=security_checkpoint, to_node=return_polished, route="human_review"
        ),
    ],
)

app = App(
    name="text_polish_agent",
    root_agent=root_agent,
)
