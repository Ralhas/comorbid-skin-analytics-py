# mcp_server.py
#
# Exposes the v4 agent (agent_graph.ask_agent) as an MCP tool, so
# other MCP clients (e.g. Claude Desktop) can call it directly.

from mcp.server import MCPServer
from agent_graph import ask_agent
from structured_llm import ask_structured

mcp = MCPServer(
    name="comorbid-skin-analytics",
    description=(
        "Cosmetic ingredient risk analysis for comorbid skin conditions "
        "(rosacea, acne, melasma, atopic dermatitis). Deterministic "
        "database matching + LLM interpretation + self-verification."
    ),
)


@mcp.tool()
def check_ingredient_safety(question: str) -> dict:
    """
    Checks risk level for one or more cosmetic ingredients against
    comorbid skin conditions. Runs the full LangGraph pipeline
    (conflict detection + deterministic self-verification).

    Args:
        question: natural language question, e.g.
                  "is salicylic acid safe for rosacea"
                  "can I use niacinamide and retinol together"

    Returns:
        risk_level, mechanism, summary, confidence, verified,
        correction_note, conflict_warning (if applicable)
    """
    return ask_agent(question)


@mcp.tool()
def ask_with_provider(question: str, provider: str = "gemini") -> dict:
    """
    Same question, different LLM provider. Only "gemini" is implemented;
    "openai", "anthropic", "deepseek" are stubbed (no API key configured)
    and return a clear error instead of failing silently.

    No self-verification here -- single LLM call. Use
    check_ingredient_safety for a verified result.

    Args:
        question: natural language question
        provider: "gemini" | "openai" | "anthropic" | "deepseek"
    """
    try:
        return ask_structured(question, provider=provider)
    except NotImplementedError as e:
        return {
            "risk_level": "unknown",
            "mechanism": "",
            "summary": str(e),
            "confidence": "low",
        }


if __name__ == "__main__":
    mcp.run()
