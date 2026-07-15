#!/usr/bin/env python3
"""
Demo of answer generation output format (without API call).
Shows the expected structure and citations.
"""

def demo_answer_output():
    """Show example output format"""

    example_query = "What is the incident response process?"

    # Mock answer that would come from LLM
    mock_answer = """The incident response process involves several key steps:

1. Identify and classify the incident
2. Contain affected systems
3. Preserve evidence
4. Eradicate the root cause
5. Restore services
6. Document lessons learned

Organizations should also establish preparation measures including governance, policies, roles, training, and tools."""

    # Mock sources from retrieval
    mock_sources = """- Response Steps (incident_response_runbook.md)
- Preparation (nist_800_61_incident_response.md)
- Purpose (incident_response_runbook.md)
- Detection and Analysis (nist_800_61_incident_response.md)"""

    final_output = f"{mock_answer}\n\nSources:\n{mock_sources}"

    print(f"Query: {example_query}")
    print("\n" + "=" * 100)
    print("EXAMPLE OUTPUT:")
    print("=" * 100)
    print(final_output)

if __name__ == "__main__":
    demo_answer_output()
