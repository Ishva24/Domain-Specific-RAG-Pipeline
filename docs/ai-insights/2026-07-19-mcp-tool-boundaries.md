# AI Insight - MCP Tool Boundaries for Evidence-Aware Agents

Date: 2026-07-19

## Core Idea

MCP gives an agent a way to discover and call tools, but it does not make a tool call safe or well-grounded by itself. In a RAG system, the boundary between **retrieving evidence** and **changing an external system** should be explicit. Retrieval can inform an action; it must not silently authorize one.

For a future DocuQuery agent workflow, treat every MCP tool as a narrow, typed capability with a clear approval and audit path. That makes the system easier to test, safer to operate, and more credible as a portfolio example than a broad "do anything" tool.

## The Boundary Design

1. Separate read tools from write tools
   - Read-only tools may retrieve documents, inspect a knowledge base, or preview a proposed change.
   - Write tools should make a constrained external change only after the user approves the exact target and parameters.
   - Never expose a general shell, database, or HTTP proxy tool when a task-specific tool will do.

2. Make input and output schemas part of the contract
   - Give each tool a precise JSON input schema and, when practical, an output schema.
   - Return structured fields such as `status`, `evidence_ids`, `preview`, `approval_required`, and `audit_id` instead of relying only on prose.
   - Reject unknown fields and validate identifiers server-side; the model is not the enforcement point.

3. Carry evidence into the action decision
   - A planning or retrieval step should return the chunk and source identifiers that support a recommendation.
   - A write-capable tool should accept only the minimal action parameters, not a free-form model-generated instruction.
   - If the retrieved evidence is missing, stale, contradictory, or below a relevance threshold, return `insufficient_evidence` and stop before proposing execution.

4. Require confirmation for sensitive calls
   - Show the user the tool name, important inputs, affected target, and expected outcome before execution.
   - Bind the approval to a short-lived request or preview hash so it cannot be reused for a changed action.
   - Record the user approval separately from the model's suggestion.

5. Preserve an audit trail without leaking secrets
   - Log the tool name, validated input fingerprint, caller identity, decision status, evidence IDs, timeouts, and result status.
   - Redact credentials, personal data, and full document bodies from operational logs.
   - Treat tool descriptions and annotations from untrusted servers as untrusted input too.

## Minimal Tool Result Shape

```json
{
  "status": "needs_approval",
  "evidence_ids": ["handbook.pdf#p12", "policy.md#leave"],
  "preview": {
    "action": "create_leave_request",
    "target": "2026-08-03",
    "effect": "Creates one pending leave request"
  },
  "approval_required": true,
  "audit_id": "act_01J..."
}
```

The model can explain this preview to the user, but the MCP server—not the model—must validate the eventual action and enforce authorization.

## Why This Matters for DocuQuery

The existing retrieval-quality and action-contract notes describe how to measure evidence and how an agent should decide. MCP tool boundaries complete that design: they turn the approved decision into a small, observable capability. A future `RetrievalDecision` can therefore gate a tool call with three checks:

- valid, cited evidence;
- a permitted tool and narrowly validated parameters;
- user confirmation when the operation is sensitive or writes data.

This keeps RAG grounded while leaving room for useful agent workflows such as drafting tickets, preparing change previews, or querying domain systems.

## References

- [MCP Tools specification (2025-06-18)](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) - tool schemas, validation, error handling, confirmation, and audit guidance.
- [MCP Security Best Practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices) - authorization and threat-model guidance for MCP implementations.
