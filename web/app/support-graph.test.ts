import assert from "node:assert/strict";
import test from "node:test";

import {
  applyExecutionEvent,
  graphMatchesOptions,
  initialNodeExecutions,
  isGraphDescription,
  isTerminalExecutionEvent,
  parseSseBuffer,
} from "./support-graph";

const graph = {
  graph_id: "unified-support",
  options: { faq_retrieval: false },
  nodes: [
    {
      id: "__start__",
      label: "User query",
      description: "Input.",
      kind: "boundary",
      required: true,
      enabled: true,
      stage: 0,
    },
    {
      id: "safety_check",
      label: "OpenAI Moderation",
      description: "Required.",
      kind: "safety",
      required: true,
      enabled: true,
      stage: 1,
    },
    {
      id: "faq_retrieval",
      label: "FAQ retrieval",
      description: "Optional.",
      kind: "retrieval",
      required: false,
      enabled: false,
      stage: 4,
    },
    {
      id: "__end__",
      label: "Final result",
      description: "Output.",
      kind: "boundary",
      required: true,
      enabled: true,
      stage: 6,
    },
  ],
  edges: [],
};

test("keeps disabled backend stages visually distinct", () => {
  assert.equal(isGraphDescription(graph), true);

  const executions = initialNodeExecutions(graph);

  assert.equal(executions.safety_check.status, "waiting");
  assert.equal(executions.faq_retrieval.status, "disabled");
});

test("rejects a graph description for stale options", () => {
  assert.equal(graphMatchesOptions(graph, false), true);
  assert.equal(graphMatchesOptions(graph, true), false);
  assert.equal(graphMatchesOptions(null, false), false);
});

test("applies actual node lifecycle events", () => {
  const initial = initialNodeExecutions(graph);
  const running = applyExecutionEvent(initial, {
    type: "node_started",
    node_id: "safety_check",
  });
  const completed = applyExecutionEvent(running, {
    type: "node_completed",
    node_id: "safety_check",
    output: { is_safe: true },
  });

  assert.equal(running.safety_check.status, "running");
  assert.deepEqual(completed.safety_check, {
    status: "completed",
    output: { is_safe: true },
  });
  assert.equal(completed.faq_retrieval.status, "disabled");

  const reset = initialNodeExecutions(graph);
  assert.deepEqual(reset.safety_check, { status: "waiting" });
  assert.equal("output" in reset.safety_check, false);
});

test("parses complete SSE frames and preserves partial data", () => {
  const parsed = parseSseBuffer(
    'event: execution\ndata: {"type":"node_started","node_id":"safety_check"}\n\n' +
      'event: execution\ndata: {"type":"node_completed"',
  );

  assert.deepEqual(parsed.events, [
    { type: "node_started", node_id: "safety_check" },
  ]);
  assert.match(parsed.remainder, /node_completed/);
});

test("distinguishes terminal run events from partial streams", () => {
  assert.equal(isTerminalExecutionEvent({ type: "node_completed" }), false);
  assert.equal(isTerminalExecutionEvent({ type: "run_completed" }), true);
  assert.equal(isTerminalExecutionEvent({ type: "run_failed" }), true);
});

test("attaches backend evidence to boundary and completed node state", () => {
  const initial = initialNodeExecutions(graph);
  const startEvidence = {
    stage_id: "__start__",
    label: "User query",
    summary: "Input accepted.",
    inputs: { user_query: "hello" },
    outputs: { accepted: true },
  };
  const nodeEvidence = {
    stage_id: "safety_check",
    label: "OpenAI Moderation",
    summary: "Allowed.",
    inputs: { user_query: "hello" },
    outputs: { decision: "allowed" },
  };
  const endEvidence = {
    stage_id: "__end__",
    label: "Final result",
    summary: "Completed.",
    inputs: { executed_path: ["safety_check"] },
    outputs: { outcome: "completed" },
  };

  const started = applyExecutionEvent(initial, {
    type: "run_started",
    evidence: startEvidence,
  });
  const completed = applyExecutionEvent(started, {
    type: "node_completed",
    node_id: "safety_check",
    evidence: nodeEvidence,
    output: nodeEvidence.outputs,
  });
  const ended = applyExecutionEvent(completed, {
    type: "run_completed",
    evidence: endEvidence,
  });

  assert.deepEqual(started.__start__.evidence, startEvidence);
  assert.deepEqual(completed.safety_check.evidence, nodeEvidence);
  assert.deepEqual(ended.__end__.evidence, endEvidence);
});
