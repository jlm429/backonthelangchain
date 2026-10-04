import assert from "node:assert/strict";
import test from "node:test";

import {
  applyExecutionEvent,
  graphMatchesOptions,
  initialNodeExecutions,
  isGraphDescription,
  parseSseBuffer,
} from "./support-graph";

const graph = {
  graph_id: "unified-support",
  options: { faq_retrieval: false },
  nodes: [
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
