export type NodeExecutionStatus =
  | "waiting"
  | "running"
  | "completed"
  | "failed"
  | "disabled";

export type GraphNode = {
  id: string;
  label: string;
  description: string;
  kind: string;
  required: boolean;
  enabled: boolean;
  stage: number;
};

export type GraphEdge = {
  source: string;
  target: string;
  conditional: boolean;
  branch: string | null;
};

export type GraphDescription = {
  graph_id: string;
  options: { faq_retrieval: boolean };
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type ExecutionEvent = {
  type: string;
  node_id?: string;
  output?: Record<string, unknown>;
  message?: string;
  result?: Record<string, unknown>;
};

export type NodeExecution = {
  status: NodeExecutionStatus;
  output?: Record<string, unknown>;
};

export function isGraphDescription(value: unknown): value is GraphDescription {
  if (typeof value !== "object" || value === null) return false;
  const graph = value as GraphDescription;
  return (
    graph.graph_id === "unified-support" &&
    Array.isArray(graph.nodes) &&
    graph.nodes.every(
      (node) =>
        typeof node.id === "string" &&
        typeof node.label === "string" &&
        typeof node.enabled === "boolean" &&
        typeof node.required === "boolean" &&
        typeof node.stage === "number",
    ) &&
    Array.isArray(graph.edges)
  );
}

export function initialNodeExecutions(
  graph: GraphDescription,
): Record<string, NodeExecution> {
  return Object.fromEntries(
    graph.nodes.map((node) => [
      node.id,
      { status: node.enabled ? "waiting" : "disabled" },
    ]),
  );
}

export function applyExecutionEvent(
  current: Record<string, NodeExecution>,
  event: ExecutionEvent,
): Record<string, NodeExecution> {
  if (event.type === "run_started" && current.__start__) {
    return { ...current, __start__: { status: "completed" } };
  }
  if (event.type === "run_completed" && current.__end__) {
    return { ...current, __end__: { status: "completed" } };
  }
  if (!event.node_id || !current[event.node_id]) return current;
  if (event.type === "node_started") {
    return {
      ...current,
      [event.node_id]: { status: "running" },
    };
  }
  if (event.type === "node_completed") {
    return {
      ...current,
      [event.node_id]: { status: "completed", output: event.output },
    };
  }
  if (event.type === "node_failed") {
    return {
      ...current,
      [event.node_id]: { status: "failed", output: event.output },
    };
  }
  return current;
}

export function parseSseBuffer(buffer: string): {
  events: ExecutionEvent[];
  remainder: string;
} {
  const normalized = buffer.replaceAll("\r\n", "\n");
  const frames = normalized.split("\n\n");
  const remainder = frames.pop() ?? "";
  const events: ExecutionEvent[] = [];

  for (const frame of frames) {
    const payload = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!payload) continue;
    try {
      const value: unknown = JSON.parse(payload);
      if (
        typeof value === "object" &&
        value !== null &&
        typeof (value as ExecutionEvent).type === "string"
      ) {
        events.push(value as ExecutionEvent);
      }
    } catch {
      continue;
    }
  }

  return { events, remainder };
}
