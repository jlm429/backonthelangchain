"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { RoutingSource } from "./routing-source";
import {
  ExecutionEvent,
  GraphDescription,
  NodeExecution,
  StageEvidence,
  applyExecutionEvent,
  graphMatchesOptions,
  initialNodeExecutions,
  isGraphDescription,
  isTerminalExecutionEvent,
  parseSseBuffer,
} from "./support-graph";

type SystemName = "authentication" | "billing" | "checkout" | "api";
type StatusLevel = "operational" | "degraded" | "outage";

type EvidenceReport = {
  system: string;
  user_reported_problem: boolean;
  simulated_status: StatusLevel | null;
  corroboration: string;
  relation: string;
};

type RetrievedDocument = {
  rank: number;
  document_id: string;
  title: string;
  source: string;
  retrieval_score: number | null;
  rerank_score: number | null;
  snippet: string;
};

type SupportResult = {
  outcome: "completed" | "blocked" | "escalated";
  answer: string | Record<string, unknown>;
  execution_path: string[];
  safety: { allowed: boolean; model: string; reason: string };
  system_status: {
    evaluated: boolean;
    simulated: true;
    statuses: Record<SystemName, StatusLevel>;
    evidence: null | {
      reports: EvidenceReport[];
      user_reported_problem: boolean;
      assessment: string;
      relevant_services: Array<{
        system: SystemName;
        configured_status: StatusLevel;
      }>;
    };
    notice: string;
  };
  routing: { destination: string; reason: string; used_fallback: boolean };
  jev: null | {
    decision_available: boolean;
    model: string;
    classified_route: string;
    selected_route: string;
    route_confidence: number | null;
    route_probabilities: Record<string, number>;
    route_confidence_threshold: number;
    route_confidence_threshold_met: boolean | null;
    human_escalation_probability: number | null;
    human_escalation_threshold: number;
    human_escalation_threshold_met: boolean | null;
    explicit_human_request_detected: boolean;
  };
  retrieval: {
    enabled: boolean;
    executed: boolean;
    query: string | null;
    result_count: number;
    supplied_document_count: number;
    documents: RetrievedDocument[];
    context_supplied_to_response: string | null;
  };
  response_generation: {
    inputs: Record<string, unknown>;
    generated_response: string | Record<string, unknown>;
    production: string;
  };
  execution_summary: {
    headline: string;
    facts: Array<{ category: string; text: string }>;
  };
  provenance: {
    user_query: string;
    simulated_statuses: Record<SystemName, StatusLevel>;
    system_evidence_assessment: string;
    jev_route: string;
    retrieved_documents: string[];
    human_escalation_triggered: boolean;
    response_production: string;
  };
  stage_evidence: StageEvidence[];
};

type SampleScenario = {
  label: string;
  description: string;
  query: string;
  faqRetrieval?: boolean;
  status?: Partial<Record<SystemName, StatusLevel>>;
};

const MAX_QUERY_LENGTH = 2000;
const SYSTEMS: Array<{ id: SystemName; label: string }> = [
  { id: "authentication", label: "Authentication" },
  { id: "billing", label: "Billing" },
  { id: "checkout", label: "Checkout" },
  { id: "api", label: "API" },
];
const STATUS_LEVELS: StatusLevel[] = ["operational", "degraded", "outage"];
const DEFAULT_STATUSES: Record<SystemName, StatusLevel> = {
  authentication: "operational",
  billing: "operational",
  checkout: "operational",
  api: "operational",
};
const ACCOUNTING_QUERY =
  "I restarted my accounting workstation and now the monthly report writer won't generate reports. What should I do?";
const CHECKOUT_REPORT =
  "Our checkout system is completely down and nobody can place an order.";
const SAMPLE_SCENARIOS: SampleScenario[] = [
  {
    label: "Ordinary MFA issue",
    description: "Technical support without an explicit handoff request.",
    query: "I cannot log in after enabling MFA.",
  },
  {
    label: "Repeated failure",
    description: "Repeated attempts, but no explicit human request.",
    query:
      "I have reset my password five times, re-enrolled MFA twice, and I still cannot access my account. I have been locked out since yesterday.",
  },
  {
    label: "Critical checkout impact",
    description: "Business impact without an explicit human request.",
    query:
      "Our checkout system has been unavailable for 45 minutes and customers cannot place orders. We are actively losing sales.",
    status: { checkout: "outage" },
  },
  {
    label: "Explicit human request",
    description: "A direct request to connect with a human.",
    query:
      "I have tried the troubleshooting steps several times and this still isn't working. Please connect me to a human.",
  },
  {
    label: "Duplicate billing charge",
    description: "A billing-domain route.",
    query: "I was charged twice for my subscription this month.",
  },
  {
    label: "Checkout report · Operational",
    description: "Contradictory simulated evidence.",
    query: CHECKOUT_REPORT,
    status: { checkout: "operational" },
  },
  {
    label: "Checkout report · Outage",
    description: "Corroborating simulated evidence.",
    query: CHECKOUT_REPORT,
    status: { checkout: "outage" },
  },
  {
    label: "Accounting · RAG off",
    description: "Generic context only.",
    query: ACCOUNTING_QUERY,
    faqRetrieval: false,
  },
  {
    label: "Accounting · RAG on",
    description: "Retrieve the fictional Acme procedure.",
    query: ACCOUNTING_QUERY,
    faqRetrieval: true,
  },
];

function formatLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function formatPercent(value: number | null): string {
  return value === null ? "Not available" : `${Math.round(value * 100)}%`;
}

function formatThresholdResult(value: boolean | null): string {
  if (value === null) return "Unknown";
  return value ? "Threshold met" : "Below threshold";
}

function safeErrorMessage(status: number): string {
  if (status === 422) return "Enter a query between 1 and 2,000 characters.";
  if (status === 429) return "Too many runs. Wait a minute and try again.";
  if (status === 503) return "Configure the backend provider keys before running.";
  return "The support workflow could not be started.";
}

function isSupportResult(value: unknown): value is SupportResult {
  if (typeof value !== "object" || value === null) return false;
  const result = value as SupportResult;
  return (
    typeof result.outcome === "string" &&
    Array.isArray(result.execution_path) &&
    typeof result.safety === "object" &&
    result.safety !== null &&
    typeof result.system_status === "object" &&
    result.system_status !== null &&
    typeof result.routing === "object" &&
    result.routing !== null &&
    typeof result.execution_summary === "object" &&
    result.execution_summary !== null &&
    typeof result.provenance === "object" &&
    result.provenance !== null &&
    Array.isArray(result.stage_evidence)
  );
}

function Answer({ value }: { value: SupportResult["answer"] }) {
  if (typeof value === "string") return <p className="answer-copy">{value}</p>;
  return (
    <dl className="answer-fields">
      {Object.entries(value).map(([key, fieldValue]) => (
        <div key={key}>
          <dt>{formatLabel(key)}</dt>
          <dd>{String(fieldValue)}</dd>
        </div>
      ))}
    </dl>
  );
}

function EvidenceFields({ value }: { value: Record<string, unknown> }) {
  return (
    <dl className="evidence-fields">
      {Object.entries(value).map(([key, fieldValue]) => {
        const isStructured = typeof fieldValue === "object" && fieldValue !== null;
        return (
          <div key={key}>
            <dt>{formatLabel(key)}</dt>
            <dd>
              {isStructured ? (
                <pre>{JSON.stringify(fieldValue, null, 2)}</pre>
              ) : fieldValue === null || fieldValue === undefined ? (
                "Not available"
              ) : (
                String(fieldValue)
              )}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

function StageInspector({ evidence }: { evidence: StageEvidence }) {
  return (
    <section className="stage-inspector" aria-labelledby="stage-inspector-title">
      <div className="inspector-heading">
        <div>
          <span className="kicker">Selected stage contribution</span>
          <h3 id="stage-inspector-title">{evidence.label}</h3>
        </div>
        <code>{evidence.stage_id}</code>
      </div>
      <p>{evidence.summary}</p>
      <div className="inspector-columns">
        <section>
          <h4>Application inputs</h4>
          <EvidenceFields value={evidence.inputs} />
        </section>
        <section>
          <h4>Application outputs</h4>
          <EvidenceFields value={evidence.outputs} />
        </section>
      </div>
    </section>
  );
}

function StatusSimulator({ values, disabled, onChange }: {
  values: Record<SystemName, StatusLevel>;
  disabled: boolean;
  onChange: (system: SystemName, level: StatusLevel) => void;
}) {
  return (
    <section className="simulator-card" aria-labelledby="simulator-title">
      <div className="panel-heading">
        <div>
          <span className="kicker">Simulated demo state</span>
          <h2 id="simulator-title">System signals</h2>
        </div>
        <span className="demo-badge">Not live monitoring</span>
      </div>
      <p className="panel-intro">
        Set evidence independently. The backend passes it to Jev as context, not as a
        forced decision.
      </p>
      <div className="status-grid">
        {SYSTEMS.map((system) => (
          <label className="status-control" key={system.id}>
            <span>{system.label}</span>
            <select
              id={`status-${system.id}`}
              name={`status-${system.id}`}
              value={values[system.id]}
              disabled={disabled}
              onChange={(event) => onChange(system.id, event.target.value as StatusLevel)}
            >
              {STATUS_LEVELS.map((level) => (
                <option value={level} key={level}>
                  {level[0].toUpperCase() + level.slice(1)}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </section>
  );
}

function GraphView({ graph, executions, loading, selectedNodeId, onSelect }: {
  graph: GraphDescription | null;
  executions: Record<string, NodeExecution>;
  loading: boolean;
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
}) {
  const stages = useMemo(() => {
    if (!graph) return [];
    return Array.from(new Set(graph.nodes.map((node) => node.stage))).map((stage) => ({
      stage,
      nodes: graph.nodes.filter((node) => node.stage === stage),
    }));
  }, [graph]);
  const nodeLabels = useMemo(
    () => new Map(graph?.nodes.map((node) => [node.id, node.label]) ?? []),
    [graph],
  );
  const selectedEvidence = selectedNodeId
    ? executions[selectedNodeId]?.evidence
    : undefined;

  return (
    <section className="graph-card" aria-labelledby="graph-title" aria-busy={loading}>
      <div className="graph-heading">
        <div>
          <span className="kicker">Executable architecture</span>
          <h2 id="graph-title">Unified support graph</h2>
          <p>Nodes and edges come from the compiled Python LangGraph.</p>
        </div>
        <div className="legend" aria-label="Execution state legend">
          {(["waiting", "running", "completed", "failed", "disabled"] as const).map((status) => (
            <span key={status} className={`legend-${status}`}>{status}</span>
          ))}
        </div>
      </div>
      {!graph && <div className="graph-loading">Loading backend graph…</div>}
      {graph && (
        <>
          <div className="graph-flow" role="list" aria-label="Support graph nodes">
            {stages.map(({ stage, nodes }) => (
              <div className="graph-stage" key={stage}>
                <div className="stage-nodes">
                  {nodes.map((node) => {
                    const execution = executions[node.id] ?? {
                      status: node.enabled ? "waiting" : "disabled",
                    };
                    return (
                      <article
                        className={`graph-node node-${execution.status} node-${node.kind} ${selectedNodeId === node.id ? "node-selected" : ""}`}
                        key={node.id}
                        role="listitem"
                      >
                        <button
                          className="node-select"
                          type="button"
                          disabled={execution.status !== "completed" || !execution.evidence}
                          aria-pressed={selectedNodeId === node.id}
                          onClick={() => onSelect(node.id)}
                        >
                          <span className="node-topline">
                            <span className="node-state">{execution.status}</span>
                            <span>{node.required ? "required" : "optional"}</span>
                          </span>
                          <strong>{node.label}</strong>
                          <span className="node-description">{node.description}</span>
                          {execution.status === "completed" && execution.evidence && (
                            <span className="inspect-label">Inspect contribution →</span>
                          )}
                        </button>
                      </article>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          {selectedEvidence && <StageInspector evidence={selectedEvidence} />}
          <section className="edge-map" aria-labelledby="edge-map-title">
            <div className="edge-map-heading">
              <span className="kicker">Backend-serialized connections</span>
              <strong id="edge-map-title">Executable flow</strong>
              <small>{graph.edges.length} edges from the compiled graph</small>
            </div>
            <div className="edge-list">
              {graph.edges.map((edge, index) => (
                <div className="graph-edge" key={`${edge.source}-${edge.target}-${index}`}>
                  <span>{nodeLabels.get(edge.source) ?? edge.source}</span>
                  <b aria-label="flows to">→</b>
                  <span>{nodeLabels.get(edge.target) ?? edge.target}</span>
                  {edge.conditional && <i>{edge.branch}</i>}
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </section>
  );
}

function ResultView({ result }: { result: SupportResult }) {
  const reports = result.system_status.evidence?.reports ?? [];
  return (
    <div className="result-content">
      <div className={`outcome outcome-${result.outcome}`}>
        <span>{formatLabel(result.outcome)}</span>
        <small>{formatLabel(result.routing.destination)}</small>
      </div>
      <section className="summary-card">
        <span className="card-kicker">What happened?</span>
        <strong>{result.execution_summary.headline}</strong>
        <ol>
          {result.execution_summary.facts.map((fact) => (
            <li key={fact.category}>
              <span>{formatLabel(fact.category)}</span>
              <p>{fact.text}</p>
            </li>
          ))}
        </ol>
      </section>
      <section className="answer-card">
        <span className="card-kicker">Response</span>
        <Answer value={result.answer} />
      </section>
      <section className="provenance-card">
        <span className="card-kicker">Final result provenance</span>
        <dl>
          <div><dt>User query</dt><dd>{result.provenance.user_query}</dd></div>
          <div><dt>System evidence</dt><dd>{formatLabel(result.provenance.system_evidence_assessment)}</dd></div>
          <div><dt>Jev route</dt><dd>{formatLabel(result.provenance.jev_route)}</dd></div>
          <div><dt>Retrieved documents</dt><dd>{result.provenance.retrieved_documents.join(", ") || "None"}</dd></div>
          <div><dt>Human escalation</dt><dd>{result.provenance.human_escalation_triggered ? "Triggered" : "Not triggered"}</dd></div>
          <div><dt>Response production</dt><dd>{formatLabel(result.provenance.response_production)}</dd></div>
        </dl>
      </section>
      <div className="decision-grid">
        <section className="decision-card">
          <span className="card-kicker">Mandatory safety</span>
          <strong>{result.safety.allowed ? "Allowed" : "Blocked"}</strong>
          <p>{result.safety.reason}</p>
          <small>{result.safety.model}</small>
        </section>
        <section className="decision-card">
          <span className="card-kicker">Routing</span>
          <strong>{formatLabel(result.routing.destination)}</strong>
          <p>{result.routing.reason}</p>
          <RoutingSource routing={result.routing} />
        </section>
      </div>
      <section className="evidence-card">
        <div className="evidence-heading">
          <div>
            <span className="card-kicker">Simulated status evidence</span>
            <strong>
              {result.system_status.evaluated
                ? formatLabel(result.system_status.evidence?.assessment ?? "unknown")
                : "Not reached"}
            </strong>
          </div>
          <span>Demo only</span>
        </div>
        <div className="status-summary">
          {SYSTEMS.map((system) => (
            <span key={system.id} className={`status-${result.system_status.statuses[system.id]}`}>
              {system.label}: {result.system_status.statuses[system.id]}
            </span>
          ))}
        </div>
        {reports.length === 0 ? (
          <p>No component-specific outage report was detected.</p>
        ) : (
          <ul className="evidence-list">
            {reports.map((report, index) => (
              <li key={`${report.system}-${index}`}>
                <strong>{formatLabel(report.system)}</strong>
                <span>User report: problem</span>
                <span>Simulated signal: {report.simulated_status ?? "none"}</span>
                <b>{formatLabel(report.relation)}</b>
              </li>
            ))}
          </ul>
        )}
        <small>{result.system_status.notice}</small>
      </section>
      {result.jev && (
        <section className="jev-card">
          <div className="evidence-heading">
            <div>
              <span className="card-kicker">Jev decision</span>
              <strong>{result.jev.model}</strong>
            </div>
            <span>{result.routing.used_fallback ? "Fallback used" : "Direct route"}</span>
          </div>
          <div className="metric-row">
            <div><span>Route confidence</span><strong>{formatPercent(result.jev.route_confidence)}</strong></div>
            <div><span>Human escalation</span><strong>{formatPercent(result.jev.human_escalation_probability)}</strong></div>
            <div><span>Route threshold · {formatPercent(result.jev.route_confidence_threshold)}</span><strong>{formatThresholdResult(result.jev.route_confidence_threshold_met)}</strong></div>
            <div><span>Escalation threshold · {formatPercent(result.jev.human_escalation_threshold)}</span><strong>{formatThresholdResult(result.jev.human_escalation_threshold_met)}</strong></div>
          </div>
          <div className="jev-observations">
            <span>Classified route: <strong>{formatLabel(result.jev.classified_route)}</strong></span>
            <span>Explicit human request: <strong>{result.jev.explicit_human_request_detected ? "Detected" : "Not detected"}</strong></span>
          </div>
          <div className="probabilities">
            {Object.entries(result.jev.route_probabilities).map(([route, probability]) => (
              <div key={route}>
                <span>{formatLabel(route)}</span>
                <div className="meter"><i style={{ width: `${Math.round(probability * 100)}%` }} /></div>
                <strong>{formatPercent(probability)}</strong>
              </div>
            ))}
          </div>
        </section>
      )}
      <section className="retrieval-summary">
        <span className="card-kicker">Optional demo knowledge RAG</span>
        <strong>
          {result.retrieval.executed
            ? `${result.retrieval.supplied_document_count} document(s) supplied from ${result.retrieval.result_count} result(s)`
            : result.retrieval.enabled
              ? "Enabled, branch not taken"
              : "Disabled"}
        </strong>
        {result.retrieval.executed && (
          <>
            <p>Query: {result.retrieval.query}</p>
            <ol className="retrieved-documents">
              {result.retrieval.documents.map((document) => (
                <li key={document.document_id}>
                  <div><strong>{document.rank}. {document.title}</strong><code>{document.document_id}</code></div>
                  <small>
                    retrieval {document.retrieval_score?.toFixed(3) ?? "n/a"} · rerank {document.rerank_score?.toFixed(3) ?? "n/a"}
                  </small>
                  <p>{document.snippet}</p>
                </li>
              ))}
            </ol>
            <details className="supplied-context">
              <summary>Exact knowledge supplied to response generation</summary>
              <pre>{result.retrieval.context_supplied_to_response}</pre>
            </details>
          </>
        )}
      </section>
      <details className="raw-json">
        <summary>Structured backend result <span>Application state only</span></summary>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </div>
  );
}

export default function Home() {
  const [graph, setGraph] = useState<GraphDescription | null>(null);
  const [executions, setExecutions] = useState<Record<string, NodeExecution>>({});
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [faqRetrieval, setFaqRetrieval] = useState(false);
  const [statuses, setStatuses] = useState<Record<SystemName, StatusLevel>>({
    ...DEFAULT_STATUSES,
  });
  const [query, setQuery] = useState(SAMPLE_SCENARIOS[0].query);
  const [result, setResult] = useState<SupportResult | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function loadGraph() {
      setGraphLoading(true);
      setGraph(null);
      setExecutions({});
      setSelectedNodeId(null);
      setResult(null);
      try {
        const response = await fetch(
          `/backend/support/graph?faq_retrieval=${faqRetrieval}`,
          { signal: controller.signal, headers: { Accept: "application/json" } },
        );
        if (!response.ok) throw new Error("graph request failed");
        const payload: unknown = await response.json();
        if (!isGraphDescription(payload)) throw new Error("invalid graph response");
        setGraph(payload);
        setExecutions(initialNodeExecutions(payload));
        setError(null);
      } catch (requestError) {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError("The backend graph is unavailable. Confirm that FastAPI is running.");
      } finally {
        if (!controller.signal.aborted) setGraphLoading(false);
      }
    }
    void loadGraph();
    return () => controller.abort();
  }, [faqRetrieval]);

  function updateStatus(system: SystemName, level: StatusLevel) {
    setStatuses((current) => ({ ...current, [system]: level }));
    setResult(null);
    setExecutions(graph ? initialNodeExecutions(graph) : {});
    setSelectedNodeId(null);
  }

  function updateQuery(nextQuery: string) {
    setQuery(nextQuery);
    setResult(null);
    setExecutions(graph ? initialNodeExecutions(graph) : {});
    setSelectedNodeId(null);
    setError(null);
  }

  function chooseSample(sample: SampleScenario) {
    setStatuses({ ...DEFAULT_STATUSES, ...sample.status });
    setFaqRetrieval(sample.faqRetrieval ?? false);
    updateQuery(sample.query);
    queryRef.current?.focus();
  }

  function handleEvent(event: ExecutionEvent) {
    setExecutions((current) => applyExecutionEvent(current, event));
    if (event.type === "run_started" && event.evidence) setSelectedNodeId("__start__");
    if (event.type === "node_completed" && event.node_id && event.evidence) {
      setSelectedNodeId(event.node_id);
    }
    if (event.type === "run_completed") {
      if (event.evidence) setSelectedNodeId("__end__");
      if (isSupportResult(event.result)) setResult(event.result);
      else setError("The backend returned an unexpected final result.");
    }
    if (event.type === "run_failed") {
      setError(event.message ?? "The support workflow could not be completed.");
    }
  }

  async function runSupportGraph(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!graphMatchesOptions(graph, faqRetrieval)) {
      setError("The backend graph is unavailable. Confirm that FastAPI is running.");
      return;
    }
    if (!normalizedQuery || normalizedQuery.length > MAX_QUERY_LENGTH) {
      setError("Enter a query between 1 and 2,000 characters.");
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    setExecutions(initialNodeExecutions(graph));
    setSelectedNodeId(null);
    let receivedTerminalEvent = false;
    try {
      const response = await fetch("/backend/support/run", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          query: normalizedQuery,
          options: { faq_retrieval: faqRetrieval },
          simulated_status: statuses,
        }),
      });
      if (!response.ok) {
        setError(safeErrorMessage(response.status));
        return;
      }
      if (!response.body) {
        setError("This browser could not read the execution stream.");
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const parsed = parseSseBuffer(done ? `${buffer}\n\n` : buffer);
        buffer = parsed.remainder;
        parsed.events.forEach((executionEvent) => {
          handleEvent(executionEvent);
          if (isTerminalExecutionEvent(executionEvent)) receivedTerminalEvent = true;
        });
        if (done) {
          if (!receivedTerminalEvent) {
            setError("The backend stream ended before the support run completed.");
          }
          break;
        }
      }
    } catch {
      if (!receivedTerminalEvent) {
        setError("The backend stream was interrupted. Check the service and try again.");
      }
    } finally {
      setRunning(false);
    }
  }

  return (
    <main>
      <header className="hero">
        <div>
          <div className="eyebrow"><span /> Observable support prototype</div>
          <h1>Inspect every support decision.</h1>
        </div>
        <p>
          Follow structured application evidence through safety, simulated system
          state, Jev routing, optional demo knowledge, and response or escalation.
        </p>
      </header>
      <StatusSimulator values={statuses} disabled={running} onChange={updateStatus} />
      <div className="graph-toolbar">
        <div>
          <span className="kicker">Graph controls</span>
          <strong>Optional stages are enforced by the backend graph.</strong>
        </div>
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={faqRetrieval}
            disabled={running}
            onChange={(event) => setFaqRetrieval(event.target.checked)}
          />
          <span>Enable demo knowledge RAG</span>
        </label>
        <span className="locked-control">OpenAI Moderation · always required</span>
      </div>
      <GraphView
        graph={graph}
        executions={executions}
        loading={graphLoading}
        selectedNodeId={selectedNodeId}
        onSelect={setSelectedNodeId}
      />
      <section className="execution-workspace" aria-label="Support graph execution">
        <form className="query-panel" onSubmit={runSupportGraph}>
          <div className="panel-heading">
            <div><span className="kicker">Run the system</span><h2>Support query</h2></div>
            <span className="query-count">{query.length.toLocaleString()} / 2,000</span>
          </div>
          <label className="sr-only" htmlFor="query">Support query</label>
          <textarea
            ref={queryRef}
            id="query"
            value={query}
            maxLength={MAX_QUERY_LENGTH}
            rows={5}
            disabled={running}
            onChange={(event) => updateQuery(event.target.value)}
          />
          <div className="samples" aria-label="Sample support scenarios">
            {SAMPLE_SCENARIOS.map((sample) => (
              <button type="button" key={sample.label} disabled={running} onClick={() => chooseSample(sample)}>
                <strong>{sample.label}</strong>
                <span>{sample.description}</span>
              </button>
            ))}
          </div>
          <div className="run-row">
            <button
              className="run-button"
              type="submit"
              disabled={
                running ||
                graphLoading ||
                !graphMatchesOptions(graph, faqRetrieval) ||
                !query.trim()
              }
            >
              {running ? <><span className="spinner" /> Streaming execution</> : <>Run support graph <span>→</span></>}
            </button>
            <p>Only application events and structured outputs are shown.</p>
          </div>
        </form>
        <aside className="result-panel" aria-live="polite" aria-busy={running}>
          <div className="panel-heading"><div><span className="kicker">Backend result</span><h2>Decision evidence</h2></div></div>
          {error && <div className="error-state" role="alert"><strong>Unable to continue</strong><p>{error}</p></div>}
          {!error && !result && (
            <div className={running ? "empty-state loading" : "empty-state"}>
              <div className="empty-orbit"><span /></div>
              <strong>{running ? "Following the live graph" : "Ready for a query"}</strong>
              <p>{running ? "Node states update from LangGraph events." : "Run the graph to inspect safety, evidence, routing, and response."}</p>
            </div>
          )}
          {result && <ResultView result={result} />}
        </aside>
      </section>
      <footer>
        <span>Next.js localhost:3000</span>
        <span>FastAPI 127.0.0.1:8000</span>
        <span>Python · LangGraph · OpenAI · Jev</span>
      </footer>
    </main>
  );
}
