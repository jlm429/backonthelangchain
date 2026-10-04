"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { RoutingSource } from "./routing-source";
import {
  ExecutionEvent,
  GraphDescription,
  NodeExecution,
  applyExecutionEvent,
  initialNodeExecutions,
  isGraphDescription,
  parseSseBuffer,
} from "./support-graph";

type SystemName = "authentication" | "billing" | "checkout" | "api";
type StatusLevel = "operational" | "degraded" | "outage";

type EvidenceReport = {
  system: string;
  user_reported_problem: boolean;
  simulated_status: StatusLevel | null;
  corroboration: string;
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
    };
    notice: string;
  };
  routing: { destination: string; reason: string; used_fallback: boolean };
  jev: null | {
    model: string | null;
    route_confidence: number | null;
    route_probabilities: Record<string, number>;
    human_escalation_probability: number | null;
  };
  retrieval: {
    enabled: boolean;
    executed: boolean;
    sources: Array<{ title?: string; source?: string }>;
  };
};

const MAX_QUERY_LENGTH = 2000;
const SYSTEMS: Array<{ id: SystemName; label: string }> = [
  { id: "authentication", label: "Authentication" },
  { id: "billing", label: "Billing" },
  { id: "checkout", label: "Checkout" },
  { id: "api", label: "API" },
];
const STATUS_LEVELS: StatusLevel[] = ["operational", "degraded", "outage"];
const SAMPLE_QUERIES = [
  "I cannot log in after enabling MFA.",
  "Checkout is down and blocking our customers.",
  "I was charged twice for my subscription.",
  "I have tried five times. Connect me to a human.",
];

function formatLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function formatPercent(value: number | null): string {
  return value === null ? "Not available" : `${Math.round(value * 100)}%`;
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
    result.routing !== null
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

function GraphView({ graph, executions, loading }: {
  graph: GraphDescription | null;
  executions: Record<string, NodeExecution>;
  loading: boolean;
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
                        className={`graph-node node-${execution.status} node-${node.kind}`}
                        key={node.id}
                        role="listitem"
                      >
                        <div className="node-topline">
                          <span className="node-state">{execution.status}</span>
                          <span>{node.required ? "required" : "optional"}</span>
                        </div>
                        <strong>{node.label}</strong>
                        <p>{node.description}</p>
                        {execution.output && Object.keys(execution.output).length > 0 && (
                          <details>
                            <summary>Stage output</summary>
                            <pre>{JSON.stringify(execution.output, null, 2)}</pre>
                          </details>
                        )}
                      </article>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
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
      <section className="answer-card">
        <span className="card-kicker">Response</span>
        <Answer value={result.answer} />
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
            <strong>{result.system_status.evaluated ? "Evaluated" : "Not reached"}</strong>
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
                <b>{formatLabel(report.corroboration)}</b>
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
              <strong>{result.jev.model ?? "Model unavailable"}</strong>
            </div>
            <span>{result.routing.used_fallback ? "Fallback used" : "Direct route"}</span>
          </div>
          <div className="metric-row">
            <div><span>Route confidence</span><strong>{formatPercent(result.jev.route_confidence)}</strong></div>
            <div><span>Human escalation</span><strong>{formatPercent(result.jev.human_escalation_probability)}</strong></div>
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
        <span className="card-kicker">Optional FAQ retrieval</span>
        <strong>
          {result.retrieval.executed
            ? `${result.retrieval.sources.length} source(s) retrieved`
            : result.retrieval.enabled
              ? "Enabled, branch not taken"
              : "Disabled"}
        </strong>
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
  const [faqRetrieval, setFaqRetrieval] = useState(false);
  const [statuses, setStatuses] = useState<Record<SystemName, StatusLevel>>({
    authentication: "operational",
    billing: "operational",
    checkout: "operational",
    api: "operational",
  });
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [result, setResult] = useState<SupportResult | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function loadGraph() {
      setGraphLoading(true);
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
        setResult(null);
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
  }

  function chooseSample(sample: string) {
    setQuery(sample);
    setError(null);
    queryRef.current?.focus();
  }

  function handleEvent(event: ExecutionEvent) {
    setExecutions((current) => applyExecutionEvent(current, event));
    if (event.type === "run_completed") {
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
    if (!graph || !normalizedQuery || normalizedQuery.length > MAX_QUERY_LENGTH) {
      setError("Enter a query between 1 and 2,000 characters.");
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    setExecutions(initialNodeExecutions(graph));
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
        parsed.events.forEach(handleEvent);
        if (done) break;
      }
    } catch {
      setError("The backend stream was interrupted. Check the service and try again.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <main>
      <header className="hero">
        <div>
          <div className="eyebrow"><span /> Live LangGraph support lab</div>
          <h1>See the support system think in states.</h1>
        </div>
        <p>
          One end-to-end workflow for safe intake, simulated operational evidence,
          Jev routing, optional retrieval, and response or escalation.
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
          <span>Enable Tier 1 FAQ retrieval</span>
        </label>
        <span className="locked-control">OpenAI Moderation · always required</span>
      </div>
      <GraphView graph={graph} executions={executions} loading={graphLoading} />
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
            onChange={(event) => setQuery(event.target.value)}
          />
          <div className="samples" aria-label="Sample support queries">
            {SAMPLE_QUERIES.map((sample) => (
              <button type="button" key={sample} disabled={running} onClick={() => chooseSample(sample)}>{sample}</button>
            ))}
          </div>
          <div className="run-row">
            <button className="run-button" type="submit" disabled={running || graphLoading || !query.trim()}>
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
