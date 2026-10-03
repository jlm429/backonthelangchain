"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { RoutingSource } from "./routing-source";

type ExampleMetadata = {
  id: string;
  display_name: string;
  description: string;
  default_prompt: string;
  sample_prompts: string[];
};

type ExampleResult = {
  outcome: "completed" | "blocked" | "escalated";
  answer: string | Record<string, unknown>;
  safety: {
    allowed: boolean;
    model: string;
    reason: string;
  };
  routing: {
    destination: string;
    reason: string;
    used_fallback: boolean;
  };
  jev: null | {
    model: string | null;
    route_confidence: number | null;
    route_probabilities: Record<string, number>;
    human_escalation_probability: number | null;
  };
};

type RunResponse = {
  example_id: string;
  result: ExampleResult;
};

const MAX_QUERY_LENGTH = 2000;

function isExampleCatalog(value: unknown): value is ExampleMetadata[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as ExampleMetadata).id === "string" &&
        typeof (item as ExampleMetadata).display_name === "string" &&
        typeof (item as ExampleMetadata).description === "string" &&
        typeof (item as ExampleMetadata).default_prompt === "string" &&
        Array.isArray((item as ExampleMetadata).sample_prompts),
    )
  );
}

function isRunResponse(value: unknown): value is RunResponse {
  if (typeof value !== "object" || value === null) return false;
  const response = value as RunResponse;
  const result = response.result;
  return (
    typeof response.example_id === "string" &&
    typeof result === "object" &&
    result !== null &&
    typeof result.outcome === "string" &&
    typeof result.safety === "object" &&
    result.safety !== null &&
    typeof result.routing === "object" &&
    result.routing !== null
  );
}

function safeErrorMessage(status: number): string {
  if (status === 422) return "Enter a query between 1 and 2,000 characters.";
  if (status === 429) return "You have run several examples. Wait a minute and try again.";
  if (status === 503) return "This example is not configured on the server yet.";
  return "The example could not be completed. Please try again.";
}

function formatLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function formatPercent(value: number | null): string {
  return value === null ? "Not available" : `${Math.round(value * 100)}%`;
}

function Answer({ value }: { value: ExampleResult["answer"] }) {
  if (typeof value === "string") {
    return <p className="answer-copy">{value}</p>;
  }

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

export default function Home() {
  const [examples, setExamples] = useState<ExampleMetadata[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RunResponse | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryRef = useRef<HTMLTextAreaElement>(null);

  const selectedExample = useMemo(
    () => examples.find((example) => example.id === selectedId) ?? null,
    [examples, selectedId],
  );

  useEffect(() => {
    const controller = new AbortController();

    async function loadCatalog() {
      try {
        const response = await fetch("/backend/examples", {
          signal: controller.signal,
          headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error("catalog request failed");
        const payload: unknown = await response.json();
        if (!isExampleCatalog(payload) || payload.length === 0) {
          throw new Error("catalog response was invalid");
        }
        setExamples(payload);
        setSelectedId(payload[0].id);
        setQuery(payload[0].default_prompt);
      } catch (requestError) {
        if (requestError instanceof DOMException && requestError.name === "AbortError") {
          return;
        }
        setError("Examples are unavailable. Confirm that the backend is running.");
      } finally {
        setCatalogLoading(false);
      }
    }

    void loadCatalog();
    return () => controller.abort();
  }, []);

  function selectExample(exampleId: string) {
    const nextExample = examples.find((example) => example.id === exampleId);
    setSelectedId(exampleId);
    setResult(null);
    setError(null);
    if (nextExample) setQuery(nextExample.default_prompt);
  }

  function chooseSample(sample: string) {
    setQuery(sample);
    setError(null);
    queryRef.current?.focus();
  }

  async function runExample(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!selectedExample || !normalizedQuery || normalizedQuery.length > MAX_QUERY_LENGTH) {
      setError("Enter a query between 1 and 2,000 characters.");
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(
        `/backend/examples/${encodeURIComponent(selectedExample.id)}/run`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({ query: normalizedQuery }),
        },
      );
      if (!response.ok) {
        setError(safeErrorMessage(response.status));
        return;
      }
      const payload: unknown = await response.json();
      if (!isRunResponse(payload)) {
        setError("The server returned an unexpected result. Please try again.");
        return;
      }
      setResult(payload);
    } catch {
      setError("The backend could not be reached. Check the service and try again.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <main>
      <header className="hero">
        <div className="eyebrow"><span /> Interactive LangGraph workshop</div>
        <h1>backonthelangchain</h1>
        <p>
          Explore production-minded LLM workflows with visible safety, routing,
          and structured decisions.
        </p>
      </header>

      <section className="workspace" aria-label="Example runner">
        <div className="runner-panel">
          <div className="section-heading">
            <span className="step">01</span>
            <div>
              <h2>Choose a workflow</h2>
              <p>The catalog comes directly from the Python backend.</p>
            </div>
          </div>

          <label className="field-label" htmlFor="example-select">Example</label>
          <div className="select-wrap">
            <select
              id="example-select"
              value={selectedId}
              disabled={catalogLoading || examples.length === 0 || running}
              onChange={(event) => selectExample(event.target.value)}
            >
              {catalogLoading && <option>Loading examples…</option>}
              {examples.map((example) => (
                <option value={example.id} key={example.id}>
                  {example.display_name}
                </option>
              ))}
            </select>
          </div>

          {selectedExample && (
            <p className="example-description">{selectedExample.description}</p>
          )}

          <form onSubmit={runExample}>
            <div className="section-heading query-heading">
              <span className="step">02</span>
              <div>
                <h2>Ask a question</h2>
                <p>Provider calls stay in Python and run only after submission.</p>
              </div>
            </div>

            <div className="textarea-label-row">
              <label className="field-label" htmlFor="query">Query</label>
              <span>{query.length.toLocaleString()} / {MAX_QUERY_LENGTH.toLocaleString()}</span>
            </div>
            <textarea
              ref={queryRef}
              id="query"
              value={query}
              maxLength={MAX_QUERY_LENGTH}
              rows={6}
              disabled={running || !selectedExample}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Describe the support issue…"
            />

            {selectedExample && (
              <div className="samples" aria-label="Sample queries">
                <span className="samples-label">Try a sample</span>
                <div className="sample-list">
                  {selectedExample.sample_prompts.map((sample) => (
                    <button
                      type="button"
                      className="sample-button"
                      key={sample}
                      disabled={running}
                      onClick={() => chooseSample(sample)}
                    >
                      {sample}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="run-row">
              <button
                type="submit"
                className="run-button"
                disabled={running || !selectedExample || !query.trim()}
              >
                {running ? <><span className="spinner" /> Running workflow…</> : <>Run example <span>→</span></>}
              </button>
              <p>Moderation runs before any routing decision.</p>
            </div>
          </form>
        </div>

        <aside className="result-panel" aria-live="polite" aria-busy={running}>
          <div className="section-heading">
            <span className="step">03</span>
            <div>
              <h2>Inspect the result</h2>
              <p>Normalized decisions, not raw provider responses.</p>
            </div>
          </div>

          {error && <div className="error-state" role="alert"><strong>Unable to run</strong><p>{error}</p></div>}

          {!error && !result && (
            <div className={running ? "empty-state loading" : "empty-state"}>
              <div className="empty-orbit"><span /></div>
              <strong>{running ? "Working through the graph" : "Ready when you are"}</strong>
              <p>{running ? "Safety, escalation, and routing are being evaluated." : "Run a query to see each structured decision."}</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div className={`outcome outcome-${result.result.outcome}`}>
                <span>{formatLabel(result.result.outcome)}</span>
                <small>{formatLabel(result.result.routing.destination)}</small>
              </div>

              <section className="answer-card">
                <h3>Response</h3>
                <Answer value={result.result.answer} />
              </section>

              <div className="decision-grid">
                <section className="decision-card">
                  <span className="card-kicker">Safety gate</span>
                  <strong>{result.result.safety.allowed ? "Allowed" : "Blocked"}</strong>
                  <p>{result.result.safety.reason}</p>
                  <small>{result.result.safety.model}</small>
                </section>
                <section className="decision-card">
                  <span className="card-kicker">Routing</span>
                  <strong>{formatLabel(result.result.routing.destination)}</strong>
                  <p>{result.result.routing.reason}</p>
                  <RoutingSource routing={result.result.routing} />
                </section>
              </div>

              {result.result.jev && (
                <section className="jev-card">
                  <div>
                    <span className="card-kicker">Jev decision</span>
                    <strong>{result.result.jev.model ?? "Model unavailable"}</strong>
                  </div>
                  <div className="metric-row">
                    <div><span>Route confidence</span><strong>{formatPercent(result.result.jev.route_confidence)}</strong></div>
                    <div><span>Human escalation</span><strong>{formatPercent(result.result.jev.human_escalation_probability)}</strong></div>
                  </div>
                  <div className="probabilities">
                    {Object.entries(result.result.jev.route_probabilities).map(([route, probability]) => (
                      <div key={route}>
                        <span>{formatLabel(route)}</span>
                        <div className="meter"><i style={{ width: `${Math.round(probability * 100)}%` }} /></div>
                        <strong>{formatPercent(probability)}</strong>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <details className="raw-json">
                <summary>Raw JSON <span>Normalized backend result</span></summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </aside>
      </section>

      <footer><span>Python + FastAPI</span><span>Next.js + TypeScript</span><span>LangGraph + Jev</span></footer>
    </main>
  );
}
