"use client";

import {
  CheckCircle2,
  Circle,
  ExternalLink,
  Loader2,
  Play,
  XCircle
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Environment } from "../types/triage";

type SampleIncident = {
  label: string;
  title: string;
  environment: Environment;
  stack_trace: string;
  description: string;
};

type TimelineState = "pending" | "running" | "done";

type NodeStreamMessage = {
  node: string;
  ts: string;
  seq: number;
  fields: Record<string, unknown>;
};

type DoneStreamMessage = {
  event: "done";
  ticket_id: string;
  status: string;
  fix_pr_url: string | null;
};

type ErrorStreamMessage = {
  event: "error";
  detail: string;
};

type RunSummary = {
  ticket_id: string;
  status: string;
  fix_pr_url: string | null;
};

const sampleIncidents: SampleIncident[] = [
  {
    label: "Checkout 504 - DB pool exhausted (todotest)",
    environment: "production",
    title: "Checkout 504s under load - DB connection pool exhausted",
    stack_trace:
      "Error: timeout exceeded when trying to connect\n    at Timeout._onTimeout (/app/node_modules/pg-pool/index.js:200:27)\n    at async checkout (/app/services/gateway/src/db.js:12:20)\nContext: POST /checkout returns HTTP 504 under load. The pg Pool max is configured in services/gateway/src/db.js. Under 30 concurrent requests, 26 return 504 with \"checkout database pool acquisition timed out\".",
    description: "Pool sizing lives in services/gateway/src/db.js. Started after a traffic ramp."
  },
  {
    label: "NullPointerException on order confirmation",
    environment: "production",
    title: "Null pointer crash on order confirmation screen",
    stack_trace:
      "Exception in thread \"main\" java.lang.NullPointerException: Cannot invoke \"Order.getShippingAddress()\" because \"order\" is null\n\tat com.flodata.orders.OrderConfirmationService.render(OrderConfirmationService.java:142)\n\tat com.flodata.orders.OrderController.confirm(OrderController.java:57)",
    description: "Happens when an order is cancelled between page load and confirmation click."
  },
  {
    label: "Redis connection refused across pods",
    environment: "production",
    title: "Redis connection refused across all app pods",
    stack_trace:
      "redis.exceptions.ConnectionError: Error 111 connecting to redis-cache:6379. Connection refused.\n  File \"cache/client.py\", line 23, in get\n    return self._client.get(key)",
    description: "All pods affected simultaneously after a deploy."
  },
  {
    label: "Staging: KeyError on webhook payload",
    environment: "staging",
    title: "KeyError when processing webhook payloads missing optional field",
    stack_trace:
      "KeyError: 'customer_id'\n  File \"webhooks/handler.py\", line 34, in process_payload\n    customer_id = payload['customer_id']",
    description: "Partner changed their webhook schema to make customer_id optional."
  }
];

const baseNodeOrder = ["log_inspector", "rag_lookup", "triage_router"];
const defaultNodeOrder = [...baseNodeOrder, "generate_fix", "verify_fix", "open_pr"];

const nodeLabels: Record<string, string> = {
  log_inspector: "Log Inspector",
  rag_lookup: "RAG Lookup",
  triage_router: "Triage Router",
  fallback_human_escalation: "Human Escalation",
  generate_fix: "Generate Fix",
  verify_fix: "Verify Fix",
  open_pr: "Open PR",
  fix_escalation: "Fix Escalation"
};

const fieldLabels: Record<string, string> = {
  extracted_error: "error",
  affected_file: "file",
  affected_line: "line",
  confidence: "confidence",
  used_llm: "used LLM",
  hits: "hits",
  top_similarity: "top similarity",
  severity: "severity",
  summary: "summary",
  status: "status",
  escalation_reason: "reason",
  affected_path: "path",
  llm_confidence: "LLM confidence",
  diff_lines: "diff lines",
  skipped_reason: "skipped",
  attempt: "attempt",
  attempts: "attempts",
  test_command: "test command",
  verified: "verified",
  pr_url: "PR",
  branch: "branch",
  keys: "keys"
};

export default function AgentRun() {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [nodeEvents, setNodeEvents] = useState<Record<string, NodeStreamMessage>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  const selectedSample = sampleIncidents[selectedIndex] ?? sampleIncidents[0];
  const visibleNodes = useMemo(() => buildVisibleNodeOrder(nodeEvents), [nodeEvents]);

  useEffect(() => {
    return () => {
      sourceRef.current?.close();
    };
  }, []);

  function closeSource(source: EventSource) {
    source.close();
    if (sourceRef.current === source) {
      sourceRef.current = null;
    }
    setIsRunning(false);
  }

  function runTriage() {
    sourceRef.current?.close();
    setNodeEvents({});
    setSummary(null);
    setError(null);
    setIsRunning(true);

    const source = new EventSource(buildStreamUrl(selectedSample));
    sourceRef.current = source;
    let terminalReceived = false;

    source.onmessage = (event) => {
      let message: unknown;
      try {
        message = JSON.parse(event.data);
      } catch {
        terminalReceived = true;
        setError("The stream returned an invalid event payload.");
        closeSource(source);
        return;
      }

      if (isDoneMessage(message)) {
        terminalReceived = true;
        setSummary({
          ticket_id: message.ticket_id,
          status: message.status,
          fix_pr_url: message.fix_pr_url
        });
        closeSource(source);
        return;
      }

      if (isErrorMessage(message)) {
        terminalReceived = true;
        setError(message.detail || "The triage stream ended with an error.");
        closeSource(source);
        return;
      }

      if (isNodeMessage(message)) {
        setNodeEvents((current) => ({ ...current, [message.node]: message }));
      }
    };

    source.onerror = () => {
      if (terminalReceived) {
        return;
      }
      terminalReceived = true;
      setError("The triage stream connection failed.");
      closeSource(source);
    };
  }

  return (
    <section className="panel agent-run-panel">
      <div className="panel-heading">
        <div>
          <h2>Live Agent Run</h2>
          <p>{isRunning ? "Run active" : summary ? "Run complete" : "Ready"}</p>
        </div>
      </div>

      <div className="agent-run-controls">
        <label className="agent-sample-field">
          <span>Sample Incident</span>
          <select
            value={selectedIndex}
            disabled={isRunning}
            onChange={(event) => setSelectedIndex(Number(event.target.value))}
          >
            {sampleIncidents.map((sample, index) => (
              <option key={sample.label} value={index}>
                {sample.label}
              </option>
            ))}
          </select>
        </label>

        <button className="primary-button agent-run-button" type="button" disabled={isRunning} onClick={runTriage}>
          {isRunning ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          {isRunning ? "Running" : "Run triage"}
        </button>
      </div>

      <div className="agent-sample-summary">
        <strong>{selectedSample.title}</strong>
        <span>{selectedSample.environment}</span>
      </div>

      {summary ? (
        <div className="agent-summary" role="status">
          <span>
            Ticket <code>{shortId(summary.ticket_id)}</code> {summary.status}
          </span>
          {summary.fix_pr_url ? (
            <a href={summary.fix_pr_url} target="_blank" rel="noreferrer">
              View PR
              <ExternalLink size={14} />
            </a>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <div className="error-box agent-run-error" role="alert">
          <XCircle size={18} />
          <div>
            <strong>Stream error</strong>
            <p>{error}</p>
          </div>
        </div>
      ) : null}

      <div className="agent-timeline" aria-live="polite">
        {visibleNodes.map((node, index) => {
          const event = nodeEvents[node];
          const state = getNodeState(node, index, visibleNodes, nodeEvents, isRunning);
          return <TimelineCard key={node} node={node} event={event} state={state} />;
        })}
      </div>
    </section>
  );
}

function TimelineCard({
  node,
  event,
  state
}: {
  node: string;
  event: NodeStreamMessage | undefined;
  state: TimelineState;
}) {
  return (
    <article className={`agent-node-card ${state}`}>
      <div className="agent-node-status" aria-hidden="true">
        {state === "done" ? (
          <CheckCircle2 size={18} />
        ) : state === "running" ? (
          <Loader2 className="spin" size={18} />
        ) : (
          <Circle size={18} />
        )}
      </div>
      <div className="agent-node-content">
        <div className="agent-node-head">
          <h3>{nodeLabels[node] ?? node}</h3>
          <span className={`badge agent-state-${state}`}>{state}</span>
        </div>
        {event ? (
          <span className="agent-node-meta">
            seq {event.seq} - {formatTime(event.ts)}
          </span>
        ) : null}
        {state === "done" && event ? (
          renderNodeFields(node, event.fields)
        ) : (
          <p className="agent-node-placeholder">{state === "running" ? "Running" : "Pending"}</p>
        )}
      </div>
    </article>
  );
}

function renderNodeFields(node: string, fields: Record<string, unknown>) {
  if (node === "verify_fix") {
    const status = typeof fields.status === "string" ? fields.status : "NOT_ATTEMPTED";
    const attempts = formatFieldValue("attempts", fields.attempts ?? 0);
    const remainingFields = Object.fromEntries(
      Object.entries(fields).filter(([key]) => key !== "status" && key !== "attempts")
    );

    return (
      <>
        <div className="verification-row">
          <span>attempt {attempts}</span>
          <span className={`verification-pill ${verificationTone(status)}`}>{verificationLabel(status)}</span>
        </div>
        <FieldList fields={remainingFields} />
      </>
    );
  }

  if (node === "open_pr" && typeof fields.pr_url === "string" && fields.pr_url) {
    const remainingFields = Object.fromEntries(Object.entries(fields).filter(([key]) => key !== "pr_url"));
    return (
      <>
        <a className="agent-pr-link" href={fields.pr_url} target="_blank" rel="noreferrer">
          View PR
          <ExternalLink size={14} />
        </a>
        <FieldList fields={remainingFields} />
      </>
    );
  }

  return <FieldList fields={fields} />;
}

function FieldList({ fields }: { fields: Record<string, unknown> }) {
  const entries = Object.entries(fields).filter(([, value]) => value !== undefined);

  if (!entries.length) {
    return <p className="agent-node-placeholder">No fields</p>;
  }

  return (
    <dl className="agent-field-list">
      {entries.map(([key, value]) => (
        <div className="agent-field-row" key={key}>
          <dt>{fieldLabels[key] ?? humanizeKey(key)}</dt>
          <dd>{formatFieldValue(key, value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function buildVisibleNodeOrder(events: Record<string, NodeStreamMessage>) {
  if (events.fallback_human_escalation) {
    return [...baseNodeOrder, "fallback_human_escalation"];
  }

  if (events.fix_escalation || Boolean(events.generate_fix?.fields.skipped_reason)) {
    const order = [...baseNodeOrder, "generate_fix"];
    if (events.verify_fix) {
      order.push("verify_fix");
    }
    order.push("fix_escalation");
    return order;
  }

  return defaultNodeOrder;
}

function getNodeState(
  node: string,
  index: number,
  visibleNodes: string[],
  events: Record<string, NodeStreamMessage>,
  isRunning: boolean
): TimelineState {
  if (events[node]) {
    return "done";
  }

  const firstPendingIndex = visibleNodes.findIndex((candidate) => !events[candidate]);
  if (isRunning && index === firstPendingIndex) {
    return "running";
  }

  return "pending";
}

function buildStreamUrl(sample: SampleIncident) {
  const params = new URLSearchParams();
  params.set("title", sample.title);
  params.set("stack_trace", sample.stack_trace);
  params.set("environment", sample.environment);
  params.set("description", sample.description);
  return `/api/triage/stream?${params.toString()}`;
}

function isDoneMessage(message: unknown): message is DoneStreamMessage {
  return isRecord(message) && message.event === "done" && typeof message.ticket_id === "string";
}

function isErrorMessage(message: unknown): message is ErrorStreamMessage {
  return isRecord(message) && message.event === "error";
}

function isNodeMessage(message: unknown): message is NodeStreamMessage {
  return (
    isRecord(message) &&
    typeof message.node === "string" &&
    typeof message.ts === "string" &&
    typeof message.seq === "number" &&
    isRecord(message.fields)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatFieldValue(key: string, value: unknown) {
  if (value === null || value === "") {
    return "none";
  }
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => String(item)).join(", ") : "none";
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (typeof value === "number") {
    if (key.includes("confidence")) {
      return `${Math.round(value * 100)}%`;
    }
    if (key.includes("similarity")) {
      return value.toFixed(2);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  return String(value);
}

function verificationTone(status: string) {
  if (status === "PASSED") {
    return "passed";
  }
  if (status.startsWith("FAILED")) {
    return "failed";
  }
  return "skipped";
}

function verificationLabel(status: string) {
  if (status === "PASSED") {
    return "PASSED";
  }
  if (status.startsWith("FAILED")) {
    return "FAILED";
  }
  return "SKIPPED";
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

function shortId(id: string) {
  return id.length > 12 ? `${id.slice(0, 8)}...${id.slice(-4)}` : id;
}

function humanizeKey(key: string) {
  return key.replace(/_/g, " ");
}