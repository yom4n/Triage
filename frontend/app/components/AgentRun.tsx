"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Circle,
  Clock3,
  ExternalLink,
  GitPullRequest,
  Loader2,
  Play,
  XCircle
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Environment } from "../../types/triage";

type SampleIncident = {
  label: string;
  title: string;
  environment: Environment;
  stack_trace: string;
  description: string;
};

type StreamFieldValue = string | number | boolean | null | undefined;

type StreamNodeEvent = {
  node: string;
  ts: string;
  duration_ms: number;
  fields: Record<string, StreamFieldValue>;
};

type StreamTerminalEvent = {
  event: "done" | "error";
  ticket_id?: string;
  status?: string;
  detail?: string;
};

type TimelineNode = {
  id: string;
  label: string;
  helper: string;
};

const SANDBOX_MAX_ATTEMPTS = 3;

const sampleIncidents: SampleIncident[] = [
  {
    label: "Checkout 504 - DB pool exhausted (todotest)",
    environment: "production",
    title: "Checkout 504s under load - DB connection pool exhausted",
    stack_trace: `Error: timeout exceeded when trying to connect
    at Timeout._onTimeout (/app/node_modules/pg-pool/index.js:200:27)
    at async checkout (/app/services/gateway/src/db.js:12:20)
Context: POST /checkout returns HTTP 504 under load. The pg Pool max is configured in services/gateway/src/db.js. Under 30 concurrent requests, 26 return 504 with "checkout database pool acquisition timed out".`,
    description: "Pool sizing lives in services/gateway/src/db.js. Started after a traffic ramp."
  },
  {
    label: "NullPointerException on order confirmation",
    environment: "production",
    title: "Null pointer crash on order confirmation screen",
    stack_trace: `Exception in thread "main" java.lang.NullPointerException: Cannot invoke "Order.getShippingAddress()" because "order" is null
	at com.flodata.orders.OrderConfirmationService.render(OrderConfirmationService.java:142)
	at com.flodata.orders.OrderController.confirm(OrderController.java:57)`,
    description: "Happens when an order is cancelled between page load and confirmation click."
  },
  {
    label: "Redis connection refused across pods",
    environment: "production",
    title: "Redis connection refused across all app pods",
    stack_trace: `redis.exceptions.ConnectionError: Error 111 connecting to redis-cache:6379. Connection refused.
  File "cache/client.py", line 23, in get
    return self._client.get(key)`,
    description: "All pods affected simultaneously after a deploy."
  },
  {
    label: "Staging: KeyError on webhook payload",
    environment: "staging",
    title: "KeyError when processing webhook payloads missing optional field",
    stack_trace: `KeyError: 'customer_id'
  File "webhooks/handler.py", line 34, in process_payload
    customer_id = payload['customer_id']`,
    description: "Partner changed their webhook schema to make customer_id optional."
  }
];

const nodeMeta: Record<string, TimelineNode> = {
  log_inspector: {
    id: "log_inspector",
    label: "Log Inspector",
    helper: "Root error and affected file"
  },
  rag_lookup: {
    id: "rag_lookup",
    label: "RAG Lookup",
    helper: "Historical matches"
  },
  triage_router: {
    id: "triage_router",
    label: "Triage Router",
    helper: "Severity and summary"
  },
  generate_fix: {
    id: "generate_fix",
    label: "Generate Fix",
    helper: "Candidate patch"
  },
  verify_fix: {
    id: "verify_fix",
    label: "Verify Fix",
    helper: "Sandbox test result"
  },
  open_pr: {
    id: "open_pr",
    label: "Open PR",
    helper: "Branch and pull request"
  },
  fallback_human_escalation: {
    id: "fallback_human_escalation",
    label: "Human Escalation",
    helper: "Manual handoff"
  },
  fix_escalation: {
    id: "fix_escalation",
    label: "Fix Escalation",
    helper: "Patch needs review"
  }
};

function buildTimeline(events: Record<string, StreamNodeEvent>): TimelineNode[] {
  const fixedPrefix = [nodeMeta.log_inspector, nodeMeta.rag_lookup, nodeMeta.triage_router];

  if (events.fallback_human_escalation) {
    return [...fixedPrefix, nodeMeta.fallback_human_escalation];
  }

  const generatedDiffLines = events.generate_fix?.fields.diff_line_count;
  const verificationStatus = events.verify_fix?.fields.fix_verification_status;
  const generateWentToEscalation = events.generate_fix && generatedDiffLines === 0;
  const verifyWentToEscalation =
    typeof verificationStatus === "string" &&
    (verificationStatus === "FAILED_MAX_ATTEMPTS" || verificationStatus === "SKIPPED_UNTESTABLE");

  if (events.fix_escalation || generateWentToEscalation || verifyWentToEscalation) {
    return [
      ...fixedPrefix,
      nodeMeta.generate_fix,
      ...(events.verify_fix ? [nodeMeta.verify_fix] : []),
      nodeMeta.fix_escalation
    ];
  }

  return [...fixedPrefix, nodeMeta.generate_fix, nodeMeta.verify_fix, nodeMeta.open_pr];
}

export default function AgentRun() {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [nodeEvents, setNodeEvents] = useState<Record<string, StreamNodeEvent>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<{ ticket_id: string; status: string } | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  const selectedIncident = sampleIncidents[selectedIndex];
  const timeline = useMemo(() => buildTimeline(nodeEvents), [nodeEvents]);
  const activeNodeId = isRunning ? timeline.find((node) => !nodeEvents[node.id])?.id ?? null : null;
  const prUrl = typeof nodeEvents.open_pr?.fields.fix_pr_url === "string" ? nodeEvents.open_pr.fields.fix_pr_url : null;

  useEffect(() => {
    return () => {
      sourceRef.current?.close();
    };
  }, []);

  function runTriage() {
    sourceRef.current?.close();
    setNodeEvents({});
    setSummary(null);
    setError(null);
    setIsRunning(true);

    const params = new URLSearchParams({
      title: selectedIncident.title,
      stack_trace: selectedIncident.stack_trace,
      environment: selectedIncident.environment
    });
    if (selectedIncident.description.trim()) {
      params.set("description", selectedIncident.description);
    }

    const source = new EventSource(`/api/triage/stream?${params.toString()}`);
    sourceRef.current = source;

    source.onmessage = (event) => {
      let payload: StreamNodeEvent | StreamTerminalEvent;
      try {
        payload = JSON.parse(event.data) as StreamNodeEvent | StreamTerminalEvent;
      } catch {
        setError("The live stream returned an unreadable event.");
        source.close();
        setIsRunning(false);
        return;
      }

      if ("event" in payload) {
        if (payload.event === "done") {
          setSummary({
            ticket_id: payload.ticket_id ?? "unknown",
            status: payload.status ?? "COMPLETED"
          });
          source.close();
          setIsRunning(false);
          return;
        }

        setError(payload.detail ?? "The live triage stream failed.");
        source.close();
        setIsRunning(false);
        return;
      }

      setNodeEvents((current) => ({ ...current, [payload.node]: payload }));
    };

    source.onerror = () => {
      setError("The live triage stream closed before completion.");
      source.close();
      setIsRunning(false);
    };
  }

  return (
    <section className="panel agent-run-panel">
      <div className="panel-heading agent-run-heading">
        <div>
          <h2>Live Agent Run</h2>
          <p>{selectedIncident.title}</p>
        </div>
        <span className={`badge environment-${selectedIncident.environment}`}>
          {selectedIncident.environment}
        </span>
      </div>

      <div className="agent-run-controls">
        <label>
          <span>Incident</span>
          <select
            value={selectedIndex}
            disabled={isRunning}
            onChange={(event) => setSelectedIndex(Number(event.target.value))}
          >
            {sampleIncidents.map((incident, index) => (
              <option key={incident.label} value={index}>
                {incident.label}
              </option>
            ))}
          </select>
        </label>

        <button className="primary-button agent-run-button" type="button" disabled={isRunning} onClick={runTriage}>
          {isRunning ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          {isRunning ? "Running" : "Run"}
        </button>
      </div>

      {error ? (
        <div className="error-box agent-run-error" role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>Stream error</strong>
            <p>{error}</p>
          </div>
        </div>
      ) : null}

      {summary ? (
        <div className="agent-run-summary">
          <CheckCircle2 size={18} />
          <span>
            ticket <code>{summary.ticket_id}</code> finished as <strong>{summary.status}</strong>
          </span>
          {prUrl ? (
            <a href={prUrl} target="_blank" rel="noreferrer">
              <ExternalLink size={14} />
              View PR
            </a>
          ) : null}
        </div>
      ) : null}

      <ol className="agent-timeline">
        {timeline.map((node) => {
          const event = nodeEvents[node.id];
          const state = event ? "done" : activeNodeId === node.id ? "running" : "pending";
          return (
            <li className={`agent-node ${state}`} key={node.id}>
              <div className="agent-node-marker" aria-hidden="true">
                {state === "done" ? <CheckCircle2 size={18} /> : state === "running" ? <Clock3 size={18} /> : <Circle size={18} />}
              </div>

              <div className="agent-node-card">
                <div className="agent-node-header">
                  <div>
                    <h3>{node.label}</h3>
                    <p>{node.helper}</p>
                  </div>
                  <span className={`badge node-state-${state}`}>{state}</span>
                </div>

                {event ? (
                  <div className="agent-node-body">
                    <span className="agent-duration">{formatDuration(event.duration_ms)}</span>
                    {node.id === "verify_fix" ? <VerificationStatus fields={event.fields} /> : null}
                    {node.id === "open_pr" && prUrl ? (
                      <a className="agent-pr-link" href={prUrl} target="_blank" rel="noreferrer">
                        <GitPullRequest size={15} />
                        View PR
                      </a>
                    ) : null}
                    <FieldList fields={event.fields} />
                  </div>
                ) : null}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function FieldList({ fields }: { fields: Record<string, StreamFieldValue> }) {
  const entries = Object.entries(fields).filter(([, value]) => value !== null && value !== undefined && value !== "");

  if (!entries.length) {
    return <div className="empty-state agent-empty-fields">No fields emitted</div>;
  }

  return (
    <dl className="agent-fields">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{formatFieldKey(key)}</dt>
          <dd>{formatFieldValue(key, value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function VerificationStatus({ fields }: { fields: Record<string, StreamFieldValue> }) {
  const rawStatus = typeof fields.fix_verification_status === "string" ? fields.fix_verification_status : "NOT_ATTEMPTED";
  const attempts = typeof fields.fix_verification_attempts === "number" ? fields.fix_verification_attempts : 0;
  const tone = rawStatus === "PASSED" ? "pass" : rawStatus.startsWith("FAILED") ? "fail" : rawStatus.startsWith("SKIPPED") ? "skip" : "idle";
  const Icon = tone === "pass" ? CheckCircle2 : tone === "fail" ? XCircle : AlertTriangle;

  return (
    <div className="verification-row">
      <span>attempt {attempts}/{SANDBOX_MAX_ATTEMPTS}</span>
      <span className={`verify-pill ${tone}`}>
        <Icon size={13} />
        {tone === "pass" ? "PASSED" : tone === "fail" ? "FAILED" : tone === "skip" ? "SKIPPED" : rawStatus}
      </span>
    </div>
  );
}

function formatFieldKey(key: string) {
  return key
    .replace(/^fix_/, "")
    .replace(/^triage_/, "")
    .replace(/^log_inspector_/, "")
    .replace(/_/g, " ");
}

function formatFieldValue(key: string, value: StreamFieldValue) {
  if (typeof value === "number") {
    if (key.includes("confidence") || key.includes("similarity")) {
      return `${Math.round(value * 100)}%`;
    }
    return value.toString();
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  return String(value);
}

function formatDuration(durationMs: number) {
  if (durationMs >= 1000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  }
  return `${Math.max(0, Math.round(durationMs))}ms`;
}
