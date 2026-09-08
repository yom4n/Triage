"use client";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  FileCode2,
  GitPullRequest,
  Loader2,
  RefreshCcw,
  Send,
  Server,
  ShieldAlert
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import AgentRun from "./components/AgentRun";
import type {
  Environment,
  IncidentRecord,
  ProblemResponse,
  Severity,
  TriagePayload,
  TriageResponse,
  TriageStatus
} from "./types/triage";

const sampleTrace = `Traceback (most recent call last):
  File "checkout.py", line 91, in charge_card
    cursor.execute(query, params)
psycopg2.OperationalError: connection to server at "db-primary" port 5432 failed: timeout expired`;

const emptyPayload: TriagePayload = {
  title: "",
  environment: "production",
  stack_trace: "",
  description: "",
  reporter_email: ""
};

const statusLabels: Record<TriageStatus, string> = {
  COMPLETED: "Completed",
  ESCALATED_TO_HUMAN: "Escalated"
};

const severityLabels: Record<Severity, string> = {
  CRITICAL: "Critical",
  HIGH: "High",
  MEDIUM: "Medium",
  LOW: "Low"
};

type HealthState = "checking" | "online" | "offline";
type FilterState = "ALL" | TriageStatus;
type ActiveTab = "live" | "dashboard";

export default function Home() {
  const [form, setForm] = useState<TriagePayload>(emptyPayload);
  const [incidents, setIncidents] = useState<IncidentRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterState>("ALL");
  const [health, setHealth] = useState<HealthState>("checking");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<ProblemResponse | null>(null);
  const [isLoadingTickets, setIsLoadingTickets] = useState(false);
  const [ticketsError, setTicketsError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("live");

  useEffect(() => {
    checkHealth();
    loadTickets();
  }, []);

  async function loadTickets() {
    setIsLoadingTickets(true);
    setTicketsError(null);
    try {
      const response = await fetch("/api/tickets?limit=100", { cache: "no-store" });
      const body = await response.json();
      if (!response.ok) {
        setTicketsError((body as ProblemResponse).detail ?? "Could not load tickets.");
        return;
      }
      const fetched = (body as TriageResponse[]).map((ticket) => ticketToIncident(ticket));
      setIncidents(fetched);
      setSelectedId((current) => current ?? fetched[0]?.id ?? null);
    } catch {
      setTicketsError("Could not reach the backend to load tickets.");
    } finally {
      setIsLoadingTickets(false);
    }
  }

  const filteredIncidents = useMemo(() => {
    if (filter === "ALL") {
      return incidents;
    }
    return incidents.filter((incident) => incident.response.status === filter);
  }, [filter, incidents]);

  const selectedIncident = useMemo(() => {
    if (!incidents.length) {
      return null;
    }
    return (
      incidents.find((incident) => incident.id === selectedId) ??
      filteredIncidents[0] ??
      incidents[0]
    );
  }, [filteredIncidents, incidents, selectedId]);

  const stats = useMemo(() => {
    const completed = incidents.filter((incident) => incident.response.status === "COMPLETED").length;
    const escalated = incidents.filter((incident) => incident.response.status === "ESCALATED_TO_HUMAN").length;
    const averageConfidence = incidents.length
      ? incidents.reduce((sum, incident) => sum + incident.response.confidence, 0) / incidents.length
      : 0;

    return {
      total: incidents.length,
      completed,
      escalated,
      averageConfidence
    };
  }, [incidents]);

  async function checkHealth() {
    setHealth("checking");
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      setHealth(response.ok ? "online" : "offline");
    } catch {
      setHealth("offline");
    }
  }

  function updateField<K extends keyof TriagePayload>(field: K, value: TriagePayload[K]) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function fillSample() {
    setForm({
      ...emptyPayload,
      title: "Checkout fails intermittently under load",
      environment: "production",
      stack_trace: sampleTrace,
      description: "Started after last night's deploy."
    });
    setError(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);

    const payload = compactPayload(form);

    try {
      const response = await fetch("/api/triage", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
      const body = await response.json();

      if (!response.ok) {
        setError(body as ProblemResponse);
        return;
      }

      const result = body as TriageResponse;
      const record = ticketToIncident(result, payload);

      setIncidents((current) => [record, ...current.filter((item) => item.id !== record.id)]);
      setSelectedId(record.id);
      setForm({ ...emptyPayload, environment: form.environment });
    } catch {
      setError({
        title: "Request Failed",
        detail: "The frontend could not complete the triage request.",
        status: 500,
        trace_id: "unavailable"
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function ticketToIncident(response: TriageResponse, request?: TriagePayload): IncidentRecord {
    return {
      id: response.ticket_id,
      submitted_at: response.created_at,
      request: request ?? {
        title: response.title,
        environment: response.environment,
        stack_trace: response.stack_trace
      },
      response,
      git_diff: response.fix_diff ?? undefined
    };
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <ShieldAlert size={20} />
          </div>
          <div>
            <h1>Incident Triage</h1>
            <p>FDE production support console</p>
          </div>
        </div>

        <div className={`backend-health ${health}`}>
          <Server size={16} />
          <span>{health === "checking" ? "Checking" : health === "online" ? "Backend online" : "Backend offline"}</span>
          <button className="icon-button" type="button" title="Refresh backend health" onClick={checkHealth}>
            <RefreshCcw size={16} />
          </button>
        </div>
      </header>
      <div className="view-tabs" role="tablist" aria-label="Primary view">
        <button
          className={activeTab === "live" ? "active" : ""}
          type="button"
          role="tab"
          aria-selected={activeTab === "live"}
          onClick={() => setActiveTab("live")}
        >
          Live Run
        </button>
        <button
          className={activeTab === "dashboard" ? "active" : ""}
          type="button"
          role="tab"
          aria-selected={activeTab === "dashboard"}
          onClick={() => setActiveTab("dashboard")}
        >
          Dashboard
        </button>
      </div>

      {activeTab === "live" ? (
        <section className="live-run-shell">
          <AgentRun />
        </section>
      ) : (
        <section className="workspace">
          <div className="left-column">
            <IncidentForm
              form={form}
              isSubmitting={isSubmitting}
              error={error}
              onSubmit={handleSubmit}
              onFieldChange={updateField}
              onFillSample={fillSample}
            />
          </div>

          <div className="right-column">
            <Dashboard
              incidents={filteredIncidents}
              allCount={incidents.length}
              selectedId={selectedIncident?.id ?? null}
              filter={filter}
              stats={stats}
              isLoading={isLoadingTickets}
              loadError={ticketsError}
              onFilterChange={setFilter}
              onSelect={setSelectedId}
              onRefresh={loadTickets}
            />

            <TicketDetail incident={selectedIncident} gitDiff={selectedIncident?.git_diff} />
          </div>
        </section>
      )}
    </main>
  );
}

function IncidentForm({
  form,
  isSubmitting,
  error,
  onSubmit,
  onFieldChange,
  onFillSample
}: {
  form: TriagePayload;
  isSubmitting: boolean;
  error: ProblemResponse | null;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onFieldChange: <K extends keyof TriagePayload>(field: K, value: TriagePayload[K]) => void;
  onFillSample: () => void;
}) {
  return (
    <section className="panel form-panel">
      <div className="panel-heading">
        <div>
          <h2>New Incident</h2>
          <p>Submit raw logs for automated triage</p>
        </div>
        <button className="secondary-button" type="button" onClick={onFillSample}>
          <FileCode2 size={16} />
          Sample
        </button>
      </div>

      <form className="triage-form" onSubmit={onSubmit}>
        <label>
          <span>Title</span>
          <input
            value={form.title}
            minLength={5}
            maxLength={200}
            required
            onChange={(event) => onFieldChange("title", event.target.value)}
            placeholder="Checkout fails intermittently under load"
          />
        </label>

        <div className="form-row">
          <label>
            <span>Environment</span>
            <select
              value={form.environment}
              onChange={(event) => onFieldChange("environment", event.target.value as Environment)}
            >
              <option value="production">Production</option>
              <option value="staging">Staging</option>
              <option value="development">Development</option>
            </select>
          </label>

          <label>
            <span>Reporter</span>
            <input
              type="email"
              value={form.reporter_email ?? ""}
              onChange={(event) => onFieldChange("reporter_email", event.target.value)}
              placeholder="ops@example.com"
            />
          </label>
        </div>

        <label>
          <span>Stack Trace</span>
          <textarea
            value={form.stack_trace}
            minLength={30}
            maxLength={8000}
            required
            onChange={(event) => onFieldChange("stack_trace", event.target.value)}
            placeholder="Paste the exception, stack trace, or error log..."
          />
        </label>

        <label>
          <span>Description</span>
          <textarea
            className="description-input"
            value={form.description ?? ""}
            maxLength={2000}
            onChange={(event) => onFieldChange("description", event.target.value)}
            placeholder="Observed impact, recent deploys, affected workflow..."
          />
        </label>

        {error ? (
          <div className="error-box" role="alert">
            <AlertTriangle size={18} />
            <div>
              <strong>{error.title ?? "Request error"}</strong>
              <p>{error.detail ?? "The request could not be processed."}</p>
              {error.trace_id ? <code>trace_id: {error.trace_id}</code> : null}
            </div>
          </div>
        ) : null}

        <button className="primary-button" type="submit" disabled={isSubmitting}>
          {isSubmitting ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          {isSubmitting ? "Triaging" : "Run Triage"}
        </button>
      </form>
    </section>
  );
}

function Dashboard({
  incidents,
  allCount,
  selectedId,
  filter,
  stats,
  isLoading,
  loadError,
  onFilterChange,
  onSelect,
  onRefresh
}: {
  incidents: IncidentRecord[];
  allCount: number;
  selectedId: string | null;
  filter: FilterState;
  stats: {
    total: number;
    completed: number;
    escalated: number;
    averageConfidence: number;
  };
  isLoading: boolean;
  loadError: string | null;
  onFilterChange: (filter: FilterState) => void;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}) {
  return (
    <section className="panel dashboard-panel">
      <div className="panel-heading">
        <div>
          <h2>Incident Dashboard</h2>
          <p>{allCount} ticket{allCount === 1 ? "" : "s"} from the database</p>
        </div>
        <button className="icon-button" type="button" title="Refresh tickets" onClick={onRefresh} disabled={isLoading}>
          <RefreshCcw size={16} className={isLoading ? "spin" : ""} />
        </button>
      </div>

      {loadError ? (
        <div className="error-box" role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>Could not load tickets</strong>
            <p>{loadError}</p>
          </div>
        </div>
      ) : null}

      <div className="stats-grid">
        <Stat icon={<ClipboardList size={18} />} label="Total" value={stats.total.toString()} />
        <Stat icon={<CheckCircle2 size={18} />} label="Completed" value={stats.completed.toString()} />
        <Stat icon={<AlertTriangle size={18} />} label="Escalated" value={stats.escalated.toString()} />
        <Stat icon={<Activity size={18} />} label="Avg Conf." value={`${Math.round(stats.averageConfidence * 100)}%`} />
      </div>

      <div className="segment-control" aria-label="Incident filter">
        <button className={filter === "ALL" ? "active" : ""} type="button" onClick={() => onFilterChange("ALL")}>
          All
        </button>
        <button
          className={filter === "COMPLETED" ? "active" : ""}
          type="button"
          onClick={() => onFilterChange("COMPLETED")}
        >
          Completed
        </button>
        <button
          className={filter === "ESCALATED_TO_HUMAN" ? "active" : ""}
          type="button"
          onClick={() => onFilterChange("ESCALATED_TO_HUMAN")}
        >
          Escalated
        </button>
      </div>

      <div className="incident-table" role="table" aria-label="Incidents">
        <div className="incident-row table-head" role="row">
          <span>Ticket</span>
          <span>Status</span>
          <span>Severity</span>
          <span>Confidence</span>
          <span />
        </div>

          {incidents.length ? (
            incidents.map((incident) => (
              <button
                className={`incident-row ${selectedId === incident.id ? "selected" : ""}`}
                key={incident.id}
                type="button"
                role="row"
                onClick={() => onSelect(incident.id)}
              >
                <span className="ticket-cell">
                  <strong>{incident.response.title}</strong>
                  <small>{formatDate(incident.response.created_at)}</small>
                </span>
                <StatusBadge status={incident.response.status} />
                <SeverityBadge severity={incident.response.severity} />
                <span>{Math.round(incident.response.confidence * 100)}%</span>
                <ChevronRight size={16} />
              </button>
            ))
        ) : (
          <div className="empty-state">No incidents</div>
        )}
      </div>
    </section>
  );
}

function TicketDetail({
  incident,
  gitDiff
}: {
  incident: IncidentRecord | null;
  gitDiff?: string;
}) {
  if (!incident) {
    return (
      <section className="panel detail-panel empty-detail">
        <ClipboardList size={28} />
        <h2>No ticket selected</h2>
      </section>
    );
  }

  const response = incident.response;
  const confidencePercent = Math.round(response.confidence * 100);

  return (
    <section className="panel detail-panel">
      <div className="detail-header">
        <div>
          <div className="detail-meta">
            <StatusBadge status={response.status} />
            <SeverityBadge severity={response.severity} />
            <span>{response.environment}</span>
          </div>
          <h2>{response.title}</h2>
          <p>{response.summary}</p>
        </div>
        <div className="confidence-meter" aria-label={`Confidence ${confidencePercent}%`}>
          <strong>{confidencePercent}%</strong>
          <span>confidence</span>
          <div>
            <i style={{ width: `${confidencePercent}%` }} />
          </div>
        </div>
      </div>

      <div className="diagnostic-grid">
        <InfoTile label="Ticket ID" value={shortId(response.ticket_id)} />
        <InfoTile label="Root Error" value={response.extracted_error} />
        <InfoTile label="Affected File" value={formatAffectedLocation(response)} />
        <InfoTile label="Vector Matches" value={response.similar_tickets_considered.toString()} />
      </div>

      {response.escalation_reason ? (
        <div className="notice-box">
          <AlertTriangle size={18} />
          <p>{response.escalation_reason}</p>
        </div>
      ) : null}

      <div className="detail-columns">
        <section className="detail-section">
          <h3>Resolution Steps</h3>
          <ol className="steps-list">
            {response.resolution_steps.map((step, index) => (
              <li key={`${step}-${index}`}>{step}</li>
            ))}
          </ol>
        </section>

        <section className="detail-section">
          <h3>Submitted Trace</h3>
          <pre className="trace-block">{incident.request.stack_trace}</pre>
        </section>
      </div>

      <section className="detail-section diff-section">
        <div className="diff-heading">
          <GitPullRequest size={18} />
          <h3>Proposed Fix</h3>
          {response.fix_pr_url ? (
            <a
              className="secondary-button"
              href={response.fix_pr_url}
              target="_blank"
              rel="noreferrer"
            >
              View Pull Request
            </a>
          ) : null}
        </div>
        {response.fix_branch_name ? (
          <p className="fix-branch-name">
            branch: <code>{response.fix_branch_name}</code>
          </p>
        ) : null}
        {!response.fix_pr_url && response.fix_skipped_reason ? (
          <div className="notice-box">
            <AlertTriangle size={18} />
            <p>{response.fix_skipped_reason}</p>
          </div>
        ) : null}
        {gitDiff ? <pre className="trace-block">{gitDiff}</pre> : <div className="empty-state">No diff attached</div>}
      </section>
    </section>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="stat-tile">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusBadge({ status }: { status: TriageStatus }) {
  return <span className={`badge status-${status.toLowerCase()}`}>{statusLabels[status]}</span>;
}

function SeverityBadge({ severity }: { severity: Severity }) {
  return <span className={`badge severity-${severity.toLowerCase()}`}>{severityLabels[severity]}</span>;
}

function compactPayload(payload: TriagePayload): TriagePayload {
  return {
    title: payload.title,
    environment: payload.environment,
    stack_trace: payload.stack_trace,
    ...(payload.description?.trim() ? { description: payload.description.trim() } : {}),
    ...(payload.reporter_email?.trim() ? { reporter_email: payload.reporter_email.trim() } : {})
  };
}

function shortId(id: string) {
  return id.length > 12 ? `${id.slice(0, 8)}...${id.slice(-4)}` : id;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function formatAffectedLocation(response: TriageResponse) {
  if (!response.affected_file) {
    return "Not detected";
  }
  return response.affected_line ? `${response.affected_file}:${response.affected_line}` : response.affected_file;
}
