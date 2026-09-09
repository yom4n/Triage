export type Environment = "development" | "staging" | "production";

export type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export type TriageStatus = "COMPLETED" | "ESCALATED_TO_HUMAN";
export type TicketSource = "human" | "crash" | "metric" | string;

export type TriagePayload = {
  title: string;
  stack_trace: string;
  environment: Environment;
  description?: string;
  reporter_email?: string;
};

export type TriageResponse = {
  ticket_id: string;
  title: string;
  environment: Environment;
  source: TicketSource;
  stack_trace: string;
  extracted_error: string;
  affected_file: string | null;
  affected_line: number | null;
  severity: Severity;
  summary: string;
  resolution_steps: string[];
  confidence: number;
  similar_tickets_considered: number;
  status: TriageStatus;
  escalation_reason: string | null;
  fix_attempted: boolean;
  fix_skipped_reason: string | null;
  fix_diff: string | null;
  fix_pr_url: string | null;
  fix_branch_name: string | null;
  fix_verified: boolean;
  fix_verification_status: string;
  fix_verification_attempts: number;
  fix_test_command: string | null;
  fix_test_output_tail: string | null;
  created_at: string;
};

export type ProblemResponse = {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  instance?: string;
  trace_id?: string;
};

export type IncidentRecord = {
  id: string;
  submitted_at: string;
  request: TriagePayload;
  response: TriageResponse;
  git_diff?: string;
};
