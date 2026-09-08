import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function POST(request: NextRequest) {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      {
        title: "Invalid Request",
        detail: "Request body must be valid JSON.",
        status: 400,
        trace_id: "unavailable"
      },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${backendBaseUrl()}/api/v1/triage`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, application/problem+json"
      },
      body: JSON.stringify(payload),
      cache: "no-store"
    });

    const text = await response.text();
    const contentType = response.headers.get("content-type") ?? "";
    let body: unknown = text ? { detail: text } : {};

    if (contentType.includes("json") && text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = { detail: text };
      }
    }

    return NextResponse.json(body, {
      status: response.status,
      headers: {
        "Content-Type": contentType.includes("problem+json")
          ? "application/problem+json"
          : "application/json"
      }
    });
  } catch {
    return NextResponse.json(
      {
        title: "Backend Unreachable",
        detail: `Could not reach the FastAPI backend at ${backendBaseUrl()}.`,
        status: 503,
        trace_id: "unavailable"
      },
      { status: 503 }
    );
  }
}
