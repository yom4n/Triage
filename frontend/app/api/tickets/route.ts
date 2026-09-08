import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET(request: NextRequest) {
  const search = request.nextUrl.search;

  try {
    const response = await fetch(`${backendBaseUrl()}/api/v1/tickets${search}`, {
      method: "GET",
      headers: {
        Accept: "application/json, application/problem+json"
      },
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
