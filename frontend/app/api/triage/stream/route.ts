import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET(request: NextRequest) {
  const search = request.nextUrl.search;

  try {
    const response = await fetch(`${backendBaseUrl()}/api/v1/triage/stream${search}`, {
      method: "GET",
      headers: {
        Accept: "text/event-stream"
      },
      cache: "no-store"
    });

    if (!response.body) {
      return NextResponse.json(
        {
          title: "Backend Stream Empty",
          detail: "The FastAPI backend returned an empty stream response.",
          status: 502,
          trace_id: "unavailable"
        },
        { status: 502 }
      );
    }

    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive"
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