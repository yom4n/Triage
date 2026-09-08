import { NextRequest } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

function sseError(detail: string) {
  return new Response(`data: ${JSON.stringify({ event: "error", detail })}\n\n`, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no"
    }
  });
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
      return sseError("The backend returned an empty stream.");
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!response.ok || !contentType.includes("text/event-stream")) {
      const text = await response.text();
      let detail = text || `The backend stream returned HTTP ${response.status}.`;
      if (contentType.includes("json") && text) {
        try {
          const parsed = JSON.parse(text) as { detail?: string };
          detail = parsed.detail ?? detail;
        } catch {
          detail = text;
        }
      }
      return sseError(detail);
    }

    const headers = new Headers();
    headers.set("Content-Type", contentType);
    headers.set("Cache-Control", "no-cache");
    headers.set("X-Accel-Buffering", "no");

    return new Response(response.body, {
      status: response.status,
      headers
    });
  } catch {
    return sseError(`Could not reach the FastAPI backend at ${backendBaseUrl()}.`);
  }
}
