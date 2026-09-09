import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET() {
  try {
    const response = await fetch(`${backendBaseUrl()}/api/v1/evals/scorecard`, {
      headers: {
        Accept: "application/json"
      },
      cache: "no-store"
    });

    const text = await response.text();
    let body: unknown = text ? { detail: text } : {};
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("json") && text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = { detail: text };
      }
    }

    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json({ available: false }, { status: 200 });
  }
}
