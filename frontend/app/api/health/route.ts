import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET() {
  try {
    const response = await fetch(`${backendBaseUrl()}/health`, {
      cache: "no-store"
    });

    if (!response.ok) {
      return NextResponse.json(
        {
          status: "offline",
          detail: `Backend health check returned HTTP ${response.status}.`
        },
        { status: 503 }
      );
    }

    const body = await response.json();
    return NextResponse.json({
      status: body.status === "ok" ? "online" : "offline"
    });
  } catch {
    return NextResponse.json(
      {
        status: "offline",
        detail: `Could not reach the FastAPI backend at ${backendBaseUrl()}.`
      },
      { status: 503 }
    );
  }
}
