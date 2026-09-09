import { NextResponse } from "next/server";

export const runtime = "nodejs";

function backendBaseUrl() {
  return (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function GET() {
  try {
    const response = await fetch(`${backendBaseUrl()}/api/v1/detector/status`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store"
    });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json({ enabled: false, rules: [] }, { status: 200 });
  }
}
