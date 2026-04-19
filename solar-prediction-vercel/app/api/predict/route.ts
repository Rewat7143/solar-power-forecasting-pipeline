import { NextRequest, NextResponse } from "next/server";
import { runTimestampPrediction } from "@/lib/predict";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const date = String(body?.date ?? "").trim();
    const time = String(body?.time ?? "").trim();

    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !/^\d{2}:\d{2}$/.test(time)) {
      return NextResponse.json(
        { ok: false, error: "Invalid input. Use date=YYYY-MM-DD and time=HH:mm." },
        { status: 400 }
      );
    }

    const result = await runTimestampPrediction(date, time);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
