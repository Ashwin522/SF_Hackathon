// app/api/court-image/route.ts
import { NextResponse } from "next/server";

export async function GET() {
  // Static image for now
  const imageUrl = "/basketball_court_grid.png"; // must be in public folder
  return NextResponse.json({ url: imageUrl });
}
