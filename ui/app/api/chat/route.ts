import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const body = await req.json();

  return NextResponse.json({
    id: 1,
    sender: "ai",
    name: "AI Assistant",
    content: `Simulated response: ${body.content}`,
    timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  });
}
