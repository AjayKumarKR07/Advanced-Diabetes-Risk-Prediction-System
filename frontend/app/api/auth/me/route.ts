import { NextResponse } from 'next/server';

export async function GET() {
  // Deprecated: NextAuth exposes session via getServerSession on the server and client hooks.
  return NextResponse.json({ error: 'This endpoint is deprecated. Use NextAuth session APIs' }, { status: 410 });
}
