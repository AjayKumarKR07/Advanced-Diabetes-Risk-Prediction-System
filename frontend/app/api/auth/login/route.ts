import { NextResponse } from 'next/server';

export async function POST() {
  // Deprecated: NextAuth handles authentication now.
  return NextResponse.json({ error: 'This endpoint is deprecated. Use NextAuth sign-in at /api/auth/signin' }, { status: 410 });
}
