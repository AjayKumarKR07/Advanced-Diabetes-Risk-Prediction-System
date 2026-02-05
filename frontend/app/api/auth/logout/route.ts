import { NextResponse } from 'next/server';

export async function POST() {
  // Deprecated: NextAuth handles sign-out now.
  return NextResponse.json({ error: 'This endpoint is deprecated. Use NextAuth sign-out at /api/auth/signout' }, { status: 410 });
}
