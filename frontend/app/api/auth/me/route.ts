import { NextResponse } from 'next/server';
import { verifyToken } from '../_utils';
import { prisma } from '../../../../lib/prisma';

export async function GET(req: Request) {
  try {
    const cookie = req.headers.get('cookie') || '';
    const match = cookie.split('; ').find((c) => c.trim().startsWith('token='));
    if (!match) return NextResponse.json({ user: null }, { status: 200 });
    const token = match.split('=')[1];
    const payload: any = verifyToken(token as string);
    const user = await prisma.user.findUnique({ where: { id: Number(payload.sub) } });
    if (!user) return NextResponse.json({ user: null }, { status: 200 });
    return NextResponse.json({ user: { id: user.id, email: user.email, name: user.name } });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ user: null }, { status: 200 });
  }
}
