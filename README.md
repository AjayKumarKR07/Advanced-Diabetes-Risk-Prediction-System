# Smart Apartment System — Auth scaffold

This repository contains an initial scaffold for JWT-based authentication using Next.js App Router + Prisma (Postgres).

Quick start

1. Copy `.env.example` to `.env` and fill values
2. Install dependencies: `npm install` (install `prisma`, `@prisma/client`, `bcrypt`, `jsonwebtoken`, `jest`, `ts-jest`, `@testing-library/react` etc.)
3. Generate Prisma client: `npm run prisma:generate`
4. Run migrations: `npm run prisma:migrate`
5. Start dev server: `npm run dev`

Files added

- `database/schema.prisma` — Prisma schema with `User` model
- `frontend/app/api/auth/*` — `register`, `login`, `logout`, `me` routes
- `frontend/app/api/auth/[...nextauth]/route.ts` — optional NextAuth credentials provider example
- `frontend/components/forms/*` — `RegisterForm` and `LoginForm` components
- Test scaffold and GitHub Actions CI workflow

Security notes

- Use a strong `JWT_SECRET` and keep it out of source control.
- In production set `NODE_ENV=production` so cookies are secure.
- The repo also includes an optional **NextAuth** credentials provider example (`frontend/app/api/auth/[...nextauth]/route.ts`) that uses the Prisma adapter. To enable NextAuth fully:
  1. Install: `npm install next-auth @next-auth/prisma-adapter`
  2. Set `NEXTAUTH_URL` and `NEXTAUTH_SECRET` in your `.env` (example in `.env.example`).
  3. Consider configuring OAuth providers (Google/GitHub) in `authOptions.providers` for social sign-in.

Notes

- The project uses **NextAuth** as the primary authentication strategy (Credentials + Google OAuth). The repo used to include a custom JWT flow; those endpoints are now deprecated in favor of NextAuth. For signup, use the `/api/auth/register` endpoint to create a user (the login flow should use NextAuth's `signIn`), and use the protected example at `GET /api/profile` to see how to access the session on the server.
