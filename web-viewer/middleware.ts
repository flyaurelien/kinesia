import { NextResponse, type NextRequest } from "next/server";

const DEFAULT_ALLOWED_ORIGINS = [
  "http://localhost:3000",
  "http://127.0.0.1:3000",
  "http://localhost:3001",
  "http://127.0.0.1:3001",
];

function configuredOrigins(): string[] {
  return (process.env.KINESIA_ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((origin) => origin.trim().replace(/\/+$/, ""))
    .filter(Boolean);
}

function originMatchesAllowed(requestOrigin: string, allowedOrigin: string): boolean {
  if (requestOrigin === allowedOrigin) {
    return true;
  }
  const wildcard = allowedOrigin.match(/^([a-z][a-z0-9+.-]*):\/\/\*\.([^/:]+(?:\.[^/:]+)+)$/i);
  if (!wildcard) {
    return false;
  }
  try {
    const originUrl = new URL(requestOrigin);
    return originUrl.protocol === `${wildcard[1].toLowerCase()}:` &&
      originUrl.hostname.toLowerCase().endsWith(`.${wildcard[2].toLowerCase()}`);
  } catch {
    return false;
  }
}

function corsOrigin(requestOrigin: string | null): string | null {
  if (!requestOrigin) {
    return null;
  }
  const allowed = configuredOrigins();
  if (allowed.includes("*")) {
    return "*";
  }
  const candidates = allowed.length > 0 ? allowed : DEFAULT_ALLOWED_ORIGINS;
  const normalizedOrigin = requestOrigin.replace(/\/+$/, "");
  return candidates.some((candidate) => originMatchesAllowed(normalizedOrigin, candidate)) ? requestOrigin : null;
}

function applyCorsHeaders(response: NextResponse, origin: string): NextResponse {
  response.headers.set("access-control-allow-origin", origin);
  response.headers.set("access-control-allow-methods", "GET,POST,DELETE,OPTIONS");
  response.headers.set("access-control-allow-headers", "Content-Type,Range,Authorization");
  response.headers.set("access-control-expose-headers", "Accept-Ranges,Content-Length,Content-Range,Content-Type");
  response.headers.set("access-control-max-age", "86400");
  response.headers.append("vary", "Origin");
  return response;
}

export function middleware(request: NextRequest) {
  const origin = corsOrigin(request.headers.get("origin"));
  if (request.method === "OPTIONS") {
    const response = new NextResponse(null, { status: origin ? 204 : 403 });
    return origin ? applyCorsHeaders(response, origin) : response;
  }
  const response = NextResponse.next();
  return origin ? applyCorsHeaders(response, origin) : response;
}

export const config = {
  matcher: "/api/:path*",
};
