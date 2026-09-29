import { NextResponse, type NextRequest } from "next/server";

/**
 * Only the Kinesia page itself may use the API.
 *
 * The API starts (billed) GPU jobs and deletes analyses, so a web page opened
 * in the same browser must not be able to call it: requests must name a local
 * host (no DNS rebinding) and changes must come from the app's own origin
 * (no cross-site requests). Extra origins can be allowed with
 * `KINESIA_ALLOWED_ORIGINS` (comma-separated), e.g. for a remote front end.
 */

const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

function allowedOrigins(): string[] {
  return (process.env.KINESIA_ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((origin) => origin.trim().replace(/\/+$/, ""))
    .filter(Boolean);
}

function hostName(host: string): string {
  return host.startsWith("[") ? host.slice(0, host.indexOf("]") + 1) : host.split(":")[0];
}

export function middleware(request: NextRequest) {
  const host = request.headers.get("host") ?? "";
  const extra = allowedOrigins();
  const extraHosts = new Set(extra.map((origin) => { try { return new URL(origin).host; } catch { return ""; } }));
  if (!LOCAL_HOSTS.has(hostName(host)) && !extraHosts.has(host)) {
    return new NextResponse("Forbidden host", { status: 403 });
  }
  const origin = request.headers.get("origin");
  const mutating = !["GET", "HEAD"].includes(request.method);
  if (origin && origin !== `${request.nextUrl.protocol}//${host}` && !extra.includes(origin)) {
    if (mutating || request.method === "OPTIONS") return new NextResponse("Forbidden origin", { status: 403 });
  }
  const response = NextResponse.next();
  if (origin && extra.includes(origin)) {
    response.headers.set("access-control-allow-origin", origin);
    response.headers.set("access-control-allow-methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS");
    response.headers.set("access-control-allow-headers", "Content-Type,Range");
    response.headers.append("vary", "Origin");
  }
  return response;
}

export const config = {
  matcher: "/api/:path*",
};
