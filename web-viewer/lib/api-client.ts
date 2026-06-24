// Optional backend origin; empty when frontend and backend are served together (same origin).
const backendUrl = normalizeBaseUrl(process.env.NEXT_PUBLIC_KINESIA_BACKEND_URL);

// Trim whitespace and trailing slashes so paths can be appended without doubling separators.
function normalizeBaseUrl(value: string | undefined): string {
  return value?.trim().replace(/\/+$/, "") ?? "";
}

// True for URLs that should be used as-is (scheme://, blob:, data:) rather than prefixed.
function isAbsoluteUrl(value: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(value) || value.startsWith("blob:") || value.startsWith("data:");
}

// Resolve a path against the backend origin; absolute URLs pass through, null in stays null out.
export function apiUrl(path: string): string;
export function apiUrl(path: string | null | undefined): string | null;
export function apiUrl(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }
  if (isAbsoluteUrl(path)) {
    return path;
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return backendUrl ? `${backendUrl}${normalizedPath}` : normalizedPath;
}

// fetch() wrapper that turns network failures into actionable "backend unreachable" errors,
// while letting AbortError (caller-initiated cancellation) propagate unchanged.
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(apiUrl(path), init);
  } catch (error) {
    if (
      typeof DOMException !== "undefined" &&
      error instanceof DOMException &&
      error.name === "AbortError"
    ) {
      throw error;
    }
    if (backendUrl) {
      throw new Error(
        `Cannot reach the Kinesia backend at ${backendUrl}. Check that the backend is running ` +
          "and that KINESIA_ALLOWED_ORIGINS on the backend allows this frontend origin.",
      );
    }
    throw new Error(
      "Cannot reach the Kinesia backend. Check that the app is running on this machine.",
    );
  }
}
