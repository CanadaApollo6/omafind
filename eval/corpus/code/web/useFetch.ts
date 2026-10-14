// useFetch: load JSON from an endpoint, re-run when the url changes,
// and abort the request if the component unmounts or the url changes mid-flight.
import { useEffect, useState } from "react";

export interface FetchState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

export function useFetch<T = unknown>(url: string | null, init?: RequestInit): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    error: null,
    loading: url !== null,
  });

  useEffect(() => {
    if (url === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }

    const controller = new AbortController();
    setState((prev) => ({ ...prev, loading: true, error: null }));

    fetch(url, { ...init, signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) {
          // keep the status in the message, the UI shows it
          throw new Error(`request failed with status ${res.status}`);
        }
        return (await res.json()) as T;
      })
      .then((data) => {
        setState({ data, error: null, loading: false });
      })
      .catch((err: unknown) => {
        // aborts are expected on unmount, don't show them as errors
        if (err instanceof DOMException && err.name === "AbortError") return;
        const error = err instanceof Error ? err : new Error(String(err));
        setState({ data: null, error, loading: false });
      });

    return () => controller.abort();
    // init is intentionally left out, callers pass inline objects
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  return state;
}
