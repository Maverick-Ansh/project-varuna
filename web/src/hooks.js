import { useEffect, useRef } from "react";

// Poll fn() every ms while enabled; fires immediately on (re)enable; pauses in hidden tabs.
export function usePolling(fn, ms, enabled = true, deps = []) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!enabled) return undefined;
    let stop = false;
    const tick = () => { if (!stop && !document.hidden) saved.current(); };
    tick();
    const id = setInterval(tick, ms);
    return () => { stop = true; clearInterval(id); };
  }, [enabled, ms, ...deps]);
}
