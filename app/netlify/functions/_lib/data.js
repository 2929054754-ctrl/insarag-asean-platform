// Baseline data loading from offline-exported artifacts. Cached across warm
// invocations via a module-level promise.
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Netlify bundle layout can vary; probe candidate locations and use the first
// that contains manifest.json.
function resolveDataDir() {
  const candidates = [
    path.join(HERE, "..", "..", "function_data"),
    path.join(HERE, "function_data"),
    path.join(process.cwd(), "function_data"),
    path.join(process.cwd(), "app", "function_data"),
    "/var/task/function_data",
  ];
  for (const c of candidates) {
    if (existsSync(path.join(c, "manifest.json"))) return c;
  }
  throw new Error(`function_data not found (tried: ${candidates.join(" | ")})`);
}

let cachePromise = null;

export async function loadBase() {
  if (cachePromise) return cachePromise;
  cachePromise = (async () => {
    const dir = resolveDataDir();
    const [manifest, meta, tables, bin] = await Promise.all([
      readFile(path.join(dir, "manifest.json"), "utf8").then(JSON.parse),
      readFile(path.join(dir, "meta.json"), "utf8").then(JSON.parse),
      readFile(path.join(dir, "tables.json"), "utf8").then(JSON.parse),
      readFile(path.join(dir, "vectors.bin")),
    ]);
    const vectors = new Float32Array(
      bin.buffer,
      bin.byteOffset,
      manifest.rows * manifest.dim
    );
    return { dir, manifest, meta, tables, vectors };
  })();
  return cachePromise;
}
