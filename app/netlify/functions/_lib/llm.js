// DeepSeek streaming chat client (OpenAI-compatible).
// Async generator yielding { kind: "content" | "reasoning", text }.
import {
  DEEPSEEK_API_KEY,
  DEEPSEEK_BASE_URL,
  DEEPSEEK_MODEL,
} from "./config.js";

export class LLMNotConfigured extends Error {}

export async function* chatStream(messages, temperature = 0.3) {
  if (!DEEPSEEK_API_KEY) {
    throw new LLMNotConfigured("DEEPSEEK_API_KEY 未配置");
  }
  const r = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: DEEPSEEK_MODEL,
      messages,
      temperature,
      stream: true,
    }),
  });
  if (!r.ok) {
    throw new Error(`DeepSeek API error ${r.status}: ${(await r.text()).slice(0, 300)}`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      let obj;
      try {
        obj = JSON.parse(data);
      } catch {
        continue;
      }
      const delta = (obj.choices && obj.choices[0] && obj.choices[0].delta) || {};
      if (delta.content) yield { kind: "content", text: delta.content };
      if (delta.reasoning_content) yield { kind: "reasoning", text: delta.reasoning_content };
    }
  }
}
