// Server-Sent Events helpers, matching the Python API's wire format exactly.
export function sse(event) {
  return `data: ${JSON.stringify(event)}\n\n`;
}

// Turn an async generator of SSE strings into a streaming Web Response.
export function sseResponse(generator) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of generator) {
          controller.enqueue(encoder.encode(chunk));
        }
      } catch (err) {
        controller.enqueue(
          encoder.encode(sse({ type: "error", message: `服务器内部错误: ${err.message}` }))
        );
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

export function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}
