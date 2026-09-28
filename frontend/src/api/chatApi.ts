import api from "./api";
import type { ConversationHistoryResponse, Source } from "../types/chat";

interface StreamDoneEvent {
    conversation_id: string;
    sources: Source[];
}

// Consumes /chat/stream's newline-delimited JSON events
// ({"type": "token", ...} | {"type": "done", ...}). Uses fetch directly
// rather than the axios instance -- axios has no browser-side streaming
// body reader, only a Node-only `responseType: "stream"`.
export async function streamQuestion(
    question: string,
    conversationId: string | null | undefined,
    onToken: (token: string) => void,
    onDone: (event: StreamDoneEvent) => void,
): Promise<void> {
    const response = await fetch(`${api.defaults.baseURL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question,
            top_k: 5,
            conversation_id: conversationId ?? null,
        }),
    });

    if (!response.ok || !response.body) {
        throw new Error(`Stream request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (!line.trim()) continue;

            const event = JSON.parse(line);
            if (event.type === "token") {
                onToken(event.token);
            } else if (event.type === "done") {
                onDone(event);
            }
        }
    }
}

export async function getConversationHistory(
    conversationId: string,
): Promise<ConversationHistoryResponse> {
    const response = await api.get(`/conversations/${conversationId}`);

    return response.data;
}