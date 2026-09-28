import api from "./api";
import type { ConversationHistoryResponse } from "../types/chat";

export async function askQuestion(
    question: string,
    conversationId?: string | null,
) {
    const response = await api.post(
        "/chat",
        {
            question,
            top_k: 5,
            conversation_id: conversationId ?? null,
        }
    );

    return response.data;
}

export async function getConversationHistory(
    conversationId: string,
): Promise<ConversationHistoryResponse> {
    const response = await api.get(`/conversations/${conversationId}`);

    return response.data;
}