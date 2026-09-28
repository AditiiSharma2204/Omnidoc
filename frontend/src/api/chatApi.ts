import api from "./api";

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