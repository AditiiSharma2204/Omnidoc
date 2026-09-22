import api from "./api";

export async function askQuestion(
    question: string
) {
    const response = await api.post(
        "/chat",
        {
            question,
            top_k: 5,
        }
    );

    return response.data;
}