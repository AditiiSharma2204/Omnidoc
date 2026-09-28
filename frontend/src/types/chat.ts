export interface Source {

    index: number;

    document: string;

    heading: string;

    page: number | null;

    cited: boolean;

}

export interface ChatResponse {

    answer: string;

    sources: Source[];

    conversation_id: string;

}

export interface MessageOut {

    role: "user" | "assistant";

    content: string;

    sources: Source[] | null;

    created_at: string;

}

export interface ConversationHistoryResponse {

    conversation_id: string;

    messages: MessageOut[];

}