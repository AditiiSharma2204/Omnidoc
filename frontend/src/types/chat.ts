export interface Source {

    document: string;

    heading: string;

    page: number | null;

}

export interface ChatResponse {

    answer: string;

    sources: Source[];

}