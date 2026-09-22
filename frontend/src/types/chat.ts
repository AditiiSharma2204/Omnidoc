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

}