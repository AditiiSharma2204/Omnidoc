export interface DocumentSummary {

    document_id: string;

    original_filename: string;

    mime_type: string;

    size: number;

    upload_time: string;

    status: string;

    parse_error: string | null;

}

export interface DocumentListResponse {

    documents: DocumentSummary[];

}
