import api from "./api";
import type { DocumentListResponse } from "../types/document";

export async function uploadDocument(file: File) {

    const formData = new FormData();

    formData.append("file", file);

    const response = await api.post(
        "/documents/upload",
        formData,
        {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        }
    );

    return response.data;
}

export async function listDocuments(): Promise<DocumentListResponse> {

    const response = await api.get("/documents");

    return response.data;
}

export async function deleteDocument(documentId: string): Promise<void> {

    await api.delete(`/documents/${documentId}`);

}