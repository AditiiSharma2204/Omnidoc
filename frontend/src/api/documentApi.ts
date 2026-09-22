import api from "./api";

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