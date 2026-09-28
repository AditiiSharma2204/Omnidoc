import { useRef, useState } from "react";

import Logo from "../common/Logo";
import { deleteDocument, uploadDocument } from "../../api/documentApi";
import { useDocuments } from "../../context/DocumentContext";

export default function Sidebar() {

    const fileInput = useRef<HTMLInputElement>(null);
    const {
        documents,
        refreshDocuments,
        removeDocument,
    } = useDocuments();
    const [deletingId, setDeletingId] = useState<string | null>(null);


    async function handleUpload(
        event: React.ChangeEvent<HTMLInputElement>
    ) {

        const file = event.target.files?.[0];

        if (!file) return;

        try {

            await uploadDocument(file);

            await refreshDocuments();

        } catch (err) {

            console.error(err);

            alert("Upload failed");

        }

    }

    async function handleDelete(documentId: string) {

        setDeletingId(documentId);

        try {

            await deleteDocument(documentId);

            removeDocument(documentId);

        } catch (err) {

            console.error(err);

            alert("Delete failed");

        } finally {

            setDeletingId(null);

        }

    }

    return (

        <aside className="w-72 bg-white border-r border-gray-200 h-screen flex flex-col">

            <div className="p-6">
                <Logo />
            </div>

            <div className="px-6">

                <input
                    ref={fileInput}
                    type="file"
                    hidden
                    accept=".pdf,.docx,.pptx,.xlsx"
                    onChange={handleUpload}
                />

                <button

                    onClick={() => fileInput.current?.click()}

                    className="w-full rounded-lg bg-blue-600 text-white py-3 font-medium hover:bg-blue-700 transition"

                >

                    + Upload Document

                </button>

            </div>

            <div className="mt-8 px-6 overflow-y-auto">

                <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">

                    Documents

                </h2>

                <div className="mt-4 space-y-3">

                    {documents.length === 0 && (

                        <div className="text-gray-400">

                            No documents uploaded

                        </div>

                    )}

                    {documents.map((doc) => (

                        <div
                            key={doc.document_id}
                            className="rounded-lg bg-gray-100 p-3 flex items-start justify-between gap-2"
                        >
                            <div>
                                📄 {doc.original_filename}
                                {doc.status !== "indexed" && (
                                    <div className="text-xs text-gray-500 mt-1">
                                        {doc.status}
                                        {doc.parse_error && `: ${doc.parse_error}`}
                                    </div>
                                )}
                            </div>

                            <button
                                onClick={() => handleDelete(doc.document_id)}
                                disabled={deletingId === doc.document_id}
                                className="text-gray-400 hover:text-red-600 disabled:opacity-50"
                                aria-label={`Delete ${doc.original_filename}`}
                            >
                                ✕
                            </button>
                        </div>

                    ))}

                </div>

            </div>

        </aside>

    );

}
