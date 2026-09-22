import { useRef } from "react";

import Logo from "../common/Logo";
import { uploadDocument } from "../../api/documentApi";
import { useDocuments } from "../../context/DocumentContext";

export default function Sidebar() {

    const fileInput = useRef<HTMLInputElement>(null);
    const {
        documents,
        addDocument,
    } = useDocuments();


    async function handleUpload(
        event: React.ChangeEvent<HTMLInputElement>
    ) {

        const file = event.target.files?.[0];

        if (!file) return;

        try {

            const result = await uploadDocument(file);

            addDocument(result.original_filename);

        } catch (err) {

            console.error(err);

            alert("Upload failed");

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

            <div className="mt-8 px-6">

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
                            key={doc}
                            className="rounded-lg bg-gray-100 p-3"
                        >
                            📄 {doc}
                        </div>

                    ))}

                </div>

            </div>

        </aside>

    );

}