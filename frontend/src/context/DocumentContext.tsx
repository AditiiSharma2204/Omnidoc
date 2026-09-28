import {
    createContext,
    useContext,
    useEffect,
    useState,
} from "react";

import type { ReactNode } from "react";

import { listDocuments } from "../api/documentApi";
import type { DocumentSummary } from "../types/document";

interface DocumentContextType {
    documents: DocumentSummary[];
    refreshDocuments: () => Promise<void>;
    removeDocument: (documentId: string) => void;
}

const DocumentContext = createContext<DocumentContextType | undefined>(
    undefined
);

export function DocumentProvider({
    children,
}: {
    children: ReactNode;
}) {
    const [documents, setDocuments] = useState<DocumentSummary[]>([]);

    // The backend is the source of truth for what's actually indexed --
    // documents uploaded in a prior session (or by a script, an eval
    // run, etc.) are just as real as ones uploaded through this
    // browser tab, so the list is fetched from the server rather than
    // built up purely from local upload events.
    async function refreshDocuments() {
        const response = await listDocuments();

        setDocuments(response.documents);
    }

    useEffect(() => {
        refreshDocuments();
    }, []);

    function removeDocument(documentId: string) {
        setDocuments((prev) =>
            prev.filter((doc) => doc.document_id !== documentId)
        );
    }

    return (
        <DocumentContext.Provider
            value={{
                documents,
                refreshDocuments,
                removeDocument,
            }}
        >
            {children}
        </DocumentContext.Provider>
    );
}

export function useDocuments() {
    const context = useContext(DocumentContext);

    if (!context) {
        throw new Error(
            "useDocuments must be used inside DocumentProvider"
        );
    }

    return context;
}
