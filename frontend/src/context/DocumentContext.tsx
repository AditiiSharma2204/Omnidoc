import {
    createContext,
    useContext,
    useState,
} from "react";

import type { ReactNode } from "react";

interface DocumentContextType {
    documents: string[];
    addDocument: (name: string) => void;
    removeDocument: (name: string) => void;
}

const DocumentContext = createContext<DocumentContextType | undefined>(
    undefined
);

export function DocumentProvider({
    children,
}: {
    children: ReactNode;
}) {
    const [documents, setDocuments] = useState<string[]>([]);

    function addDocument(name: string) {
        setDocuments((prev) => [...prev, name]);
    }

    function removeDocument(name: string) {
        setDocuments((prev) =>
            prev.filter((doc) => doc !== name)
        );
    }

    return (
        <DocumentContext.Provider
            value={{
                documents,
                addDocument,
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