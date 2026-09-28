import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

interface Props {

    role: "user" | "assistant";

    text: string;

}

// Assistant answers are markdown (the LLM is asked for lists/bold
// formatting -- see PromptBuilder's system prompt rule 6). Default
// browser/Tailwind-reset spacing for markdown elements is too tight
// inside a chat bubble without these overrides (no @tailwindcss/
// typography plugin in this project, so styling each element
// directly rather than pulling in a whole prose plugin for one
// component).
const markdownComponents: Components = {
    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
    ul: ({ children }) => (
        <ul className="mb-2 last:mb-0 list-disc pl-5 space-y-1">
            {children}
        </ul>
    ),
    ol: ({ children }) => (
        <ol className="mb-2 last:mb-0 list-decimal pl-5 space-y-1">
            {children}
        </ol>
    ),
    li: ({ children }) => <li>{children}</li>,
    strong: ({ children }) => (
        <strong className="font-semibold">{children}</strong>
    ),
    h1: ({ children }) => (
        <h1 className="text-lg font-semibold mb-2 mt-1">{children}</h1>
    ),
    h2: ({ children }) => (
        <h2 className="text-base font-semibold mb-2 mt-1">{children}</h2>
    ),
    h3: ({ children }) => (
        <h3 className="text-sm font-semibold mb-1 mt-1">{children}</h3>
    ),
    code: ({ children }) => (
        <code className="bg-gray-100 rounded px-1 py-0.5 text-sm font-mono">
            {children}
        </code>
    ),
    a: ({ children, href }) => (
        <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline"
        >
            {children}
        </a>
    ),
    table: ({ children }) => (
        <div className="overflow-x-auto mb-2">
            <table className="border-collapse text-sm">{children}</table>
        </div>
    ),
    th: ({ children }) => (
        <th className="border px-2 py-1 bg-gray-50 text-left">
            {children}
        </th>
    ),
    td: ({ children }) => <td className="border px-2 py-1">{children}</td>,
};

export default function Message({

    role,

    text,

}: Props) {

    const isUser = role === "user";

    return (

        <div
            className={`flex ${
                isUser
                    ? "justify-end"
                    : "justify-start"
            }`}
        >

            <div
                className={`max-w-2xl rounded-xl px-5 py-4 ${
                    isUser
                        ? "bg-blue-600 text-white whitespace-pre-wrap"
                        : "bg-white border"
                }`}
            >

                {isUser ? (
                    text
                ) : (
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={markdownComponents}
                    >
                        {text}
                    </ReactMarkdown>
                )}

            </div>

        </div>

    );

}
