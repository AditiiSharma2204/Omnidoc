import { useState } from "react";

import { askQuestion } from "../../api/chatApi";
import Message from "./Message";
import type { ChatResponse } from "../../types/chat";
import { useDocuments } from "../../context/DocumentContext";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  sources?: ChatResponse["sources"];
}

export default function ChatWindow() {

  const { documents } = useDocuments();

  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const [question, setQuestion] = useState("");

  const [loading, setLoading] = useState(false);

  async function sendMessage(customQuestion?: string) {
    const userQuestion = customQuestion ?? question;

    if (!userQuestion.trim() || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",   
        text: userQuestion,
      },
    ]);

    setQuestion("");

    setLoading(true);

    try {
      const response: ChatResponse = await askQuestion(userQuestion);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (err) {
      console.error(err);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Something went wrong.",
        },
      ]);
    }

    setLoading(false);
  }

  function askSuggestion(question: string) {
    sendMessage(question);
  }

  return (
    <div className="flex flex-col flex-1 bg-gray-50 h-screen">

      {/* Messages */}

      <div className="flex-1 overflow-y-auto p-10 space-y-6">

        {messages.length === 0 && (

            <div className="h-full flex items-center justify-center">

                <div className="max-w-3xl text-center">

                <h1 className="text-5xl font-bold mb-5">

                    👋 Welcome to OmniDoc AI

                </h1>

                {documents.length === 0 ? (

                    <>

                    <p className="text-gray-600 text-lg mb-8">

                        I'm your intelligent document assistant.

                    </p>

                    <div className="bg-white rounded-xl border p-8 shadow-sm">

                        <h2 className="text-xl font-semibold mb-6">

                        I can help you:

                        </h2>

                        <div className="grid grid-cols-2 gap-4 text-left">

                        <div>📄 Summarize documents</div>

                        <div>🔍 Answer questions</div>

                        <div>📊 Extract information</div>

                        <div>📚 Explain technical reports</div>

                        <div>📅 Find important dates</div>

                        <div>⚖ Compare documents</div>

                        </div>

                        <p className="mt-8 text-gray-500">

                        Upload a document using the button on the left to get started.

                        </p>

                    </div>

                    </>

                ) : (

                    <>

                    <div className="text-gray-600 text-lg mb-8">

                        <strong>
                            {documents.length} document
                            {documents.length > 1 ? "s" : ""}
                            uploaded successfully.
                        </strong>

                        <div className="mt-4 flex flex-wrap justify-center gap-2">

                            {documents.map((doc) => (

                                <span
                                    key={doc}
                                    className="px-3 py-1 rounded-full bg-gray-100 border text-sm"
                                >
                                    📄 {doc}
                                </span>

                            ))}

                        </div>

                    </div>

                    <div className="bg-white rounded-xl border p-8 shadow-sm">

                        <h2 className="text-xl font-semibold mb-6">

                        Try asking:

                        </h2>

                        <div className="flex flex-wrap gap-4 justify-center">

                        <button
                            onClick={() => askSuggestion("What internships are mentioned?")}
                            className="px-5 py-3 rounded-full bg-blue-100 hover:bg-blue-200"
                        >
                            What internships are mentioned?
                        </button>

                        <button
                            onClick={() => askSuggestion("Summarize this document.")}
                            className="px-5 py-3 rounded-full bg-blue-100 hover:bg-blue-200"
                        >
                            Summarize this document
                        </button>

                        <button
                            onClick={() => askSuggestion("What technical skills are listed?")}
                            className="px-5 py-3 rounded-full bg-blue-100 hover:bg-blue-200"
                        >
                            Technical skills
                        </button>

                        <button
                            onClick={() => askSuggestion("Tell me about the projects.")}
                            className="px-5 py-3 rounded-full bg-blue-100 hover:bg-blue-200"
                        >
                            Projects
                        </button>

                        </div>

                    </div>

                    </>

                )}

                </div>

            </div>

            )}

        {messages.map((message, index) => (

          <div key={index} className="space-y-2">

            <Message
              role={message.role}
              text={message.text}
            />

            {message.sources && (

              <div className="ml-3">

                <div className="text-sm font-semibold text-gray-500 mb-2">
                  Sources
                </div>

                <div className="space-y-2">

                  {message.sources.map((source, idx) => (

                    <div
                      key={idx}
                      className="bg-white border rounded-lg p-3 text-sm"
                    >
                      <div className="font-medium">
                        📄 {source.document}
                      </div>

                      <div className="text-gray-500">
                        {source.heading}
                      </div>
                    </div>

                  ))}

                </div>

              </div>

            )}

          </div>

        ))}

      </div>

      {/* Input */}

      <div className="border-t bg-white p-6">

        <div className="flex gap-3">

          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                sendMessage();
              }
            }}
            placeholder="Ask anything about your documents..."
            className="flex-1 border rounded-lg px-4 py-3 outline-none"
          />

          <button
            onClick={() => sendMessage()}
            disabled={loading}
            className="bg-blue-600 text-white px-8 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "..." : "Send"}
          </button>

        </div>

      </div>

    </div>
  );
}