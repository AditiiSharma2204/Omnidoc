import { useEffect, useState } from "react";

import { streamQuestion, getConversationHistory } from "../../api/chatApi";
import Message from "./Message";
import type { ChatResponse } from "../../types/chat";
import { useDocuments } from "../../context/DocumentContext";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  sources?: ChatResponse["sources"];
}

const CONVERSATION_ID_STORAGE_KEY = "omnidoc:conversationId";

export default function ChatWindow() {

  const { documents } = useDocuments();

  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const [question, setQuestion] = useState("");

  const [loading, setLoading] = useState(false);

  // The backend hands back a conversation_id on every response; once
  // we have one, every following question is sent with it so the
  // model sees prior turns as real conversation history, not just
  // retrieved document context. Persisted to localStorage so a page
  // reload continues the same conversation instead of silently
  // starting a new one while the old one sits orphaned in the DB.
  const [conversationId, setConversationId] = useState<string | null>(
    () => localStorage.getItem(CONVERSATION_ID_STORAGE_KEY),
  );

  // On mount, if a conversation id survived a reload, fetch its
  // history from the backend so the UI isn't blank while the server
  // still has the full transcript. A 404 means the conversation no
  // longer exists (e.g. a fresh DB) -- clear the stale id rather than
  // send it on the next request.
  useEffect(() => {
    if (!conversationId) return;

    let cancelled = false;

    getConversationHistory(conversationId)
      .then((history) => {
        if (cancelled) return;

        setMessages(
          history.messages.map((m) => ({
            role: m.role,
            text: m.content,
            sources: m.sources ?? undefined,
          })),
        );
      })
      .catch(() => {
        if (cancelled) return;

        localStorage.removeItem(CONVERSATION_ID_STORAGE_KEY);
        setConversationId(null);
      });

    return () => {
      cancelled = true;
    };
    // Only ever run for the id we loaded from localStorage on mount --
    // this isn't meant to re-fetch on every new conversation_id the
    // chat flow itself produces.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function sendMessage(customQuestion?: string) {
    const userQuestion = customQuestion ?? question;

    if (!userQuestion.trim() || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        text: userQuestion,
      },
      {
        role: "assistant",
        text: "",
      },
    ]);

    setQuestion("");

    setLoading(true);

    // Both callbacks below only ever touch the last message (the
    // assistant placeholder just pushed above) -- streaming can't
    // overlap with another sendMessage() call since the input is
    // disabled via `loading` while a stream is in flight.
    function updateLastMessage(patch: Partial<ChatMessage>) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });
    }

    try {
      await streamQuestion(
        userQuestion,
        conversationId,
        (token) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, text: last.text + token };
            return next;
          });
        },
        (event) => {
          setConversationId(event.conversation_id);
          localStorage.setItem(
            CONVERSATION_ID_STORAGE_KEY,
            event.conversation_id,
          );
          updateLastMessage({ sources: event.sources });
        },
      );
    } catch (err) {
      console.error(err);

      updateLastMessage({ text: "Something went wrong." });
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

            {message.sources && message.sources.length > 0 && (

              <div className="ml-3">

                <div className="text-sm font-semibold text-gray-500 mb-2">
                  Sources
                </div>

                <div className="space-y-2">

                  {message.sources.map((source) => (

                    <div
                      key={source.index}
                      className={`border rounded-lg p-3 text-sm ${
                        source.cited
                          ? "bg-white border-blue-200"
                          : "bg-gray-50 border-gray-200 opacity-60"
                      }`}
                    >
                      <div className="font-medium flex items-center gap-2">
                        <span
                          className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-xs font-semibold ${
                            source.cited
                              ? "bg-blue-100 text-blue-700"
                              : "bg-gray-200 text-gray-500"
                          }`}
                        >
                          {source.index}
                        </span>
                        📄 {source.document}
                      </div>

                      <div className="text-gray-500">
                        {source.heading}
                        {source.page != null && ` · p.${source.page}`}
                      </div>

                      {!source.cited && (
                        <div className="text-xs text-gray-400 mt-1">
                          Retrieved but not cited in the answer
                        </div>
                      )}
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