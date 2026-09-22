interface Props {

    role: "user" | "assistant";

    text: string;

}

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
                className={`max-w-2xl rounded-xl px-5 py-4 whitespace-pre-wrap ${
                    isUser
                        ? "bg-blue-600 text-white"
                        : "bg-white border"
                }`}
            >

                {text}

            </div>

        </div>

    );

}