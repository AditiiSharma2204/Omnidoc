import Sidebar from "./Sidebar";
import ChatWindow from "../chat/ChatWindow";

export default function MainLayout() {
  return (
    <div className="flex h-screen">

      <Sidebar />

      <ChatWindow />

    </div>
  );
}