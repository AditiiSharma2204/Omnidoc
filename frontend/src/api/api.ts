import axios from "axios";

// Overridable at build time (VITE_API_BASE_URL in a .env file or
// Docker build arg) -- defaults to the local dev backend so nothing
// changes for the normal `npm run dev` workflow.
const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1",
});

export default api;