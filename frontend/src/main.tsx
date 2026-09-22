import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { DocumentProvider } from "./context/DocumentContext";
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>

    <DocumentProvider>

        <App />

    </DocumentProvider>

  </StrictMode>
)
