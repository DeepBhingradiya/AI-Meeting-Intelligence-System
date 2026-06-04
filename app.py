import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import pipeline dependencies from your core module
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# Load environment variables
load_dotenv()

app = FastAPI(title="DeepInsight - AI Intelligence Dashboard")

# Global session state
session_state = {
    "title": "",
    "transcript": "",
    "summary": "",
    "action_items": "",
    "key_decisions": "",
    "open_questions": "",
    "rag_chain": None
}

class PipelineRequest(BaseModel):
    source: str
    language: str = "english"

class ChatRequest(BaseModel):
    question: str


@app.post("/api/run")
def run_pipeline_endpoint(payload: PipelineRequest):
    try:
        source = payload.source.strip()
        language = payload.language.strip() or "english"

        if not source:
            raise HTTPException(status_code=400, detail="Source input is empty.")

        chunks = process_input(source)
        transcript = transcribe_all(chunks, language)
        
        session_state["title"] = generate_title(transcript)
        session_state["summary"] = summarize(transcript)
        session_state["action_items"] = extract_action_items(transcript)
        session_state["key_decisions"] = extract_key_decisions(transcript)
        session_state["open_questions"] = extract_questions(transcript)
        session_state["rag_chain"] = build_rag_chain(transcript)
        session_state["transcript"] = transcript

        return {
            "title": session_state["title"],
            "summary": session_state["summary"],
            "action_items": session_state["action_items"],
            "key_decisions": session_state["key_decisions"],
            "open_questions": session_state["open_questions"],
            "transcript": session_state["transcript"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat_endpoint(payload: ChatRequest):
    if not session_state["rag_chain"]:
        raise HTTPException(status_code=400, detail="RAG system offline.")
    try:
        answer = ask_question(session_state["rag_chain"], payload.question.strip())
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DeepInsight — AI Meeting Intelligence</title>
        <!-- Tailwind CSS -->
        <script src="https://cdn.tailwindcss.com"></script>
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        fontFamily: {
                            sans: ['Inter', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        }
                    }
                }
            }
        </script>
        <style>
            /* Cyberpunk-esque gradient scrollbar styling */
            ::-webkit-scrollbar {
                width: 5px;
                height: 5px;
            }
            ::-webkit-scrollbar-track {
                background: #090a10;
            }
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #6366f1 0%, #ec4899 100%);
                border-radius: 99px;
            }
            body {
                background-color: #08090e;
                background-image: 
                    radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 85% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.05) 0%, transparent 60%);
            }
            /* Translucent glowing card class */
            .glass-card {
                background: rgba(17, 18, 27, 0.65);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.06);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .glass-card:hover {
                border-color: rgba(139, 92, 246, 0.25);
                box-shadow: 0 10px 40px -10px rgba(124, 58, 237, 0.15);
                transform: translateY(-1px);
            }
        </style>
    </head>
    <body class="text-slate-100 font-sans flex h-screen overflow-hidden antialiased">

        <!-- SIDEBAR -->
        <aside class="w-80 bg-[#0c0d15]/80 backdrop-blur-3xl border-r border-slate-800/60 flex flex-col justify-between p-7 z-20">
            <div>
                <!-- Brand Signature Logo -->
                <div class="mb-10 flex items-center gap-3">
                    <div class="relative flex h-3 w-3">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-3 w-3 bg-pink-500 shadow-[0_0_10px_rgba(236,72,153,0.5)]"></span>
                    </div>
                    <div>
                        <span class="text-[9px] font-bold tracking-[0.25em] text-indigo-400 block uppercase">Intelligence System</span>
                        <h1 class="text-sm font-bold text-slate-100 mt-0.5 tracking-tight">DeepInsight</h1>
                    </div>
                </div>
                
                <div class="space-y-6">
                    <div>
                        <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-2.5">Pipeline Input Source</label>
                        <input id="sourceInput" type="text" placeholder="YouTube URL or video path..." class="w-full bg-[#08090e]/90 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all">
                    </div>

                    <div>
                        <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-2.5">Language Profile</label>
                        <select id="languageInput" class="w-full bg-[#08090e]/90 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all">
                            <option value="english">English</option>
                            <option value="hinglish">Hinglish</option>
                        </select>
                    </div>

                    <!-- Vibrant Gradient Pulse Button -->
                    <button id="runBtn" onclick="runPipeline()" class="w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90 text-white font-bold text-xs tracking-wider py-3.5 px-4 rounded-lg transition-all mt-6 uppercase shadow-[0_4px_20px_rgba(139,92,246,0.3)] hover:shadow-[0_4px_25px_rgba(139,92,246,0.5)]">
                        INITIALIZE ANALYTICS
                    </button>
                </div>
            </div>

            <!-- FOOTER BADGE -->
            <div class="border-t border-slate-800/80 pt-5 text-center">
                <span class="text-[9px] text-slate-400 uppercase tracking-widest block">DESIGNED & DEVELOPED BY</span>
                <span class="text-xs text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-pink-400 font-bold tracking-wide block mt-1">Deep Bhingradiya</span>
            </div>
        </aside>

        <!-- MAIN DASHBOARD -->
        <main class="flex-1 flex flex-col overflow-hidden relative">
            <!-- HEADER -->
            <header class="h-16 border-b border-slate-800/60 flex items-center justify-between px-8 bg-[#0c0d15]/30 backdrop-blur-md z-10">
                <div class="flex items-center gap-3">
                    <span class="text-[10px] text-indigo-400 tracking-[0.2em] font-bold uppercase">Workspace Console</span>
                    <span class="text-xs font-semibold text-slate-300 tracking-tight" id="resourceHeader">No analysis initialized</span>
                </div>
                <div id="statusBadge" class="text-[9px] px-3.5 py-1 rounded-full bg-slate-950 border border-slate-800 text-slate-400 font-bold tracking-[0.1em] uppercase">
                    Status: Standby
                </div>
            </header>

            <!-- EMPTY INTRO STATE (With requested brand paragraph) -->
            <div id="emptyState" class="flex-grow flex flex-col items-center justify-center p-8 z-10">
                <div class="max-w-xl text-center border border-slate-800/60 p-10 rounded-xl bg-[#0c0d15]/60 backdrop-blur-2xl shadow-2xl relative overflow-hidden">
                    <!-- Subtle visual background accent within empty card -->
                    <div class="absolute -top-10 -right-10 w-28 h-28 bg-indigo-500/10 rounded-full blur-2xl"></div>
                    <div class="absolute -bottom-10 -left-10 w-28 h-28 bg-pink-500/10 rounded-full blur-2xl"></div>

                    <span class="text-[9px] font-bold tracking-[0.25em] text-indigo-400 uppercase block mb-3.5">Cognitive Core</span>
                    <h3 class="text-base font-bold text-slate-100 mb-4 tracking-tight">DeepInsight Intelligence System</h3>
                    <p class="text-xs text-slate-300 leading-relaxed max-w-md mx-auto mb-6">
                        DeepInsight is an AI-powered meeting intelligence system that converts audio and video content into structured insights. It generates transcripts, summaries, key decisions, and action items, and allows users to interact with meeting data through a contextual chat interface.
                    </p>
                    <div class="text-[9px] tracking-[0.15em] text-indigo-300/40 border-t border-slate-800/80 pt-4 font-mono">
                        CONFIGURE DESTINATION IN LEFT PARAMETERS PANEL TO RUN SCAN
                    </div>
                </div>
            </div>

            <!-- CYBERPUNK-STYLE RADAR LOADER VIEW -->
            <div id="loadingState" class="hidden flex-grow flex flex-col items-center justify-center p-8 z-10">
                <div class="flex flex-col items-center space-y-6 max-w-xs text-center">
                    <!-- Dynamic Pulse Radar Loader -->
                    <div class="relative flex items-center justify-center w-20 h-20">
                        <div class="absolute w-full h-full rounded-full border border-indigo-500/30 animate-ping"></div>
                        <div class="absolute w-3/4 h-3/4 rounded-full border border-purple-500/20 animate-pulse"></div>
                        <div class="absolute w-12 h-12 rounded-full border-2 border-t-pink-500 border-indigo-500/10 animate-spin"></div>
                    </div>
                    <div>
                        <p class="text-xs font-bold tracking-[0.25em] text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 uppercase">RUNNING PIPELINE SCAN</p>
                        <p class="text-[11px] text-slate-400 leading-relaxed mt-2.5">
                            Processing multimedia chunks, mapping metadata summaries, and establishing interactive RAG context models...
                        </p>
                    </div>
                </div>
            </div>

            <!-- ACTIVE DATA WORKSPACE CONTAINER -->
            <div id="dashboardTabs" class="hidden flex-grow flex flex-col overflow-hidden p-9 z-10">
                <!-- Glowing Custom Tabs -->
                <div class="flex border-b border-slate-800/80 mb-7 gap-8">
                    <button id="tabBtnSummary" onclick="switchTab('summary')" class="border-b-2 border-pink-500 pb-3 text-xs font-semibold tracking-wider text-slate-100 transition-all">SUMMARY BRIEF</button>
                    <button id="tabBtnOutcomes" onclick="switchTab('outcomes')" class="border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all">EXTRACTED OUTCOMES</button>
                    <button id="tabBtnTranscript" onclick="switchTab('transcript')" class="border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all">RAW TRANSCRIPT</button>
                    <button id="tabBtnConsult" onclick="switchTab('consult')" class="border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all">CONSULT CHAT</button>
                </div>

                <!-- Tab Area Contents -->
                <div class="flex-grow overflow-y-auto pr-1">
                    <!-- Tab 1: Summary -->
                    <div id="tabContentSummary" class="block">
                        <div class="glass-card rounded-xl p-6">
                            <span class="text-[9px] font-bold tracking-[0.25em] text-indigo-400 block mb-4 uppercase">Executive Briefing</span>
                            <div class="text-xs text-slate-300 leading-relaxed space-y-4" id="dataSummary"></div>
                        </div>
                    </div>

                    <!-- Tab 2: OutComes Cards -->
                    <div id="tabContentOutcomes" class="hidden space-y-6">
                        <div class="grid grid-cols-2 gap-6">
                            <div>
                                <span class="text-[9px] font-bold tracking-[0.25em] text-emerald-400 block mb-3 uppercase">Decisions Register</span>
                                <div class="space-y-3.5" id="dataDecisions"></div>
                            </div>
                            <div>
                                <span class="text-[9px] font-bold tracking-[0.25em] text-pink-400 block mb-3 uppercase">Action Items Checklist</span>
                                <div class="space-y-3.5" id="dataActionItems"></div>
                            </div>
                        </div>
                        <div class="pt-2">
                            <span class="text-[9px] font-bold tracking-[0.25em] text-amber-400 block mb-3 uppercase">Gaps & Unresolved Questions</span>
                            <div class="space-y-3.5" id="dataQuestions"></div>
                        </div>
                    </div>

                    <!-- Tab 3: Monospace Transcript -->
                    <div id="tabContentTranscript" class="hidden">
                        <div class="bg-[#0c0d15]/50 border border-slate-800/80 rounded-xl p-6 font-mono text-[11px] text-slate-400 leading-relaxed max-h-[500px] overflow-y-auto" id="dataTranscript"></div>
                    </div>

                    <!-- Tab 4: Agent Chat System -->
                    <div id="tabContentConsult" class="hidden flex flex-col h-[520px] bg-[#0c0d15]/30 border border-slate-800/60 rounded-xl overflow-hidden shadow-2xl">
                        <div class="flex-grow overflow-y-auto p-6 space-y-5" id="chatArea">
                            <div class="text-center text-[8px] text-slate-500 uppercase tracking-[0.25em] py-4">Contextual Cognitive Mapping Complete</div>
                        </div>
                        
                        <!-- Conversational input row with glowing interactive focus outline -->
                        <div class="p-3 border-t border-slate-800 bg-[#06070a]/90 flex gap-3 m-2.5 rounded-lg">
                            <input id="chatInput" type="text" placeholder="Query specific statements, commitments, timelines..." class="flex-grow bg-[#0c0d15] border border-slate-800 rounded-lg px-4 py-2.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all">
                            <button onclick="submitChat()" class="bg-gradient-to-r from-indigo-500 to-pink-500 hover:opacity-90 text-white font-bold text-xs px-6 py-2.5 rounded-lg tracking-wider uppercase transition-all shadow-md">QUERY</button>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- DASHBOARD RENDERING & RECONSTRUCTION SCRIPT -->
        <script>
            // Parses unstructured outputs into beautiful visual capsules
            function parseMarkdownList(text, mode) {
                if (!text || text.trim() === "") return `<div class="text-xs text-slate-500">No entries recorded.</div>`;
                const lines = text.split('\\n');
                let html = "";
                
                lines.forEach(line => {
                    const cleaned = line.replace(/^[\\*\\-\\s\\d\\.]+/g, '').trim();
                    if (cleaned.length < 3) return;
                    
                    if (mode === 'decision') {
                        html += `
                            <div class="flex items-start gap-3 bg-emerald-950/10 border border-emerald-900/20 rounded-xl p-4 shadow-sm hover:border-emerald-500/20 transition-all">
                                <span class="h-2 w-2 rounded-full bg-emerald-400 mt-1.5 shrink-0 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></span>
                                <p class="text-xs text-slate-200 leading-relaxed">${cleaned}</p>
                            </div>
                        `;
                    } else if (mode === 'action') {
                        html += `
                            <div class="flex items-start gap-3 bg-pink-950/10 border border-pink-900/20 rounded-xl p-4 shadow-sm hover:border-pink-500/20 transition-all">
                                <div class="h-4 w-4 rounded border border-pink-500/30 flex items-center justify-center shrink-0 mt-0.5 bg-zinc-950 cursor-pointer hover:bg-pink-500/10 transition-colors"></div>
                                <p class="text-xs text-slate-200 leading-relaxed">${cleaned}</p>
                            </div>
                        `;
                    } else if (mode === 'question') {
                        html += `
                            <div class="flex items-start gap-3 bg-amber-950/10 border border-amber-900/20 rounded-xl p-4 shadow-sm hover:border-amber-500/20 transition-all">
                                <span class="h-2 w-2 rounded-full bg-amber-400 mt-1.5 shrink-0 shadow-[0_0_8px_rgba(245,158,11,0.5)]"></span>
                                <p class="text-xs text-slate-200 leading-relaxed">${cleaned}</p>
                            </div>
                        `;
                    }
                });
                return html || `<div class="text-xs text-slate-200">${text}</div>`;
            }

            function switchTab(tabName) {
                document.getElementById(`tabContentSummary`).classList.add('hidden');
                document.getElementById(`tabContentOutcomes`).classList.add('hidden');
                document.getElementById(`tabContentTranscript`).classList.add('hidden');
                document.getElementById(`tabContentConsult`).classList.add('hidden');

                document.getElementById(`tabBtnSummary`).className = "border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all";
                document.getElementById(`tabBtnOutcomes`).className = "border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all";
                document.getElementById(`tabBtnTranscript`).className = "border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all";
                document.getElementById(`tabBtnConsult`).className = "border-b-2 border-transparent pb-3 text-xs font-semibold tracking-wider text-slate-400 hover:text-slate-200 transition-all";

                let formattedTabName = tabName.charAt(0).toUpperCase() + tabName.slice(1);
                document.getElementById(`tabContent${formattedTabName}`).classList.remove('hidden');
                document.getElementById(`tabBtn${formattedTabName}`).className = "border-b-2 border-pink-500 pb-3 text-xs font-semibold tracking-wider text-slate-100 transition-all";
            }

            async function runPipeline() {
                const source = document.getElementById('sourceInput').value;
                const language = document.getElementById('languageInput').value;
                const runBtn = document.getElementById('runBtn');

                if (!source) {
                    alert("A pipeline media source target configuration is required.");
                    return;
                }

                // Layout transitions: Hide dynamic templates, trigger custom animation
                document.getElementById('emptyState').classList.add('hidden');
                document.getElementById('dashboardTabs').classList.add('hidden');
                document.getElementById('loadingState').classList.remove('hidden');

                runBtn.disabled = true;
                runBtn.innerText = "PROCESSING";
                document.getElementById('statusBadge').innerText = "STATUS: PROCESSING";
                document.getElementById('statusBadge').className = "text-[9px] px-3.5 py-1 rounded-full bg-indigo-950/20 border border-indigo-900/40 text-indigo-400 font-bold tracking-[0.1em]";

                try {
                    const response = await fetch('/api/run', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ source, language })
                    });

                    if (!response.ok) throw new Error(await response.text());
                    const data = await response.json();

                    // Map analysis dashboard outcomes
                    document.getElementById('resourceHeader').innerText = data.title;
                    document.getElementById('dataSummary').innerHTML = data.summary;
                    document.getElementById('dataDecisions').innerHTML = parseMarkdownList(data.key_decisions, 'decision');
                    document.getElementById('dataActionItems').innerHTML = parseMarkdownList(data.action_items, 'action');
                    document.getElementById('dataQuestions').innerHTML = parseMarkdownList(data.open_questions, 'question');
                    document.getElementById('dataTranscript').innerText = data.transcript;

                    // Swap active states
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('dashboardTabs').classList.remove('hidden');

                    document.getElementById('statusBadge').innerText = "STATUS: COMPLETE";
                    document.getElementById('statusBadge').className = "text-[9px] px-3.5 py-1 rounded-full bg-emerald-950/20 border border-emerald-900/40 text-emerald-400 font-bold tracking-[0.1em]";
                } catch (err) {
                    alert("Pipeline execution error: " + err.message);
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('emptyState').classList.remove('hidden');
                    document.getElementById('statusBadge').innerText = "STATUS: FAILED";
                    document.getElementById('statusBadge').className = "text-[9px] px-3.5 py-1 rounded-full bg-rose-950/20 border border-rose-900/40 text-rose-400 font-bold tracking-[0.1em]";
                } finally {
                    runBtn.disabled = false;
                    runBtn.innerText = "INITIALIZE ANALYTICS";
                }
            }

            async function submitChat() {
                const inputElement = document.getElementById('chatInput');
                const query = inputElement.value.trim();
                const chatArea = document.getElementById('chatArea');

                if (!query) return;

                // User message rendering
                chatArea.innerHTML += `
                    <div class="flex justify-end">
                        <div class="bg-indigo-950/20 border border-indigo-500/30 rounded-lg px-4.5 py-3 max-w-[70%] text-xs shadow-md">
                            <span class="text-[7px] uppercase tracking-[0.2em] block font-bold text-indigo-400 mb-1.5">CLIENT DELEGATE</span>
                            <span class="text-slate-200 leading-relaxed">${query}</span>
                        </div>
                    </div>
                `;
                inputElement.value = '';
                chatArea.scrollTop = chatArea.scrollHeight;

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: query })
                    });

                    if (!response.ok) throw new Error(await response.text());
                    const data = await response.json();

                    // Assistant response rendering
                    chatArea.innerHTML += `
                        <div class="flex justify-start">
                            <div class="bg-[#0b0c15] border border-slate-800/80 rounded-lg px-4.5 py-3 max-w-[70%] text-xs shadow-md">
                                <span class="text-[7px] uppercase tracking-[0.2em] block font-bold text-pink-400 mb-1.5">SYSTEM COGNITION</span>
                                <span class="text-slate-300 leading-relaxed">${data.answer}</span>
                            </div>
                        </div>
                    `;
                } catch (err) {
                    chatArea.innerHTML += `
                        <div class="flex justify-start">
                            <div class="bg-rose-950/10 border border-rose-900/30 text-rose-400 rounded-lg px-4 py-3 max-w-[70%] text-xs">
                                <span class="text-[7px] uppercase tracking-[0.2em] block font-bold text-rose-500 mb-1.5">ROUTING ERROR</span>
                                Error connecting to backend query endpoint.
                            </div>
                        </div>
                    `;
                } finally {
                    chatArea.scrollTop = chatArea.scrollHeight;
                }
            }
        </script>
    </body>
    </html>
    """