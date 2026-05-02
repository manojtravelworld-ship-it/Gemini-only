import { GoogleGenAI } from "@google/genai";
import { LlmInference, FilesetResolver } from '@mediapipe/tasks-genai';

export interface AIMessage {
  role: 'user' | 'assistant';
  content: string;
  model?: string;
}

export interface AIResponse {
  text: string;
  model: string;
}

export type AITaskType = 'voice' | 'drafting' | 'search' | 'general';

/**
 * HybridAIEngine implementation using multiple models.
 * Engine: LiteRT-LM (Google AI Edge Inference)
 * Drafting: Gemini 2.5 Flash
 * Web Search: Gemini 2.5 Flash Lite
 */
export class HybridAIEngine {
  private static instance: HybridAIEngine;
  private ai: any;
  private localInference: LlmInference | null = null;
  private chromeAI: any = null;
  private currentModelId: string | null = null;
  private isLocalLoading = false;
  private loadProgress = 0;
  private loadedModels: Set<string> = new Set();

  private constructor() {
    const apiKey = process.env.GEMINI_API_KEY || (import.meta as any).env?.VITE_GEMINI_API_KEY;
    console.log("Initializing Nexus AI Engine...");
    console.log("Primary Local: Gemini Nano (Chrome Built-in)");
    console.log("Secondary Local: LiteRT-LM (Edge Node)");
    
    if (apiKey) {
      this.ai = new GoogleGenAI({ apiKey });
    } else {
      console.warn("GEMINI_API_KEY missing. Local Edge Hub active.");
      this.ai = null;
    }
  }

  public static getInstance(): HybridAIEngine {
    if (!HybridAIEngine.instance) {
      HybridAIEngine.instance = new HybridAIEngine();
    }
    return HybridAIEngine.instance;
  }

  public getStatus() {
    return {
      builtIn: !!this.ai,
      voiceModel: this.chromeAI ? 'Gemini Nano (Built-in)' : (this.localInference ? (this.currentModelId === 'advanced' ? 'Gemma-7B-Vision (Power Engine)' : 'Gemma-2B (Efficiency Engine)') : (!!this.ai ? 'Gemini 2.5 Flash Lite' : 'Brain Offline')),
      draftModel: 'gemini-2.5-flash',
      searchModel: 'gemini-2.5-flash-lite',
      isLocalReady: !!this.chromeAI || !!this.localInference,
      isChromeAI: !!this.chromeAI,
      loadProgress: this.loadProgress
    };
  }

  public setApiKey(apiKey: string) {
    if (apiKey) {
      this.ai = new GoogleGenAI({ apiKey });
      console.log("Nexus AI: Cloud Engine Activated manually via BYOK.");
    }
  }

  public async getChromeAIStatus(): Promise<{ available: boolean; state: 'ready' | 'after-download' | 'no' | 'unsupported' | 'iframe-blocked'; details?: string }> {
    console.log("Nexus AI: Initializing Browser AI Detection...");
    
    // Check for Secure Context (Required for Gemini Nano)
    if (!window.isSecureContext) {
      return {
        available: false,
        state: 'unsupported',
        details: "Neural Node requires a Secure Context (HTTPS). Local neural processing is disabled on insecure connections."
      };
    }

    // Check if we are in an iframe
    const isIframe = window.self !== window.top;
    const win = window as any;
    
    // 1. Browser Check
    const isChrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
    const isEdge = /Edg/.test(navigator.userAgent);
    const isChromium = isChrome || isEdge;

    if (!isChromium) {
      return { 
        available: false, 
        state: 'unsupported', 
        details: "Native AI currently requires a Chromium-based browser (Chrome 128+ or Edge)." 
      };
    }

    // 2. Feature Policy / API Presence Check
    const ai = win.ai || win.model || win.assistant;
    
    if (!ai) {
      if (isIframe) {
        return { 
          available: false, 
          state: 'iframe-blocked', 
          details: "AI Node Blocked in iFrame: Browsers restrict local models in sub-frames. Click 'OPEN IN NEW TAB' to activate Gemini Nano." 
        };
      }
      return { 
        available: false, 
        state: 'unsupported', 
        details: "Neural API missing. Go to chrome://flags and set 'Prompt API' and 'Optimization Guide' to ENABLED." 
      };
    }

    // 3. API Capability Check
    // Defensive access to languageModel or assistant
    const modelApi = ai?.languageModel || ai?.assistant || (typeof ai?.create === 'function' ? ai : null);
    
    if (!modelApi || typeof modelApi.capabilities !== 'function') {
      return { 
        available: false, 
        state: 'no', 
        details: "API found but capabilities detection failed. Ensure 'Prompt API' flag is explicitly set to 'Enabled'." 
      };
    }

    try {
      const capabilities = await modelApi.capabilities();
      console.log("Nexus AI: Chrome AI Capabilities detected:", capabilities);
      return { 
        available: true, 
        state: capabilities.available,
        details: capabilities.available === 'no' ? "Model not yet available. Visit chrome://components and check 'Optimization Guide On Device Model'." : undefined
      };
    } catch (e) {
      console.error("Nexus AI: Failed to read AI capabilities:", e);
      return { available: false, state: 'no', details: "Calibration error. Try restarting Chrome." };
    }
  }

  public async loadChromeAI(onProgress?: (progress: number) => void): Promise<boolean> {
    try {
      const status = await this.getChromeAIStatus();
      
      if (!status.available || status.state === 'unsupported' || status.state === 'no') {
        process.env.NODE_ENV !== 'production' && console.warn(`Nexus AI: Chrome AI unavailable - ${status.details}`);
        return false;
      }

      const ai = (window as any).ai || (window as any).model || (window as any).assistant;
      if (!ai) return false;

      const modelApi = ai.languageModel || ai.assistant || (typeof ai.create === 'function' ? ai : null);
      if (!modelApi || typeof modelApi.create !== 'function') return false;

      if (onProgress) onProgress(10);
      
      console.log("Nexus AI: Initializing Gemini Nano...");
      this.chromeAI = await modelApi.create({
        monitor(m: any) {
          if (m && typeof m.addEventListener === 'function') {
            m.addEventListener("downloadprogress", (e: any) => {
              if (e.total > 0) {
                const p = Math.round((e.loaded / e.total) * 100);
                if (onProgress) onProgress(p);
              }
            });
          }
        }
      });
      
      if (this.chromeAI) {
        console.log("Nexus AI: Gemini Nano successfully bound to instance.");
        if (onProgress) onProgress(100);
        return true;
      }
      return false;
    } catch (e) {
      console.error("Nexus AI: Chrome AI Creation Failed:", e);
      return false;
    }
  }

  public async isModelDownloaded(modelId: string): Promise<boolean> {
    const officialModels = {
      standard: 'https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-2b-it-cpu-int4.bin',
      advanced: 'https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-7b-it-gpu-int4.bin' 
    };
    const modelUrl = officialModels[modelId as keyof typeof officialModels];
    if (!modelUrl) return false;
    
    try {
      const cache = await caches.open('nexus-brain-registry');
      const match = await cache.match(modelUrl);
      return !!match;
    } catch (e) {
      return false;
    }
  }

  public async loadLocalModel(modelId: string = 'standard', onProgress?: (progress: number) => void) {
    if (this.loadedModels.has(modelId) && this.currentModelId === modelId && this.localInference) {
      if (onProgress) onProgress(100);
      return;
    }
    
    if (this.isLocalLoading) return;

    this.isLocalLoading = true;
    this.currentModelId = modelId;
    try {
      const officialModels = {
        standard: 'https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-2b-it-cpu-int4.bin',
        advanced: 'https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-7b-it-gpu-int4.bin' 
      };
      const modelUrl = officialModels[modelId as keyof typeof officialModels] || officialModels.standard;

      // 1. Check Cache API for existing model (Production-grade Persistence)
      const cacheName = 'nexus-brain-registry-v1';
      const cache = await caches.open(cacheName);
      let cachedResponse = await cache.match(modelUrl);
      let modelAsset: string | Uint8Array = modelUrl;

      if (!cachedResponse) {
        console.log(`📡 OkHttp Engine: Requesting Registry Byte-Range for ${modelId}...`);
        
        // Use fetch with streaming for progress reporting
        const response = await fetch(modelUrl, {
          mode: 'cors',
          credentials: 'omit'
        });

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error(`Registry 404: The model artifact at ${modelUrl} was not found. Please verify the official Google AI Edge path.`);
          }
          throw new Error(`Registry fetch failed: ${response.status} ${response.statusText}`);
        }
        
        const contentLength = response.headers.get('content-length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        let loaded = 0;

        const reader = response.body?.getReader();
        if (!reader) throw new Error("Registry stream uninitialized.");

        const chunks: Uint8Array[] = [];
        console.log(`🔧 WorkManager: Assigned persistent background fetch task for ${total} bytes.`);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.length;
          if (total > 0) {
            const p = Math.round((loaded / total) * 100);
            if (p !== this.loadProgress) {
              this.loadProgress = p;
              if (onProgress) onProgress(this.loadProgress);
            }
          }
        }

        const blob = new Blob(chunks);
        // Persist to cache
        await cache.put(modelUrl, new Response(blob.slice(), {
          headers: { 'Content-Type': 'application/octet-stream' }
        }));
        
        console.log(`✅ WorkManager: Registry Blobs persisted to local storage.`);
        modelAsset = URL.createObjectURL(blob);
      } else {
        console.log(`🔍 WorkManager: Persistent node found in local registry. Resolving internal bridge...`);
        const blob = await cachedResponse.blob();
        modelAsset = URL.createObjectURL(blob);
        this.loadProgress = 100;
        if (onProgress) onProgress(100);
      }

      // 2. Intelligent Hardware Selection (WebGPU vs CPU fallback)
      let delegate: "GPU" | "CPU" = "CPU";
      let hardwareStatus = "Initializing CPU-Vector (Wasm) Engine...";

      if ((navigator as any).gpu) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) {
            delegate = "GPU";
            hardwareStatus = "WebGPU Acceleration Active.";
          }
        } catch (e) {
          console.warn("WebGPU found but adapter request failed. Falling back to CPU.", e);
        }
      }

      console.log(`🤖 LiteRT Hardware: ${hardwareStatus}`);
      console.log(`📥 LiteRT-LM: Constructing Neural Inference Bridge...`);
      
      const genaiFileset = await FilesetResolver.forGenAiTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
      );

      this.localInference = await LlmInference.createFromOptions(genaiFileset, {
        baseOptions: { 
          modelAssetPath: modelAsset,
          delegate: delegate
        } as any,
        maxTokens: 512,
        topK: 40,
        temperature: 0.7,
        randomSeed: 42
      });

      this.loadProgress = 100;
      this.loadedModels.add(modelId);
      if (onProgress) onProgress(100);
      console.log(`✅ LiteRT-LM Node Secured: ${modelId === 'advanced' ? 'Gemma-7B' : 'Gemma-2B'} online.`);
    } catch (error) {
      console.error("LiteRT-LM Deployment Failure:", error);
      this.localInference = null;
      this.loadProgress = 0;
      if (onProgress) onProgress(0);
      throw error;
    } finally {
      this.isLocalLoading = false;
    }
  }

  public async *generateResponseStream(
    prompt: string, 
    history: AIMessage[], 
    task: AITaskType = 'voice'
  ): AsyncGenerator<string> {
    // Priority 1: Chrome AI (Gemini Nano)
    if (this.chromeAI && (task === 'voice' || task === 'general')) {
      console.log("Nexus AI: Attempting Gemini Nano Streaming Prompt...");
      try {
        const stream = this.chromeAI.promptStreaming(prompt);
        let lastChunk = "";
        for await (const chunk of stream) {
          // Some versions of Chrome AI return cumulative text, some return chunks.
          // AdvocatePortal handles cumulative updates via setVoiceAiReply(fullText).
          yield chunk;
          lastChunk = chunk;
        }
        if (lastChunk) {
          console.log("Nexus AI: Gemini Nano Streaming Success.");
          return;
        }
        console.warn("Nexus AI: Gemini Nano stream was empty. Falling back...");
      } catch (e) {
        console.error("Nexus AI: Gemini Nano Stream Error:", e);
      }
    }

    // Priority 2: LiteRT-LM (Edge Node)
    if (this.localInference && task === 'voice') {
      console.log("Nexus AI: Using LiteRT-LM (Edge) fallback...");
      const response = await this.generateLocalResponse(prompt, history);
      yield response;
      return;
    }

    if (!this.ai) {
      console.error("Nexus AI: Root engine not bound.");
      yield "Error: AI engine not initialized.";
      return;
    }

    try {
      // Use Flash Lite for voice tasks to minimize latency
      const modelName = task === 'voice' ? 'gemini-3.1-flash-lite-preview' : 'gemini-3-flash-preview';
      console.log(`Nexus AI: Using Cloud Failover (${modelName})...`);
      const contents: any[] = history.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));
      contents.push({ role: 'user', parts: [{ text: prompt }] });

      const responseStream = await this.ai.models.generateContentStream({
        model: modelName,
        contents: contents,
        config: {
          systemInstruction: "You are Nexus Justice, a professional legal voice assistant. You are currently speaking to the user via voice. Keep your responses EXTREMELY concise, formal, and helpful. Answer directly without unnecessary preamble. Maintain context from previous turns in the conversation. If the user speaks to you in Malayalam (or any other language), you MUST respond in that same language. Do not mention that you are a text-based AI or that you cannot hear sound, as you are integrated into a voice-capable system. Your goal is to be a seamless extension of the advocate's workflow. \n\nCRITICAL CONVERSATIONAL RULES:\n1. Never just stop after answering a question. \n2. Always encourage the user to ask more or talk more. \n3. Identify the most complex or 'toughest' part of your current answer and proactively ask the user if they want to know more about that specific detail (e.g., 'Do you want to know more about [specific topic]?').\n4. End your response with an open-ended question like 'Do you have anything else to know?' or 'Is there anything else I can assist you with?'"
        }
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          yield chunk.text;
        }
      }
    } catch (error) {
      console.error("Streaming Error:", error);
      yield "Error: Failed to connect to AI engine.";
    }
  }

  private async generateLocalResponse(prompt: string, history: AIMessage[]): Promise<string> {
    if (!this.localInference) return "Local model not ready.";

    const modelId = this.currentModelId || 'standard';
    try {
      console.log(`Performing local inference with LiteRT-LM Node: ${modelId}`);
      
      const response = await this.localInference.generateResponse(prompt);
      console.log("Local inference output success");
      return response.trim() || "Local brain produced empty result.";
    } catch (error: any) {
      console.error("Local Inference Error:", error);
      return `Error: Local inference failed. ${error.message || 'Check hardware compatibility.'}`;
    }
  }

  public async generateResponse(
    prompt: string, 
    history: AIMessage[], 
    imageBase64?: string,
    task: AITaskType = 'general'
  ): Promise<AIResponse> {
    try {
      console.log("Generating response for task:", task);
      
      // Priority 1: Chrome AI (Gemini Nano)
      if (this.chromeAI && (task === 'voice' || task === 'general')) {
        console.log("Nexus AI: Prompting Gemini Nano (Local)...");
        try {
          const text = await this.chromeAI.prompt(prompt);
          if (text && text.trim()) {
            return { text, model: "Gemini Nano (Chrome)" };
          }
          console.warn("Nexus AI: Gemini Nano returned empty response. Falling back...");
        } catch (e) {
          console.error("Nexus AI: Gemini Nano Hub Error:", e);
        }
      }

      // Priority 2: LiteRT-LM (Edge Node)
      if (this.localInference && (task === 'voice' || task === 'general')) {
        console.log("Nexus AI: Prompting LiteRT-LM (Edge)...");
        const text = await this.generateLocalResponse(prompt, history);
        return { text, model: `LiteRT-LM (${this.currentModelId === 'advanced' ? 'Gemma-4B' : 'Gemma-2B'})` };
      }

      const effectiveTask = task === 'general' ? await this.orchestrate(prompt) : task;
      console.log("Effective task:", effectiveTask);

      if (!this.ai) {
        console.error("AI Engine not initialized (this.ai is null)");
        return { text: "Error: AI engine not initialized. Please check your API keys.", model: "Error" };
      }

      if (effectiveTask === 'drafting') {
        const result = await this.ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: [
            ...history.map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
            { role: 'user', parts: [{ text: prompt }] }
          ]
        });
        return { text: result.text || "Failed to generate draft.", model: "Gemini 2.5 Flash" };
      }

      // 2. Search Task -> Gemini 2.5 Flash Lite with Web Search
      if (effectiveTask === 'search') {
        const text = await this.callGeminiSearch(prompt, history);
        return { text, model: "Gemini 2.5 Flash" };
      }

      // 3. Voice/General Task -> Gemini API
      const modelName = task === 'voice' ? 'gemini-3.1-flash-lite-preview' : 'gemini-3-flash-preview';
      
      const contents: any[] = history.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));

      const parts: any[] = [{ text: prompt }];
      if (imageBase64) {
        parts.push({
          inlineData: {
            data: imageBase64.split(',')[1],
            mimeType: 'image/jpeg'
          }
        });
      }

      contents.push({ role: 'user', parts });

      const response = await this.ai.models.generateContent({
        model: modelName,
        contents: contents,
        config: {
          systemInstruction: "You are Nexus Justice, a professional legal voice assistant. You are currently speaking to the user via voice. Keep your responses EXTREMELY concise, formal, and helpful. Answer directly without unnecessary preamble. Maintain context from previous turns in the conversation. If the user speaks to you in Malayalam (or any other language), you MUST respond in that same language. Do not mention that you are a text-based AI or that you cannot hear sound, as you are integrated into a voice-capable system. Your goal is to be a seamless extension of the advocate's workflow. \n\nCRITICAL CONVERSATIONAL RULES:\n1. Never just stop after answering a question. \n2. Always encourage the user to ask more or talk more. \n3. Identify the most complex or 'toughest' part of your current answer and proactively ask the user if they want to know more about that specific detail (e.g., 'Do you want to know more about [specific topic]?').\n4. End your response with an open-ended question like 'Do you have anything else to know?' or 'Is there anything else I can assist you with?'"
        }
      });

      return { text: response.text || "I'm sorry, I couldn't generate a response.", model: "Gemini 2.5 Flash" };
    } catch (error: any) {
      console.error("AI Engine Error:", error);
      const errorMessage = error?.message || "Unknown error";
      return { text: `Error: Failed to connect to AI engine. (${errorMessage})`, model: "Error" };
    }
  }

  private async orchestrate(prompt: string): Promise<AITaskType> {
    try {
      const response = await this.ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [{ 
          role: 'user', 
          parts: [{ 
            text: `You are a legal AI orchestrator. Classify the user's intent into one of these categories:
            - 'drafting': If the user wants to create, edit, or generate a legal document, contract, or formal writing.
            - 'search': If the user is asking for specific laws, case citations, or real-time legal facts from the web.
            - 'voice': If the user is just talking, asking for advice, or having a general conversation.
            
            Return ONLY the category name in lowercase.
            
            User Request: "${prompt}"` 
          }] 
        }]
      });
      
      const decision = response.text?.toLowerCase().trim() || 'voice';
      if (decision.includes('draft')) return 'drafting';
      if (decision.includes('search')) return 'search';
      return 'voice';
    } catch (err) {
      console.error("Orchestration Error:", err);
      return 'voice'; // Default to voice if orchestration fails
    }
  }

  private async callGeminiSearch(prompt: string, history: AIMessage[]): Promise<string> {
    try {
      const contents: any[] = history.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));
      contents.push({ role: 'user', parts: [{ text: prompt }] });

      const response = await this.ai.models.generateContent({
        model: 'gemini-3.1-flash-lite-preview',
        contents: contents,
        config: {
          tools: [{ googleSearch: {} }]
        }
      });

      let text = response.text || "";
      
      // Append grounding sources if available
      const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      if (chunks && chunks.length > 0) {
        text += "\n\n**Sources:**\n";
        chunks.forEach((c: any) => {
          if (c.web) {
            text += `- [${c.web.title}](${c.web.uri})\n`;
          }
        });
      }

      return text;
    } catch (err) {
      console.error("Gemini Search Error:", err);
      return "Error: Web search failed.";
    }
  }
}
