// src/types.ts
export enum Provider {
    Google = 'google',
    OpenAI = 'openai',
}

export enum ModelName {
    // Google models
    Gemini2Flash = 'Gemini 2.0 Flash',
    Gemini15 = 'Gemini 1.5',

    // OpenAI models
    GPT4oMini = 'GPT-4o-mini',
}

export interface ModelCapabilities {
    structuredOutput: boolean;
    batchRequests: boolean;
}

export interface ModelInfo {
    id: string;
    name: ModelName;
    provider: Provider;
    capabilities: ModelCapabilities;
}

export type ModelType = `${Provider.Google}/${ModelName.Gemini2Flash}` | `${Provider.Google}/${ModelName.Gemini15}` | `${Provider.OpenAI}/${ModelName.GPT4oMini}`;

export interface UsageInfo {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
}

export interface CompletionResponse<T = string> {
    text: string;
    data: T;
    usage: UsageInfo;
    model: ModelType;
    provider: Provider;
}

export interface BatchStatus {
    id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    outputFileId?: string;
    error?: string;
}

export type PromptMessage = {
    role: 'developer' | 'user' | 'assistant';
    content: string;
};

// Provider interface instead of an abstract class
// Common options for LLM requests
export interface LLMRequestOptions<T = any> {
    // Generation parameters
    temperature?: number; // Controls randomness (0-1, default: 0.7)
    topK?: number; // Limits token selection to top K options
    topP?: number; // Nucleus sampling parameter (0-1)
    maxTokens?: number; // Maximum number of tokens to generate
    frequencyPenalty?: number; // Penalizes repeated tokens (0-2)
    presencePenalty?: number; // Penalizes tokens already in the prompt (0-2)

    // Structured output
    sampleObj?: T; // Sample object for structured output formatting

    // Metadata
    agentName?: string; // Name of the agent making the request
    user?: string; // End-user identifier for tracking/monitoring

    // Provider-specific parameters
    providerParams?: Record<string, any>; // For provider-specific options
}

// Options specific to batch operations
export interface BatchRequestOptions<T = any> extends LLMRequestOptions<T> {
    batchName?: string; // Custom name for the batch
    priority?: 'high' | 'normal' | 'low'; // Priority level for batch processing
    timeoutSeconds?: number; // Timeout for batch completion
    failureThreshold?: number; // Percentage of allowable failures before cancelling
}

export interface LLMProvider {
    getModels(): ModelInfo[];

    prompt<T = string>(messages: PromptMessage[], model: ModelName, options?: LLMRequestOptions<T>): Promise<CompletionResponse<T>>;

    createBatch<T = string>(prompts: PromptMessage[][], model: ModelName, options?: BatchRequestOptions<T>): Promise<string>; // Returns batch ID

    retrieveBatch<T = string>(batchId: string, model: ModelName, sampleObj?: T): Promise<CompletionResponse<T>[]>;

    checkBatch(batchId: string, model: ModelName): Promise<BatchStatus>;

    cancelBatch(batchId: string, model: ModelName): Promise<boolean>;
}
