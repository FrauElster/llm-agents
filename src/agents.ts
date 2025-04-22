// src/agents.ts
import {
    Provider,
    type ModelName,
    type ModelType,
    type PromptMessage,
    type CompletionResponse,
    type BatchStatus,
    type LLMRequestOptions,
    type BatchRequestOptions,
    type BatchRequest,
    isBatchRequest,
} from './types.js';
import { GoogleProvider } from './providers/google.js';
import { OpenAIProvider } from './providers/openai.js';

export class LLMAgent<T = string> {
    name: string;
    model: ModelType;
    modelName: ModelName;
    provider: Provider;
    basePrompt: string;
    private apiKey: string;
    private baseUrl?: string;
    private fetchFn: typeof fetch;
    private providerInstance: GoogleProvider | OpenAIProvider;
    private sampleObj?: T;

    constructor(options: {
        name: string;
        basePrompt: string;
        model: ModelType;
        apiKey: string;
        baseUrl?: string;
        fetchFn?: typeof fetch;
        sampleObj?: T;
    }) {
        this.name = options.name;
        this.basePrompt = options.basePrompt;
        this.model = options.model;
        this.apiKey = options.apiKey;
        this.baseUrl = options.baseUrl;
        this.fetchFn = options.fetchFn || fetch;

        // Parse the model type into provider and model name
        const [providerStr, modelStr] = options.model.split('/');
        this.provider = providerStr as Provider;
        this.modelName = modelStr as ModelName;

        // Initialize the provider
        if (this.provider === Provider.Google) {
            this.providerInstance = new GoogleProvider(this.apiKey, this.baseUrl, this.fetchFn);
        } else if (this.provider === Provider.OpenAI) {
            this.providerInstance = new OpenAIProvider(this.apiKey, this.baseUrl, this.fetchFn);
        } else {
            throw new Error(`Unsupported provider: ${this.provider}`);
        }

        if (options.sampleObj) {
            this.validateBasePrompt(options.sampleObj);
            this.sampleObj = options.sampleObj;
        }
    }

    /**
     * Validates that all keys in the sample object are mentioned in the base prompt
     * This helps ensure the LLM will include all expected fields in responses
     */
    private validateBasePrompt(sampleObj: T): void {
        // Skip validation for primitive types
        if (typeof sampleObj !== 'object' || sampleObj === null) {
            return;
        }

        const basePromptLower = this.basePrompt.toLowerCase();
        const missingKeys: string[] = [];

        // Helper function to check keys recursively
        const checkKeys = (obj: any, prefix = '') => {
            if (Array.isArray(obj)) {
                // For arrays, check the first item if it exists
                if (obj.length > 0 && typeof obj[0] === 'object' && obj[0] !== null) {
                    checkKeys(obj[0], prefix);
                }
            } else if (typeof obj === 'object' && obj !== null) {
                // For objects, check each key
                for (const key of Object.keys(obj)) {
                    const fullKey = prefix ? `${prefix}.${key}` : key;

                    // Check if the key is mentioned in the base prompt
                    if (!basePromptLower.includes(key.toLowerCase())) {
                        missingKeys.push(fullKey);
                    }

                    // Recursively check nested objects
                    const value = obj[key];
                    if (typeof value === 'object' && value !== null) {
                        checkKeys(value, fullKey);
                    }
                }
            }
        };

        // Start recursive key check
        checkKeys(sampleObj);

        // Warn about missing keys
        if (missingKeys.length > 0) {
            console.warn(`⚠️ Warning: The following fields are not mentioned in the base prompt and may be ignored by the LLM: ${missingKeys.join(', ')}`);
            console.warn('Consider adding these fields to your base prompt for better results.');
        }
    }

    /**
     * Helper to prepend the base prompt to messages
     */
    private prependBasePrompt(messages: PromptMessage[]): PromptMessage[] {
        if (!this.basePrompt) {
            return messages;
        }

        return [
            { role: 'developer', content: this.basePrompt },
            ...messages.filter(m => m.role !== 'developer'), // Remove any existing system messages
        ];
    }

    /**
     * Send a prompt to the model and get a completion
     */
    async prompt(messages: PromptMessage[], options: Omit<LLMRequestOptions<T>, 'sampleObj'> = {}): Promise<CompletionResponse<T>> {
        const messagesWithBasePrompt = this.prependBasePrompt(messages);

        // Add agent name to options
        const optionsWithAgent = {
            ...options,
            agentName: this.name,
            sampleObj: this.sampleObj,
        };

        return this.providerInstance.prompt<T>(messagesWithBasePrompt, this.modelName, optionsWithAgent);
    }

    /**
     * Create a batch of prompts
     */
    async createBatch(prompts: PromptMessage[][] | BatchRequest[], options: Omit<BatchRequestOptions<T>, 'sampleObj'> = {}): Promise<string> {
        // Add base prompt to each message set
        for (const idx in prompts) {
            const prompt = prompts[idx];
            if (isBatchRequest(prompt)) {
                prompts[idx] = { ...prompt, messages: this.prependBasePrompt(prompt.messages) };
            } else {
                prompts[idx] = this.prependBasePrompt(prompt);
            }
        }

        // Add agent name to options
        const optionsWithAgent = {
            ...options,
            agentName: this.name,
            sampleObj: this.sampleObj,
        };

        try {
            return await this.providerInstance.createBatch<T>(prompts, this.modelName, optionsWithAgent);
        } catch (error) {
            // If this is a Google provider and it doesn't support batching, throw a more specific error
            if (this.provider === Provider.Google) {
                throw new Error(`The agent "${this.name}" uses Google's ${this.modelName} which doesn't support batch operations`);
            }
            throw error;
        }
    }

    /**
     * Check the status of a batch
     */
    async checkBatch(batchId: string): Promise<BatchStatus> {
        try {
            return await this.providerInstance.checkBatch(batchId, this.modelName);
        } catch (error) {
            // If this is a Google provider and it doesn't support batching, throw a more specific error
            if (this.provider === Provider.Google) {
                throw new Error(`The agent "${this.name}" uses Google's ${this.modelName} which doesn't support batch operations`);
            }
            throw error;
        }
    }

    /**
     * Retrieve results from a completed batch
     */
    async retrieveBatch(batchId: string): Promise<CompletionResponse<T>[]> {
        try {
            return await this.providerInstance.retrieveBatch<T>(batchId, this.modelName, this.sampleObj);
        } catch (error) {
            // If this is a Google provider and it doesn't support batching, throw a more specific error
            if (this.provider === Provider.Google) {
                throw new Error(`The agent "${this.name}" uses Google's ${this.modelName} which doesn't support batch operations`);
            }
            throw error;
        }
    }

    /**
     * Cancel a batch that is in progress
     */
    async cancelBatch(batchId: string): Promise<boolean> {
        try {
            return await this.providerInstance.cancelBatch(batchId, this.modelName);
        } catch (error) {
            // If this is a Google provider and it doesn't support batching, throw a more specific error
            if (this.provider === Provider.Google) {
                throw new Error(`The agent "${this.name}" uses Google's ${this.modelName} which doesn't support batch operations`);
            }
            throw error;
        }
    }
}

/**
 * Helper factory function to create agents
 */
export function createAgent<T = string>(options: {
    name: string;
    basePrompt: string;
    model: ModelType;
    apiKey: string;
    baseUrl?: string;
    fetchFn?: typeof fetch;
    sampleObj?: T;
}): LLMAgent<T> {
    return new LLMAgent<T>(options);
}
