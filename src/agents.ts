// src/agents.ts
import { Provider, type ModelName, type ModelType, type PromptMessage, type CompletionResponse, type BatchStatus } from './types.ts';
import { GoogleProvider } from './providers/google.ts';
import { OpenAIProvider } from './providers/openai.ts';

export class LLMAgent<T = string> {
    private name: string;
    private basePrompt: string;
    private model: ModelType;
    private provider: Provider;
    private modelName: ModelName;
    private apiKey: string;
    private baseUrl?: string;
    private fetchFn: typeof fetch;
    private providerInstance: GoogleProvider | OpenAIProvider;

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
     * Get the name of this agent
     */
    getName(): string {
        return this.name;
    }

    /**
     * Get the model being used by this agent
     */
    getModel(): ModelType {
        return this.model;
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
    async prompt(messages: PromptMessage[], options: any = {}): Promise<CompletionResponse<T>> {
        const messagesWithBasePrompt = this.prependBasePrompt(messages);

        // Add agent name to options
        const optionsWithAgent = {
            ...options,
            agentName: this.name,
        };

        return this.providerInstance.prompt<T>(messagesWithBasePrompt, this.modelName, optionsWithAgent);
    }

    /**
     * Create a batch of prompts
     */
    async createBatch(prompts: PromptMessage[][], options: any = {}): Promise<string> {
        // Add base prompt to each message set
        const promptsWithBasePrompt = prompts.map(messages => this.prependBasePrompt(messages));

        // Add agent name to options
        const optionsWithAgent = {
            ...options,
            agentName: this.name,
        };

        try {
            return await this.providerInstance.createBatch<T>(promptsWithBasePrompt, this.modelName, optionsWithAgent);
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
    async retrieveBatch(batchId: string, sampleObj?: T): Promise<CompletionResponse<T>[]> {
        try {
            return await this.providerInstance.retrieveBatch<T>(batchId, this.modelName, sampleObj);
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
}): LLMAgent<T> {
    return new LLMAgent<T>(options);
}
