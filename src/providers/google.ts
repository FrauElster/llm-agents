// src/providers/google.ts
import {
    type ModelInfo,
    ModelName,
    Provider,
    type PromptMessage,
    type CompletionResponse,
    type BatchStatus,
    type LLMProvider,
    type LLMRequestOptions,
    type BatchRequestOptions,
    type ModelType,
} from '../types.ts';

// Define Google-specific types
namespace GoogleTypes {
    export interface GeminiPromptRequest {
        contents: {
            role: string;
            parts: {
                text: string;
            }[];
        }[];
        generationConfig: {
            temperature?: number;
            topK?: number;
            topP?: number;
            maxOutputTokens?: number;
            response_mime_type?: string;
            response_schema?: {
                type: string;
                properties?: Record<string, any>;
                items?: any;
            };
        };
    }

    export interface GeminiPromptResponse {
        candidates: {
            content: {
                parts: {
                    text: string;
                }[];
            };
            finishReason: string;
        }[];
        usageMetadata: {
            promptTokenCount: number;
            candidatesTokenCount: number;
            totalTokenCount: number;
        };
    }
}

export class GoogleProvider implements LLMProvider {
    private apiKey: string;
    private baseUrl: string;
    private fetchFn: typeof fetch;

    constructor(apiKey: string, baseUrl = 'https://generativelanguage.googleapis.com/v1', fetchFn: typeof fetch = fetch) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.fetchFn = fetchFn;
    }

    getModels(): ModelInfo[] {
        return [
            {
                id: 'gemini-2.0-flash',
                name: ModelName.Gemini2Flash,
                provider: Provider.Google,
                capabilities: {
                    structuredOutput: true,
                    batchRequests: false,
                },
            },
            {
                id: 'gemini-1.5-pro',
                name: ModelName.Gemini15,
                provider: Provider.Google,
                capabilities: {
                    structuredOutput: true,
                    batchRequests: false,
                },
            },
        ];
    }

    private formatMessages(messages: PromptMessage[]): GoogleTypes.GeminiPromptRequest['contents'] {
        return messages.map(msg => ({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content }],
        }));
    }

    private parseJsonResponse<T>(text: string): T {
        try {
            return JSON.parse(text) as T;
        } catch (error) {
            throw new Error(`Failed to parse JSON response: ${error}`);
        }
    }

    private enhancePromptWithStructure<T>(messages: PromptMessage[], sampleObj: T): PromptMessage[] {
        // Create a copy of the messages
        const enhancedMessages = [...messages];

        // Get the structure of the sample object
        const structure =
            typeof sampleObj === 'object'
                ? JSON.stringify(
                      sampleObj,
                      (key, value) => {
                          if (Array.isArray(value) && value.length > 0) {
                              // For arrays, return an array with a single example item
                              return [typeof value[0] === 'object' ? {} : 'example'];
                          }
                          return value;
                      },
                      2,
                  )
                : typeof sampleObj;

        // Add structure information to the last user message, or add a new one
        const lastUserMessageIndex = enhancedMessages.findLastIndex(m => m.role === 'user');

        if (lastUserMessageIndex >= 0) {
            enhancedMessages[lastUserMessageIndex] = {
                ...enhancedMessages[lastUserMessageIndex],
                content: `${enhancedMessages[lastUserMessageIndex].content}\n\nPlease provide the response in the following JSON structure: ${structure}`,
            };
        } else {
            enhancedMessages.push({
                role: 'user',
                content: `Please provide the response in the following JSON structure: ${structure}`,
            });
        }

        return enhancedMessages;
    }

    async prompt<T = string>(messages: PromptMessage[], model: ModelName, options: LLMRequestOptions<T> = {}): Promise<CompletionResponse<T>> {
        const modelInfo = this.getModels().find(m => m.name === model);
        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }

        let formattedMessages = messages;

        // If T is not string and the model supports structured output, use it
        const needsStructuredOutput = options.sampleObj && typeof options.sampleObj !== 'string';

        // Prepare request
        const requestBody: GoogleTypes.GeminiPromptRequest = {
            contents: this.formatMessages(formattedMessages),
            generationConfig: {
                temperature: options.temperature ?? 0.7,
                topK: options.topK,
                topP: options.topP,
                maxOutputTokens: options.maxTokens,
            },
        };

        // Add JSON response format if needed
        let baseUrl = this.baseUrl;
        if (needsStructuredOutput) {
            baseUrl = baseUrl.replace(/\/v1$/, '/v1beta');
            // Always set response_mime_type when we want JSON
            requestBody.generationConfig.response_mime_type = 'application/json';

            if (modelInfo.capabilities.structuredOutput) {
                // Add schema information for structured output
                requestBody.generationConfig.response_schema = this.objectToGoogleSchema(options.sampleObj);
            } else {
                // Enhance prompt with structure information
                formattedMessages = this.enhancePromptWithStructure(messages, options.sampleObj);
                requestBody.contents = this.formatMessages(formattedMessages);
            }
        }

        // Make API request
        const response = await this.fetchFn(`${baseUrl}/models/${modelInfo.id}:generateContent?key=${this.apiKey}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Google API error: ${JSON.stringify(error)}`);
        }

        const responseData = (await response.json()) as GoogleTypes.GeminiPromptResponse;
        const text = responseData.candidates[0].content.parts[0].text;

        let data: T;
        if (needsStructuredOutput) {
            try {
                data = this.parseJsonResponse<T>(text);
            } catch (error) {
                data = text as unknown as T;
            }
        } else {
            data = text as unknown as T;
        }

        return {
            text,
            data,
            usage: {
                promptTokens: responseData.usageMetadata.promptTokenCount,
                completionTokens: responseData.usageMetadata.candidatesTokenCount,
                totalTokens: responseData.usageMetadata.totalTokenCount,
            },
            model: `${Provider.Google}/${model}` as ModelType,
            provider: Provider.Google,
        };
    }

    async createBatch<T = string>(prompts: PromptMessage[][], model: ModelName, options: BatchRequestOptions<T> = {}): Promise<string> {
        // Check for batch support
        const modelInfo = this.getModels().find(m => m.name === model);

        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }

        // Throw error as Google Gemini API doesn't support batch requests as of April 2025
        throw new Error(`Batch requests are not supported by the Google Gemini API for model ${model}`);
    }

    async checkBatch(batchId: string, model: ModelName): Promise<BatchStatus> {
        // Check for batch support
        const modelInfo = this.getModels().find(m => m.name === model);

        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }

        // Throw error as Google Gemini API doesn't support batch requests as of April 2025
        throw new Error(`Batch operations are not supported by the Google Gemini API for model ${model}`);
    }

    async retrieveBatch<T = string>(batchId: string, model: ModelName, sampleObj?: T): Promise<CompletionResponse<T>[]> {
        // Check for batch support
        const modelInfo = this.getModels().find(m => m.name === model);

        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }

        // Throw error as Google Gemini API doesn't support batch requests as of April 2025
        throw new Error(`Batch operations are not supported by the Google Gemini API for model ${model}`);
    }

    async cancelBatch(batchId: string, model: ModelName): Promise<boolean> {
        // Check for batch support
        const modelInfo = this.getModels().find(m => m.name === model);

        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }

        // Throw error as Google Gemini API doesn't support batch requests as of April 2025
        throw new Error(`Batch operations are not supported by the Google Gemini API for model ${model}`);
    }

    private objectToGoogleSchema<T>(sampleObj: T): any {
        if (sampleObj === null || sampleObj === undefined) {
            return null;
        }

        // Handle primitive types
        if (typeof sampleObj !== 'object') {
            return { type: this.getGoogleSchemaType(typeof sampleObj) };
        }

        // Handle arrays
        if (Array.isArray(sampleObj)) {
            if (sampleObj.length === 0) {
                // Empty array - can't infer type, default to array of strings
                return {
                    type: 'ARRAY',
                    items: { type: 'STRING' },
                };
            }

            // Use the first item to determine array type
            return {
                type: 'ARRAY',
                items: this.objectToGoogleSchema(sampleObj[0]),
            };
        }

        // Handle objects
        const properties: Record<string, any> = {};
        for (const [key, value] of Object.entries(sampleObj)) {
            properties[key] = this.objectToGoogleSchema(value);
        }

        return {
            type: 'OBJECT',
            properties,
        };
    }

    /**
     * Map JavaScript types to Google schema types
     */
    private getGoogleSchemaType(jsType: string): string {
        switch (jsType) {
            case 'string':
                return 'STRING';
            case 'number':
                return 'NUMBER';
            case 'boolean':
                return 'BOOLEAN';
            default:
                return 'STRING'; // Default fallback
        }
    }
}
