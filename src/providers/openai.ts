// src/providers/openai.ts
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
    type BatchRequest,
    type CreateBatchResponse,
} from '../types.js';

// Define OpenAI-specific types
namespace OpenAITypes {
    export interface ChatCompletionRequest {
        model: string;
        messages: {
            role: string;
            content: string;
        }[];
        temperature?: number;
        max_tokens?: number;
        top_p?: number;
        frequency_penalty?: number;
        presence_penalty?: number;
        response_format?: {
            type: 'json_object' | 'text';
        };
        user?: string;
        seed?: number;
    }

    export interface ChatCompletionResponse {
        id: string;
        object: string;
        created: number;
        model: string;
        choices: {
            index: number;
            message: {
                role: string;
                content: string;
            };
            finish_reason: string;
        }[];
        usage: {
            prompt_tokens: number;
            completion_tokens: number;
            total_tokens: number;
        };
    }

    export interface FileUploadResponse {
        id: string;
        object: 'file';
        bytes: number;
        created_at: number;
        filename: string;
        purpose: string;
    }

    export interface BatchRequest {
        custom_id: string;
        method: 'POST';
        url: string;
        body: ChatCompletionRequest;
    }

    export interface BatchCreateResponse {
        id: string;
        object: 'batch';
        created_at: number;
        endpoint: string;
        status: 'validating' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'cancelling' | 'expired';
        input_file_id: string;
        completion_window: string;
        error_file_id?: string;
        output_file_id?: string;
        request_counts: {
            total: number;
            completed: number;
            failed: number;
        };
    }

    export interface BatchStatusResponse {
        id: string;
        object: 'batch';
        created_at: number;
        endpoint: string;
        status: 'validating' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'cancelling' | 'expired';
        input_file_id: string;
        completion_window: string;
        error_file_id?: string;
        output_file_id?: string;
        request_counts: {
            total: number;
            completed: number;
            failed: number;
        };
    }

    export interface BatchChatCompletionResponse {
        id: string;
        custom_id: string;
        response?: {
            body: ChatCompletionResponse;
            request_id: string;
            status_code: number;
        };
        error?: {
            message: string;
            type: string;
            code?: string;
            param?: string;
        };
    }
}

export class OpenAIProvider implements LLMProvider {
    private apiKey: string;
    private baseUrl: string;
    private fetchFn: typeof fetch;

    constructor(apiKey: string, baseUrl = 'https://api.openai.com/v1', fetchFn: typeof fetch = fetch) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.fetchFn = fetchFn;
    }

    getModels(): ModelInfo[] {
        return [
            {
                id: 'gpt-4o-mini',
                name: ModelName.GPT4oMini,
                provider: Provider.OpenAI,
                capabilities: {
                    structuredOutput: true,
                    batchRequests: true,
                },
            },
        ];
    }

    private formatMessages(messages: PromptMessage[]): OpenAITypes.ChatCompletionRequest['messages'] {
        return messages.map(msg => ({
            role: msg.role,
            content: msg.content,
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

    /**
     * Upload a file to OpenAI for batch processing
     */
    private async uploadFile(content: string): Promise<string> {
        const formData = new FormData();
        formData.append('purpose', 'batch');
        formData.append('file', new Blob([content], { type: 'application/jsonl' }), 'batch.jsonl');

        const response = await this.fetchFn(`${this.baseUrl}/files`, {
            method: 'POST',
            headers: {
                Authorization: `${this.apiKey.startsWith('Bearer ') ? '' : 'Bearer '}${this.apiKey}`,
            },
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Failed to upload file: ${JSON.stringify(error)}`);
        }

        const data = (await response.json()) as OpenAITypes.FileUploadResponse;
        return data.id;
    }

    async prompt<T = string>(messages: PromptMessage[], model: ModelName, options: LLMRequestOptions<T> = {}): Promise<CompletionResponse<T>> {
        const modelInfo = this.getModels().find(m => m.name === model);
        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }

        let formattedMessages = messages;

        // If T is not string and the model supports structured output, use it
        const needsStructuredOutput = options.sampleObj && typeof options.sampleObj !== 'string';
        if (needsStructuredOutput && !messages.some(m => m.content.toLowerCase().includes('json'))) {
            throw new Error('A message for a structured output must contain "json"');
        }

        // Prepare request
        const requestBody: OpenAITypes.ChatCompletionRequest = {
            model: modelInfo.id,
            messages: this.formatMessages(formattedMessages),
            temperature: options.temperature ?? 0.7,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            frequency_penalty: options.frequencyPenalty,
            presence_penalty: options.presencePenalty,
            seed: options.seed ?? (options.providerParams?.seed as number),
        };

        // Add agent name as user identifier if provided
        if (options.agentName || options.user) {
            requestBody.user = options.user || options.agentName;
        }

        // Add JSON response format if needed
        if (needsStructuredOutput && modelInfo.capabilities.structuredOutput) {
            requestBody.response_format = { type: 'json_object' };
        } else if (needsStructuredOutput && !modelInfo.capabilities.structuredOutput) {
            // Enhance prompt with structure information
            formattedMessages = this.enhancePromptWithStructure(messages, options.sampleObj);
            requestBody.messages = this.formatMessages(formattedMessages);
        }

        // Make API request
        const response = await this.fetchFn(`${this.baseUrl}/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `${this.apiKey.startsWith('Bearer ') ? '' : 'Bearer '}${this.apiKey}`,
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`OpenAI API error: ${JSON.stringify(error)}`);
        }

        const responseData = (await response.json()) as OpenAITypes.ChatCompletionResponse;
        const text = responseData.choices[0].message.content;

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
                promptTokens: responseData.usage.prompt_tokens,
                completionTokens: responseData.usage.completion_tokens,
                totalTokens: responseData.usage.total_tokens,
            },
            model: `${Provider.OpenAI}/${model}` as ModelType,
            provider: Provider.OpenAI,
        };
    }

    async createBatch<T = string>(prompts: BatchRequest[], model: ModelName, options: BatchRequestOptions<T> = {}): Promise<CreateBatchResponse> {
        const modelInfo = this.getModels().find(m => m.name === model);
        if (!modelInfo) {
            throw new Error(`Model not found: ${model}`);
        }
        if (!modelInfo.capabilities.batchRequests) {
            throw new Error(`Model ${model} does not support batch requests`);
        }

        // Maximum number of requests allowed in a batch
        const MAX_BATCH_REQUESTS = 50000;
        if (prompts.length > MAX_BATCH_REQUESTS) {
            throw new Error(`Maximum number of requests in a batch is ${MAX_BATCH_REQUESTS}`);
        }

        const needsStructuredOutput = options.sampleObj && typeof options.sampleObj !== 'string';
        if (needsStructuredOutput && !prompts.some(m => m.messages.some(msg => msg.content.toLowerCase().includes('json')))) {
            throw new Error('A message for a structured output must contain "json"');
        }

        // Need to create a JSONL file with batch requests
        let jsonl = '';
        const batchRequests: OpenAITypes.BatchRequest[] = [];

        for (let i = 0; i < prompts.length; i++) {
            const prompt = prompts[i];

            let requestId = `${Date.now()}_${i}`;
            if (options.batchName) {
                requestId = `${options.batchName.replace(/[^a-zA-Z0-9]/g, '')}_${requestId}`;
            }
            if (options.agentName) {
                requestId = `${options.agentName.replace(/[^a-zA-Z0-9]/g, '')}_${requestId}`;
            }
            requestId = prompt.id?.replace(/[^a-zA-Z0-9]/g, '') || requestId;

            const messages = prompt.messages;
            if (!messages || messages.length === 0) {
                throw new Error(`Prompt ${i} is empty`);
            }
            if (messages.length > 2048) {
                throw new Error(`Prompt ${i} exceeds the maximum number of messages (2048)`);
            }

            // Create OpenAI request body
            const chatRequest: OpenAITypes.ChatCompletionRequest = {
                model: modelInfo.id,
                messages: this.formatMessages(messages),
                temperature: options.temperature ?? 0.7,
                max_tokens: options.maxTokens,
                top_p: options.topP,
                frequency_penalty: options.frequencyPenalty,
                presence_penalty: options.presencePenalty,
            };

            // Add user if provided
            if (options.user || options.agentName) {
                chatRequest.user = options.user || options.agentName;
            }

            // Add seed if provided
            if (options.providerParams?.seed) {
                chatRequest.seed = options.providerParams.seed as number;
            }

            // Add JSON response format if needed and supported
            if (needsStructuredOutput && modelInfo.capabilities.structuredOutput) {
                chatRequest.response_format = { type: 'json_object' };
            } else if (needsStructuredOutput && !modelInfo.capabilities.structuredOutput) {
                // Enhance prompt with structure information
                const enhancedMessages = this.enhancePromptWithStructure(messages, options.sampleObj);
                chatRequest.messages = this.formatMessages(enhancedMessages);
            }

            // Create batch request
            const batchRequest: OpenAITypes.BatchRequest = {
                custom_id: requestId,
                method: 'POST',
                url: '/v1/chat/completions',
                body: chatRequest,
            };

            batchRequests.push(batchRequest);
            jsonl += `${JSON.stringify(batchRequest)}\n`;

            // Check JSONL size limit (200MB)
            if (jsonl.length > 200000000) {
                throw new Error('The total size of the batch requests exceeds 200MB');
            }
        }

        // Upload the JSONL file
        const fileId = await this.uploadFile(jsonl);

        // Create batch with uploaded file
        const batchOptions = {
            input_file_id: fileId,
            endpoint: '/v1/chat/completions',
            completion_window: '24h', // Default completion window
        };

        // Add timeout if specified
        if (options.timeoutSeconds) {
            // Convert seconds to hours for OpenAI's format, minimum 1h, maximum 24h
            const hours = Math.min(Math.max(Math.ceil(options.timeoutSeconds / 3600), 1), 24);
            batchOptions.completion_window = `${hours}h`;
        }

        const response = await this.fetchFn(`${this.baseUrl}/batches`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `${this.apiKey.startsWith('Bearer ') ? '' : 'Bearer '}${this.apiKey}`,
            },
            body: JSON.stringify(batchOptions),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Failed to create batch: ${JSON.stringify(error)}`);
        }

        const batchResponse = (await response.json()) as OpenAITypes.BatchCreateResponse;

        return {
            requestIds: batchRequests.map(request => request.custom_id),
            batchId: batchResponse.id,
        }
    }

    async checkBatch(batchId: string, model: ModelName): Promise<BatchStatus> {
        // Extract the actual batch ID if it's prefixed with an agent name
        const actualBatchId = batchId.includes('_') ? batchId.split('_').slice(1).join('_') : batchId;

        const response = await this.fetchFn(`${this.baseUrl}/batches/${actualBatchId}`, {
            headers: {
                Authorization: `${this.apiKey.startsWith('Bearer ') ? '' : 'Bearer '}${this.apiKey}`,
            },
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Failed to check batch status: ${JSON.stringify(error)}`);
        }

        const statusData = (await response.json()) as OpenAITypes.BatchStatusResponse;

        return {
            id: batchId, // Keep the original ID with potential prefix
            status: this.mapOpenAIBatchStatus(statusData.status),
            outputFileId: statusData.output_file_id,
            error: statusData.error_file_id ? `Error file ID: ${statusData.error_file_id}` : undefined,
        };
    }

    private mapOpenAIBatchStatus(status: string): 'pending' | 'processing' | 'completed' | 'failed' {
        switch (status) {
            case 'validating':
            case 'processing':
                return 'processing';
            case 'completed':
                return 'completed';
            case 'failed':
            case 'cancelled':
            case 'cancelling':
            case 'expired':
                return 'failed';
            default:
                return 'pending';
        }
    }

    async retrieveBatch<T = string>(batchId: string, model: ModelName, sampleObj?: T): Promise<CompletionResponse<T>[]> {
        // Extract the actual batch ID if it's prefixed with an agent name
        const actualBatchId = batchId.includes('_') ? batchId.split('_').slice(1).join('_') : batchId;

        // First check if the batch is completed
        const status = await this.checkBatch(batchId, model);
        if (status.status !== 'completed') {
            throw new Error(`Batch ${batchId} is not completed yet (status: ${status.status})`);
        }

        if (!status.outputFileId) {
            throw new Error('No output file ID found for completed batch');
        }

        // Get the file content
        const fileContentResponse = await this.fetchFn(`${this.baseUrl}/files/${status.outputFileId}/content`, {
            headers: {
                Authorization: `${this.apiKey.startsWith('Bearer ') ? '' : 'Bearer '}${this.apiKey}`,
            },
        });

        if (!fileContentResponse.ok) {
            const error = await fileContentResponse.json();
            throw new Error(`Failed to retrieve batch results: ${JSON.stringify(error)}`);
        }

        const jsonlContent = await fileContentResponse.text();
        const rawResponses = jsonlContent.trim().split('\n');
        const responses: CompletionResponse<T>[] = [];

        for (const rawResponse of rawResponses) {
            let response: OpenAITypes.BatchChatCompletionResponse;
            try {
                response = JSON.parse(rawResponse) as OpenAITypes.BatchChatCompletionResponse;
            } catch (error) {
                // Skip invalid responses
                console.warn(`Failed to parse batch response: ${error}`);
                continue;
            }

            // Skip responses with errors
            if (response.error || !response.response) {
                console.warn(`Batch response error for ID ${response.custom_id}: ${response.error?.message || 'No response'}`);
                continue;
            }

            const responseText = response.response.body.choices[0].message.content;
            let parsedData: T;

            // Parse the response based on the expected type
            if (sampleObj && typeof sampleObj === 'object') {
                try {
                    parsedData = this.parseJsonResponse<T>(responseText);
                } catch (error) {
                    // Skip responses that can't be properly parsed
                    console.warn(`Failed to parse JSON in response: ${error}`);
                    continue;
                }
            } else {
                parsedData = responseText as unknown as T;
            }

            // Extract request ID parts - helps identify which prompt this response is for
            const requestIdParts = response.custom_id.split('_');
            const requestIndex = requestIdParts.length > 1 ? Number.parseInt(requestIdParts[1]) : -1;

            responses.push({
                text: responseText,
                data: parsedData,
                usage: {
                    promptTokens: response.response.body.usage.prompt_tokens,
                    completionTokens: response.response.body.usage.completion_tokens,
                    totalTokens: response.response.body.usage.total_tokens,
                },
                model: `${Provider.OpenAI}/${model}` as ModelType,
                provider: Provider.OpenAI,
            });
        }

        return responses;
    }

    async cancelBatch(batchId: string, model: ModelName): Promise<boolean> {
        // Extract the actual batch ID if it's prefixed with an agent name
        const actualBatchId = batchId.includes('_') ? batchId.split('_').slice(1).join('_') : batchId;

        const response = await this.fetchFn(`${this.baseUrl}/batches/${actualBatchId}/cancel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `${this.apiKey.startsWith('Bearer ') ? '' : 'Bearer '}${this.apiKey}`,
            },
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Failed to cancel batch: ${JSON.stringify(error)}`);
        }

        return true;
    }
}
