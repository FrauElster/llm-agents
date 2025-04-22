// src/agents.test.ts
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import { setTimeout } from 'node:timers/promises';

import { LLMAgent, createAgent, Provider, ModelName, type LLMRequestOptions, type BatchRequestOptions, type PromptMessage } from './index.ts';

// Test interface for structured output testing
interface TestResponse {
    answer: string;
    confidence: number;
    reasoning: string;
    sources?: string[];
}

// Helper to check if a value is a valid UUID
function isUUID(str: string): boolean {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    return uuidRegex.test(str);
}

// Read API keys from environment variables
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

// Skip tests if credentials are missing
const skipGoogleTests = !GOOGLE_API_KEY;
const skipOpenAITests = !OPENAI_API_KEY;

// Basic validation of API keys before starting tests
if (skipGoogleTests) {
    console.warn('⚠️ GOOGLE_API_KEY not found in environment. Google tests will be skipped.');
}

if (skipOpenAITests) {
    console.warn('⚠️ OPENAI_API_KEY not found in environment. OpenAI tests will be skipped.');
}

if (skipGoogleTests && skipOpenAITests) {
    console.error('❌ No API keys found. All tests will be skipped.');
    process.exit(1);
}

// Test Agent Creation
describe('Agent Creation', () => {
    it('should create a Google agent', () => {
        if (skipGoogleTests) {
            console.log('Skipping Google agent creation test');
            return;
        }

        const agent = createAgent({
            name: 'TestGeminiAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.Google}/${ModelName.Gemini2Flash}`,
            apiKey: GOOGLE_API_KEY as string,
        });

        assert.strictEqual(agent.getName(), 'TestGeminiAgent');
        assert.strictEqual(agent.getModel(), `${Provider.Google}/${ModelName.Gemini2Flash}`);
    });

    it('should create an OpenAI agent', () => {
        if (skipOpenAITests) {
            console.log('Skipping OpenAI agent creation test');
            return;
        }

        const agent = createAgent({
            name: 'TestOpenAIAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: OPENAI_API_KEY as string,
        });

        assert.strictEqual(agent.getName(), 'TestOpenAIAgent');
        assert.strictEqual(agent.getModel(), `${Provider.OpenAI}/${ModelName.GPT4oMini}`);
    });

    it('should throw error for invalid provider', () => {
        assert.throws(() => {
            createAgent({
                name: 'InvalidAgent',
                basePrompt: 'Test prompt',
                // @ts-expect-error
                model: 'invalid/model',
                apiKey: 'fake-key',
            });
        }, /Unsupported provider/);
    });
});

// Node.js test context type
interface TestContext {
    timeout?: number;
    signal?: AbortSignal;
}

// Test Basic Prompting
describe('Basic Prompting', function (this: TestContext) {
    // Extend timeout for API calls
    this.timeout = 30000;

    it('should get a text response from Google Gemini', async () => {
        if (skipGoogleTests) {
            console.log('Skipping Google prompting test');
            return;
        }

        const agent = createAgent({
            name: 'TestGeminiPromptAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.Google}/${ModelName.Gemini2Flash}`,
            apiKey: GOOGLE_API_KEY as string,
        });

        const response = await agent.prompt([{ role: 'user', content: 'What is the capital of France?' }]);

        assert.ok(response.text, 'Response text should exist');
        assert.strictEqual(typeof response.text, 'string');
        assert.ok(response.text.toLowerCase().includes('paris'), 'Response should mention Paris');
        assert.ok(response.usage.promptTokens > 0, 'Should include prompt token count');
        assert.ok(response.usage.completionTokens > 0, 'Should include completion token count');
        assert.strictEqual(response.provider, Provider.Google);
    });

    it('should get a text response from OpenAI', async () => {
        if (skipOpenAITests) {
            console.log('Skipping OpenAI prompting test');
            return;
        }

        const agent = createAgent({
            name: 'TestOpenAIPromptAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: OPENAI_API_KEY as string,
        });

        const response = await agent.prompt([{ role: 'user', content: 'What is the capital of France?' }]);

        assert.ok(response.text, 'Response text should exist');
        assert.strictEqual(typeof response.text, 'string');
        assert.ok(response.text.toLowerCase().includes('paris'), 'Response should mention Paris');
        assert.ok(response.usage.promptTokens > 0, 'Should include prompt token count');
        assert.ok(response.usage.completionTokens > 0, 'Should include completion token count');
        assert.strictEqual(response.provider, Provider.OpenAI);
    });
});

// Test Structured Output
describe('Structured Output', function (this: TestContext) {
    // Extend timeout for API calls
    this.timeout = 30000;

    const sampleObj: TestResponse = {
        answer: '',
        confidence: 0,
        reasoning: '',
    };

    it('should return structured output from Google Gemini', async () => {
        if (skipGoogleTests) {
            console.log('Skipping Google structured output test');
            return;
        }

        const agent = createAgent<TestResponse>({
            name: 'TestGeminiStructuredAgent',
            basePrompt: 'You are a helpful assistant that provides structured responses.',
            model: `${Provider.Google}/${ModelName.Gemini2Flash}`,
            apiKey: GOOGLE_API_KEY as string,
        });

        const options: LLMRequestOptions<TestResponse> = {
            temperature: 0.2,
            sampleObj,
        };

        const response = await agent.prompt(
            [
                {
                    role: 'user',
                    content:
                        'What is the capital of France? Rate your confidence on a scale of 0-1. Supply a reasoning explaining your confidence. Answer in json format, with keys "answer", "confidence", and "reasoning".',
                },
            ],
            options,
        );

        assert.ok(response.text, 'Response text should exist');
        assert.ok(response.data, 'Structured data should exist');
        assert.strictEqual(typeof response.data, 'object');
        assert.strictEqual(typeof response.data.answer, 'string');
        assert.strictEqual(typeof response.data.confidence, 'number');
        assert.strictEqual(typeof response.data.reasoning, 'string');
        assert.ok(response.data.answer.toLowerCase().includes('paris'), 'Answer should mention Paris');
        assert.ok(response.data.confidence >= 0 && response.data.confidence <= 1, 'Confidence should be between 0 and 1');
    });

    it('should return structured output from OpenAI', async () => {
        if (skipOpenAITests) {
            console.log('Skipping OpenAI structured output test');
            return;
        }

        const agent = createAgent<TestResponse>({
            name: 'TestOpenAIStructuredAgent',
            basePrompt: 'You are a helpful assistant that provides structured responses.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: OPENAI_API_KEY as string,
        });

        const options: LLMRequestOptions<TestResponse> = {
            temperature: 0.2,
            sampleObj,
        };

        const response = await agent.prompt(
            [
                {
                    role: 'user',
                    content:
                        'What is the capital of France? Rate your confidence on a scale of 0-1. Supply a reasoning explaining your confidence. Answer in json format, with keys "answer", "confidence", and "reasoning".',
                },
            ],
            options,
        );

        assert.ok(response.text, 'Response text should exist');
        assert.ok(response.data, 'Structured data should exist');
        assert.strictEqual(typeof response.data, 'object');
        assert.strictEqual(typeof response.data.answer, 'string');
        assert.strictEqual(typeof response.data.confidence, 'number');
        assert.strictEqual(typeof response.data.reasoning, 'string');
        assert.ok(response.data.answer.toLowerCase().includes('paris'), 'Answer should mention Paris');
        assert.ok(response.data.confidence >= 0 && response.data.confidence <= 1, 'Confidence should be between 0 and 1');
    });
});

// Test Batch Operations (OpenAI only)
describe('Batch Operations', function (this: TestContext) {
    // Extend timeout for API calls (batches can take longer)
    this.timeout = 120000;

    let batchId: string;

    it('should create a batch with OpenAI', async () => {
        if (skipOpenAITests) {
            console.log('Skipping OpenAI batch creation test');
            return;
        }

        const agent = createAgent({
            name: 'TestBatchAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: OPENAI_API_KEY as string,
        });

        const prompts: PromptMessage[][] = [
            [{ role: 'user', content: 'What is the capital of France?' }],
            [{ role: 'user', content: 'What is the capital of Germany?' }],
            [{ role: 'user', content: 'What is the capital of Italy?' }],
        ];

        const options: BatchRequestOptions = {
            temperature: 0.2,
            batchName: 'CapitalsBatch',
        };

        batchId = await agent.createBatch(prompts, options);

        assert.ok(batchId, 'Batch ID should be returned');
        assert.ok(batchId.includes('TestBatchAgent'), 'Batch ID should include agent name');
    });

    it('should check batch status with OpenAI', async () => {
        if (skipOpenAITests || !batchId) {
            console.log('Skipping OpenAI batch status check test');
            return;
        }

        const agent = createAgent({
            name: 'TestBatchAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: OPENAI_API_KEY as string,
        });

        const status = await agent.checkBatch(batchId);

        assert.strictEqual(status.id, batchId);
        assert.ok(['pending', 'processing', 'completed', 'failed'].includes(status.status), 'Status should be a valid batch status');
    });

    it('should retrieve batch results when completed with OpenAI', async () => {
        if (skipOpenAITests || !batchId) {
            console.log('Skipping OpenAI batch retrieval test');
            return;
        }

        const agent = createAgent({
            name: 'TestBatchAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: OPENAI_API_KEY as string,
        });

        // Poll for completion (with timeout)
        let status = await agent.checkBatch(batchId);
        let attempts = 0;
        const maxAttempts = 20;

        while (status.status !== 'completed' && status.status !== 'failed' && attempts < maxAttempts) {
            console.log(`Batch status: ${status.status}. Waiting for completion...`);
            await setTimeout(5000); // Wait 5 seconds between checks
            status = await agent.checkBatch(batchId);
            attempts++;
        }

        if (status.status === 'failed') {
            console.error('Batch processing failed:', status.error);
            assert.fail('Batch processing failed');
        }

        if (attempts >= maxAttempts) {
            console.warn('Timed out waiting for batch completion');
            return; // Skip further testing
        }

        assert.strictEqual(status.status, 'completed', 'Batch should be completed');

        // Retrieve the results
        const results = await agent.retrieveBatch(batchId);

        assert.ok(Array.isArray(results), 'Results should be an array');
        assert.ok(results.length > 0, 'Should have at least one result');

        // Verify first result
        const firstResult = results[0];
        assert.ok(firstResult.text, 'Response text should exist');
        assert.ok(firstResult.usage.promptTokens > 0, 'Should include prompt token count');
        assert.ok(firstResult.usage.completionTokens > 0, 'Should include completion token count');
        assert.strictEqual(firstResult.provider, Provider.OpenAI);
    });
});

// Test for expected errors and edge cases
describe('Error Handling', function (this: TestContext) {
    // Extend timeout for API calls
    this.timeout = 30000;

    it('should throw error when using batch operations with Google', async () => {
        if (skipGoogleTests) {
            console.log('Skipping Google batch error test');
            return;
        }

        const agent = createAgent({
            name: 'TestGoogleBatchErrorAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.Google}/${ModelName.Gemini2Flash}`,
            apiKey: GOOGLE_API_KEY as string,
        });

        const prompts: PromptMessage[][] = [[{ role: 'user', content: 'What is the capital of France?' }], [{ role: 'user', content: 'What is the capital of Germany?' }]];

        try {
            await agent.createBatch(prompts);
            assert.fail('Should have thrown an error for Google batch operations');
        } catch (error) {
            assert.ok((error as Error).message.includes("doesn't support batch operations"), "Error should mention that Google doesn't support batch operations");
        }
    });

    it('should throw error with invalid API key for OpenAI', async () => {
        if (skipOpenAITests) {
            console.log('Skipping OpenAI invalid key test');
            return;
        }

        const agent = createAgent({
            name: 'TestInvalidKeyAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
            apiKey: 'invalid-key',
        });

        try {
            await agent.prompt([{ role: 'user', content: 'What is the capital of France?' }]);
            assert.fail('Should have thrown an error for invalid API key');
        } catch (error) {
            assert.ok((error as Error).message.includes('API'), 'Error should mention API issue');
        }
    });

    it('should throw error with invalid API key for Google', async () => {
        if (skipGoogleTests) {
            console.log('Skipping Google invalid key test');
            return;
        }

        const agent = createAgent({
            name: 'TestInvalidKeyAgent',
            basePrompt: 'You are a helpful assistant.',
            model: `${Provider.Google}/${ModelName.Gemini2Flash}`,
            apiKey: 'invalid-key',
        });

        try {
            await agent.prompt([{ role: 'user', content: 'What is the capital of France?' }]);
            assert.fail('Should have thrown an error for invalid API key');
        } catch (error) {
            assert.ok((error as Error).message.includes('API'), 'Error should mention API issue');
        }
    });
});
