# LLM Provider Abstraction Library

A TypeScript library that abstracts away differences between LLM providers, allowing you to easily swap models and providers in your application.

![License](https://img.shields.io/badge/license-MIT-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-blue)

## Features

✨ **Provider Agnostic API**: Use the same code for OpenAI and Google (Gemini) models  
🔄 **Easy Provider Switching**: Swap LLM providers without changing your application code  
🧠 **Agent-Based Design**: Create named agents with specific configurations and base prompts  
📊 **Structured Output**: Automatic handling for JSON-formatted responses  
🛠️ **Type Safety**: Full TypeScript support with generics for response types  
⚡ **Batch Processing**: Complete batch request lifecycle management (for supported providers)  
🌐 **System Instructions**: Support for system prompts across providers  

## Installation

```bash
# Install with pnpm
pnpm add llm-abstraction

# Create .env file for API keys
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "GOOGLE_API_KEY=your_google_key_here" >> .env
```

## Quick Start

### Creating Agents

```typescript
import { createAgent, Provider, ModelName } from 'llm-abstraction';

// Create an OpenAI agent
const openaiAgent = createAgent({
  name: "MyAssistant",
  basePrompt: "You are a helpful, concise assistant.",
  model: `${Provider.OpenAI}/${ModelName.GPT4oMini}`,
  apiKey: process.env.OPENAI_API_KEY as string
});

// Create a Google Gemini agent
const geminiAgent = createAgent({
  name: "MyGeminiAssistant",
  basePrompt: "You are a helpful, concise assistant.",
  model: `${Provider.Google}/${ModelName.Gemini2Flash}`,
  apiKey: process.env.GOOGLE_API_KEY as string
});
```

### Basic Prompting

```typescript
// Send a prompt to your agent
const response = await openaiAgent.prompt([
  { role: 'user', content: 'What is the capital of France?' }
]);

console.log(response.text);  // Paris
console.log(`Tokens used: ${response.usage.totalTokens}`);
```

### Structured Output

```typescript
// Define a type for structured output
interface CountryInfo {
  name: string;
  capital: string;
  population: number;
  continent: string;
  languages: string[];
}

// Create a sample object to define the structure
const sampleCountry: CountryInfo = {
  name: "",
  capital: "",
  population: 0,
  continent: "",
  languages: [""]
};

// Get structured output from your agent
const structuredResponse = await openaiAgent.prompt<CountryInfo>(
  [{ role: 'user', content: 'Provide information about France.' }],
  { sampleObj: sampleCountry }
);

// Access typed data
console.log(`Country: ${structuredResponse.data.name}`);
console.log(`Capital: ${structuredResponse.data.capital}`);
console.log(`Population: ${structuredResponse.data.population.toLocaleString()}`);
console.log(`Languages: ${structuredResponse.data.languages.join(', ')}`);
```

### Batch Processing

```typescript
// Create a batch of prompts (OpenAI only)
const batchId = await openaiAgent.createBatch([
  [{ role: 'user', content: 'What is the capital of France?' }],
  [{ role: 'user', content: 'What is the capital of Japan?' }],
  [{ role: 'user', content: 'What is the capital of Brazil?' }]
]);

console.log(`Created batch ${batchId}`);

// Check batch status
let status = await openaiAgent.checkBatch(batchId);
console.log(`Batch status: ${status.status}`);

// Poll until batch is complete
while (status.status !== 'completed') {
  await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
  status = await openaiAgent.checkBatch(batchId);
  console.log(`Batch status: ${status.status}`);
}

// Retrieve batch results
const results = await openaiAgent.retrieveBatch(batchId);
results.forEach((result, i) => {
  console.log(`Result ${i+1}: ${result.text}`);
});

// Cancel a batch if needed
await openaiAgent.cancelBatch(batchId);
```

## Supported Models

### OpenAI
- `ModelName.GPT4oMini` ("GPT-4o-mini")

### Google Gemini
- `ModelName.Gemini2Flash` ("Gemini 2.0 Flash")
- `ModelName.Gemini15` ("Gemini 1.5")

## Advanced Configuration

### Request Options

```typescript
import { LLMRequestOptions } from 'llm-abstraction';

// Configure request parameters
const options: LLMRequestOptions = {
  temperature: 0.3,           // Lower temperature for more deterministic outputs
  maxTokens: 1000,            // Limit response length
  topP: 0.9,                  // Nucleus sampling parameter
  frequencyPenalty: 0.5,      // Reduce repetition
  presencePenalty: 0.5,       // Encourage topic diversity
  
  // Provider-specific parameters
  providerParams: {
    seed: 12345              // For reproducible results (if supported)
  }
};

const response = await myAgent.prompt(messages, options);
```

### Batch Options

```typescript
import { BatchRequestOptions } from 'llm-abstraction';

// Configure batch options
const batchOptions: BatchRequestOptions = {
  temperature: 0.5,
  maxTokens: 500,
  batchName: "Capitals Batch",     // Custom name for the batch
  priority: "high",                // Priority level
  timeoutSeconds: 3600,            // 1 hour timeout
  failureThreshold: 10             // Allow up to 10% failures
};

const batchId = await openaiAgent.createBatch(prompts, batchOptions);
```

## Project Structure

```
src/
├── types.ts              # Common type definitions
├── agents.ts             # Agent implementation
├── providers/
│   ├── google.ts         # Google Gemini provider
│   └── openai.ts         # OpenAI provider
├── index.ts              # Main exports
└── agents.test.ts        # Tests
```

## Running Tests

This project uses Node's built-in test runner with ts-node.

```bash
# Run tests
pnpm test

# With environment variables
pnpm test -r dotenv/config
```

The tests make real API calls to validate functionality, so you'll need valid API keys in your environment.

## Architecture

### Core Components

1. **LLMProvider Interface**: Defines the contract for provider implementations
2. **Provider Implementations**: Concrete implementations for Google and OpenAI
3. **LLMAgent Class**: Wraps providers with agent-specific functionality
4. **Type System**: Rich TypeScript types for request/response handling

### Key Design Principles

- **Interface-based design**: Clear contracts between components
- **Generic type support**: Strongly typed structured outputs
- **Provider-specific optimizations**: Uses each provider's native capabilities
- **Consistent error handling**: Clear error messages across providers

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request