import tiktoken from 'tiktoken';
import { logger } from './logger';

const maxTokenLimit = {
  "gpt-3.5-turbo": 4096,
  "gpt-3.5-turbo-0301": 4096,
  "gpt-3.5-turbo-0613": 4096,
  "gpt-3.5-turbo-instruct": 4096,
  "gpt-3.5-turbo-16k": 16384,
  "gpt-35-turbo": 4096,
  "gpt-35-turbo-16k": 16384,
  "gpt-35-turbo-instruct": 4096,
  "gpt-4": 8192,
  "gpt-4-32k": 32768,
  "gpt-4-32k-0314": 32768,  //deprecate in Sep
  "gpt-4-0314": 8192,  //deprecate in Sep
  "gpt-4-0613": 8192,
  "gpt-4-32k-0613": 32768,
}


function getMaxTokenLimit(model = 'gpt-3.5-turbo-0613') {
    return maxTokenLimit[model];
}

function percentileUsed(input, model = 'gpt-3.5-turbo-0613') {
    return countToken(input, model) / getMaxTokenLimit(model);
}

// Count number of tokens left for an OpenAI model.
function tokenLeft(input, model = 'gpt-3.5-turbo-0613') {
    return getMaxTokenLimit(model) - countToken(input, model);
}

// Count number of tokens used by an OpenAI model.
function countToken(input, model = 'gpt-3.5-turbo-0613') {
    if (typeof input === 'string') {
        return numTokenFromText(input, model);
    } else if (Array.isArray(input) || typeof input === 'object') {
        return numTokenFromMessages(input, model);
    } else {
        throw new Error('Input must be a string, array, or object');
    }
}

// Return the number of tokens used by a string.
function numTokenFromText(text, model = 'gpt-3.5-turbo-0613') {
    try {
        const encoding = tiktoken.encoding_for_model(model);
        return encoding.encode(text).length;
    } catch (e) {
        logger.warn(`Model ${model} not found. Using cl100k_base encoding.`);
        const encoding = tiktoken.get_encoding('cl100k_base');
        return encoding.encode(text).length;
    }
}

function numTokenFromMessages(messages, model = 'gpt-3.5-turbo-0613') {
  if (!Array.isArray(messages)) {
      messages = [messages];
  }

  let encoding;
  try {
      encoding = tiktoken.encoding_for_model(model);
  } catch (error) {
      console.warn("Warning: model not found. Using cl100k_base encoding.");
      encoding = tiktoken.get_encoding('cl100k_base');
  }

  let tokensPerMessage, tokensPerName;
  const modelTokensConfig = {
      "gpt-3.5-turbo-0613": { tokensPerMessage: 3, tokensPerName: 1 },
      "gpt-3.5-turbo-16k-0613": { tokensPerMessage: 3, tokensPerName: 1 },
      "gpt-4-0314": { tokensPerMessage: 3, tokensPerName: 1 },
      "gpt-4-32k-0314": { tokensPerMessage: 3, tokensPerName: 1 },
      "gpt-4-0613": { tokensPerMessage: 3, tokensPerName: 1 },
      "gpt-4-32k-0613": { tokensPerMessage: 3, tokensPerName: 1 },
      "gpt-3.5-turbo-0301": { tokensPerMessage: 4, tokensPerName: -1 }
  };

  if (model in modelTokensConfig) {
      tokensPerMessage = modelTokensConfig[model].tokensPerMessage;
      tokensPerName = modelTokensConfig[model].tokensPerName;
  } else if (model.includes("gpt-3.5-turbo")) {
      // Assume latest gpt-3.5-turbo model
      return numTokenFromMessages(messages, "gpt-3.5-turbo-0613");
  } else if (model.includes("gpt-4")) {
      // Assume latest gpt-4 model
      return numTokenFromMessages(messages, "gpt-4-0613");
  } else {
      throw new Error(`numTokenFromMessages() is not implemented for model ${model}.`);
  }

  let numTokens = 0;
  messages.forEach(message => {
      numTokens += tokensPerMessage;
      Object.entries(message).forEach(([key, value]) => {
          if (value === null || value === undefined) return;

          if (typeof value !== 'string') {
              try {
                  value = JSON.stringify(value);
              } catch (error) {
                  console.warn(`Value ${value} is not a string and cannot be converted to JSON. It is a type: ${typeof value}. Skipping.`);
                  return;
              }
          }

          numTokens += encoding.encode(value).length;
          if (key === 'name') {
              numTokens += tokensPerName;
          }
      });
  });

  numTokens += 3; // Every reply is primed with assistant
  return numTokens;
}

function numTokensFromFunctions(functions, model = 'gpt-3.5-turbo-0613') {
  let encoding;
  try {
      encoding = tiktoken.encoding_for_model(model);
  } catch (error) {
      console.warn("Warning: model not found. Using cl100k_base encoding.");
      encoding = tiktoken.get_encoding('cl100k_base');
  }

  let numTokens = 0;
  functions.forEach(functionObj => {
      let functionTokens = encoding.encode(functionObj.name).length;
      functionTokens += encoding.encode(functionObj.description).length;
      functionTokens -= 2;

      if (functionObj.parameters && functionObj.parameters.properties) {
          const properties = functionObj.parameters.properties;
          for (let key in properties) {
              functionTokens += encoding.encode(key).length;
              let value = properties[key];
              for (let field in value) {
                  if (field === 'type') {
                      functionTokens += 2 + encoding.encode(value.type).length;
                  } else if (field === 'description') {
                      functionTokens += 2 + encoding.encode(value.description).length;
                  } else if (field === 'enum') {
                      functionTokens -= 3;
                      value.enum.forEach(enumOption => {
                          functionTokens += 3 + encoding.encode(enumOption).length;
                      });
                  } else {
                      console.warn(`Warning: not supported field ${field}`);
                  }
              }
          }
          functionTokens += 11;
          if (Object.keys(properties).length === 0) {
              functionTokens -= 2;
          }
      }

      numTokens += functionTokens;
  });

  numTokens += 12;
  return numTokens;
}

