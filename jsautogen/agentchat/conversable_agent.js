// import { OpenAIWrapper } from 'autogen';
import { Agent } from './agent';
import { DEFAULT_MODEL, UNKNOWN, executeCode, extractCode, inferLang } from 'autogen/codeUtils';
import { colored } from 'termcolor';
import { logger } from '../logger';

/**
 * (In preview) A class for generic conversable agents which can be configured as assistant or user proxy.
 * After receiving each message, the agent will send a reply to the sender unless the msg is a termination msg.
 * For example, AssistantAgent and UserProxyAgent are subclasses of this class,
 * configured with different default settings.

 * To modify auto reply, override `generate_reply` method.
 * To disable/enable human response in every turn, set `human_input_mode` to "NEVER" or "ALWAYS".
 * To modify the way to get human input, override `get_human_input` method.
 * To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`,
 * `run_code`, and `execute_function` methods respectively.
 * To customize the initial message when a conversation starts, override `generate_init_message` method.
 */
export class ConversableAgent extends Agent {
  static DEFAULT_CONFIG = { model: DEFAULT_MODEL };
  static MAX_CONSECUTIVE_AUTO_REPLY = 100;

  constructor(
    name,
    systemMessage = "You are a helpful AI Assistant.",
    isTerminationMsg = null,
    maxConsecutiveAutoReply = null,
    humanInputMode = "TERMINATE",
    functionMap = null,
    codeExecutionConfig = null,
    llmConfig = null,
    defaultAutoReply = ""
  ) {
    super(name);
    this._oaiMessages = new Map();
    this._oaiSystemMessage = [{ content: systemMessage, role: "system" }];
    this._isTerminationMsg = isTerminationMsg ?? (x => x.get("content") === "TERMINATE");
    this.llmConfig = llmConfig === false ? false : { ...ConversableAgent.DEFAULT_CONFIG, ...llmConfig };
    this.client = llmConfig !== false ? new OpenAIWrapper(this.llmConfig) : null;
    this._codeExecutionConfig = codeExecutionConfig ?? {};
    this.humanInputMode = humanInputMode;
    this._maxConsecutiveAutoReply = maxConsecutiveAutoReply ?? ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY;
    this._consecutiveAutoReplyCounter = new Map();
    this._maxConsecutiveAutoReplyDict = new Map();
    this._functionMap = functionMap ?? {};
    this._defaultAutoReply = defaultAutoReply;
    this._replyFuncList = [];
    this.replyAtReceive = new Map();
    this.registerReply([Agent, null], this.generateOAIReply.bind(this));
    this.registerReply([Agent, null], this.generateCodeExecutionReply.bind(this));
    this.registerReply([Agent, null], this.generateFunctionCallReply.bind(this));
    this.registerReply([Agent, null], this.generateAsyncFunctionCallReply.bind(this));
    this.registerReply([Agent, null], this.checkTerminationAndHumanReply.bind(this));
  }

  registerReply(trigger, replyFunc, position = 0, config = null, resetConfig = null) {
    if (!['string', 'function', 'object'].includes(typeof trigger) && !Array.isArray(trigger)) {
      throw new Error("trigger must be a class, a string, an agent, a callable, or a list.");
    }

    const replyFuncTuple = {
      trigger,
      replyFunc,
      config: { ...config },
      initConfig: config,
      resetConfig,
    };

    this._replyFuncList.splice(position, 0, replyFuncTuple);
  }

  get systemMessage() {
    return this._oaiSystemMessage[0].content;
  }

  updateSystemMessage(systemMessage) {
    this._oaiSystemMessage[0].content = systemMessage;
  }

  updateMaxConsecutiveAutoReply(value, sender = null) {
    if (sender === null) {
      this._maxConsecutiveAutoReply = value;
      this._maxConsecutiveAutoReplyDict.forEach((_, key) => {
        this._maxConsecutiveAutoReplyDict.set(key, value);
      });
    } else {
      this._maxConsecutiveAutoReplyDict.set(sender, value);
    }
  }

  maxConsecutiveAutoReply(sender = null) {
    return sender === null
      ? this._maxConsecutiveAutoReply
      : this._maxConsecutiveAutoReplyDict.get(sender);
  }

  get chatMessages() {
    return this._oaiMessages;
  }

  lastMessage(agent = null) {
    if (agent === null) {
      const nConversations = this._oaiMessages.size;
      if (nConversations === 0) {
        return null;
      }
      if (nConversations === 1) {
        return [...this._oaiMessages.values()][0].slice(-1)[0];
      }
      throw new Error("More than one conversation is found. Please specify the sender to get the last message.");
    }
    if (!this._oaiMessages.has(agent)) {
      throw new Error(`The agent '${agent.name}' is not present in any conversation. No history available for this agent.`);
    }
    return this._oaiMessages.get(agent).slice(-1)[0];
  }

  get useDocker() {
    return this._codeExecutionConfig === false ? null : this._codeExecutionConfig.useDocker;
  }

  static _messageToDict(message) {
    if (typeof message === 'string') {
      return { content: message };
    } else if (typeof message === 'object' && message !== null) {
      return message;
    } else {
      return {}; // or throw an error depending on how you want to handle invalid message types
    }
  }

  _appendOaiMessage(message, role, conversationId) {
    const oaiMessage = {};

    // Convert message to object if it's a string
    if (typeof message === 'string') {
      message = { content: message };
    }

    // Populate oaiMessage with necessary fields
    ['content', 'function_call', 'name', 'context'].forEach(key => {
      if (message[key]) {
        oaiMessage[key] = message[key];
      }
    });

    // Validating message
    if (!oaiMessage.content && !oaiMessage.function_call) {
      return false;
    }

    oaiMessage.role = message.role === 'function' ? 'function' : role;

    if (oaiMessage.function_call) {
      oaiMessage.role = 'assistant'; // Force role to 'assistant' if function_call is present
    }

    // Append message to conversation
    if (!this._oaiMessages.has(conversationId)) {
      this._oaiMessages.set(conversationId, []);
    }
    this._oaiMessages.get(conversationId).push(oaiMessage);

    return true;
  }

  send(message, recipient, requestReply = null, silent = false) {
    const isValid = this._appendOaiMessage(message, 'assistant', recipient);

    if (!isValid) {
      throw new Error("Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided.");
    }

    recipient.receive(message, this, requestReply, silent);
  }

  async aSend(message, recipient, requestReply = null, silent = false) {
    const isValid = this._appendOaiMessage(message, 'assistant', recipient);

    if (!isValid) {
      throw new Error("Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided.");
    }

    await recipient.aReceive(message, this, requestReply, silent);
  }

  _printReceivedMessage(message, sender) {
    console.log(colored(`${sender.name} (to ${this.name}):`, 'yellow'));
    message = ConversableAgent._messageToDict(message);

    if (message.role === 'function') {
      const funcPrint = `***** Response from calling function "${message.name}" *****`;
      console.log(colored(funcPrint, 'green'));
      console.log(message.content);
      console.log(colored('*'.repeat(funcPrint.length), 'green'));
    } else {
      let content = message.content;
      if (content && message.context) {
        content = OpenAIWrapper.instantiate(content, message.context, this.llmConfig && this.llmConfig.allowFormatStrTemplate);
      }
      if (content) console.log(content);
      if (message.functionCall) {
        const functionCall = { ...message.functionCall };
        const funcPrint = `***** Suggested function Call: ${functionCall.name || '(No function name found)'} *****`;
        console.log(colored(funcPrint, 'green'));
        console.log('Arguments: \n', functionCall.args || '(No arguments found)');
        console.log(colored('*'.repeat(funcPrint.length), 'green'));
      }
    }
    console.log('\n' + '-'.repeat(80));
  }

  _processReceivedMessage(message, sender, silent) {
    message = ConversableAgent._messageToDict(message);
    const valid = this._appendOAIMessage(message, 'user', sender);
    if (!valid) {
      throw new Error("Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided.");
    }
    if (!silent) {
      this._printReceivedMessage(message, sender);
    }
  }

  receive(message, sender, requestReply = null, silent = false) {
    this._processReceivedMessage(message, sender, silent);
    if (requestReply === false || (requestReply === null && !this.replyAtReceive.get(sender))) {
      return;
    }
    const reply = this.generateReply(this._oaiMessages.get(sender), sender);
    if (reply !== null) {
      this.send(reply, sender, silent);
    }
  }

  async aReceive(message, sender, requestReply = null, silent = false) {
    this._processReceivedMessage(message, sender, silent);
    if (requestReply === false || (requestReply === null && !this.replyAtReceive.get(sender))) {
      return;
    }
    const reply = await this.aGenerateReply(sender);
    if (reply !== null) {
      await this.aSend(reply, sender, silent);
    }
  }

  _prepareChat(recipient, clearHistory) {
    this.resetConsecutiveAutoReplyCounter(recipient);
    recipient.resetConsecutiveAutoReplyCounter(this);
    this.replyAtReceive.set(recipient, true);
    recipient.replyAtReceive.set(this, true);
    if (clearHistory) {
      this.clearHistory(recipient);
      recipient.clearHistory(this);
    }
  }

  initiateChat(recipient, clearHistory = true, silent = false, context = {}) {
    this._prepareChat(recipient, clearHistory);
    this.send(this.generateInitMessage(context), recipient, silent);
  }

  async aInitiateChat(recipient, clearHistory = true, silent = false, context = {}) {
    this._prepareChat(recipient, clearHistory);
    await this.aSend(this.generateInitMessage(context), recipient, silent);
  }

  reset() {
    this.clearHistory();
    this.resetConsecutiveAutoReplyCounter();
    this.stopReplyAtReceive();
    this._replyFuncList.forEach(replyFuncTuple => {
      if (replyFuncTuple.resetConfig) {
        replyFuncTuple.resetConfig(replyFuncTuple.config);
      } else {
        replyFuncTuple.config = { ...replyFuncTuple.initConfig };
      }
    });
  }

  stopReplyAtReceive(sender = null) {
    if (sender === null) {
      this.replyAtReceive.clear();
    } else {
      this.replyAtReceive.set(sender, false);
    }
  }

  resetConsecutiveAutoReplyCounter(sender = null) {
    if (sender === null) {
      this._consecutiveAutoReplyCounter.clear();
    } else {
      this._consecutiveAutoReplyCounter.set(sender, 0);
    }
  }

  clearHistory(agent = null) {
    if (agent === null) {
      this._oaiMessages.clear();
    } else {
      if (this._oaiMessages.has(agent)) {
        this._oaiMessages.get(agent).clear();
      }
    }
  }

  generateOAIReply(messages = null, sender = null, config = null) {
    let client = config ?? this.client;
    if (!client) return [false, null];
    if (!messages) messages = this._oaiMessages.get(sender);

    // TODO: handle token limit exceeded error
    let response = client.create(
      { context: messages.at(-1)?.context ?? null, messages: [...this._oaiSystemMessage, ...messages] }
    );
    return [true, client.extractTextOrFunctionCall(response)[0]];
  }

  generateCodeExecutionReply(messages = null, sender = null, config = null) {
    let codeExecutionConfig = config !== null ? config : this._codeExecutionConfig;
    if (codeExecutionConfig === false) {
      return [false, null];
    }
    messages = messages ?? this._oaiMessages.get(sender);
    let lastNMessages = codeExecutionConfig.lastNMessages ?? 1;

    let messagesToScan = lastNMessages;
    if (lastNMessages === "auto") {
      // Find when the agent last spoke
      messagesToScan = 0;
      for (let i = messages.length - 1; i >= 0; i--) {
        let message = messages[i];
        if (!("role" in message) || message.role !== "user") {
          break;
        }
        messagesToScan++;
      }
    }

    // Scan the last n messages for code blocks
    for (let i = Math.max(messages.length - messagesToScan, 0); i < messages.length; i++) {
      let message = messages[i];
      if (!message.content) {
        continue;
      }
      let codeBlocks = extractCode(message.content);
      if (codeBlocks.length === 1 && codeBlocks[0][0] === UNKNOWN) {
        continue;
      }

      // Execute code blocks and return output
      let [exitcode, logs] = this.executeCodeBlocks(codeBlocks);
      codeExecutionConfig.lastNMessages = lastNMessages;
      let exitcodeStr = exitcode === 0 ? "execution succeeded" : "execution failed";
      return [true, `exitcode: ${exitcode} (${exitcodeStr})\nCode output: ${logs}`];
    }

    // No code blocks found, return null
    codeExecutionConfig.lastNMessages = lastNMessages;
    return [false, null];
  }

  generateFunctionCallReply(messages = null, sender = null, config = null) {
    if (!config) config = this;
    if (!messages) messages = this._oaiMessages.get(sender);
    let message = messages.at(-1);
    if (message.functionCall) {
      let [_, funcReturn] = this.executeFunction(message.functionCall);
      return [true, funcReturn];
    }
    return [false, null];
  }

  async generateAsyncFunctionCallReply(messages = null, sender = null, config = null) {
    if (!config) config = this;
    if (!messages) messages = this._oaiMessages.get(sender);
    let message = messages.at(-1);
    if (message.functionCall) {
      let funcCall = message.functionCall;
      let funcName = funcCall.name;
      let func = this._functionMap.get(funcName);
      if (func && func instanceof AsyncFunction) {
        let [_, funcReturn] = await this.aExecuteFunction(funcCall);
        return [true, funcReturn];
      }
    }
    return [false, null];
  }

  async checkTerminationAndHumanReply(messages = null, sender = null, config = null) {
    config = config ?? this;
    messages = messages ?? this._oaiMessages.get(sender);
    const message = messages[messages.length - 1];
    let reply = "";
    let noHumanInputMsg = "";

    if (this.humanInputMode === "ALWAYS") {
      reply = await this.getHumanInput(`Provide feedback to ${sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: `);
      noHumanInputMsg = reply ? "" : "NO HUMAN INPUT RECEIVED.";
      reply = reply || (this._isTerminationMsg(message) ? "exit" : "");
    } else {
      if (this._consecutiveAutoReplyCounter.get(sender) >= this._maxConsecutiveAutoReplyDict.get(sender)) {
        if (this.humanInputMode === "NEVER") {
          reply = "exit";
        } else {
          const terminate = this._isTerminationMsg(message);
          reply = await this.getHumanInput(terminate ?
            `Please give feedback to ${sender.name}. Press enter or type 'exit' to stop the conversation: ` :
            `Please give feedback to ${sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: `
          );
          noHumanInputMsg = reply ? "" : "NO HUMAN INPUT RECEIVED.";
          reply = reply || (terminate ? "exit" : "");
        }
      } else if (this._isTerminationMsg(message)) {
        reply = "exit";
      }
    }

    if (noHumanInputMsg) {
      console.log(`\n>>>>>>>> ${noHumanInputMsg}`);
    }

    if (reply === "exit") {
      this._consecutiveAutoReplyCounter.set(sender, 0);
      return [true, null];
    }

    if (reply || this._maxConsecutiveAutoReplyDict.get(sender) === 0) {
      this._consecutiveAutoReplyCounter.set(sender, 0);
      return [true, reply];
    }

    this._consecutiveAutoReplyCounter.set(sender, this._consecutiveAutoReplyCounter.get(sender) + 1);
    if (this.humanInputMode !== "NEVER") {
      console.log("\n>>>>>>>> USING AUTO REPLY...");
    }

    return [false, null];
  }

  async aCheckTerminationAndHumanReply(messages = null, sender = null, config = null) {
    if (config === null) {
      config = this;
    }
    if (messages === null) {
      messages = this._oaiMessages.get(sender);
    }
    const message = messages[messages.length - 1];
    let reply = "";
    let noHumanInputMsg = "";

    if (this.humanInputMode === "ALWAYS") {
      reply = await this.aGetHumanInput(`Provide feedback to ${sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: `);
      noHumanInputMsg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
      reply = reply || (!this._isTerminationMsg(message) ? "" : "exit");
    } else {
      if (this._consecutiveAutoReplyCounter.get(sender) >= this._maxConsecutiveAutoReplyDict.get(sender)) {
        if (this.humanInputMode === "NEVER") {
          reply = "exit";
        } else {
          const terminate = this._isTerminationMsg(message);
          reply = await this.aGetHumanInput(terminate ? `Please give feedback to ${sender.name}. Press enter or type 'exit' to stop the conversation: ` : `Please give feedback to ${sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: `);
          noHumanInputMsg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
          reply = reply || (terminate ? "exit" : "");
        }
      } else if (this._isTerminationMsg(message)) {
        if (this.humanInputMode === "NEVER") {
          reply = "exit";
        } else {
          reply = await this.aGetHumanInput(`Please give feedback to ${sender.name}. Press enter or type 'exit' to stop the conversation: `);
          noHumanInputMsg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
          reply = reply || "exit";
        }
      }
    }

    if (noHumanInputMsg) {
      console.log(noHumanInputMsg);  // Using console.log for simplicity
    }

    if (reply === "exit") {
      this._consecutiveAutoReplyCounter.set(sender, 0);
      return [true, null];
    }

    if (reply || this._maxConsecutiveAutoReplyDict.get(sender) === 0) {
      this._consecutiveAutoReplyCounter.set(sender, 0);
      return [true, reply];
    }

    this._consecutiveAutoReplyCounter.set(sender, this._consecutiveAutoReplyCounter.get(sender) + 1);
    return [false, null];
  }

  generateReply(messages = null, sender = null, exclude = null) {
    if (!messages && !sender) {
      throw new Error("Either 'messages' or 'sender' must be provided.");
    }

    messages = messages ?? this._oaiMessages.get(sender);

    for (const replyFuncTuple of this._replyFuncList) {
      const replyFunc = replyFuncTuple.replyFunc;
      if (exclude && exclude.includes(replyFunc)) {
        continue;
      }
      if (this._matchTrigger(replyFuncTuple.trigger, sender)) {
        const [final, reply] = replyFunc(this, messages, sender, replyFuncTuple.config);
        if (final) {
          return reply;
        }
      }
    }
    return this._defaultAutoReply;
  }

  async aGenerateReply(messages = null, sender = null, exclude = null) {
    if (!messages && !sender) {
      throw new Error("Either 'messages' or 'sender' must be provided.");
    }

    messages = messages ?? this._oaiMessages.get(sender);

    for (const replyFuncTuple of this._replyFuncList) {
      const replyFunc = replyFuncTuple.replyFunc;
      if (exclude && exclude.includes(replyFunc)) {
        continue;
      }
      if (this._matchTrigger(replyFuncTuple.trigger, sender)) {
        if (typeof replyFunc === 'function') {
          const [final, reply] = await replyFunc(this, messages, sender, replyFuncTuple.config);
          if (final) {
            return reply;
          }
        }
      }
    }
    return this._defaultAutoReply;
  }

  _matchTrigger(trigger, sender) {
    if (trigger === null) {
      return sender === null;
    } else if (typeof trigger === 'string') {
      return trigger === sender.name;
    } else if (typeof trigger === 'function') {
      return trigger(sender);
    } else if (Array.isArray(trigger)) {
      return trigger.some(t => this._matchTrigger(t, sender));
    } else if (trigger instanceof Agent) {
      return trigger === sender;
    } else {
      throw new Error(`Unsupported trigger type: ${typeof trigger}`);
    }
  }

  getHumanInput(prompt) {
    return new Promise((resolve) => {
      // Implement a mechanism to get input from the user
      // For example, this could be a prompt in a web interface
      console.log(prompt);
      const input = ""; // Replace with actual user input mechanism
      resolve(input);
    });
  }

  async aGetHumanInput(prompt) {
    // Asynchronous variant of getHumanInput, if needed
    return this.getHumanInput(prompt);
  }

  runCode(code, options = {}) {
    // Implement the logic to execute the code
    // This might involve a sandboxed environment or an external API call
    // TODO: implement sandboxing `executeCode` function
    console.log(`Executing code: ${code}`);
    const result = {}; // Replace with actual code execution logic
    return result;
  }

  executeCodeBlocks(codeBlocks) {
    let logsAll = "";
    let exitcode = 0;

    codeBlocks.forEach((codeBlock, i) => {
      const [lang, code] = codeBlock;
      const inferredLang = lang || inferLang(code);
      console.log(`\n>>>>>>>> EXECUTING CODE BLOCK ${i} (inferred language is ${inferredLang})...`);

      let result;
      if (['bash', 'shell', 'sh'].includes(inferredLang)) {
        result = this.runCode(code, inferredLang, ...this._codeExecutionConfig);
      } else if (['python', 'Python'].includes(inferredLang)) {
        let filename;
        if (code.startsWith('# filename: ')) {
          filename = code.substring(11, code.indexOf('\n')).trim();
        }
        result = this.runCode(code, 'python', filename, ...this._codeExecutionConfig);
      } else {
        result = { exitcode: 1, logs: `unknown language ${inferredLang}`, image: null };
      }

      const { exitcode: codeExit, logs, image } = result;
      if (image !== null) {
        this._codeExecutionConfig.useDocker = image;
      }
      logsAll += "\n" + logs;
      if (codeExit !== 0) {
        exitcode = codeExit;
      }
    });

    return [exitcode, logsAll];
  }

  static formatJsonStr(jstr) {
    let result = [];
    let insideQuotes = false;
    let lastChar = ' ';

    jstr.split('').forEach(char => {
      if (lastChar !== '\\' && char === '"') {
        insideQuotes = !insideQuotes;
      }
      lastChar = char;
      if (!insideQuotes && char === '\n') {
        return;
      }
      if (insideQuotes && char === '\n') {
        char = '\\n';
      }
      if (insideQuotes && char === '\t') {
        char = '\\t';
      }
      result.push(char);
    });

    return result.join('');
  }

  executeFunction(funcCall) {
    const funcName = funcCall.name || '';
    const func = this._functionMap[funcName];

    let isExecSuccess = false;
    let content;

    if (func) {
      const inputString = ConversableAgent.formatJsonStr(funcCall.args || '{}');
      let args;

      try {
        args = JSON.parse(inputString);
      } catch (e) {
        args = null;
        content = `Error: ${e}\n Your argument should follow JSON format.`;
      }

      if (args !== null) {
        console.log(`\n>>>>>>>> EXECUTING FUNCTION ${funcName}...`);
        try {
          content = func(...args);
          isExecSuccess = true;
        } catch (e) {
          content = `Error: ${e}`;
        }
      }
    } else {
      content = `Error: Function ${funcName} not found.`;
    }

    return [isExecSuccess, { name: funcName, role: "function", content: String(content) }];
  }

  async aExecuteFunction(funcCall) {
    const funcName = funcCall.name || '';
    const func = this._functionMap[funcName];

    let isExecSuccess = false;
    let content;

    if (func) {
      const inputString = ConversableAgent.formatJsonStr(funcCall.args || '{}');
      let args;

      try {
        args = JSON.parse(inputString);
      } catch (e) {
        args = null;
        content = `Error: ${e}\n Your argument should follow JSON format.`;
      }

      if (args !== null) {
        console.log(`\n>>>>>>>> EXECUTING ASYNC FUNCTION ${funcName}...`);
        try {
          if (typeof func === 'function' && func.constructor.name === 'AsyncFunction') {
            content = await func(...args);
          } else {
            content = func(...args);
          }
          isExecSuccess = true;
        } catch (e) {
          content = `Error: ${e}`;
        }
      }
    } else {
      content = `Error: Function ${funcName} not found.`;
    }

    return [isExecSuccess, { name: funcName, role: "function", content: String(content) }];
  }

  generateInitMessage(context = {}) {
    return context.message;
  }

  registerFunction(functionMap) {
    this._functionMap = { ...this._functionMap, ...functionMap };
  }

  canExecuteFunction(name) {
    return Object.hasOwnProperty.call(this._functionMap, name);
  }

  get functionMap() {
    return this._functionMap;
  }

}
