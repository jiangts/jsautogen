/**
 * (In preview) An abstract class for AI agent.
 *
 * An agent can communicate with other agents and perform actions.
 * Different agents can differ in what actions they perform in the `receive`
 * method.
 */
export class Agent {
  /**
   * Creates an instance of Agent.
   * @param {string} name - Name of the agent.
   */
  constructor(name) {
    this._name = name;
  }

  /**
   * Get the name of the agent.
   * @returns {string} The name of the agent.
   */
  get name() {
    return this._name;
  }

  /**
   * (Abstract method) Send a message to another agent.
   * @param {(Object|string)} message - The message to send.
   * @param {Agent} recipient - The recipient agent.
   * @param {boolean} [requestReply=false] - Whether a reply is requested.
   */
  send(message, recipient, requestReply = false) {
    // Abstract method, should be implemented in subclass.
  }

  /**
   * (Abstract async method) Send a message to another agent.
   * @param {(Object|string)} message - The message to send.
   * @param {Agent} recipient - The recipient agent.
   * @param {boolean} [requestReply=false] - Whether a reply is requested.
   * @returns {Promise<void>}
   */
  async aSend(message, recipient, requestReply = false) {
    // Abstract method, should be implemented in subclass.
  }

  /**
   * (Abstract method) Receive a message from another agent.
   * @param {(Object|string)} message - The received message.
   * @param {Agent} sender - The sender agent.
   * @param {boolean} [requestReply=false] - Whether a reply is requested.
   */
  receive(message, sender, requestReply = false) {
    // Abstract method, should be implemented in subclass.
  }

  /**
   * (Abstract async method) Receive a message from another agent.
   * @param {(Object|string)} message - The received message.
   * @param {Agent} sender - The sender agent.
   * @param {boolean} [requestReply=false] - Whether a reply is requested.
   * @returns {Promise<void>}
   */
  async aReceive(message, sender, requestReply = false) {
    // Abstract method, should be implemented in subclass.
  }

  /**
   * (Abstract method) Reset the agent.
   */
  reset() {
    // Abstract method, should be implemented in subclass.
  }

  /**
   * (Abstract method) Generate a reply based on the received messages.
   * @param {Array<Object>} [messages] - A list of messages received.
   * @param {Agent} [sender] - Sender of an Agent instance.
   * @param {...any} kwargs - Additional arguments.
   * @returns {(string|Object|null)} The generated reply. If null, no reply is generated.
   */
  generateReply(messages = null, sender = null, ...kwargs) {
    // Abstract method, should be implemented in subclass.
  }

  /**
   * (Abstract async method) Generate a reply based on the received messages.
   * @param {Array<Object>} [messages] - A list of messages received.
   * @param {Agent} [sender] - Sender of an Agent instance.
   * @param {...any} kwargs - Additional arguments.
   * @returns {Promise<(string|Object|null)>} The generated reply. If null, no reply is generated.
   */
  async aGenerateReply(messages = null, sender = null, ...kwargs) {
    // Abstract method, should be implemented in subclass.
  }
}