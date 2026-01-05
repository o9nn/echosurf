/**
 * echo-ml - JavaScript wrapper for the Echo ML native addon
 * 
 * This provides a clean API for using the C/C++ ML framework from Node.js/Electron.
 */

const path = require('path');

// Load the native addon
let native;
try {
    native = require('../build/Release/echo_ml.node');
} catch (e) {
    try {
        native = require('../build/Debug/echo_ml.node');
    } catch (e2) {
        throw new Error('Failed to load echo_ml native addon. Run `npm install` first.');
    }
}

/**
 * EchoML - Main class for interacting with the ML engine
 */
class EchoML {
    constructor() {
        this._initialized = false;
        this._dictLoaded = false;
    }

    /**
     * Initialize the ML engine with the given configuration.
     * 
     * @param {Object} config - Configuration object
     * @param {number} config.vocab_size - Size of the vocabulary
     * @param {number} config.embedding_dim - Dimension of embeddings
     * @param {number} config.reservoir_size - Size of the reservoir
     * @param {number} [config.hidden_dim] - Hidden dimension (defaults to reservoir_size)
     * @param {number} config.output_dim - Output dimension
     * @param {number} [config.spectral_radius=0.9] - Spectral radius for reservoir
     * @param {number} [config.reservoir_sparsity=0.9] - Sparsity of reservoir connections
     * @returns {boolean} Success
     */
    init(config) {
        const result = native.init(config);
        this._initialized = result;
        return result;
    }

    /**
     * Load a pre-trained model from a file.
     * 
     * @param {string} modelPath - Path to the model file
     * @returns {boolean} Success
     */
    load(modelPath) {
        const result = native.load(modelPath);
        this._initialized = result;
        return result;
    }

    /**
     * Save the current model to a file.
     * 
     * @param {string} modelPath - Path to save the model
     * @returns {boolean} Success
     */
    save(modelPath) {
        return native.save(modelPath);
    }

    /**
     * Run inference on a sequence of token IDs.
     * 
     * @param {number[]} tokens - Array of token IDs
     * @returns {number[]} Output vector
     */
    forward(tokens) {
        if (!this._initialized) {
            throw new Error('Engine not initialized. Call init() or load() first.');
        }
        return native.forward(tokens);
    }

    /**
     * Reset the engine's internal state (reservoir state).
     */
    reset() {
        if (this._initialized) {
            native.reset();
        }
    }

    /**
     * Free the engine and release all resources.
     */
    free() {
        if (this._initialized) {
            native.free();
            this._initialized = false;
        }
    }

    /**
     * Load a dictionary file for tokenization.
     * 
     * @param {string} dictPath - Path to dictionary file (one token per line)
     * @returns {number} Number of tokens loaded
     */
    loadDict(dictPath) {
        const count = native.loadDict(dictPath);
        this._dictLoaded = count > 0;
        return count;
    }

    /**
     * Tokenize a string using the loaded dictionary.
     * 
     * @param {string} text - Text to tokenize
     * @returns {number[]} Array of token IDs
     */
    tokenize(text) {
        if (!this._dictLoaded) {
            throw new Error('Dictionary not loaded. Call loadDict() first.');
        }
        return native.tokenize(text);
    }

    /**
     * Process text through the full pipeline: tokenize and run inference.
     * 
     * @param {string} text - Input text
     * @returns {number[]} Output vector
     */
    process(text) {
        const tokens = this.tokenize(text);
        return this.forward(tokens);
    }

    /**
     * Get the current engine configuration.
     * 
     * @returns {Object} Configuration object
     */
    getConfig() {
        return native.getConfig();
    }

    /**
     * Check if the engine is initialized.
     * 
     * @returns {boolean}
     */
    get initialized() {
        return this._initialized;
    }

    /**
     * Check if a dictionary is loaded.
     * 
     * @returns {boolean}
     */
    get dictLoaded() {
        return this._dictLoaded;
    }
}

/**
 * Create a default configuration for Deep Tree Echo.
 * 
 * @param {Object} [overrides] - Override default values
 * @returns {Object} Configuration object
 */
function createDefaultConfig(overrides = {}) {
    return {
        vocab_size: 8192,
        embedding_dim: 128,
        reservoir_size: 512,
        hidden_dim: 512,
        output_dim: 64,
        spectral_radius: 0.9,
        reservoir_sparsity: 0.9,
        ...overrides
    };
}

/**
 * Create a minimal configuration for testing.
 * 
 * @returns {Object} Configuration object
 */
function createMinimalConfig() {
    return {
        vocab_size: 1024,
        embedding_dim: 32,
        reservoir_size: 64,
        hidden_dim: 64,
        output_dim: 16,
        spectral_radius: 0.9,
        reservoir_sparsity: 0.95
    };
}

// Export
module.exports = {
    EchoML,
    createDefaultConfig,
    createMinimalConfig,
    native  // Export native module for advanced use
};
