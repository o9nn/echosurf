/**
 * echo-ml - TypeScript definitions
 */

export interface EchoConfig {
    /** Size of the vocabulary */
    vocab_size: number;
    /** Dimension of embeddings */
    embedding_dim: number;
    /** Size of the reservoir */
    reservoir_size: number;
    /** Hidden dimension (defaults to reservoir_size) */
    hidden_dim?: number;
    /** Output dimension */
    output_dim: number;
    /** Spectral radius for reservoir (default: 0.9) */
    spectral_radius?: number;
    /** Sparsity of reservoir connections (default: 0.9) */
    reservoir_sparsity?: number;
}

export class EchoML {
    constructor();
    
    /** Initialize the ML engine with the given configuration */
    init(config: EchoConfig): boolean;
    
    /** Load a pre-trained model from a file */
    load(modelPath: string): boolean;
    
    /** Save the current model to a file */
    save(modelPath: string): boolean;
    
    /** Run inference on a sequence of token IDs */
    forward(tokens: number[]): number[];
    
    /** Reset the engine's internal state */
    reset(): void;
    
    /** Free the engine and release all resources */
    free(): void;
    
    /** Load a dictionary file for tokenization */
    loadDict(dictPath: string): number;
    
    /** Tokenize a string using the loaded dictionary */
    tokenize(text: string): number[];
    
    /** Process text through the full pipeline */
    process(text: string): number[];
    
    /** Get the current engine configuration */
    getConfig(): EchoConfig;
    
    /** Check if the engine is initialized */
    readonly initialized: boolean;
    
    /** Check if a dictionary is loaded */
    readonly dictLoaded: boolean;
}

/** Create a default configuration for Deep Tree Echo */
export function createDefaultConfig(overrides?: Partial<EchoConfig>): EchoConfig;

/** Create a minimal configuration for testing */
export function createMinimalConfig(): EchoConfig;

/** Native module for advanced use */
export const native: {
    init(config: EchoConfig): boolean;
    load(path: string): boolean;
    save(path: string): boolean;
    forward(tokens: number[]): number[];
    reset(): boolean;
    free(): boolean;
    loadDict(path: string): number;
    tokenize(text: string): number[];
    getConfig(): EchoConfig;
};
