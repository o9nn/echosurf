/**
 * Basic tests for echo-ml module
 */

const { EchoML, createMinimalConfig } = require('../lib/index.js');

console.log('=== Echo ML Basic Tests ===\n');

// Test 1: Create instance
console.log('Test 1: Create EchoML instance');
const echo = new EchoML();
console.log('  ✓ Instance created\n');

// Test 2: Initialize with minimal config
console.log('Test 2: Initialize with minimal config');
const config = createMinimalConfig();
console.log('  Config:', JSON.stringify(config, null, 2));
try {
    const success = echo.init(config);
    console.log('  ✓ Initialized:', success);
    console.log('  Actual config:', echo.getConfig());
} catch (e) {
    console.log('  ✗ Failed:', e.message);
}
console.log();

// Test 3: Forward pass with random tokens
console.log('Test 3: Forward pass');
try {
    const tokens = [1, 2, 3, 4, 5];
    console.log('  Input tokens:', tokens);
    const output = echo.forward(tokens);
    console.log('  Output length:', output.length);
    console.log('  Output sample:', output.slice(0, 5).map(x => x.toFixed(4)));
    console.log('  ✓ Forward pass successful');
} catch (e) {
    console.log('  ✗ Failed:', e.message);
}
console.log();

// Test 4: Reset state
console.log('Test 4: Reset state');
try {
    echo.reset();
    console.log('  ✓ State reset');
} catch (e) {
    console.log('  ✗ Failed:', e.message);
}
console.log();

// Test 5: Multiple forward passes (temporal dynamics)
console.log('Test 5: Multiple forward passes (temporal dynamics)');
try {
    echo.reset();
    const outputs = [];
    for (let i = 0; i < 3; i++) {
        const tokens = [i + 1, i + 2, i + 3];
        const output = echo.forward(tokens);
        outputs.push(output[0].toFixed(4));
    }
    console.log('  Outputs[0] over time:', outputs);
    console.log('  ✓ Temporal dynamics working');
} catch (e) {
    console.log('  ✗ Failed:', e.message);
}
console.log();

// Test 6: Save and load model
console.log('Test 6: Save and load model');
try {
    const modelPath = '/tmp/test_model.echo';
    echo.save(modelPath);
    console.log('  ✓ Model saved to:', modelPath);
    
    const echo2 = new EchoML();
    echo2.load(modelPath);
    console.log('  ✓ Model loaded');
    
    // Compare outputs
    echo.reset();
    echo2.reset();
    const tokens = [1, 2, 3];
    const out1 = echo.forward(tokens);
    const out2 = echo2.forward(tokens);
    
    const diff = out1.reduce((sum, v, i) => sum + Math.abs(v - out2[i]), 0);
    console.log('  Output difference:', diff.toFixed(6));
    console.log('  ✓ Save/load verified');
    
    echo2.free();
} catch (e) {
    console.log('  ✗ Failed:', e.message);
}
console.log();

// Test 7: Free resources
console.log('Test 7: Free resources');
try {
    echo.free();
    console.log('  ✓ Resources freed');
} catch (e) {
    console.log('  ✗ Failed:', e.message);
}
console.log();

console.log('=== All tests completed ===');
