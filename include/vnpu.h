/**
 * vnpu.h - Virtual Neural Processing Unit
 * 
 * A membrane-bound neural substrate for Deep Tree Echo.
 * 
 * This defines the core abstractions for a "learnable processor" that represents
 * potential process promises—like stable diffusion for the imago dei. The vNPU
 * provides a body for intelligence, not just a bigger brain.
 * 
 * Key concepts:
 * - Membranes: inner | trans | outer (isolation boundaries)
 * - Isolates: actor/process with private heap/state
 * - Ports: typed channels (in/out/ctl/stat)
 * - Packets: IntentPacket, EvidencePacket
 * - Tensors: shape, dtype, device, layout
 * - Kernels: callable ops (ATen/GGML style)
 * - Graphs: DAGs of kernels (glyphs compile to these)
 * - Schedulers: phases/ticks/barriers (μ/σ/φ style)
 * - Policies: gating (trust, provenance, budgets)
 * 
 * The vNPU is designed to be:
 * - Tensor-native (ATen/TH)
 * - Event-driven (WorkerD / Plan 9)
 * - Actor-isolated (V8 isolates)
 * - Cache-coherent (GGML)
 * - Time-continuous (RWKV)
 * - Accelerator-agnostic (CUDA when available, CPU otherwise)
 * - Low-energy autonomous (runs without the cloud)
 * - Composable cognition (glyphs / skills / kernels)
 * - LLM-optional (LLMs are plugins, not gods)
 */

#ifndef VNPU_H
#define VNPU_H

#include "echo_ml.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * VERSION AND MAGIC
 * ============================================================================ */

#define VNPU_VERSION_MAJOR 1
#define VNPU_VERSION_MINOR 0
#define VNPU_MAGIC 0x564E5055  /* "VNPU" */

/* ============================================================================
 * MEMBRANE TYPES
 * 
 * Membranes define isolation boundaries. They control what can flow between
 * different regions of the cognitive substrate.
 * 
 * - INNER: The protected core. No external tool calls. Pure computation.
 * - TRANS: The transactional boundary. Evidence flows here with provenance checks.
 * - OUTER: The interface to the world. Intent packets arrive here.
 * ============================================================================ */

typedef enum {
    VNPU_MEMBRANE_INNER = 0,   /* Protected core - no external calls */
    VNPU_MEMBRANE_TRANS = 1,   /* Transactional boundary - evidence with provenance */
    VNPU_MEMBRANE_OUTER = 2    /* World interface - intent packets */
} VnpuMembrane;

/* ============================================================================
 * PORT TYPES
 * 
 * Ports are typed channels for communication between isolates.
 * Inspired by Plan 9's file-based everything.
 * ============================================================================ */

typedef enum {
    VNPU_PORT_IN = 0,      /* Input channel */
    VNPU_PORT_OUT = 1,     /* Output channel */
    VNPU_PORT_CTL = 2,     /* Control channel */
    VNPU_PORT_STAT = 3     /* Status/telemetry channel */
} VnpuPortDirection;

typedef enum {
    VNPU_PORT_TYPE_INTENT = 0,     /* IntentPacket */
    VNPU_PORT_TYPE_EVIDENCE = 1,   /* EvidencePacket */
    VNPU_PORT_TYPE_TENSOR = 2,     /* Raw tensor data */
    VNPU_PORT_TYPE_BYTES = 3       /* Raw bytes */
} VnpuPortType;

typedef struct {
    char name[64];
    VnpuPortDirection direction;
    VnpuPortType type;
    size_t buffer_size;
    void* buffer;
    bool connected;
} VnpuPort;

/* ============================================================================
 * PACKET TYPES
 * 
 * Packets are the units of communication between isolates.
 * ============================================================================ */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t timestamp;
    uint32_t intent_type;
    float urgency;           /* 0.0 - 1.0 */
    float confidence;        /* 0.0 - 1.0 */
    size_t payload_size;
    void* payload;
} VnpuIntentPacket;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t timestamp;
    uint32_t evidence_type;
    float provenance;        /* 0.0 - 1.0 (trust score) */
    float relevance;         /* 0.0 - 1.0 */
    size_t payload_size;
    void* payload;
} VnpuEvidencePacket;

/* ============================================================================
 * DEVICE ABSTRACTION
 * 
 * Devices represent compute resources (CPU, GPU, NPU, etc.)
 * ============================================================================ */

typedef enum {
    VNPU_DEVICE_CPU = 0,
    VNPU_DEVICE_CUDA = 1,
    VNPU_DEVICE_METAL = 2,
    VNPU_DEVICE_VULKAN = 3,
    VNPU_DEVICE_NPU = 4      /* Actual hardware NPU */
} VnpuDeviceKind;

typedef struct {
    char name[64];
    VnpuDeviceKind kind;
    uint32_t threads;        /* For CPU */
    uint32_t sm_count;       /* For CUDA (streaming multiprocessors) */
    size_t memory_bytes;
    bool available;
} VnpuDevice;

/* ============================================================================
 * KERNEL ABSTRACTION
 * 
 * Kernels are callable operations. They form the grammar terminals of
 * the cognitive computation.
 * ============================================================================ */

typedef enum {
    VNPU_KERNEL_ATEN = 0,    /* ATen-style ops */
    VNPU_KERNEL_GGML = 1,    /* GGML-style ops */
    VNPU_KERNEL_RWKV = 2,    /* RWKV recurrent ops */
    VNPU_KERNEL_ECHO = 3,    /* Echo ML reservoir ops */
    VNPU_KERNEL_CUSTOM = 4   /* Custom user-defined */
} VnpuKernelFamily;

typedef void (*VnpuKernelFunc)(void* output, const void* const* inputs, 
                               size_t num_inputs, void* params);

typedef struct {
    char name[64];
    char qualified_name[128];  /* e.g., "aten.matmul", "rwkv.step" */
    VnpuKernelFamily family;
    VnpuKernelFunc func;
    size_t num_inputs;
    size_t num_outputs;
    void* params;
} VnpuKernel;

/* ============================================================================
 * GRAPH ABSTRACTION
 * 
 * Graphs are DAGs of kernels. Glyphs compile to graphs.
 * ============================================================================ */

typedef struct VnpuGraphNode {
    uint32_t id;
    VnpuKernel* kernel;
    uint32_t* input_node_ids;
    size_t num_inputs;
    EchoTensor* output;
    bool executed;
} VnpuGraphNode;

typedef struct {
    char name[64];
    VnpuGraphNode* nodes;
    size_t num_nodes;
    size_t capacity;
    uint32_t* execution_order;  /* Topologically sorted */
} VnpuGraph;

/* ============================================================================
 * POLICY ABSTRACTION
 * 
 * Policies control what can flow through membranes.
 * ============================================================================ */

typedef enum {
    VNPU_POLICY_ALLOW = 0,
    VNPU_POLICY_DENY = 1
} VnpuPolicyAction;

typedef struct {
    VnpuMembrane membrane;
    VnpuPolicyAction action;
    char target[64];           /* What is being allowed/denied */
    float min_provenance;      /* Minimum provenance score required */
    uint32_t max_tokens;       /* Token budget */
    bool active;
} VnpuPolicyRule;

typedef struct {
    char name[64];
    VnpuPolicyRule* rules;
    size_t num_rules;
    size_t capacity;
} VnpuPolicy;

/* ============================================================================
 * SCHEDULER ABSTRACTION
 * 
 * Schedulers manage the execution phases (μ/σ/φ style).
 * Maps to the 12-step cognitive loop.
 * ============================================================================ */

typedef enum {
    VNPU_PHASE_MU = 0,     /* μ - Perception phase */
    VNPU_PHASE_SIGMA = 1,  /* σ - Action phase */
    VNPU_PHASE_PHI = 2     /* φ - Simulation phase */
} VnpuPhase;

typedef struct {
    uint8_t current_step;      /* 0-11 in the 12-step cycle */
    VnpuPhase current_phase;   /* Which of the 3 phases */
    uint64_t tick_count;
    uint64_t last_tick_ns;
    bool running;
} VnpuScheduler;

/* ============================================================================
 * ISOLATE ABSTRACTION
 * 
 * Isolates are actor/processes with private heap/state.
 * They are the "software organelles" of the vNPU.
 * ============================================================================ */

typedef struct VnpuIsolate {
    char name[64];
    uint32_t id;
    VnpuMembrane membrane;
    
    /* State */
    EchoEngine* engine;        /* The ML engine for this isolate */
    void* heap;                /* Private heap */
    size_t heap_size;
    
    /* Communication */
    VnpuPort* ports;
    size_t num_ports;
    
    /* Execution */
    VnpuGraph* entry_graph;
    VnpuScheduler scheduler;
    VnpuPolicy* policy;
    
    /* Lifecycle */
    bool initialized;
    bool running;
    struct VnpuIsolate* parent;
    struct VnpuIsolate** children;
    size_t num_children;
} VnpuIsolate;

/* ============================================================================
 * VNPU RUNTIME
 * 
 * The main vNPU runtime that manages all isolates.
 * ============================================================================ */

typedef struct {
    uint32_t magic;
    uint32_t version;
    
    /* Devices */
    VnpuDevice* devices;
    size_t num_devices;
    
    /* Isolates */
    VnpuIsolate* isolates;
    size_t num_isolates;
    size_t isolate_capacity;
    
    /* Global state */
    uint64_t global_tick;
    bool initialized;
} VnpuRuntime;

/* ============================================================================
 * API FUNCTIONS
 * ============================================================================ */

/* Runtime lifecycle */
EchoError vnpu_runtime_init(VnpuRuntime* rt);
void vnpu_runtime_free(VnpuRuntime* rt);

/* Device management */
EchoError vnpu_device_add(VnpuRuntime* rt, const char* name, VnpuDeviceKind kind);
VnpuDevice* vnpu_device_get(VnpuRuntime* rt, const char* name);

/* Isolate management */
EchoError vnpu_isolate_create(VnpuRuntime* rt, VnpuIsolate** out, 
                              const char* name, VnpuMembrane membrane);
void vnpu_isolate_free(VnpuIsolate* isolate);
EchoError vnpu_isolate_start(VnpuIsolate* isolate);
EchoError vnpu_isolate_stop(VnpuIsolate* isolate);

/* Port management */
EchoError vnpu_port_create(VnpuIsolate* isolate, const char* name,
                           VnpuPortDirection dir, VnpuPortType type);
EchoError vnpu_port_connect(VnpuPort* from, VnpuPort* to);
EchoError vnpu_port_send(VnpuPort* port, const void* data, size_t size);
EchoError vnpu_port_recv(VnpuPort* port, void* data, size_t* size);

/* Packet creation */
EchoError vnpu_intent_packet_create(VnpuIntentPacket* pkt, uint32_t type,
                                    float urgency, const void* payload, size_t size);
EchoError vnpu_evidence_packet_create(VnpuEvidencePacket* pkt, uint32_t type,
                                      float provenance, const void* payload, size_t size);

/* Graph management */
EchoError vnpu_graph_create(VnpuGraph* graph, const char* name);
void vnpu_graph_free(VnpuGraph* graph);
EchoError vnpu_graph_add_node(VnpuGraph* graph, VnpuKernel* kernel,
                              const uint32_t* input_ids, size_t num_inputs);
EchoError vnpu_graph_execute(VnpuGraph* graph, VnpuIsolate* isolate);

/* Policy management */
EchoError vnpu_policy_create(VnpuPolicy* policy, const char* name);
void vnpu_policy_free(VnpuPolicy* policy);
EchoError vnpu_policy_add_rule(VnpuPolicy* policy, VnpuMembrane membrane,
                               VnpuPolicyAction action, const char* target,
                               float min_provenance, uint32_t max_tokens);
bool vnpu_policy_check(const VnpuPolicy* policy, VnpuMembrane membrane,
                       const char* target, float provenance, uint32_t tokens);

/* Scheduler */
void vnpu_scheduler_init(VnpuScheduler* sched);
void vnpu_scheduler_tick(VnpuScheduler* sched);
VnpuPhase vnpu_scheduler_get_phase(const VnpuScheduler* sched);

/* Kernel registry */
EchoError vnpu_kernel_register(const char* qualified_name, VnpuKernelFamily family,
                               VnpuKernelFunc func, size_t num_inputs, size_t num_outputs);
VnpuKernel* vnpu_kernel_lookup(const char* qualified_name);

/* ============================================================================
 * IR PARSING (for .vnpu files)
 * ============================================================================ */

typedef struct {
    VnpuRuntime* runtime;
    char* source;
    size_t source_len;
    size_t pos;
    int line;
    int col;
    char error_msg[256];
} VnpuParser;

EchoError vnpu_parse_file(VnpuRuntime* rt, const char* path);
EchoError vnpu_parse_string(VnpuRuntime* rt, const char* source);

#ifdef __cplusplus
}
#endif

#endif /* VNPU_H */
