/**
 * vnpu_runtime.c - Virtual Neural Processing Unit Runtime Implementation
 * 
 * This implements the core vNPU runtime: membranes, isolates, ports, packets,
 * graphs, policies, and the scheduler.
 * 
 * The vNPU is a "learnable processor" - a potential process promise that
 * converges on an image of the future through the unpredictable descent
 * of all forks in the road.
 */

#define _POSIX_C_SOURCE 199309L

#include "vnpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ============================================================================
 * RUNTIME LIFECYCLE
 * ============================================================================ */

EchoError vnpu_runtime_init(VnpuRuntime* rt) {
    memset(rt, 0, sizeof(VnpuRuntime));
    
    rt->magic = VNPU_MAGIC;
    rt->version = (VNPU_VERSION_MAJOR << 16) | VNPU_VERSION_MINOR;
    
    /* Allocate device array */
    rt->devices = (VnpuDevice*)calloc(8, sizeof(VnpuDevice));
    if (!rt->devices) return ECHO_ERR_ALLOC;
    rt->num_devices = 0;
    
    /* Allocate isolate array */
    rt->isolate_capacity = 16;
    rt->isolates = (VnpuIsolate*)calloc(rt->isolate_capacity, sizeof(VnpuIsolate));
    if (!rt->isolates) {
        free(rt->devices);
        return ECHO_ERR_ALLOC;
    }
    rt->num_isolates = 0;
    
    /* Add default CPU device */
    vnpu_device_add(rt, "cpu0", VNPU_DEVICE_CPU);
    
    rt->global_tick = 0;
    rt->initialized = true;
    
    return ECHO_OK;
}

void vnpu_runtime_free(VnpuRuntime* rt) {
    if (!rt->initialized) return;
    
    /* Free all isolates */
    for (size_t i = 0; i < rt->num_isolates; i++) {
        vnpu_isolate_free(&rt->isolates[i]);
    }
    free(rt->isolates);
    
    /* Free devices */
    free(rt->devices);
    
    rt->initialized = false;
}

/* ============================================================================
 * DEVICE MANAGEMENT
 * ============================================================================ */

EchoError vnpu_device_add(VnpuRuntime* rt, const char* name, VnpuDeviceKind kind) {
    if (rt->num_devices >= 8) return ECHO_ERR_INVALID;
    
    VnpuDevice* dev = &rt->devices[rt->num_devices];
    strncpy(dev->name, name, sizeof(dev->name) - 1);
    dev->kind = kind;
    dev->available = true;
    
    switch (kind) {
        case VNPU_DEVICE_CPU:
            dev->threads = 4;  /* Default */
            dev->memory_bytes = 8ULL * 1024 * 1024 * 1024;  /* 8GB */
            break;
        case VNPU_DEVICE_CUDA:
            dev->sm_count = 80;  /* Default for modern GPU */
            dev->memory_bytes = 16ULL * 1024 * 1024 * 1024;  /* 16GB */
            break;
        default:
            dev->memory_bytes = 1ULL * 1024 * 1024 * 1024;  /* 1GB */
            break;
    }
    
    rt->num_devices++;
    return ECHO_OK;
}

VnpuDevice* vnpu_device_get(VnpuRuntime* rt, const char* name) {
    for (size_t i = 0; i < rt->num_devices; i++) {
        if (strcmp(rt->devices[i].name, name) == 0) {
            return &rt->devices[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * ISOLATE MANAGEMENT
 * ============================================================================ */

EchoError vnpu_isolate_create(VnpuRuntime* rt, VnpuIsolate** out,
                              const char* name, VnpuMembrane membrane) {
    if (rt->num_isolates >= rt->isolate_capacity) {
        /* Grow capacity */
        size_t new_cap = rt->isolate_capacity * 2;
        VnpuIsolate* new_isolates = (VnpuIsolate*)realloc(
            rt->isolates, new_cap * sizeof(VnpuIsolate));
        if (!new_isolates) return ECHO_ERR_ALLOC;
        rt->isolates = new_isolates;
        rt->isolate_capacity = new_cap;
    }
    
    VnpuIsolate* isolate = &rt->isolates[rt->num_isolates];
    memset(isolate, 0, sizeof(VnpuIsolate));
    
    strncpy(isolate->name, name, sizeof(isolate->name) - 1);
    isolate->id = rt->num_isolates;
    isolate->membrane = membrane;
    
    /* Allocate private heap */
    isolate->heap_size = 1024 * 1024;  /* 1MB default */
    isolate->heap = malloc(isolate->heap_size);
    if (!isolate->heap) return ECHO_ERR_ALLOC;
    
    /* Initialize scheduler */
    vnpu_scheduler_init(&isolate->scheduler);
    
    /* Allocate port array */
    isolate->ports = (VnpuPort*)calloc(16, sizeof(VnpuPort));
    if (!isolate->ports) {
        free(isolate->heap);
        return ECHO_ERR_ALLOC;
    }
    isolate->num_ports = 0;
    
    isolate->initialized = true;
    isolate->running = false;
    
    rt->num_isolates++;
    *out = isolate;
    
    return ECHO_OK;
}

void vnpu_isolate_free(VnpuIsolate* isolate) {
    if (!isolate->initialized) return;
    
    /* Free ML engine if attached */
    if (isolate->engine) {
        echo_engine_free(isolate->engine);
        free(isolate->engine);
    }
    
    /* Free ports */
    for (size_t i = 0; i < isolate->num_ports; i++) {
        free(isolate->ports[i].buffer);
    }
    free(isolate->ports);
    
    /* Free heap */
    free(isolate->heap);
    
    /* Free policy */
    if (isolate->policy) {
        vnpu_policy_free(isolate->policy);
        free(isolate->policy);
    }
    
    /* Free children array */
    free(isolate->children);
    
    isolate->initialized = false;
}

EchoError vnpu_isolate_start(VnpuIsolate* isolate) {
    if (!isolate->initialized) return ECHO_ERR_INVALID;
    isolate->running = true;
    isolate->scheduler.running = true;
    return ECHO_OK;
}

EchoError vnpu_isolate_stop(VnpuIsolate* isolate) {
    if (!isolate->initialized) return ECHO_ERR_INVALID;
    isolate->running = false;
    isolate->scheduler.running = false;
    return ECHO_OK;
}

/* ============================================================================
 * PORT MANAGEMENT
 * ============================================================================ */

EchoError vnpu_port_create(VnpuIsolate* isolate, const char* name,
                           VnpuPortDirection dir, VnpuPortType type) {
    if (isolate->num_ports >= 16) return ECHO_ERR_INVALID;
    
    VnpuPort* port = &isolate->ports[isolate->num_ports];
    strncpy(port->name, name, sizeof(port->name) - 1);
    port->direction = dir;
    port->type = type;
    port->connected = false;
    
    /* Allocate buffer based on type */
    switch (type) {
        case VNPU_PORT_TYPE_INTENT:
            port->buffer_size = sizeof(VnpuIntentPacket) + 4096;
            break;
        case VNPU_PORT_TYPE_EVIDENCE:
            port->buffer_size = sizeof(VnpuEvidencePacket) + 4096;
            break;
        case VNPU_PORT_TYPE_TENSOR:
            port->buffer_size = 64 * 1024;  /* 64KB */
            break;
        case VNPU_PORT_TYPE_BYTES:
            port->buffer_size = 4096;
            break;
    }
    
    port->buffer = malloc(port->buffer_size);
    if (!port->buffer) return ECHO_ERR_ALLOC;
    
    isolate->num_ports++;
    return ECHO_OK;
}

EchoError vnpu_port_connect(VnpuPort* from, VnpuPort* to) {
    if (from->direction != VNPU_PORT_OUT || to->direction != VNPU_PORT_IN) {
        return ECHO_ERR_INVALID;
    }
    if (from->type != to->type) {
        return ECHO_ERR_INVALID;
    }
    
    from->connected = true;
    to->connected = true;
    
    /* In a real implementation, we'd set up a channel here */
    return ECHO_OK;
}

EchoError vnpu_port_send(VnpuPort* port, const void* data, size_t size) {
    if (port->direction != VNPU_PORT_OUT) return ECHO_ERR_INVALID;
    if (size > port->buffer_size) return ECHO_ERR_INVALID;
    
    memcpy(port->buffer, data, size);
    return ECHO_OK;
}

EchoError vnpu_port_recv(VnpuPort* port, void* data, size_t* size) {
    if (port->direction != VNPU_PORT_IN) return ECHO_ERR_INVALID;
    
    /* In a real implementation, this would block or poll */
    if (*size > port->buffer_size) *size = port->buffer_size;
    memcpy(data, port->buffer, *size);
    
    return ECHO_OK;
}

/* ============================================================================
 * PACKET CREATION
 * ============================================================================ */

#define INTENT_MAGIC 0x494E5454   /* "INTT" */
#define EVIDENCE_MAGIC 0x45564944 /* "EVID" */

EchoError vnpu_intent_packet_create(VnpuIntentPacket* pkt, uint32_t type,
                                    float urgency, const void* payload, size_t size) {
    pkt->magic = INTENT_MAGIC;
    pkt->version = 1;
    pkt->timestamp = get_timestamp_ns();
    pkt->intent_type = type;
    pkt->urgency = urgency;
    pkt->confidence = 1.0f;  /* Default */
    pkt->payload_size = size;
    
    if (size > 0 && payload) {
        pkt->payload = malloc(size);
        if (!pkt->payload) return ECHO_ERR_ALLOC;
        memcpy(pkt->payload, payload, size);
    } else {
        pkt->payload = NULL;
    }
    
    return ECHO_OK;
}

EchoError vnpu_evidence_packet_create(VnpuEvidencePacket* pkt, uint32_t type,
                                      float provenance, const void* payload, size_t size) {
    pkt->magic = EVIDENCE_MAGIC;
    pkt->version = 1;
    pkt->timestamp = get_timestamp_ns();
    pkt->evidence_type = type;
    pkt->provenance = provenance;
    pkt->relevance = 1.0f;  /* Default */
    pkt->payload_size = size;
    
    if (size > 0 && payload) {
        pkt->payload = malloc(size);
        if (!pkt->payload) return ECHO_ERR_ALLOC;
        memcpy(pkt->payload, payload, size);
    } else {
        pkt->payload = NULL;
    }
    
    return ECHO_OK;
}

/* ============================================================================
 * GRAPH MANAGEMENT
 * ============================================================================ */

EchoError vnpu_graph_create(VnpuGraph* graph, const char* name) {
    memset(graph, 0, sizeof(VnpuGraph));
    strncpy(graph->name, name, sizeof(graph->name) - 1);
    
    graph->capacity = 64;
    graph->nodes = (VnpuGraphNode*)calloc(graph->capacity, sizeof(VnpuGraphNode));
    if (!graph->nodes) return ECHO_ERR_ALLOC;
    
    graph->execution_order = (uint32_t*)calloc(graph->capacity, sizeof(uint32_t));
    if (!graph->execution_order) {
        free(graph->nodes);
        return ECHO_ERR_ALLOC;
    }
    
    graph->num_nodes = 0;
    return ECHO_OK;
}

void vnpu_graph_free(VnpuGraph* graph) {
    for (size_t i = 0; i < graph->num_nodes; i++) {
        free(graph->nodes[i].input_node_ids);
        /* Note: output tensors are owned elsewhere */
    }
    free(graph->nodes);
    free(graph->execution_order);
}

EchoError vnpu_graph_add_node(VnpuGraph* graph, VnpuKernel* kernel,
                              const uint32_t* input_ids, size_t num_inputs) {
    if (graph->num_nodes >= graph->capacity) {
        /* Grow capacity */
        size_t new_cap = graph->capacity * 2;
        VnpuGraphNode* new_nodes = (VnpuGraphNode*)realloc(
            graph->nodes, new_cap * sizeof(VnpuGraphNode));
        if (!new_nodes) return ECHO_ERR_ALLOC;
        graph->nodes = new_nodes;
        
        uint32_t* new_order = (uint32_t*)realloc(
            graph->execution_order, new_cap * sizeof(uint32_t));
        if (!new_order) return ECHO_ERR_ALLOC;
        graph->execution_order = new_order;
        
        graph->capacity = new_cap;
    }
    
    VnpuGraphNode* node = &graph->nodes[graph->num_nodes];
    node->id = graph->num_nodes;
    node->kernel = kernel;
    node->num_inputs = num_inputs;
    node->executed = false;
    node->output = NULL;
    
    if (num_inputs > 0) {
        node->input_node_ids = (uint32_t*)malloc(num_inputs * sizeof(uint32_t));
        if (!node->input_node_ids) return ECHO_ERR_ALLOC;
        memcpy(node->input_node_ids, input_ids, num_inputs * sizeof(uint32_t));
    } else {
        node->input_node_ids = NULL;
    }
    
    /* Add to execution order (simple linear for now) */
    graph->execution_order[graph->num_nodes] = graph->num_nodes;
    
    graph->num_nodes++;
    return ECHO_OK;
}

EchoError vnpu_graph_execute(VnpuGraph* graph, VnpuIsolate* isolate) {
    /* Execute nodes in topological order */
    for (size_t i = 0; i < graph->num_nodes; i++) {
        uint32_t node_id = graph->execution_order[i];
        VnpuGraphNode* node = &graph->nodes[node_id];
        
        if (node->executed) continue;
        
        /* Gather inputs */
        const void* inputs[16];
        for (size_t j = 0; j < node->num_inputs; j++) {
            uint32_t input_id = node->input_node_ids[j];
            inputs[j] = graph->nodes[input_id].output;
        }
        
        /* Execute kernel */
        if (node->kernel && node->kernel->func) {
            node->kernel->func(node->output, inputs, node->num_inputs, 
                              node->kernel->params);
        }
        
        node->executed = true;
    }
    
    return ECHO_OK;
}

/* ============================================================================
 * POLICY MANAGEMENT
 * ============================================================================ */

EchoError vnpu_policy_create(VnpuPolicy* policy, const char* name) {
    memset(policy, 0, sizeof(VnpuPolicy));
    strncpy(policy->name, name, sizeof(policy->name) - 1);
    
    policy->capacity = 16;
    policy->rules = (VnpuPolicyRule*)calloc(policy->capacity, sizeof(VnpuPolicyRule));
    if (!policy->rules) return ECHO_ERR_ALLOC;
    
    policy->num_rules = 0;
    return ECHO_OK;
}

void vnpu_policy_free(VnpuPolicy* policy) {
    free(policy->rules);
}

EchoError vnpu_policy_add_rule(VnpuPolicy* policy, VnpuMembrane membrane,
                               VnpuPolicyAction action, const char* target,
                               float min_provenance, uint32_t max_tokens) {
    if (policy->num_rules >= policy->capacity) {
        size_t new_cap = policy->capacity * 2;
        VnpuPolicyRule* new_rules = (VnpuPolicyRule*)realloc(
            policy->rules, new_cap * sizeof(VnpuPolicyRule));
        if (!new_rules) return ECHO_ERR_ALLOC;
        policy->rules = new_rules;
        policy->capacity = new_cap;
    }
    
    VnpuPolicyRule* rule = &policy->rules[policy->num_rules];
    rule->membrane = membrane;
    rule->action = action;
    strncpy(rule->target, target, sizeof(rule->target) - 1);
    rule->min_provenance = min_provenance;
    rule->max_tokens = max_tokens;
    rule->active = true;
    
    policy->num_rules++;
    return ECHO_OK;
}

bool vnpu_policy_check(const VnpuPolicy* policy, VnpuMembrane membrane,
                       const char* target, float provenance, uint32_t tokens) {
    for (size_t i = 0; i < policy->num_rules; i++) {
        const VnpuPolicyRule* rule = &policy->rules[i];
        if (!rule->active) continue;
        if (rule->membrane != membrane) continue;
        if (strcmp(rule->target, target) != 0 && strcmp(rule->target, "*") != 0) continue;
        
        /* Check conditions */
        if (provenance < rule->min_provenance) {
            return rule->action == VNPU_POLICY_DENY;
        }
        if (tokens > rule->max_tokens) {
            return rule->action == VNPU_POLICY_DENY;
        }
        
        return rule->action == VNPU_POLICY_ALLOW;
    }
    
    /* Default: deny if no matching rule */
    return false;
}

/* ============================================================================
 * SCHEDULER
 * 
 * Implements the 12-step cognitive loop with 3 phases (μ/σ/φ).
 * The three streams are phased 4 steps apart (120 degrees).
 * 
 * Step mapping:
 *   Steps {0, 3, 6, 9}  -> Phase μ (Perception)
 *   Steps {1, 4, 7, 10} -> Phase σ (Action)
 *   Steps {2, 5, 8, 11} -> Phase φ (Simulation)
 * ============================================================================ */

void vnpu_scheduler_init(VnpuScheduler* sched) {
    sched->current_step = 0;
    sched->current_phase = VNPU_PHASE_MU;
    sched->tick_count = 0;
    sched->last_tick_ns = get_timestamp_ns();
    sched->running = false;
}

void vnpu_scheduler_tick(VnpuScheduler* sched) {
    if (!sched->running) return;
    
    /* Advance step */
    sched->current_step = (sched->current_step + 1) % 12;
    
    /* Update phase based on step */
    switch (sched->current_step % 3) {
        case 0: sched->current_phase = VNPU_PHASE_MU; break;     /* Perception */
        case 1: sched->current_phase = VNPU_PHASE_SIGMA; break;  /* Action */
        case 2: sched->current_phase = VNPU_PHASE_PHI; break;    /* Simulation */
    }
    
    sched->tick_count++;
    sched->last_tick_ns = get_timestamp_ns();
}

VnpuPhase vnpu_scheduler_get_phase(const VnpuScheduler* sched) {
    return sched->current_phase;
}

/* ============================================================================
 * KERNEL REGISTRY
 * ============================================================================ */

#define MAX_KERNELS 256
static VnpuKernel g_kernel_registry[MAX_KERNELS];
static size_t g_num_kernels = 0;

EchoError vnpu_kernel_register(const char* qualified_name, VnpuKernelFamily family,
                               VnpuKernelFunc func, size_t num_inputs, size_t num_outputs) {
    if (g_num_kernels >= MAX_KERNELS) return ECHO_ERR_INVALID;
    
    VnpuKernel* kernel = &g_kernel_registry[g_num_kernels];
    strncpy(kernel->qualified_name, qualified_name, sizeof(kernel->qualified_name) - 1);
    
    /* Extract short name from qualified name */
    const char* dot = strrchr(qualified_name, '.');
    if (dot) {
        strncpy(kernel->name, dot + 1, sizeof(kernel->name) - 1);
    } else {
        strncpy(kernel->name, qualified_name, sizeof(kernel->name) - 1);
    }
    
    kernel->family = family;
    kernel->func = func;
    kernel->num_inputs = num_inputs;
    kernel->num_outputs = num_outputs;
    kernel->params = NULL;
    
    g_num_kernels++;
    return ECHO_OK;
}

VnpuKernel* vnpu_kernel_lookup(const char* qualified_name) {
    for (size_t i = 0; i < g_num_kernels; i++) {
        if (strcmp(g_kernel_registry[i].qualified_name, qualified_name) == 0) {
            return &g_kernel_registry[i];
        }
    }
    return NULL;
}
