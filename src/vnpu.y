/**
 * vnpu.y - Parser for vNPU IR
 * 
 * This is a yacc/bison specification for parsing the vNPU intermediate
 * representation language. It is designed to be compatible with Plan 9's
 * yacc and standard bison.
 * 
 * The vNPU IR defines:
 * - Devices: compute resources (CPU, GPU, NPU)
 * - Tensors: data with shape, dtype, device
 * - Kernels: callable operations
 * - Graphs: DAGs of kernels
 * - Isolates: actor/processes with membranes
 * - Policies: access control rules
 */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vnpu.h"

/* External declarations */
extern int yylex(void);
extern int vnpu_line;
extern int vnpu_col;
extern FILE* yyin;

/* Error handling */
void yyerror(const char* s);

/* AST building context */
static VnpuRuntime* g_parse_runtime = NULL;
static VnpuIsolate* g_current_isolate = NULL;
static VnpuGraph* g_current_graph = NULL;
static VnpuPolicy* g_current_policy = NULL;

/* Forward declarations for AST building */
static void add_device(const char* name, const char* kind, int threads);
static void add_tensor(const char* name, const char* dtype, int* dims, int ndims, const char* device);
static void add_kernel(const char* name, const char* op, const char** args, int nargs, const char* output);
static void start_graph(const char* name);
static void end_graph(void);
static void add_graph_node(const char* kernel_name);
static void start_isolate(const char* name);
static void end_isolate(void);
static void set_isolate_membrane(int membrane);
static void set_isolate_entry(const char* graph_name);
static void start_policy(const char* name);
static void end_policy(void);
static void add_policy_rule(int membrane, int action, const char* target, float min_prov, int max_tokens);

%}

/* Value types */
%union {
    int i;
    double f;
    char* s;
    int* dims;
    char** args;
}

/* Tokens */
%token VNPU DEVICE TENSOR KERNEL GRAPH ISOLATE POLICY
%token MEMBRANE ENTRY PORTS
%token INNER TRANS OUTER
%token ALLOWS DENIES WHEN
%token AND OR
%token INTENT EVIDENCE TENSOR_TYPE BYTES
%token AT ARROW
%token <s> ID DTYPE STRING COP VERSION
%token <i> INT BOOL
%token <f> FLOAT

/* Non-terminal types */
%type <i> membrane policy_action
%type <s> qualified_id
%type <dims> shape dims
%type <args> args opt_args

/* Precedence for expressions */
%left OR
%left AND

%%

/* Top-level program */
program
    : header decls
    ;

header
    : VNPU VERSION ';'
        { printf("vNPU IR version: %s\n", $2); free($2); }
    ;

decls
    : /* empty */
    | decls decl
    ;

decl
    : device_decl
    | tensor_decl
    | kernel_decl
    | graph_decl
    | isolate_decl
    | policy_decl
    ;

/* Device declarations */
device_decl
    : DEVICE ID '{' device_props '}'
        { /* Device already added in device_props */ }
    ;

device_props
    : /* empty */
    | device_props device_prop
    ;

device_prop
    : ID '=' ID ';'
        { /* Handle kind=cpu, kind=cuda, etc. */ }
    | ID '=' INT ';'
        { /* Handle threads=4, sm=80, etc. */ }
    ;

/* Tensor declarations */
tensor_decl
    : TENSOR ID ':' DTYPE shape opt_location ';'
        { 
            /* TODO: Build tensor in runtime */
            printf("Tensor: %s : %s\n", $2, $4);
            free($2); free($4);
        }
    ;

shape
    : '[' dims ']'
        { $$ = $2; }
    ;

dims
    : dim
        { 
            $$ = (int*)malloc(16 * sizeof(int));
            $$[0] = 1;  /* Count */
            $$[1] = $<i>1;
        }
    | dims ',' dim
        {
            $$ = $1;
            $$[0]++;
            $$[$$[0]] = $<i>3;
        }
    ;

dim
    : INT
        { $<i>$ = $1; }
    | ID
        { $<i>$ = -1; /* Symbolic dimension */ }
    ;

opt_location
    : /* empty */
    | AT ID
        { free($2); }
    ;

/* Kernel declarations */
kernel_decl
    : KERNEL ID '=' call ARROW ID ';'
        {
            printf("Kernel: %s -> %s\n", $2, $6);
            free($2); free($6);
        }
    ;

call
    : qualified_id '(' opt_args ')'
        { free($1); }
    ;

qualified_id
    : ID '.' ID
        {
            $$ = (char*)malloc(strlen($1) + strlen($3) + 2);
            sprintf($$, "%s.%s", $1, $3);
            free($1); free($3);
        }
    | qualified_id '.' ID
        {
            $$ = (char*)realloc($1, strlen($1) + strlen($3) + 2);
            strcat($$, ".");
            strcat($$, $3);
            free($3);
        }
    ;

opt_args
    : /* empty */
        { $$ = NULL; }
    | args
        { $$ = $1; }
    ;

args
    : arg
        {
            $$ = (char**)malloc(16 * sizeof(char*));
            $$[0] = (char*)(intptr_t)1;  /* Count stored as first element */
            $$[1] = $<s>1;
        }
    | args ',' arg
        {
            $$ = $1;
            int count = (int)(intptr_t)$$[0];
            $$[count + 1] = $<s>3;
            $$[0] = (char*)(intptr_t)(count + 1);
        }
    ;

arg
    : ID
        { $<s>$ = $1; }
    | INT
        { 
            $<s>$ = (char*)malloc(16);
            sprintf($<s>$, "%d", $1);
        }
    | FLOAT
        {
            $<s>$ = (char*)malloc(32);
            sprintf($<s>$, "%f", $1);
        }
    | STRING
        { $<s>$ = $1; }
    ;

/* Graph declarations */
graph_decl
    : GRAPH ID '{' graph_body '}'
        {
            printf("Graph: %s\n", $2);
            free($2);
        }
    ;

graph_body
    : /* empty */
    | graph_body graph_stmt
    ;

graph_stmt
    : ID ';'
        { 
            printf("  Node: %s\n", $1);
            free($1); 
        }
    ;

/* Isolate declarations */
isolate_decl
    : ISOLATE ID '{' isolate_body '}'
        {
            printf("Isolate: %s\n", $2);
            free($2);
        }
    ;

isolate_body
    : /* empty */
    | isolate_body isolate_prop
    ;

isolate_prop
    : MEMBRANE '=' membrane ';'
        { printf("  Membrane: %d\n", $3); }
    | ENTRY ID ';'
        { printf("  Entry: %s\n", $2); free($2); }
    | PORTS '{' port_decls '}'
    ;

membrane
    : INNER     { $$ = VNPU_MEMBRANE_INNER; }
    | TRANS     { $$ = VNPU_MEMBRANE_TRANS; }
    | OUTER     { $$ = VNPU_MEMBRANE_OUTER; }
    ;

port_decls
    : /* empty */
    | port_decls port_decl
    ;

port_decl
    : ID ':' port_type ';'
        { printf("  Port: %s\n", $1); free($1); }
    ;

port_type
    : INTENT
    | EVIDENCE
    | TENSOR_TYPE
    | BYTES
    ;

/* Policy declarations */
policy_decl
    : POLICY ID '{' policy_body '}'
        {
            printf("Policy: %s\n", $2);
            free($2);
        }
    ;

policy_body
    : /* empty */
    | policy_body policy_stmt
    ;

policy_stmt
    : MEMBRANE membrane policy_action ID opt_condition ';'
        {
            printf("  Rule: membrane=%d action=%d target=%s\n", $2, $3, $4);
            free($4);
        }
    ;

policy_action
    : ALLOWS     { $$ = VNPU_POLICY_ALLOW; }
    | DENIES     { $$ = VNPU_POLICY_DENY; }
    ;

opt_condition
    : /* empty */
    | WHEN expr
    ;

expr
    : expr AND expr
    | expr OR expr
    | ID COP literal
        { free($1); free($2); }
    | '(' expr ')'
    ;

literal
    : INT
    | FLOAT
    | STRING
        { free($1); }
    | BOOL
    ;

%%

/* Error handler */
void yyerror(const char* s) {
    fprintf(stderr, "Parse error at line %d, col %d: %s\n", 
            vnpu_line, vnpu_col, s);
}

/* Public API: Parse a vNPU IR file */
EchoError vnpu_parse_file(VnpuRuntime* rt, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return ECHO_ERR_FILE;
    
    g_parse_runtime = rt;
    yyin = f;
    
    int result = yyparse();
    
    fclose(f);
    g_parse_runtime = NULL;
    
    return (result == 0) ? ECHO_OK : ECHO_ERR_INVALID;
}

/* Public API: Parse a vNPU IR string */
EchoError vnpu_parse_string(VnpuRuntime* rt, const char* source) {
    /* For string parsing, we'd need to use yy_scan_string */
    /* This is a simplified implementation */
    g_parse_runtime = rt;
    
    /* Create a temporary file (simple approach) */
    FILE* f = tmpfile();
    if (!f) return ECHO_ERR_FILE;
    
    fputs(source, f);
    rewind(f);
    
    yyin = f;
    int result = yyparse();
    
    fclose(f);
    g_parse_runtime = NULL;
    
    return (result == 0) ? ECHO_OK : ECHO_ERR_INVALID;
}
