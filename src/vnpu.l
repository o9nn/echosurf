/**
 * vnpu.l - Lexical analyzer for vNPU IR
 * 
 * This is a lex/flex specification for tokenizing the vNPU intermediate
 * representation language. It is designed to be compatible with Plan 9's
 * lex and standard flex.
 * 
 * The vNPU IR is the language that glyphs compile into, and that isolates
 * execute. It defines devices, tensors, kernels, graphs, isolates, and policies.
 * 
 * Example vNPU IR:
 *   vnpu v1;
 *   device cpu0 { kind=cpu; threads=4; }
 *   tensor x : f16[1,128,4096] @cpu0;
 *   kernel k0 = aten.matmul(x,w) -> y;
 *   graph g_main { k0; k1; }
 *   isolate core { membrane=inner; entry g_main; }
 */

%{
#include "vnpu.tab.h"
#include <string.h>
#include <stdlib.h>

/* Line tracking for error messages */
int vnpu_line = 1;
int vnpu_col = 1;

/* Update column position */
#define YY_USER_ACTION vnpu_col += yyleng;
%}

%option noyywrap
%option nounput
%option noinput

%%

 /* Keywords */
"vnpu"          { return VNPU; }
"device"        { return DEVICE; }
"tensor"        { return TENSOR; }
"kernel"        { return KERNEL; }
"graph"         { return GRAPH; }
"isolate"       { return ISOLATE; }
"policy"        { return POLICY; }
"membrane"      { return MEMBRANE; }
"entry"         { return ENTRY; }
"ports"         { return PORTS; }

 /* Membrane types */
"inner"         { return INNER; }
"trans"         { return TRANS; }
"outer"         { return OUTER; }

 /* Policy actions */
"allows"        { return ALLOWS; }
"denies"        { return DENIES; }
"when"          { return WHEN; }

 /* Logical operators */
"and"           { return AND; }
"or"            { return OR; }

 /* Port types */
"Intent"        { return INTENT; }
"Evidence"      { return EVIDENCE; }
"Tensor"        { return TENSOR_TYPE; }
"Bytes"         { return BYTES; }

 /* Data types */
"f16"           { yylval.s = strdup(yytext); return DTYPE; }
"f32"           { yylval.s = strdup(yytext); return DTYPE; }
"bf16"          { yylval.s = strdup(yytext); return DTYPE; }
"i8"            { yylval.s = strdup(yytext); return DTYPE; }
"i16"           { yylval.s = strdup(yytext); return DTYPE; }
"i32"           { yylval.s = strdup(yytext); return DTYPE; }
"i64"           { yylval.s = strdup(yytext); return DTYPE; }
"u8"            { yylval.s = strdup(yytext); return DTYPE; }

 /* Boolean literals */
"true"          { yylval.i = 1; return BOOL; }
"false"         { yylval.i = 0; return BOOL; }

 /* Operators and punctuation */
"->"            { return ARROW; }
">="            { yylval.s = strdup(yytext); return COP; }
"<="            { yylval.s = strdup(yytext); return COP; }
"=="            { yylval.s = strdup(yytext); return COP; }
"!="            { yylval.s = strdup(yytext); return COP; }
">"             { yylval.s = strdup(yytext); return COP; }
"<"             { yylval.s = strdup(yytext); return COP; }
"@"             { return AT; }
":"             { return ':'; }
";"             { return ';'; }
","             { return ','; }
"{"             { return '{'; }
"}"             { return '}'; }
"["             { return '['; }
"]"             { return ']'; }
"("             { return '('; }
")"             { return ')'; }
"="             { return '='; }
"."             { return '.'; }

 /* Version string (e.g., v1, v1.0, v2.1.3) */
v[0-9]+(\.[0-9]+)*  { yylval.s = strdup(yytext); return VERSION; }

 /* Floating point numbers */
[0-9]+"."[0-9]+([eE][+-]?[0-9]+)?  { yylval.f = atof(yytext); return FLOAT; }

 /* Integer numbers */
[0-9]+          { yylval.i = atoi(yytext); return INT; }

 /* Identifiers */
[a-zA-Z_][a-zA-Z0-9_]*  { yylval.s = strdup(yytext); return ID; }

 /* String literals */
\"([^\"\\]|\\.)*\"  { 
    /* Remove quotes and handle escapes */
    yylval.s = strdup(yytext + 1);
    yylval.s[strlen(yylval.s) - 1] = '\0';
    return STRING; 
}

 /* Comments */
"//".*          { /* Single-line comment, skip */ }
"/*"([^*]|\*+[^*/])*\*+"/"  { 
    /* Multi-line comment, count newlines */
    for (int i = 0; yytext[i]; i++) {
        if (yytext[i] == '\n') {
            vnpu_line++;
            vnpu_col = 1;
        }
    }
}

 /* Whitespace */
[ \t\r]+        { /* Skip horizontal whitespace */ }
\n              { vnpu_line++; vnpu_col = 1; }

 /* Catch-all for single characters */
.               { return yytext[0]; }

%%

/* Error reporting helper */
void vnpu_lex_error(const char* msg) {
    fprintf(stderr, "Lexical error at line %d, col %d: %s\n", 
            vnpu_line, vnpu_col, msg);
}
