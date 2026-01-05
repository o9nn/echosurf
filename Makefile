# Echo ML - Makefile for standalone C/C++ compilation with vNPU support
#
# This builds the library for use outside of Node.js (e.g., embedded systems,
# direct C/C++ integration, or as a shared library).
#
# The vNPU (Virtual Neural Processing Unit) provides a membrane-bound neural
# substrate for Deep Tree Echo - a "learnable processor" that represents
# potential process promises.

CC = gcc
CXX = g++
AR = ar

# Optimization flags
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -std=c11
CXXFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -std=c++17

# SIMD flags (adjust for target architecture)
SIMD_FLAGS = -mavx2 -mfma

# Include paths
INCLUDES = -Iinclude

# Source files - Core ML
C_SOURCES_CORE = src/echo_tensor.c src/echo_reservoir.c src/echo_layers.c

# Source files - vNPU Runtime
C_SOURCES_VNPU = src/vnpu_runtime.c

# All sources
C_SOURCES = $(C_SOURCES_CORE) $(C_SOURCES_VNPU)
OBJECTS = $(C_SOURCES:.c=.o)

# Parser sources (optional, requires flex/bison)
LEX_SRC = src/vnpu.l
YACC_SRC = src/vnpu.y
PARSER_OBJS = src/vnpu.tab.o src/lex.yy.o

# Output
LIB_STATIC = libecho_ml.a
LIB_SHARED = libecho_ml.so
TEST_BIN = test_echo_ml

# Default target (without parser)
all: $(LIB_STATIC) $(LIB_SHARED)

# Static library
$(LIB_STATIC): $(OBJECTS)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"
	@ls -lh $@

# Shared library
$(LIB_SHARED): $(OBJECTS)
	$(CC) -shared -o $@ $^ -lm
	@echo "Built shared library: $@"
	@ls -lh $@

# Object files
src/echo_tensor.o: src/echo_tensor.c include/echo_ml.h
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(INCLUDES) -fPIC -c $< -o $@

src/echo_reservoir.o: src/echo_reservoir.c include/echo_ml.h
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(INCLUDES) -fPIC -c $< -o $@

src/echo_layers.o: src/echo_layers.c include/echo_ml.h
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(INCLUDES) -fPIC -c $< -o $@

src/vnpu_runtime.o: src/vnpu_runtime.c include/vnpu.h include/echo_ml.h
	$(CC) $(CFLAGS) $(INCLUDES) -fPIC -c $< -o $@

# Test binary
test: $(LIB_STATIC) tests/test_standalone.c
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(INCLUDES) tests/test_standalone.c -L. -lecho_ml -lm -o $(TEST_BIN)
	./$(TEST_BIN)

# ============================================================================
# PARSER TARGETS (optional, requires flex/bison)
# ============================================================================

# Generate parser from lex/yacc specs
parser: src/vnpu.tab.c src/lex.yy.c
	@echo "Parser generated successfully"

src/vnpu.tab.c src/vnpu.tab.h: $(YACC_SRC)
	bison -d -o src/vnpu.tab.c $(YACC_SRC)

src/lex.yy.c: $(LEX_SRC) src/vnpu.tab.h
	flex -o src/lex.yy.c $(LEX_SRC)

src/vnpu.tab.o: src/vnpu.tab.c
	$(CC) $(CFLAGS) $(INCLUDES) -fPIC -c $< -o $@

src/lex.yy.o: src/lex.yy.c
	$(CC) $(CFLAGS) $(INCLUDES) -fPIC -Wno-unused-function -c $< -o $@

# Build with parser support (requires flex/bison)
with-parser: parser $(OBJECTS) $(PARSER_OBJS)
	$(AR) rcs $(LIB_STATIC) $(OBJECTS) $(PARSER_OBJS)
	$(CC) -shared -o $(LIB_SHARED) $(OBJECTS) $(PARSER_OBJS) -lm
	@echo "Built library with parser support"
	@ls -lh $(LIB_STATIC) $(LIB_SHARED)

# ============================================================================
# UTILITY TARGETS
# ============================================================================

# Clean
clean:
	rm -f $(OBJECTS) $(PARSER_OBJS) $(LIB_STATIC) $(LIB_SHARED) $(TEST_BIN)
	rm -f src/vnpu.tab.c src/vnpu.tab.h src/lex.yy.c

# Install (to /usr/local by default)
PREFIX ?= /usr/local
install: $(LIB_STATIC) $(LIB_SHARED)
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIB_STATIC) $(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(PREFIX)/lib/
	install -m 644 include/echo_ml.h $(PREFIX)/include/
	install -m 644 include/vnpu.h $(PREFIX)/include/

# Size analysis
size: $(LIB_STATIC) $(LIB_SHARED)
	@echo "\n=== Library Size Analysis ==="
	@echo "Static library:"
	@ls -lh $(LIB_STATIC)
	@size $(LIB_STATIC) 2>/dev/null || true
	@echo "\nShared library:"
	@ls -lh $(LIB_SHARED)
	@echo "\nObject files:"
	@ls -lh $(OBJECTS)
	@echo "\nTotal source lines:"
	@wc -l src/*.c include/*.h | tail -1

# Debug build
debug: CFLAGS = -O0 -g -Wall -Wextra -std=c11 -DDEBUG
debug: clean all

# Architecture info
arch:
	@echo "=== Architecture Info ==="
	@echo "Target: $(shell uname -m)"
	@echo "Compiler: $(CC) $(shell $(CC) --version | head -1)"
	@echo "SIMD flags: $(SIMD_FLAGS)"

.PHONY: all test clean install size debug arch parser with-parser
