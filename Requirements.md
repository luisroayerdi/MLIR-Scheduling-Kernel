# MLIR Attention Pipeline - Requirements Specification

**Version:** 1.0  
**Date:** April 2026  
**Target:** FlashAttention 1 & 2 Compiler Implementation

---

## 1. Project Overview

### 1.1 Core Problem Statement

**Problem:**  
Modern attention kernels (FlashAttention) achieve 2-10x speedups over naive implementations through hand-written CUDA code. These optimizations are:
- Hardware-specific (NVIDIA-only)
- Opaque (thousands of lines of low-level code)
- Not reusable (each new operation needs new hand-tuning)
- Not portable (AMD/Intel GPUs require complete rewrites)

**Research Question:**  
Can we express FlashAttention 1 & 2 optimizations as modular, reusable compiler passes in MLIR, and how much of the performance gap can we close?

**What We're NOT Doing:**  
❌ Beating FlashAttention-2's performance  
❌ Production-ready system  
❌ Inventing new attention algorithms  

**What We ARE Doing:**  
✅ Systematically implementing FA1/FA2 optimizations as MLIR passes  
✅ Measuring what compilers CAN achieve  
✅ Identifying where compilers CANNOT achieve hand-tuned performance  
✅ Documenting tradeoffs and design decisions  
✅ Providing recommendations for MLIR/compiler developers  

---

### 1.2 Success Criteria

**Minimum Viable Success:**
- [ ] Fusion pass implemented and tested (correctness)
- [ ] Tiling pass implemented and tested (correctness)
- [ ] Pipeline generates executable CPU code
- [ ] Outperforms unfused baseline measurably (>20% speedup)
- [ ] Numerical correctness validated against PyTorch (error < 1e-5)

**Target Success:**
- [ ] All minimum viable criteria met
- [ ] GPU lowering functional (nvgpu dialect)
- [ ] Tensor cores utilized (verified via profiler)
- [ ] Performance comparison table complete (all baselines measured)
- [ ] Gap analysis documented (why we're X% slower than FA2)

**Stretch Success:**
- [ ] Matches torch.compile performance
- [ ] Multi-backend support (NVIDIA + AMD)
- [ ] One FA2 optimization fully working (e.g., better work distribution)

---

## 2. FlashAttention 1 & 2 Optimization Taxonomy

### 2.1 FlashAttention 1 Optimizations (Dao et al., 2022)

**Paper Reference:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

| Optimization | Description | Compiler Expressibility | Implementation Priority |
|-------------|-------------|------------------------|------------------------|
| **Op Fusion** | Merge MatMul→Scale→Mask→Softmax into single kernel | ✅ High | **REQUIRED** |
| **Tiling** | Break computation into tiles that fit in SRAM | ✅ High | **REQUIRED** |
| **Recomputation** | Recompute attention in backward pass instead of storing | ⚠️ Medium | Out of scope (forward only) |
| **Online Softmax** | Incremental softmax computation during tiling | ✅ Medium | Implicit in fusion |

**Rationale for FA1 Focus:**
- Core techniques (fusion + tiling) are well within compiler capabilities
- Well-documented with clear algorithms
- Hardware-agnostic (works on any GPU)
- Sufficient for complete research study

---

### 2.2 FlashAttention 2 Optimizations (Dao, 2023)

**Paper Reference:** [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

| Optimization | Description | Compiler Expressibility | Implementation Priority |
|-------------|-------------|------------------------|------------------------|
| **Reduced Non-MatMul FLOPs** | Optimize softmax, avoid redundant ops | ✅ Medium | **SHOULD HAVE** |
| **Better Parallelization** | Optimize thread block dimensions | ✅ Medium | **SHOULD HAVE** |
| **Tensor Core Utilization** | Explicit tensor core lowering | ✅ High | **REQUIRED** |
| **Work Partitioning** | Better work distribution across SMs | ⚠️ Medium-Low | **STRETCH GOAL** |
| **Register Optimization** | Minimize register spills | ❌ Low | Out of scope (backend) |

**Rationale for FA2 Inclusion:**
- Tensor core lowering demonstrates backend-specific optimizations
- Shows where compiler abstractions start to struggle
- Provides data for "what's achievable" analysis
- Work partitioning exploration = research contribution even if it fails

**Important:** We attempt ALL FA2 optimizations that are plausibly compiler-expressible. Failures are valuable research data.

---

### 2.3 Out of Scope (FA3/FA4)

**Not Implementing:**
- ❌ Warp specialization (FA3) - requires low-level control not in MLIR
- ❌ Async memory operations (FA3) - limited MLIR support
- ❌ Dynamic tiling (FA4) - requires runtime compilation
- ❌ Blackwell-specific optimizations (FA4) - too hardware-specific, too recent

**Use In Project:**
- Reference in "Limitations" section
- Discuss in "Future Work"
- Cite as examples of "where compilers struggle"

---

## 3. MLIR Pass Specifications

### 3.1 Pass 1: Operation Fusion

**Purpose:**  
Merge linalg.matmul → scale → mask → softmax into single `attention.fused` operation.

**Input IR:**
```mlir
%qk = linalg.matmul ins(%Q, %K : memref<MxDxf32>, memref<NxDxf32>) outs(%qk_out : memref<MxNxf32>)
%scaled = linalg.generic { arith.divf %qk, %scale }
%masked = linalg.generic { arith.select %mask, %scaled, %neg_inf }
%probs = linalg.softmax ins(%masked)
```

**Output IR:**
```mlir
%probs = attention.fused ins(%Q, %K, %scale, %mask : ...) outs(%output : ...)
```

**Algorithm:**
1. Pattern match 4-op sequence
2. Verify data dependencies (no intervening uses)
3. Create fused operation
4. Replace sequence with fused op

**Correctness Tests:**
- [ ] Numerical: Output matches unfused within 1e-5
- [ ] Edge cases: Empty sequences, all-masked, numerical stability
- [ ] Compilation: Fused op verifies correctly

**Performance Expectation:**
- Memory traffic reduction: 40-60%
- Speedup over unfused: 1.4-1.8x

**Justification:**
- FA1 core technique
- Standard compiler optimization (operation fusion)
- Enables all downstream passes

---

### 3.2 Pass 2: Memory-Aware Tiling

**Purpose:**  
Insert tiling loops to ensure tiles fit in GPU SRAM.

**Input IR:**
```mlir
attention.fused ins(%Q, %K : memref<1024x64xf32>, memref<1024x64xf32>) ...
```

**Output IR:**
```mlir
for %tile_i = 0 to 8 step 1 {  // 1024/128 = 8 tiles
  for %tile_j = 0 to 8 step 1 {
    %Q_tile = memref.subview %Q[%tile_i*128, 0][128, 64]
    %K_tile = memref.subview %K[%tile_j*128, 0][128, 64]
    attention.fused ins(%Q_tile, %K_tile : ...) ...
  }
}
```

**Algorithm:**
1. Query target SRAM size (e.g., 192KB for A100)
2. Calculate max tile size: `tile_size = sqrt(SRAM_size / (3 * sizeof(float)))`
3. Round down to tensor core alignment (16×16)
4. Insert affine.for loops with tile size

**Correctness Tests:**
- [ ] Numerical: Tiled output matches untiled within 1e-5
- [ ] Memory: Tile size fits in SRAM (analytical check)
- [ ] Coverage: All elements processed exactly once

**Performance Expectation:**
- SRAM utilization: >80%
- Speedup over unfused+untiled: 1.8-2.5x

**Justification:**
- FA1 core technique
- Memory hierarchy optimization fundamental to GPU performance
- Enables backend to generate efficient memory access patterns

---

### 3.3 Pass 3: Vectorization

**Purpose:**  
Convert scalar operations to vector operations (SIMD).

**Input IR:**
```mlir
for %i in range(1024) {
  %val = memref.load %input[%i]
  %result = arith.addf %val, %const
  memref.store %result, %output[%i]
}
```

**Output IR:**
```mlir
for %i in range(128) {  // 1024/8 = 128
  %vec = vector.load %input[%i*8 : %i*8+8]
  %result = vector.addf %vec, %const_vec
  vector.store %result, %output[%i*8 : %i*8+8]
}
```

**Algorithm:**
1. Detect vectorizable loops (no loop-carried dependencies)
2. Determine vector width (8 for f32 on typical GPUs)
3. Insert vector.load/vector.store
4. Generate remainder loop for non-divisible sizes

**Correctness Tests:**
- [ ] Numerical: Vectorized output matches scalar
- [ ] Remainder: Non-divisible sizes handled correctly

**Performance Expectation:**
- Throughput increase: 1.5-2x over scalar
- Approach peak memory bandwidth

**Justification:**
- Standard compiler optimization
- Necessary for competitive performance
- Demonstrates MLIR's vector dialect capabilities

---

### 3.4 Pass 4: Causal Mask Specialization

**Purpose:**  
Generate specialized kernels for different causal mask tile types.

**Algorithm:**
1. Classify tiles into three categories:
   - **Full tiles:** All elements valid (below diagonal)
   - **Masked tiles:** All elements invalid (above diagonal)
   - **Boundary tiles:** Partial mask (straddles diagonal)
2. Generate three kernel variants:
   - Full: No mask checks
   - Masked: Skip computation
   - Boundary: Check mask only at edges
3. Insert dispatch logic

**Correctness Tests:**
- [ ] Numerical: Output matches generic masked attention
- [ ] Classification: All tiles correctly categorized
- [ ] Edge cases: Square matrices, non-square, small sizes

**Performance Expectation:**
- Speedup over generic masking: 1.15-1.3x
- Reduced branch divergence (verify via profiler)

**Justification:**
- Domain-specific optimization
- Demonstrates compiler can encode expert knowledge
- Branching kills GPU performance - specialization avoids it

---

### 3.5 Pass 5: GPU Backend Lowering (Tensor Cores)

**Purpose:**  
Lower to nvgpu dialect to utilize tensor cores.

**Input IR:**
```mlir
linalg.matmul ins(%A, %B : memref<16x16xf32>, memref<16x16xf32>) outs(%C : memref<16x16xf32>)
```

**Output IR:**
```mlir
%A_frag = nvgpu.ldmatrix %A : memref<16x16xf32> -> !nvgpu.mma_matrix<16x16xf32>
%B_frag = nvgpu.ldmatrix %B : memref<16x16xf32> -> !nvgpu.mma_matrix<16x16xf32>
%C_frag = nvgpu.mma %A_frag, %B_frag, %C_acc : ... -> !nvgpu.mma_matrix<16x16xf32>
nvgpu.stmatrix %C_frag, %C : !nvgpu.mma_matrix<16x16xf32> -> memref<16x16xf32>
```

**Algorithm:**
1. Detect matmul candidates (must be 16×16 aligned)
2. Insert data layout transformations (row-major → tensor core format)
3. Replace linalg.matmul with nvgpu.mma
4. Insert matrix load/store operations

**Correctness Tests:**
- [ ] Numerical: Tensor core output matches standard matmul
- [ ] Alignment: Verify 16×16 tile alignment
- [ ] PTX inspection: Verify `mma.sync` instructions generated

**Performance Expectation:**
- Tensor core utilization: >70% (verify via profiler)
- Speedup over standard matmul: 8-12x
- TFLOPS: Approach 70-80% of theoretical peak

**Justification:**
- FA2 technique
- Shows backend-specific optimization in compiler
- Demonstrates gap between high-level fusion and low-level hardware exploitation

---

### 3.6 Pass 6: Work Distribution Optimization (FA2)

**Purpose:**  
Improve thread block dimensions and work partitioning.

**Algorithm:**
1. Analyze workload characteristics (sequence length, head dim)
2. Compute optimal block dimensions based on:
   - SM count
   - Register pressure
   - SRAM capacity
3. Generate backend hints for work distribution

**Correctness Tests:**
- [ ] Numerical: Output unchanged
- [ ] Resource usage: No register spills (verify via profiler)

**Performance Expectation:**
- GPU occupancy: >75%
- Speedup: 1.1-1.2x over naive distribution

**Priority:** **STRETCH GOAL**  
**Rationale:** FA2 technique, but high risk - may not be expressible at MLIR level. Attempt if core passes succeed.

---

## 4. Testing & Validation Strategy

### 4.1 Correctness Testing

**Test Suite Structure:**
```
test/
├── unit/                    # Individual pass tests
│   ├── fusion_pass.mlir     # Does fusion preserve semantics?
│   ├── tiling_pass.mlir     # Does tiling cover all elements?
│   └── vectorization.mlir   # Does vectorization match scalar?
├── integration/             # Full pipeline tests
│   ├── cpu_pipeline.mlir    # End-to-end CPU execution
│   └── gpu_pipeline.mlir    # End-to-end GPU execution
└── numerical/               # Accuracy validation
    ├── pytorch_reference.py # Generate ground truth
    └── validate_output.py   # Compare MLIR vs PyTorch
```

**Numerical Validation Protocol:**
```python
def validate_correctness(mlir_output, pytorch_output):
    """
    Validates MLIR implementation matches PyTorch reference.
    
    Passing criteria:
    - Max absolute error < 1e-5 (fp32 precision)
    - Mean absolute error < 1e-6
    - 99.9% of elements within 1e-5
    """
    max_error = np.max(np.abs(mlir_output - pytorch_output))
    mean_error = np.mean(np.abs(mlir_output - pytorch_output))
    within_tol = np.sum(np.abs(mlir_output - pytorch_output) < 1e-5) / mlir_output.size
    
    assert max_error < 1e-5, f"Max error {max_error} exceeds tolerance"
    assert mean_error < 1e-6, f"Mean error {mean_error} exceeds tolerance"
    assert within_tol > 0.999, f"Only {within_tol*100}% within tolerance"
```

**Test Cases:**
- [ ] Sequence lengths: [128, 256, 512, 1024, 2048, 4096]
- [ ] Batch sizes: [1, 4, 8, 16, 32]
- [ ] Head dimensions: [64, 128]
- [ ] Edge cases: Empty sequences, all-masked, numerical stability (large values)

---

### 4.2 Performance Testing (Pre-Hardware)

**CPU Validation:**

Before GPU testing, validate on CPU:
```bash
# Build and run
./build/bin/attention-opt test.mlir --fusion-pass --tiling-pass | \
  mlir-cpu-runner -e main -entry-point-result=void

# Profiling
perf stat -e cycles,instructions,cache-misses ./cpu_executable
```

**Passing Criteria (CPU):**
- [ ] Executes without errors
- [ ] Numerical correctness validated
- [ ] Memory usage reasonable (no excessive allocations)
- [ ] Speedup over unfused baseline: >1.2x

**If CPU tests fail:**  
❌ Do NOT proceed to GPU testing  
✅ Debug and fix CPU implementation first

---

### 4.3 Hardware Testing (GPU) - MAIN MILESTONE

**Hardware Requirements:**
- **Primary:** NVIDIA A100 GPU (80GB)
- **Secondary:** NVIDIA H100 (if available)
- **Fallback:** RTX 4090 (consumer GPU)

**Testing Protocol:**

**Warmup Phase:**
```python
# Discard first 10 iterations (cold cache, JIT compilation)
for _ in range(10):
    run_kernel(Q, K, V)
torch.cuda.synchronize()
```

**Measurement Phase:**
```python
# Run 100 iterations, report median
times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    run_kernel(Q, K, V)
    end.record()
    
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

median_time = np.median(times)
std_dev = np.std(times)
```

**Statistical Validity:**
- Report median (not mean) - robust to outliers
- Report standard deviation
- Flag high variance (>5%) as measurement issue
- Control for thermal throttling, background processes

---

### 4.4 Baseline Comparisons

**Baseline 1: Unfused PyTorch**
```python
import torch.nn.functional as F

def unfused_attention(Q, K, V, mask):
    qk = torch.matmul(Q, K.transpose(-2, -1))
    scaled = qk / math.sqrt(d_k)
    masked = scaled.masked_fill(mask, float('-inf'))
    probs = F.softmax(masked, dim=-1)
    output = torch.matmul(probs, V)
    return output
```

**Purpose:** Demonstrates cost of NOT fusing  
**Expected Result:** Our worst-case lower bound

---

**Baseline 2: torch.compile (Triton)**
```python
@torch.compile
def compiled_attention(Q, K, V, mask):
    # Same as unfused
    return unfused_attention(Q, K, V, mask)
```

**Purpose:** State-of-art automatic fusion  
**Expected Result:** Our competitive target (should get within 20-30%)

---

**Baseline 3: FlashAttention-2 (Reference)**
```python
from flash_attn import flash_attn_func

output = flash_attn_func(Q, K, V, causal=True)
```

**Purpose:** Hand-optimized reference  
**Expected Result:** Upper bound (we won't beat this, that's OK)

---

**Baseline 4: Our MLIR Pipeline (Ablation Study)**

Test each pass incrementally:

| Configuration | Passes Enabled | Purpose |
|--------------|----------------|---------|
| MLIR-Unfused | None | Sanity check |
| MLIR-Fused | Fusion only | Measure fusion impact |
| MLIR-Tiled | Fusion + Tiling | Measure tiling impact |
| MLIR-Vector | Fusion + Tiling + Vectorization | Measure vectorization |
| MLIR-GPU | All + Tensor cores | Full pipeline |

**This ablation study IS the core research contribution.**

---

### 4.5 Performance Metrics

**Primary Metrics:**

1. **Throughput (Tokens/second)**
   ```python
   throughput = (batch_size * seq_len) / median_time
   ```

2. **Memory Bandwidth (GB/s)**
   ```python
   bytes_moved = (Q.numel() + K.numel() + V.numel() + output.numel()) * 4  # fp32
   bandwidth = bytes_moved / median_time
   ```

3. **TFLOPS (GPU only)**
   ```python
   # For attention: 2 matmuls + softmax overhead
   flops = 2 * batch * seq * seq * head_dim
   tflops = flops / (median_time * 1e12)
   ```

4. **Speedup vs Baseline**
   ```python
   speedup = baseline_time / our_time
   ```

**Secondary Metrics (Profiler):**

Using `nsight compute`:
```bash
ncu --set full --export profile.ncu-rep ./mlir_attention
```

Analyze:
- [ ] Memory bandwidth utilization (target: >80%)
- [ ] Tensor core utilization (target: >70%)
- [ ] SM occupancy (target: >75%)
- [ ] L2 cache hit rate
- [ ] Register spills (target: 0)

---

### 4.6 Acceptance Criteria for Hardware Testing

**Go/No-Go Decision:**

**PROCEED to full benchmarking if:**
- ✅ Correctness: All tests pass (error < 1e-5)
- ✅ Functionality: No crashes, no hangs
- ✅ Minimum performance: >1.5x speedup over unfused baseline
- ✅ Profiler sanity: Tensor cores being used (if Pass 5 enabled)

**STOP and debug if:**
- ❌ Correctness fails
- ❌ Performance < 1.2x vs unfused (something is very wrong)
- ❌ Profiler shows red flags (0% tensor core usage, excessive memory traffic)

---

## 5. Documentation Requirements

### 5.1 Repository Structure

```
MLIR-Scheduling-Kernel/
├── README.md                        # Quick start, build instructions
├── REQUIREMENTS.md                  # This document
├── DESIGN.md                        # Design decisions (created by Claude Code)
├── TRADEOFFS.md                     # Performance/design tradeoffs (updated continuously)
├── docs/
│   ├── passes/
│   │   ├── fusion.md                # Pass 1 design doc
│   │   ├── tiling.md                # Pass 2 design doc
│   │   └── ...
│   ├── benchmarking.md              # How to run benchmarks
│   └── profiling.md                 # Profiler guide
├── include/Attention/               # Headers
├── lib/Attention/                   # Implementation
├── test/                            # Tests
├── benchmarks/                      # Benchmark scripts
│   ├── baselines/                   # PyTorch, torch.compile, FA2
│   ├── mlir/                        # Our pipeline
│   └── analysis/                    # Result plotting
└── results/                         # Experimental data (gitignored)
```

---

### 5.2 Documentation Ownership

**README.md** (User-facing):
- Quick start guide
- Build instructions
- How to run basic tests
- Links to detailed docs

**DESIGN.md** (Created by Claude Code):
- **CRITICAL:** Claude Code creates this FIRST, before implementing
- One section per pass
- Each section includes:
  - Algorithm pseudocode
  - IR transformation example
  - Rationale for design choices
  - Known limitations
- Updated as design evolves

**TRADEOFFS.md** (Living document):
- **Updated continuously during development**
- Documents every performance/correctness/complexity tradeoff
- Format:
  ```markdown
  ## Tradeoff: Tile Size Selection
  
  **Decision:** Static 128×128 tiles
  
  **Alternatives Considered:**
  - Dynamic tile sizing (runtime)
  - Multiple fixed sizes (compile-time selection)
  
  **Chosen Because:**
  - Simplicity: Static easier to reason about
  - Performance: 128×128 optimal for A100 SRAM (192KB)
  
  **Cost:**
  - Not optimal for other GPUs (H100 has more SRAM)
  - Future: Make configurable via pass option
  
  **Measurement:**
  - SRAM utilization: 85% on A100
  - Would be only 60% optimal on H100
  ```

**docs/passes/*.md** (Technical deep-dives):
- Detailed implementation notes
- Edge cases handled
- Testing strategy
- Future improvements

---

### 5.3 Documentation Requirements for Claude Code

**Before implementing any pass, Claude Code MUST:**

1. **Update DESIGN.md** with proposed design
2. **Wait for human approval** of design
3. **Implement** the pass
4. **Update TRADEOFFS.md** with any decisions made during implementation
5. **Create docs/passes/{pass_name}.md** with implementation details

**Claude Code should ASK human:**
- "I'm about to implement tiling pass with 128×128 tiles. Design doc updated. Approve?"
- "I found a tradeoff: static vs dynamic tiling. Should I document in TRADEOFFS.md now or after measurement?"

---

## 6. Development Workflow

### 6.1 Iterative Development Loop

```
Phase 1: Requirements (THIS DOCUMENT)
    ↓
Phase 2: Design Document (Claude Code creates DESIGN.md)
    ↓ (Human reviews and approves)
Phase 3: Implementation (Pass by Pass)
    ↓
    For each pass:
        Claude Code: Implement
            ↓
        Human: Review code
            ↓
        Claude Code: Write tests
            ↓
        Human: Run tests, verify
            ↓
        Claude Code: Update TRADEOFFS.md
            ↓
        Human: Approve and merge
    ↓
Phase 4: Integration Testing
    ↓
Phase 5: CPU Validation
    ↓ (If pass → proceed, if fail → debug)
Phase 6: GPU Hardware Testing (MILESTONE)
    ↓
Phase 7: Benchmarking & Analysis
    ↓
Phase 8: Paper/Report Writing
```

---

### 6.2 Pass Implementation Order

**Week 1-2: Pass 1 (Fusion)**
- Claude Code: Create design doc section
- Human: Approve design
- Claude Code: Implement pattern matching
- Claude Code: Write unit tests
- Human: Review and test
- Claude Code: Document tradeoffs

**Week 3-4: Pass 2 (Tiling)**
- Repeat above workflow

**Week 5-6: Pass 3 (Vectorization)**
- Repeat above workflow

**Week 7-8: Integration & CPU Testing**
- Claude Code: Write integration tests
- Human: Run CPU benchmarks
- Claude Code: Document results

**CHECKPOINT: CPU validation must pass before proceeding**

**Week 9-10: Pass 5 (GPU Lowering)**
- Repeat pass workflow
- Add profiler validation

**Week 11-12: Pass 4 (Mask Specialization)**
- Repeat pass workflow

**Week 13-14: Hardware Testing**
- Human: Run on GPU
- Claude Code: Analyze profiler output
- Human: Iterate based on results

**Week 15-16: Benchmarking & Analysis**
- Human: Run full benchmark suite
- Claude Code: Generate plots
- Human: Write analysis

---

### 6.3 Git Workflow

**Branch Strategy:**
```
main
├── pass/fusion         # Pass 1 development
├── pass/tiling         # Pass 2 development
├── pass/vectorization  # Pass 3 development
└── integration         # Integration testing
```

**Commit Message Format:**
```
[PASS/FUSION] Implement pattern matcher for matmul sequence

- Detects linalg.matmul -> scale -> mask -> softmax
- Handles edge case: intervening uses
- Tests: 15 added, all passing

Tradeoffs:
- Only matches exact sequence (not reordered ops)
- Future: Use more flexible matching

Refs: #issue-number
```

---

## 7. FA2 Optimization Implementation Details

### 7.1 Mandatory FA2 Explorations

**These MUST be attempted (even if they fail):**

1. **Tensor Core Utilization**
   - **Status:** REQUIRED
   - **Expected:** High chance of success
   - **Rationale:** Well-defined MLIR support (nvgpu dialect)

2. **Reduced Non-MatMul FLOPs**
   - **Status:** SHOULD HAVE
   - **Expected:** Medium chance of success
   - **Rationale:** Compiler can optimize softmax, but may not match hand-tuning

3. **Better Parallelization**
   - **Status:** SHOULD HAVE
   - **Expected:** Medium chance of success
   - **Rationale:** Backend can specify block dims, but optimal params are heuristic

4. **Work Partitioning**
   - **Status:** STRETCH GOAL
   - **Expected:** Low chance of success
   - **Rationale:** May require runtime scheduling, hard to express statically

**Important:** Even if #4 fails, we document WHY it failed. This is research value.

---

### 7.2 Claude Code FA2 Implementation Protocol

**Before implementing ANY FA2 optimization:**

1. **Claude Code reads FA2 paper section**
2. **Claude Code extracts key technique**
3. **Claude Code asks human:**
   - "FA2 describes {technique}. I interpret this as {MLIR approach}. Does this match the literature?"
4. **Human confirms or corrects interpretation**
5. **Claude Code updates DESIGN.md**
6. **Claude Code implements**

**Example:**
```
Claude Code: "FA2 Section 3.1 describes 'reducing non-matmul FLOPs by 
avoiding redundant rescaling in softmax'. I interpret this as: instead 
of computing exp(x - max) then dividing by sum, we compute 
exp(x) * scale where scale absorbs both max and sum. 

Should I implement this as a pattern rewrite on linalg.softmax, or as 
part of the fusion pass?"
