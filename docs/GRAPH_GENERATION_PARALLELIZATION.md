# Graph Generation Parallelization Analysis

## Question: Why not 3 concurrent SLURM jobs?

**User's suggestion:** Instead of 1 SLURM job running 3 Python processes in parallel, use 3 separate SLURM jobs with job 02 dependent on all 3 completing.

**User's assumption:** "I am pretty sure this is a gpu bound process, not a cpu bound one?"

## Critical Finding: CPU-Bound, Not GPU-Bound ❌

**After analyzing the code, graph generation is CPU-bound, NOT GPU-bound:**

1. **No GPU usage in current code:**
   - Scripts import `torch` but don't use `.cuda()` or `.to(device)`
   - TorchANI defaults to CPU when device not specified
   - No `--gres=gpu` in job 01 specification
   - All tensor operations run on CPU

2. **Code evidence:**
```python
# In aev_plig/loaders.py::compute_aevs() - line 202
AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, len(atom_symbols))
# ^ No device parameter = runs on CPU

aev = AEVC.forward((sc.species, sc.coordinates), mol_len)
# ^ Tensor operations on CPU
```

3. **Computational profile:**
   - **AEV computation:** CPU-bound (TorchANI on CPU)
   - **File I/O:** I/O-bound (reading PDB/SDF files)
   - **RDKit parsing:** CPU-bound
   - **Memory:** ~10GB per dataset

**Bottom line:** Restructuring to separate jobs won't improve performance. Graph generation is CPU-bound and doesn't use GPU.

## Current vs Proposed Architecture

### Current (1 Job, 3 Parallel Processes)
```
SLURM Job 01: 4 CPUs, 32GB, 8h
├─ python generate_pdbbind_graphs.py    & (background)
├─ python generate_bindingnet_graphs.py & (background)  
└─ python generate_bindingdb_graphs.py  & (background)
   wait for all
```

### Proposed (3 Separate Jobs)
```
Job 01a: generate_pdbbind.sh    (2 CPUs, 12GB, 6h) ─┐
Job 01b: generate_bindingnet.sh (1 CPU, 6GB, 3h)  ─┼─→ Job 02
Job 01c: generate_bindingdb.sh  (1 CPU, 8GB, 3h)  ─┘
```

## Trade-off Analysis

| Aspect | Current (1 Job) | Proposed (3 Jobs) |
|--------|----------------|-------------------|
| **Queue wait** | 1x | 3x (worse) |
| **Failure isolation** | All or nothing | Independent (better) |
| **Resource flexibility** | Shared 32GB | Per-job allocation (better) |
| **Progress visibility** | Single log | 3 separate logs (better) |
| **Scheduler overhead** | Low | Higher |
| **Simplicity** | Simple | More complex |
| **Memory sharing** | Flexible | Fixed per job |
| **Job 02 start time** | After all finish | After slowest (worse) |

## Recommendations

### ✅ Keep Current for Most Cases

**Recommended when:**
- All 3 datasets always needed
- 32GB, 4 CPUs sufficient
- Queue time valuable
- Standard workflow

**Why it works:**
- Proven and documented
- Efficient resource use
- Single queue wait
- Memory flexibility

### ⚠️ Consider Separate Jobs Only If:

1. **Failure rate high** - Need to preserve partial progress
2. **Memory constrained** - 32GB insufficient for parallel
3. **Dataset size varies greatly** - Need custom resources
4. **Selective processing** - Often run only 1-2 datasets

### Implementation Requirements (If Switching)

**Prerequisites:**
```bash
# Profile each dataset
/usr/bin/time -v python scripts/generate_pdbbind_graphs.py
/usr/bin/time -v python scripts/generate_bindingnet_graphs.py
/usr/bin/time -v python scripts/generate_bindingdb_graphs.py
```

**Example submission script:**
```bash
# Submit 3 jobs in parallel
J1a=$(sbatch --parsable jobs/01a_generate_graphs_pdbbind.sh)
J1b=$(sbatch --parsable jobs/01b_generate_graphs_bindingnet.sh)
J1c=$(sbatch --parsable jobs/01c_generate_graphs_bindingdb.sh)

# Job 02 depends on ALL THREE
J2=$(sbatch --parsable --dependency=afterok:$J1a:$J1b:$J1c jobs/02_create_data.sh)
```

**Resource estimates (need validation):**
- PDBbind: 2 CPUs, 12GB, 6h
- BindingNet: 1 CPU, 6GB, 3h
- BindingDB: 1 CPU, 8GB, 3h

## GPU Acceleration (Separate Question)

**Want to ADD GPU acceleration?** That's a different change:

**Current:** No GPU usage at all  
**Would require:**
1. Modify `compute_aevs()` to accept device parameter
2. Move tensors to GPU in code
3. Add `--gres=gpu:1` to job scripts
4. Test if GPU actually helps (may not for small batches)

**Note:** Can do GPU acceleration with either job structure (1 job or 3 jobs).

## Questions Before Proceeding

1. **What problem are you solving?**
   - Failures too frequent?
   - Memory issues?
   - Just architectural preference?

2. **Dataset characteristics?**
   - Sizes: PDBbind vs BindingNet vs BindingDB?
   - Ever run subset?
   - Memory usage observed?

3. **GPU question:**
   - Want GPU acceleration (code changes)?
   - Or just restructure jobs (no code changes)?

## Implementation Checklist (If Proceeding)

- [ ] Profile resource usage per dataset
- [ ] Create 3 job scripts (01a, 01b, 01c)
- [ ] Update submission scripts
- [ ] Update documentation
- [ ] Test with devel partition
- [ ] Validate multiple dependencies work
- [ ] Update monitoring procedures

## Summary

**Key Points:**
1. ❌ Graph generation is **CPU-bound, not GPU-bound**
2. ✅ Current approach (1 job, 3 processes) works well for most cases
3. ⚠️ Separate jobs adds complexity for failure isolation benefit
4. 📊 Profile before deciding to ensure problem exists
5. �� GPU acceleration is separate decision requiring code changes

**Recommendation:** Keep current unless specific problem identified.
