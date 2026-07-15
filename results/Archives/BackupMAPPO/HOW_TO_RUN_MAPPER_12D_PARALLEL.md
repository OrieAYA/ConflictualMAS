# Running MAPPER (12-feature) in parallel with the current training

## Plan

The current training (running) uses 13 features (with `deliverability`).
To produce a comparison-grade MAPPER on the 12-feature setup, we need to:

1. Create a separate workspace via `git worktree` so the current training is
   undisturbed
2. Apply a minimal patch: revert `kPolicySz` from 13 to 12, remove the
   `deliverability` computation
3. Build in that workspace, train MAPPER, save its checkpoint with a distinct
   filename

## Step-by-step

### 1. Create a parallel workspace

From the project root, with the current training still running:

```bash
# Create a new branch from main (current code)
git branch mapper-12d-train

# Create a worktree on that branch in a sibling directory
git worktree add ../ConflictualMAS-mapper12 mapper-12d-train
```

This gives you a complete second copy of the project at
`../ConflictualMAS-mapper12/`, on its own branch. Both directories share the
same `.git/`, so you can pull/push/diff between them.

### 2. Apply the 12-feature patch

In the new workspace `../ConflictualMAS-mapper12/`:

**File: `src/DMASforPD/Policy/ObjectiveDMPolicy.hpp`**
```diff
-static constexpr int   kPolicySz   = 13;
+static constexpr int   kPolicySz   = 12;
```

**File: `src/DMASforPD/Policy/ObjectiveDMPolicy.hpp` (PolicyFeatures struct)**
Remove the `deliverability` line:
```diff
-    float deliverability    = 0.f; // steps_remaining / (delivery_steps + 1)
```

Update `PolicyFeatures::to_array` in `ObjectiveDMPolicy.cpp` to copy 12 fields
instead of 13.

**File: `src/DMASforPD/DeliveryAgent/DeliveryAgent.cpp`** (in `try_accept_task`)
Remove the `deliverability` calculation block:
```diff
-    {
-        const float delivery_steps = insertion_cost / std::max(memory.speed_mps, 0.1f);
-        const float steps_remaining = std::max(0.f, 1.f - memory.cur_time_ratio)
-            * static_cast<float>(memory.total_steps);
-        f.deliverability = std::clamp(
-            steps_remaining / (delivery_steps + 1.f), 0.f, 1.f);
-    }
```

### 3. Build and train in the parallel workspace

```bash
cd ../ConflictualMAS-mapper12
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/libs/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Then run with a DIFFERENT output_dir so the parallel run doesn't overwrite the
13-d MAPPER results:

In `src/main.cpp` option 'T', set:
```cpp
const std::string output_dir = "C:\\ConflictualMAS\\results_mapper12";
```

Run:
```bash
./Release/main.exe
# choose T
```

When training completes, `results_mapper12/mapper_seed42.bin` is the 12-d
MAPPER checkpoint, comparable to the 12-d MAPPO of 2026-05-17.

### 4. Run the final comparison (after both trainings finish)

Copy the checkpoints into a single results dir:
```bash
cp ../ConflictualMAS-mapper12/results_mapper12/mapper_seed42.bin \
   results/accepted/mapper_seed42_12d.bin
cp results/accepted/policy_seed42_mappo_working_2026-05-17.bin \
   results/accepted/mappo_seed42_12d.bin
```

For the final 5-mode comparison, you need to checkout the 12-d branch (since
the loadable checkpoints are 12-d):
```bash
git checkout mapper-12d-train
# point option X to the accepted checkpoints
# rebuild, run option X
```

OR keep the 13-d and 12-d evals SEPARATE:
- 13-d run on main: MAPPER(13d) + Hybrid + TamAA + Greedy
- 12-d run on mapper-12d-train: MAPPER(12d) + MAPPO(12d) + TamAA + Greedy

Both produce `episodes_seedXX.csv` with comparable city names, so post-hoc
analysis can merge them.

### 5. Clean up

When done with the parallel workspace:
```bash
git worktree remove ../ConflictualMAS-mapper12
# the mapper-12d-train branch stays in git history for reproducibility
```
