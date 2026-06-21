#ifndef SOTA_RMCA_HPP
#define SOTA_RMCA_HPP

class DeliveryAgent;
struct PDPTask;
class PDPGlobalMemory;

// RMCA(r) [Chen et al. 2021] used as a non-learning scoring baseline: it
// replaces the RL policy inside the same TAM pipeline. Returns a bid score in
// [0,1] that decreases with the agent's marginal insertion cost of `task`
// (eq.13 marginal cost + eq.9 capacity feasibility), so the TAM's argmax picks
// the cheapest-insertion agent. Deterministic, no training buffer.
float rmca_score(const DeliveryAgent& a, const PDPTask& task,
                 PDPGlobalMemory& memory);

#endif // SOTA_RMCA_HPP
