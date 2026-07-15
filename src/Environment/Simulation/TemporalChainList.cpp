#include "Environment/Simulation/TemporalChainList.hpp"
#include <algorithm>

TemporalChainList::TemporalChainList(float episode_end) {
    start_ = new TemporalNode();              // t = 0 entry point
    end_   = new TemporalNode();              // episode boundary
    end_->time_end  = episode_end;
    start_->next    = end_;
    end_->before    = start_;
}

TemporalChainList::~TemporalChainList() {
    for (TemporalNode* n = start_; n; ) { TemporalNode* nx = n->next; delete n; n = nx; }
}

void TemporalChainList::unlink(TemporalNode* node) {
    if (!node || node->is_sentinel()) return;
    node->before->next = node->next;
    node->next->before = node->before;
    delete node;
}

void TemporalChainList::purge_expired(float t_now) {
    // Segments are ordered by time_end, so all expired ones form a front prefix.
    while (start_->next != end_ && start_->next->time_end < t_now)
        unlink(start_->next);
}

TemporalNode* TemporalChainList::insert(float time_end, float time_spend, int weight, int agent_id) {
    purge_expired(last_time_);

    TemporalNode* n = new TemporalNode();
    n->time_end   = time_end;
    n->time_spend = time_spend;
    n->weight     = weight;
    n->agent_id   = agent_id;

    // Keep the list ordered by ascending time_end. Fast path: when time_end is
    // non-decreasing (the case for bulk sorted injection), append just before
    // the end sentinel in O(1); otherwise fall back to a front position search.
    TemporalNode* cur;
    if (start_->next != end_ && end_->before->time_end <= time_end)
        cur = end_;
    else {
        cur = start_->next;
        while (cur != end_ && cur->time_end < time_end) cur = cur->next;
    }
    n->next           = cur;
    n->before         = cur->before;
    cur->before->next = n;
    cur->before       = n;
    return n;
}

void TemporalChainList::remove(TemporalNode* node) {
    purge_expired(last_time_);
    unlink(node);
}

bool TemporalChainList::remove_interval(float time_end, float time_spend, int weight) {
    purge_expired(last_time_);
    for (TemporalNode* x = start_->next; x != end_; x = x->next)
        if (x->time_end == time_end && x->time_spend == time_spend && x->weight == weight) {
            unlink(x);
            return true;
        }
    return false;
}

int TemporalChainList::load_at(float t) const {
    int load = 0;
    for (TemporalNode* x = start_->next; x != end_; x = x->next)
        if (x->time_begin() <= t && t <= x->time_end) load += x->weight;
    return load;
}

TemporalNode* TemporalChainList::find(float t) {
    last_time_ = std::max(last_time_, t);
    purge_expired(last_time_);
    TemporalNode* cur = start_->next;
    while (cur != end_ && t > cur->time_end) cur = cur->next;
    return cur;
}

int TemporalChainList::size() const {
    int n = 0;
    for (TemporalNode* c = start_->next; c != end_; c = c->next) ++n;
    return n;
}
