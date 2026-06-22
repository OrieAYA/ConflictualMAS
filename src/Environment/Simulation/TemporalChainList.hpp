#ifndef ENVIRONMENT_TEMPORAL_CHAIN_LIST_HPP
#define ENVIRONMENT_TEMPORAL_CHAIN_LIST_HPP

#include <osmium/osm/types.hpp>
#include <set>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// Temporal chained list — event stream attached to one spatial element.
// ════════════════════════════════════════════════════════════════════════════
//
// One TemporalChainList is the event stream of a SINGLE node or edge. It is a
// doubly-linked list of time segments ordered by time_end, bounded by two
// sentinels: `start` (t = 0, the entry point, never purged) and `end` (the
// end-of-episode time). The same structure serves nodes and edges — only the
// meaning of incident_elements differs (see TemporalGraph).
//
// Each segment is the interval [time_end - time_spend, time_end].
// present_agent = number of agents concurrently occupying that interval; it is
// maintained incrementally so two segments sharing an interval both reflect the
// shared occupancy ("certified" on every insert/remove).
//
// Three operations, each first purging expired segments (time_end < t*, the
// last observed time) EXCEPT the t = 0 start sentinel:
//   insert(time_end, time_spend) — add an occupancy in time order; overlapping
//                                  segments get present_agent bumped both ways.
//   remove(node)                 — the inverse: relink before↔after, decrement
//                                  the overlaps it had bumped.
//   find(t)                      — walk while t > time_end → next; returns the
//                                  segment covering t (or the end sentinel).
struct TemporalNode {
    TemporalNode* before = nullptr;   // null only for the start sentinel
    TemporalNode* next   = nullptr;   // null only for the end sentinel

    float time_end   = 0.f;           // until when the event holds (absolute time)
    float time_spend = 0.f;           // for how long; begin = time_end - time_spend

    // Static relations of the element this chain belongs to, by OSM id:
    //   node chain → incident edges      |   edge chain → its two endpoint nodes
    std::vector<osmium::object_id_type> incident_elements;

    std::set<int> objective_id;        // task ids active in this interval (nodes)
    int           present_agent = 0;   // concurrent agent count over this interval

    float time_begin() const { return time_end - time_spend; }
    bool  is_sentinel() const { return before == nullptr || next == nullptr; }
};

class TemporalChainList {
public:
    explicit TemporalChainList(float episode_end = 1e30f);
    ~TemporalChainList();

    // Pointer-owning linked list: non-copyable and non-movable. Held by value
    // in node-based containers (unordered_map) via in-place construction, so it
    // is never moved/copied after creation.
    TemporalChainList(const TemporalChainList&)            = delete;
    TemporalChainList& operator=(const TemporalChainList&) = delete;
    TemporalChainList(TemporalChainList&&)                 = delete;
    TemporalChainList& operator=(TemporalChainList&&)      = delete;

    TemporalNode* start() const { return start_; }
    TemporalNode* end()   const { return end_; }

    // Insert an occupancy [time_end - time_spend, time_end] (one agent). Returns
    // the new segment so the caller can fill objective_id / incident_elements.
    TemporalNode* insert(float time_end, float time_spend);

    // Remove a segment previously returned by insert(). No-op on sentinels.
    void remove(TemporalNode* node);

    // First segment with time_end >= t (the one covering / following t), or the
    // end sentinel if none. Advances t* to t and purges what is now expired.
    TemporalNode* find(float t);

    // Drop every segment that ends strictly before t_now (a prefix of the list,
    // since segments are time_end-ordered). The start sentinel is never dropped.
    void purge_expired(float t_now);

    int   size()      const;            // real segments (excludes both sentinels)
    float last_time() const { return last_time_; }

private:
    TemporalNode* start_ = nullptr;     // sentinel at t = 0
    TemporalNode* end_   = nullptr;     // sentinel at episode end
    float         last_time_ = 0.f;     // t* — last observed time, drives expiry

    void unlink_and_decrement(TemporalNode* node);   // splice out + undo overlaps
};

#endif // ENVIRONMENT_TEMPORAL_CHAIN_LIST_HPP
