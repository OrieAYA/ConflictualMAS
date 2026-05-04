#ifndef PDP_TASK_HPP
#define PDP_TASK_HPP

#include "DMASforPD/Utility/ObjectiveNode.hpp"

// Fine-grained lifecycle of a PDP task.
// GlobalMemory three-list mapping:
//   available  ← Idle, Allocating
//   allocated  ← Allocated … Delivered
//   finished   ← Finished
enum class TaskStatus {
    Idle,             // created, TAM not yet started
    Allocating,       // TAM running, no confirmed agent
    Allocated,        // delivery agent accepted the task
    TowardsPickup,    // agent moving to pickup node
    Picked,           // pickup reached and done
    TowardsDelivery,  // agent moving to delivery node
    Delivered,        // delivery reached and done
    Finished          // task fully complete
};

// Timestamped milestones (simulation steps; -1 = not yet reached).
struct TaskTimeline {
    int created_step    = -1;
    int allocated_step  = -1;
    int picked_step     = -1;
    int delivered_step  = -1;
    int finished_step   = -1;

    int fill_time()       const { return (delivered_step >= 0 && allocated_step >= 0)
                                         ? delivered_step - allocated_step : -1; }
    int allocation_time() const { return (allocated_step >= 0 && created_step >= 0)
                                         ? allocated_step - created_step  : -1; }
    int total_time()      const { return (finished_step  >= 0 && created_step >= 0)
                                         ? finished_step  - created_step  : -1; }
};

struct PDPTask {
    int           task_id    = 0;
    ObjectiveNode pickup;
    ObjectiveNode delivery;
    TaskStatus    status     = TaskStatus::Idle;
    int           agent_id   = -1;
    float         reward     = 1.0f;   // current reward (updated by TAM on recall)
    float         importance = 1.0f;   // urgency / priority
    TaskTimeline  timeline;

    // ---- Status predicates -----------------------------------------------

    bool is_available()   const { return status == TaskStatus::Idle
                                      || status == TaskStatus::Allocating; }
    bool is_allocating()  const { return status == TaskStatus::Allocating; }
    bool is_allocated()   const { return status >= TaskStatus::Allocated
                                      && status <  TaskStatus::Finished; }
    bool is_finished()    const { return status == TaskStatus::Finished; }
    bool pickup_done()    const { return status >= TaskStatus::Picked; }
    bool delivery_done()  const { return status >= TaskStatus::Delivered; }

    // ---- Lifecycle transitions -------------------------------------------

    void start_allocation()  { status = TaskStatus::Allocating; }

    void assign(int aid, int step = -1) {
        agent_id = aid; status = TaskStatus::Allocated;
        timeline.allocated_step = step;
    }
    void release() { agent_id = -1; status = TaskStatus::Idle; timeline.allocated_step = -1; }

    void mark_towards_pickup  (int /*step*/ = -1)  { status = TaskStatus::TowardsPickup;   }
    void mark_picked          (int step    = -1)   { status = TaskStatus::Picked;
                                                     timeline.picked_step = step; }
    void mark_towards_delivery(int /*step*/ = -1)  { status = TaskStatus::TowardsDelivery; }
    void mark_delivered       (int step    = -1)   { status = TaskStatus::Delivered;
                                                     timeline.delivered_step = step; }

    void complete(int step = -1) { status = TaskStatus::Finished; timeline.finished_step = step; }
};

#endif // PDP_TASK_HPP
