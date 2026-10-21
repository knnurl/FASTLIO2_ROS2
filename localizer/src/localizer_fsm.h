#pragma once
// Decisions of the localizer state machine, kept free of ROS and PCL so they
// can be unit-tested (test/test_localizer_fsm.cpp). LocalizerNode applies them;
// locking, logging and the ICP/global-search calls stay in the node.
#include <algorithm>

namespace localizer_fsm
{

// Consecutive ICP failures tolerated before tracking is given up.
inline int maxIcpFailures(double lost_patience_sec, double update_hz)
{
    return std::max(1, static_cast<int>(lost_patience_sec * update_hz));
}

struct IcpFailureDecision
{
    bool widen_correspondence = false;  // halfway: widen ICP correspondence distances
    bool give_up = false;               // patience used up: restore distances, reset the count
    bool start_search = false;          // give up into a global search (IDLE)
};

// After the `failures`-th consecutive ICP failure. When the patience runs out
// while LIO dead-reckoning is already active, the global search has failed
// before, so the node keeps dead-reckoning instead of searching again.
inline IcpFailureDecision onIcpFailure(int failures, int max_failures, bool dead_reckoning)
{
    IcpFailureDecision d;
    d.widen_correspondence = failures == std::max(1, max_failures / 2);
    d.give_up = failures >= max_failures;
    d.start_search = d.give_up && !dead_reckoning;
    return d;
}

enum class SearchOutcome
{
    TRACK,        // search found the pose: switch to ICP tracking
    DEAD_RECKON,  // search failed after a good pose: likely unmapped area, hold LIO
    RETRY         // search failed and never localized: search again (IDLE)
};

inline SearchOutcome searchOutcome(bool found, bool had_pose)
{
    if (found)
        return SearchOutcome::TRACK;
    return had_pose ? SearchOutcome::DEAD_RECKON : SearchOutcome::RETRY;
}

}  // namespace localizer_fsm
