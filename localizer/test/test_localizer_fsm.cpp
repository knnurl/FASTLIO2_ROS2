#include <gtest/gtest.h>

#include "localizer_fsm.h"

using localizer_fsm::IcpFailureDecision;
using localizer_fsm::SearchOutcome;
using localizer_fsm::maxIcpFailures;
using localizer_fsm::onIcpFailure;
using localizer_fsm::searchOutcome;

TEST(MaxIcpFailures, IsPatienceTimesRate)
{
    // The shipped localizer.yaml: lost_patience_sec 5.0 at update_hz 2.0.
    EXPECT_EQ(maxIcpFailures(5.0, 2.0), 10);
    EXPECT_EQ(maxIcpFailures(2.0, 2.0), 4);
}

TEST(MaxIcpFailures, NeverBelowOne)
{
    EXPECT_EQ(maxIcpFailures(0.1, 2.0), 1);
    EXPECT_EQ(maxIcpFailures(0.0, 2.0), 1);
}

TEST(OnIcpFailure, WidensOnceAtHalfwayThenSearchesAtTheLimit)
{
    const int max_failures = 10;
    for (int failures = 1; failures <= max_failures; ++failures)
    {
        const IcpFailureDecision d = onIcpFailure(failures, max_failures, false);
        EXPECT_EQ(d.widen_correspondence, failures == 5) << "failure " << failures;
        EXPECT_EQ(d.give_up, failures == 10) << "failure " << failures;
        EXPECT_EQ(d.start_search, failures == 10) << "failure " << failures;
    }
}

TEST(OnIcpFailure, KeepsDeadReckoningInsteadOfSearchingAgain)
{
    const IcpFailureDecision d = onIcpFailure(10, 10, true);
    EXPECT_TRUE(d.give_up);
    EXPECT_FALSE(d.start_search);
}

TEST(OnIcpFailure, SingleFailurePatienceWidensAndGivesUpTogether)
{
    // max 1 -> halfway is also 1; the node widens, then restores the
    // distances and starts the search in the same tick.
    const IcpFailureDecision d = onIcpFailure(1, 1, false);
    EXPECT_TRUE(d.widen_correspondence);
    EXPECT_TRUE(d.give_up);
    EXPECT_TRUE(d.start_search);
}

TEST(OnIcpFailure, OddLimitWidensAtFloorOfHalf)
{
    EXPECT_TRUE(onIcpFailure(1, 3, false).widen_correspondence);
    EXPECT_FALSE(onIcpFailure(2, 3, false).widen_correspondence);
    EXPECT_TRUE(onIcpFailure(3, 3, false).give_up);
}

TEST(SearchOutcome, FoundPoseAlwaysTracks)
{
    EXPECT_EQ(searchOutcome(true, false), SearchOutcome::TRACK);
    EXPECT_EQ(searchOutcome(true, true), SearchOutcome::TRACK);
}

TEST(SearchOutcome, FailureAfterAGoodPoseHoldsDeadReckoning)
{
    EXPECT_EQ(searchOutcome(false, true), SearchOutcome::DEAD_RECKON);
}

TEST(SearchOutcome, FailureBeforeAnyPoseRetries)
{
    EXPECT_EQ(searchOutcome(false, false), SearchOutcome::RETRY);
}
