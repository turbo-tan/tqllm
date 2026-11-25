#pragma once
// Partial residency (the routed experts do not all fit on the device, e.g. 24 GB at 128K context): the experts kept on
// the device are re-planned from recent routing. Plain tokens and drafted verify blocks both count as decode steps;
// prompt chunks restart the schedule. Before #106 only plain tokens counted, so a drafted reply kept the plan made from
// its prompt to the end (3.5x slower first request after a short prompt). Re-plans that move almost nothing double the
// interval (up to kMax): re-planning every 32 drafted steps kept moving experts and cost ~10 % once the plan had settled.
struct Inkling2ReplanClock {
    static constexpr int kEvery = 32, kMax = 1024;
    int steps = 0, interval = kEvery;
    // a forward of n tokens; true: re-plan now
    bool step(int n, bool verifying, bool partial) {
        if (!partial) return false;
        if (n > 1 && !verifying) { steps = 0; interval = kEvery; return false; }  // a prompt chunk: a new text
        if (++steps < interval) return false;
        steps = 0;
        return true;
    }
    // after a re-plan: `moved` expert matrices uploaded, `settled` = few enough that the plan is stable
    void planned(int moved, int settled) { interval = moved <= settled ? (interval * 2 > kMax ? kMax : interval * 2) : kEvery; }
};
