#pragma once
// Touch routing (portrait):
//   pointer that lands in the bottom ship band  -> drags the ship
//   pointer that lands anywhere above           -> draws a gesture stroke
// Both can be active simultaneously (two fingers). Finger-up timestamps come
// from the motion event itself (CLOCK_MONOTONIC base) for input-path latency.

#include <cstdint>
#include <vector>

#include "common.h"

struct GameActivityMotionEvent;
class Renderer;

struct Stroke {
    std::vector<Vec2> pts;   // internal-resolution coords
    int64_t fingerUpNs = 0;  // event timestamp, CLOCK_MONOTONIC ns
};

class InputState {
public:
    void attach(const Renderer* r) { renderer_ = r; }
    void handle(const GameActivityMotionEvent& ev);
    void reset();

    bool shipActive() const { return shipPid_ >= 0; }
    Vec2 shipTarget() const { return shipTarget_; }

    std::vector<Stroke> takeStrokes();
    std::vector<Vec2> takeTaps();
    const std::vector<Vec2>& liveStroke() const { return strokePts_; }

private:
    Vec2 toInternal(const GameActivityMotionEvent& ev, uint32_t idx) const;

    const Renderer* renderer_ = nullptr;
    int shipPid_ = -1, strokePid_ = -1;
    Vec2 shipTarget_;
    std::vector<Vec2> strokePts_;
    std::vector<Stroke> strokes_;
    std::vector<Vec2> taps_;
};
