#include "input.h"

#include <game-activity/GameActivityEvents.h>

#include "renderer.h"

namespace {
// GameActivity delivers eventTime in ns on current versions; be defensive in
// case a build hands us ms (uptime ms ~1e8..1e9, uptime ns >1e12 within hours).
int64_t eventTimeNs(int64_t t) { return t < 1000000000000LL ? t * 1000000LL : t; }
// Ship is stationary with autofire: the whole screen is gesture canvas.
constexpr float kShipBand = 2.0f;    // nothing lands below internalH*2 -> no ship pointer
constexpr size_t kMaxStrokePts = 512;
}  // namespace

Vec2 InputState::toInternal(const GameActivityMotionEvent& ev, uint32_t idx) const {
    float ix = 0, iy = 0;
    renderer_->screenToInternal(GameActivityPointerAxes_getX(&ev.pointers[idx]),
                                GameActivityPointerAxes_getY(&ev.pointers[idx]), &ix, &iy);
    return {ix, iy};
}

void InputState::handle(const GameActivityMotionEvent& ev) {
    if (!renderer_ || !renderer_->ready()) return;
    const int action = ev.action & AMOTION_EVENT_ACTION_MASK;
    const uint32_t actionIdx =
        (ev.action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >>
        AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    const float bandY = (float)renderer_->internalH() * kShipBand;

    switch (action) {
        case AMOTION_EVENT_ACTION_DOWN:
        case AMOTION_EVENT_ACTION_POINTER_DOWN: {
            int id = ev.pointers[actionIdx].id;
            Vec2 p = toInternal(ev, actionIdx);
            if (p.y >= bandY && shipPid_ < 0) {
                shipPid_ = id;
                shipTarget_ = p;
            } else if (strokePid_ < 0) {
                strokePid_ = id;
                strokePts_.clear();
                strokePts_.push_back(p);
            }
            break;
        }
        case AMOTION_EVENT_ACTION_MOVE: {
            for (uint32_t i = 0; i < ev.pointerCount; i++) {
                int id = ev.pointers[i].id;
                Vec2 p = toInternal(ev, i);
                if (id == shipPid_) shipTarget_ = p;
                if (id == strokePid_ && strokePts_.size() < kMaxStrokePts) {
                    if ((p - strokePts_.back()).len() >= 0.8f) strokePts_.push_back(p);
                }
            }
            break;
        }
        case AMOTION_EVENT_ACTION_UP:
        case AMOTION_EVENT_ACTION_POINTER_UP: {
            int id = ev.pointers[actionIdx].id;
            Vec2 p = toInternal(ev, actionIdx);
            if (id == shipPid_) shipPid_ = -1;
            if (id == strokePid_) {
                if (strokePts_.size() < kMaxStrokePts) strokePts_.push_back(p);
                strokes_.push_back({std::move(strokePts_), eventTimeNs(ev.eventTime)});
                strokePts_ = {};
                strokePid_ = -1;
            }
            taps_.push_back(p);
            break;
        }
        case AMOTION_EVENT_ACTION_CANCEL:
            shipPid_ = strokePid_ = -1;
            strokePts_.clear();
            break;
        default:
            break;
    }
}

void InputState::reset() {
    shipPid_ = strokePid_ = -1;
    strokePts_.clear();
    strokes_.clear();
    taps_.clear();
}

std::vector<Stroke> InputState::takeStrokes() {
    std::vector<Stroke> out = std::move(strokes_);
    strokes_ = {};
    return out;
}

std::vector<Vec2> InputState::takeTaps() {
    std::vector<Vec2> out = std::move(taps_);
    taps_ = {};
    return out;
}
