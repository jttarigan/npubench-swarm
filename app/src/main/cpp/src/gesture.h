#pragma once
// Gesture spec GS1 (must match training/train_all.py):
// raw stroke -> resample to 64 equidistant points along arc length ->
// translate centroid to origin -> divide by half of max bbox dimension.

#include <vector>

#include "common.h"
#include "nn.h"

struct GestureResult {
    bool ok = false;          // false: stroke too short / degenerate (no inference ran)
    int cls = 0;              // argmax over kGestureNames
    float conf = 0.f;
    double resampleMs = 0.0;
    double inferMs = 0.0;
};

// Returns false if the stroke is too short to be meaningful.
bool resampleStroke(const std::vector<Vec2>& raw, float* out /* [kGesturePts*2] */);

GestureResult classifyStroke(NnModel& model, const std::vector<Vec2>& raw);
