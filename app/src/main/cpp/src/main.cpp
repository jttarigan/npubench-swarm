// Entry point: GameActivity native_app_glue pump + the one JNI hook
// (battery temperature push — outside all measured paths).

#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <jni.h>

#include "engine.h"

extern "C" JNIEXPORT void JNICALL
Java_com_npubench_swarm_MainActivity_nativeSetBatteryTemp(JNIEnv*, jclass, jfloat celsius) {
    Engine::batteryTemp.store(celsius);
}

static void onAppCmd(android_app* app, int32_t cmd) {
    Engine* e = static_cast<Engine*>(app->userData);
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (app->window) {
                const char* dir = app->activity->externalDataPath
                                      ? app->activity->externalDataPath
                                      : app->activity->internalDataPath;
                e->onWindowInit(app->window, app->activity->assetManager, dir);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            e->onWindowLost();
            break;
        case APP_CMD_GAINED_FOCUS:
            e->setFocused(true);
            break;
        case APP_CMD_LOST_FOCUS:
            e->setFocused(false);
            break;
        case APP_CMD_PAUSE:
        case APP_CMD_STOP:
            e->onPause();
            break;
        default:
            break;
    }
}

void android_main(android_app* app) {
    Engine engine;
    app->userData = &engine;
    app->onAppCmd = onAppCmd;

    while (!app->destroyRequested) {
        android_poll_source* source = nullptr;
        int result = ALooper_pollOnce(engine.animating() ? 0 : -1, nullptr, nullptr,
                                      (void**)&source);
        if (result == ALOOPER_POLL_ERROR) {
            LOGE("looper error");
            break;
        }
        if (source) source->process(app, source);
        if (app->destroyRequested) break;

        android_input_buffer* ib = android_app_swap_input_buffers(app);
        if (ib && ib->motionEventsCount > 0) {
            for (uint64_t i = 0; i < ib->motionEventsCount; i++)
                engine.handleMotionEvent(ib->motionEvents[i]);
            android_app_clear_motion_events(ib);
        }
        if (ib && ib->keyEventsCount > 0) android_app_clear_key_events(ib);

        if (engine.animating()) engine.frame();
    }
}
