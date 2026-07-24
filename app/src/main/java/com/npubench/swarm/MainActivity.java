package com.npubench.swarm;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;

import com.google.androidgamesdk.GameActivity;

/**
 * Launcher shim ONLY (see PROJECT_BRIEF): all game/benchmark logic lives in native C++.
 * The single Java->native call pushes battery temperature — a non-latency value the
 * brief explicitly allows to come through the shim — entirely outside measured paths.
 */
public class MainActivity extends GameActivity {
    static {
        System.loadLibrary("npubench");
    }

    private static native void nativeSetBatteryTemp(float celsius);

    private final BroadcastReceiver batteryReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            int tenths = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1);
            if (tenths > 0) nativeSetBatteryTemp(tenths / 10.0f);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        hideSystemBars();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // ACTION_BATTERY_CHANGED is sticky: delivers the last known value immediately.
        registerReceiver(batteryReceiver, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
    }

    @Override
    protected void onPause() {
        super.onPause();
        unregisterReceiver(batteryReceiver);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) hideSystemBars();
    }

    private void hideSystemBars() {
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION);
    }
}
