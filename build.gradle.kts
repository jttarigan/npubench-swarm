plugins {
    id("com.android.application") version "8.13.0" apply false
}

// The repo lives inside OneDrive; sync turns build outputs into reparse-point
// placeholders that Gradle cannot snapshot. Keep ALL build output outside.
layout.buildDirectory.set(File(System.getProperty("user.home"), ".npubench-build/root"))
