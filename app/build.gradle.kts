plugins {
    id("com.android.application")
}

// build output outside OneDrive (see root build.gradle.kts)
layout.buildDirectory.set(File(System.getProperty("user.home"), ".npubench-build/app"))

// OneDrive marks synced files with a cloud reparse tag; AGP hard-links the
// linked .so files into its obj dir and the links inherit the tag, which
// Gradle then refuses to snapshot. Stage clean copies outside OneDrive and
// point both CMake and jniLibs packaging at the staged dir.
val litertStageDir = File(System.getProperty("user.home"), ".npubench-build/litert-libs")
val syncLitertLibs = tasks.register<Copy>("syncLitertLibs") {
    from("src/main/jniLibs")
    into(litertStageDir)
}
tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(syncLitertLibs) }

android {
    namespace = "com.npubench.swarm"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.npubench.swarm"
        minSdk = 31            // Poco X6 Pro ships Android 14; 31+ gives us AThermal headroom natively
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }   // benchmark targets one physical device
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DLITERT_LIB_DIR=" + File(litertStageDir, "arm64-v8a").path.replace('\\', '/'),
                )
            }
        }
    }

    buildFeatures { prefab = true }   // androidx.games:games-activity native lib

    sourceSets {
        getByName("main") {
            jniLibs.setSrcDirs(listOf(litertStageDir))   // staged clean copies
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
            // .cxx staging outside OneDrive too
            buildStagingDirectory = File(System.getProperty("user.home"), ".npubench-build/cxx")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // benchmark build is sideloaded; debug signing is fine and keeps installs simple
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("androidx.games:games-activity:4.4.2")
    implementation("androidx.appcompat:appcompat:1.7.0")   // GameActivity's base class
    // align transitive kotlin-stdlib-jdk7/jdk8 with the merged 1.8+ stdlib
    implementation(platform("org.jetbrains.kotlin:kotlin-bom:1.8.22"))
}
