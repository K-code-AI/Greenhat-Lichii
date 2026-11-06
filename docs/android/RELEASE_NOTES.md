# Lichii for Android 0.1.0

First Android release of this fork, tested on a Samsung Galaxy S26 (Exynos 2600, Xclipse 960, 12 GB, Android 17) with a custom Mesa RADV driver.

## Highlights

- **The Legend of Zelda: Ocarina of Time** (Switch 2) boots. The title screen and the pre-rendered intro movies now run at **60 FPS** (about 8 FPS before).
  - The game's movies went through a neural-network enhancer built for NVIDIA tensor cores. Lichii now skips it and scales the decoded frame directly, which looks practically the same.
  - **Known issue:** the game hangs while loading the first gameplay scene after the intro. This is being worked on.
- AV1 movies are decoded with Android's hardware decoder (MediaCodec), and the movie player is woken as soon as a frame is ready.
- Emulated tensor-core math (HMMA) is about twice as fast on RADV, for games that run such networks themselves.

## For testers

- Install the APK, add your games folder, and set up keys and firmware as usual. None are included.
- Recommended: Handheld mode, NCE CPU backend, 1× resolution, and a custom RADV driver on Xclipse GPUs.
- Diagnostic switches can be set in `Android/data/dev.lichii.android/files/env.txt` (see the README).
- Report problems at https://github.com/pavelgajdo/Lichii-Android/issues with your phone, driver and `eden_log.txt`.
