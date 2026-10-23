# Viewer GUI fixes - combined project folder

This folder is based on the uploaded `multi_target_tracker.zip`. It contains the existing project sources plus local video-viewer updates. The public `tracker_nanosam.py` launcher is byte-for-byte unchanged.

This is a **viewer-only update**. Aircraft roll/pitch steering calculations and centre-distance aiming colours are **not implemented**. Object outlines use the existing fixed green colour. No new aircraft commands, command transmission, or remote shell capability has been added.

## Install over the working project

Close the running application. Extract this ZIP **outside** the existing `~/multi_target_tracker` folder, for example into a separate Downloads folder. Open a terminal inside the newly extracted `multi_target_tracker` folder and run, in the same Python environment used by the working application:

```bash
python3 apply_gui_update.py --target ~/multi_target_tracker
```

The helper validates the Python files and backs up the old GUI package under:

```text
~/multi_target_tracker/backups/gui_viewer_fix_<timestamp>_<id>/uav_gui/
```

It replaces the GUI package only. The launcher, `.gui_deps`, installed AI packages, model weights, video server, C++ sources and unrelated project files are not replaced. Existing package-local assets are retained. The helper refuses to apply the update over its own extracted folder or a symlinked installation target.

There is no new third-party dependency in this patch beyond the existing GUI stack. Keep the already working Python environment. If Qt has not yet been installed for that interpreter, the original one-time helper remains available:

```bash
python3 ~/multi_target_tracker/setup_gui.py
```

Do not copy only `tracker_nanosam.py`: it imports the accompanying `uav_gui` package.

## Daily commands - unchanged

Video:

```bash
python3 ~/multi_target_tracker/tracker_nanosam.py /mnt/c/Users/Aryan/Downloads/FVC_6.mp4
```

Existing camera feed:

```bash
python3 ~/multi_target_tracker/tracker_nanosam.py
```

Without a positional video argument, the source remains `http://localhost:5000/video`. The existing server must be running. Other file paths, quoted paths with spaces, source URLs, `--prompt` / `-p`, and `--deploy` / `-d` remain accepted.

## Changes

### Recording

Recordings now go directly into the folder containing `tracker_nanosam.py`, for example:

```text
~/multi_target_tracker/recording_<timestamp>_<id>.mp4
```

The existing AVI/MJPG fallback remains available when the MP4 encoder cannot open. Unique filenames prevent overwriting previous recordings. Screenshots keep their existing `captures/` folder.

The old recorder stopped when its pending-frame queue filled. The new recorder retains a bounded pending queue; on encoder overload it replaces an older queued recording frame with a newer one instead of stopping recording. `Record skips` and the recording tooltip show the number skipped. This does not remove frames from detector/tracker processing.

**Trade-off:** this is a processed-video recording, not a wall-clock screen recording. The duration is saved frames divided by the encoder FPS. During slow processing or writer overload, a saved clip can be shorter than the time spent recording. Skipped frames are reported, not silently represented as a lossless recording. Resolution changes, unavailable encoders, and write errors are still reported. Stop recording before switching inputs and allow finalization to finish. No audio is recorded.

### Process FPS

The old value was a reciprocal single-frame interval. File input also ran without source-rate pacing, so a 25 FPS file could be decoded hundreds of times faster than normal viewing.

The updated viewer measures completed frame intervals over a two-second window and refreshes the rate twice per second. File viewing is paced to the source's nominal FPS without skipping detector/tracker inputs or attempting a fast catch-up after a slow frame. When a file supplies no valid FPS, viewing uses a 30 FPS pacing fallback while Source FPS remains unavailable.

Live capture remains source-driven. The measured value is not artificially clamped to source metadata: an actual feed or its reported metadata can differ. Processing rate returns to zero after the worker stops.

### Settings

The settings icon now opens a viewer-settings dialog:

- **Main camera 1:** initially the existing `http://localhost:5000/video` source.
- **Camera 2:** initially camera index `1`; it can be changed to another index, device path or video URL.
- **RTSP:** enter the complete stream URL, including the correct port and path for that camera.
- **Remote IP:** optional IP/hostname used only to substitute `{remote_ip}` in a video URL. Example syntax: `rtsp://{remote_ip}:8554/your-stream-path`. There is no remote-control or remote-shell output.

`Save` saves preferences without changing the active source. `Apply selected source` saves them and requests a switch for the current viewer session. Source switching happens on the processing owner; the detector object is not reloaded for a successful switch. If a requested input fails to open, the previous input continues and a visible notice reports the failure. An input switch clears the current local analysis session.

Saved profiles **never override the existing launch commands**. A no-argument launch continues to use the original default; a positional video path still takes precedence. Camera device indices refer to devices visible inside the running Linux/WSL environment, not automatically to all Windows cameras.

An existing blocking backend read or inference call must return before a source switch/close can complete. No universal network timeout is claimed. Camera/RTSP reachability requires testing on the recipient's system.

### History

Recent entries and pins are saved in `gui_preferences.json` beside the code. History stores the submitted text, including an object-colour adjective. Selecting a history item restores that exact text without adding the current colour again.

Right-click a history chip to pin, unpin or remove it. The settings dialog also contains a full history list. Pinned items stay first. `Clear History` removes recent history but retains pins. `Clear All` removes both after confirmation. Settings dialog changes take effect when saved/applied.

Camera URLs containing credentials are stored in plain text in this local preferences file. The saver requests owner-only file permissions on Linux; filesystem/mount semantics can affect permissions. Do not include the file when sharing a project containing real credentials. Status labels redact URL user information and queries.

### Outline

Full object rectangles are replaced by four fixed-green corner brackets in the main image and the object-detail inset. Labels and the inset panel remain. No distance-to-centre colour mapping was added.

## Preserved and not implemented

The detector/tracker source, display-filter function, public launcher, setup helper, legacy standalone GUI and C++ source files were compared against the uploaded archive and remain byte-identical. The active Qt viewer has the new presentation/input/recording helpers; `standalone_gui.py` is not the launched Qt application and was not modified.

The uploaded Qt status table contains Source FPS, Process FPS, Resolution and Model score. It does not contain the legacy X/Y and Roll/Pitch calculation/display path. This update does not add those calculations or centre-distance aiming cues.

## Verification

See `TEST_REPORT.md`. At packaging: **37 tests passed; 10 Qt interaction/layout tests were skipped** because PySide6 could not be installed in the execution environment. Passing tests cover actual file recording/decoding, bounded writer overload, pacing, source switching using controlled video backends, history and preference persistence, compatibility, backup installation and protected-source hashes.

The Qt window, real detector/model, GPU environment, recipient's video, cameras and RTSP server have not been run here. Do not treat the skipped GUI tests as passed. This update was self-reviewed; no independent reviewer was available.

To run the included tests in the working project environment:

```bash
python3 -m pytest -q
```

The test bootstrap looks for the same project-local Qt dependency directory as the application. `pytest` is a development/test tool and is not needed to launch the viewer.

## Rollback

Close the application. Use the backup path printed by the update helper. Move the current `uav_gui` folder aside and copy the backup `uav_gui` folder back into `~/multi_target_tracker`. The launcher and installed dependencies were never replaced.

For example, with the actual printed backup directory substituted for `YOUR_BACKUP_DIRECTORY`:

```bash
cd ~/multi_target_tracker
mv uav_gui "uav_gui.disabled.$(date +%Y%m%d_%H%M%S)"
cp -a "backups/YOUR_BACKUP_DIRECTORY/uav_gui" ./uav_gui
```

## Audit files

`INPUT_MANIFEST.json` records hashes from this turn's uploaded archive. `PATCH_AUDIT.json` lists changed/new source files and protected comparisons. The inherited `source_audit.json` and `GUI_README.md` belong to the earlier version; this README supersedes their recording, FPS, history and settings descriptions.

## API references consulted

- Qt dialog API: https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QDialog.html
- Qt timer API: https://doc.qt.io/qtforpython-6/PySide6/QtCore/QTimer.html
- OpenCV video-writing API overview: https://docs.opencv.org/4.12.0/d7/d9e/tutorial_video_write.html

Those references explain library interfaces; they do not substitute for runtime verification of this package.
