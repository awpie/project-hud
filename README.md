# Project HUD

##  What is this?

**Project HUD** is a modular AR + computer vision platform designed for real-time task recognition and heads-up visual feedback. It supports pluggable inference models and can be adapted to a variety of use cases — from activity tracking to hazard detection — all within a wearable, low-latency HUD system.

---

##  What does it do?

In its first hardware iteration, Project HUD uses:

- An **ESP32-CAM module** that streams video over Wi-Fi
- A more powerful **external inference machine** (e.g., laptop) that:
  - Runs classification/inference
  - Sends results back to the ESP32
- An **OLED display** mounted in the AR headset, which:
  - Projects a real-time HUD directly into the user’s Field-Of-View (FoV)
  - Displays progress bars, warnings, or task feedback based on inference output

The system is modular; any vision model can be swapped in, as long as it runs fast enough and the Wi-Fi latency is tolerable for your use case.

---

##  Example Modules / Use Cases

###  Activity Detection (current default)
- Uses OpenCLIP to classify real-world tasks like:
  - Eating
  - Studying
  - Exercising
  - Socializing
- Displays RPG-style progress bars and time-tracking on the HUD.
- Example: _“Studying – Level 3”_ with visual XP bar.

### ⚠ Hazard Detection
- Trained on datasets (e.g., OSHA violations) to flag potential dangers in the user’s environment.
- Can be combined with a **rear-facing camera** to detect hazards behind the user.
- Useful for workers, cyclists, or visually impaired users.

###  Education Module
- Uses posture/hand-position keypoints (e.g., YOLOv8 or MediaPipe) to provide **real-time feedback** on task performance.
- Example: Detects incorrect wire-stripping hand motion, or improper piano hand technique.
- Offers **instant, visual correction** via HUD.

###  Pilot Safety Module
- Trained on dashcam footage to detect early signs of car crashes.
- Alerts the user visually (e.g., flashing red HUD) even if they’re not directly looking.
- Can optionally integrate microphone input to detect car horns or screeches, aiding audio-impaired drivers.

---

##  Why Project HUD?

- **Modular** — plug in your own model and task logic
- **Latency-aware** — optimized for low-resolution streaming and real-time updates
- **Open-ended** — build your own app on top of the HUD interface
*The real value of this sort of device is that it functions as a second set of eyes that can feed you fully-parsed information in real-time.*

---

## 🛠 Quickstart

_(Coming soon — will include setup instructions for ESP32, Python inference server, and example modules)_

---

##  Planned Features

-  WebSocket support for higher-speed bi-directional updates
-  Optical alignment / collimation improvements for see-through HUD if possible
-  Onboard inference with distilled models (TinyCLIP, MobileNet, etc, removing the need for a server setup, easily possible with more expensive hardware)

---
