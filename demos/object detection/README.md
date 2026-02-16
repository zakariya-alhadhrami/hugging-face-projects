

## 📋 Project Metadata

| Field       | Value                    |
| ----------- | ------------------------ |
| Title       | Trashify Demo 🚮 |
| Emoji       | 🗑️                      |
| Color From  | green                     |
| Color To    | blue                      |
| SDK         | Gradio                   |
| SDK Version | 5.34.0                  |
| App File    | app.py                   |
| Pinned      | false                    |
| License     | Apache-2.0               |

# 🌍 Trashify : Gamified Community Cleanup

Turn environmental action into an interactive experience.

Trashify is an AI-powered object detection tool designed to incentivize local cleanup efforts. 
By using state-of-the-art computer vision, the app validates real-world cleaning actions in real-time.


# 🎮 The +1 Challenge
To encourage authentic participation, the system rewards a +1 point only when it detects the "Cleanup Trifecta" in a single frame:

1. Trash: The litter being removed.

2. Hand: Proof of human effort.

3. Bin: The proper disposal destination.


# 🛠️ Technical Overview
Built on RT-DETRv2, this model is fine-tuned to distinguish between successful cleanups and false positives:

- Target Classes: `trash`, `bin`, `hand`.

- Validation Classes: `trash_arm`, `not_trash`, `not_bin`, `not_hand` (used to refine accuracy and prevent "cheating" the system).

