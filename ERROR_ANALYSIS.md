```markdown
# ERROR_ANALYSIS.md

## Overview
To understand the model's limitations, a validation split of the training data was used to isolate and extract 10 specific failure cases. An analysis of these failures reveals that the model struggles primarily with three real-world data constraints: extreme label noise, temporal emotional shifts, and highly ambiguous short-form text.

## The 10 Failure Cases

| ID | Journal Text | True State | Predicted State |
| :--- | :--- | :--- | :--- |
| 820 | "that helped a little" | calm | overwhelmed |
| 523 | "felt heavy" | calm | overwhelmed |
| 765 | "got distracted again" | mixed | focused |
| 838 | "i guess mind was all over the place." | focused | restless |
| 380 | "Gradually my breathing slowed down even if onl..." | calm | neutral |
| 609 | "Honestly still anxious a bit. After some time ..." | focused | mixed |
| 676 | "okay session ..." | neutral | focused |
| 109 | "I came in distracted, but I left the cafe sess..." | focused | calm |
| 677 | "Honestly felt good for a moment. Later it chan..." | calm | restless |
| 598 | "back to normal after ..." | overwhelmed | restless |

---

## Key Insights & Why the Model Failed

### 1. Extreme Label Noise (Imperfect Labels)
* **Examples:** Case 523 ("felt heavy" -> True: calm) and Case 838 ("mind was all over the place" -> True: focused).
* **Why it failed:** The model correctly learned the semantic weights of "heavy" and "all over the place" as negative/restless indicators and predicted accordingly. The ground-truth labels here are highly contradictory to the text. This suggests either user error during self-reporting (e.g., clicking the wrong button) or severe sarcasm/context not captured in the text.
* **How to improve:** Implement a label-smoothing technique or a confident learning framework (like `cleanlab`) to automatically identify and drop/re-weight training samples where the label is highly likely to be corrupted.

### 2. Temporal Shifts & Conflicting Signals
* **Examples:** Case 609 ("still anxious... After some time...") and Case 677 ("felt good... Later it changed").
* **Why it failed:** The current TF-IDF vectorization approach looks at word frequencies but loses *word order and sequence*. It sees the words "anxious" and "good" in the same bucket, leading the model to average the sentiment out and predict "mixed" or "restless," entirely missing the sequential "before and after" journey.
* **How to improve:** Move away from pure TF-IDF. A lightweight local embedding model (like DistilBERT or a quantized sentence-transformer) running on-device would capture the semantic sequence and understand that the *final* state overrides the initial state. 

### 3. Ambiguous & Short Inputs
* **Examples:** Case 820 ("that helped a little") and Case 598 ("back to normal after").
* **Why it failed:** "Normal" is a subjective baseline. If a user's "normal" is being overwhelmed (Case 598), the model has no way of knowing this without a historical baseline of that specific user. Similarly, "helped a little" implies a shift, but doesn't explicitly state the destination emotion.
* **How to improve:** For highly ambiguous, short inputs (under 5 words), the system should rely much heavier on the continuous metadata. We could increase the weight of the `stress_level` and `energy_level` features dynamically when the text string is extremely short.