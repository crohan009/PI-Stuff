# Physical Intelligence | The $\pi$ Research Arc

This document outlines the research arc of **Physical Intelligence (PI)** from October 2024 through April 2026. The series documents the incremental engineering of a single, universal foundation model capable of controlling any robot embodiment for any task.

---

## 1. $\pi_0$: The Foundation (Oct 2024)
**Core Problem:** Traditional robot learning often requires specialized models for specific tasks or embodiments, limiting generality and dexterity.
**Technical Contribution:** PI introduced **$\pi_0$**, a Vision-Language-Action (VLA) model that initializes from a pre-trained VLM (PaliGemma) and augments it with a **flow matching** (a variant of diffusion) action expert. 
**Evolution:** It established the base architecture of late-fusion transformers that can handle proprioceptive state and output continuous actions at high frequencies (up to 50Hz).

## 2. FAST: Efficient Action Tokenization (Jan 2025)
**Core Problem:** Autoregressive models (like LLMs) struggle with continuous robot action signals because standard binning methods lead to high redundancy and slow convergence at high frequencies.
**Technical Contribution:** **Frequency-space Action Sequence Tokenization (FAST)** uses the **Discrete Cosine Transform (DCT)** and **Byte-Pair Encoding (BPE)** to compress 1-second action chunks into roughly 30 dense tokens.
**Evolution:** This builds on $\pi_0$ by enabling autoregressive VLA training that is **5x faster** while matching the performance of diffusion-based models.

## 3. Hi Robot: Hierarchical Reasoning (Feb 2025)
**Core Problem:** While VLAs can follow simple commands, they lack "System 2" reasoning to handle complex, ambiguous, or mid-task human feedback.
**Technical Contribution:** **Hi Robot** introduces a hierarchical structure where a high-level VLM (System 2) parses complex prompts and user interjections into atomic commands, which a low-level VLA (System 1) then executes.
**Evolution:** It moves beyond the "flat" $\pi_0$ model by using **synthetic data generation** to teach the model how to respond to open-ended instructions like "can you get me something sweet?".

## 4. $\pi_{0.5}$: Open-World Generalization (Apr 2025)
**Core Problem:** Most robot models fail when moved from the lab to "in-the-wild" environments like real homes.
**Technical Contribution:** **$\pi_{0.5}$** utilizes **co-training on heterogeneous tasks**, mixing mobile manipulator data with web data, VQA, and high-level semantic subtask prediction.
**Evolution:** It refines the hierarchy from Hi Robot, using a single unified model for both high-level subtask prediction and low-level action generation, enabling tasks lasting **10–15 minutes** in unseen environments.

## 5. Knowledge Insulation (KI): Training Stability (May 2025)
**Core Problem:** Adding continuous action experts to a VLM backbone can "damage" the model's pre-trained semantic knowledge through unfavorable gradient flow.
**Technical Contribution:** The **Knowledge Insulation (KI)** recipe trains the VLM backbone on discrete FAST tokens while the action expert is trained on continuous actions, using a **stop-gradient** to prevent the expert from degrading the VLM.
**Evolution:** This ensures that as $\pi$ models scale, they **generalize better** and maintain their ability to follow complex language instructions.

## 6. Real-Time Chunking (RTC): Inference Efficiency (Jun 2025)
**Core Problem:** Large VLAs have high inference latency, leading to pauses or jerky robot movements at the boundaries between action chunks.
**Technical Contribution:** **Real-Time Chunking (RTC)** treats action generation as an **asynchronous inpainting problem**, computing the next chunk while the current one is still executing.
**Evolution:** It allows $\pi$ models to be robust to delays exceeding 300ms, enabling smooth performance in dynamic tasks like **lighting a match**.

## 7. $\pi^*_{0.6}$: Learning from Experience (Nov 2025)
**Core Problem:** Imitation learning (behavior cloning) is limited by the quality of human demonstrations and cannot easily improve through its own practice.
**Technical Contribution:** **RECAP** (Advantage-conditioned policies) enables **reinforcement learning** for VLAs by conditioning the model on state-action advantages estimated by a distributional value function.
**Evolution:** It allows the $\pi$ architecture to **self-improve** through real-world deployment, reaching robustness levels required for tasks like 13-hour continuous coffee making.

## 8. Human-to-Robot: Cross-Domain Transfer (Dec 2025)
**Core Problem:** High-quality robot teleoperation data is expensive and difficult to scale compared to human video data.
**Technical Contribution:** This research demonstrated the **emergence of transfer** where a VLA can learn manipulation strategies directly from egocentric human videos without explicit alignment steps.
**Evolution:** It scales the $\pi$ series by unlocking massive datasets of human behavior for generalist robot control.

## 9. MEM: Multi-Scale Embodied Memory (Mar 2026)
**Core Problem:** Robots need to remember events from minutes ago (e.g., "did I already add salt?"), but raw image history is too computationally expensive to store.
**Technical Contribution:** **MEM** uses a dual-memory system: a **short-term video memory** for resolving occlusions and a **long-term language memory** that stores semantic summaries of past subtasks.
**Evolution:** It equips the $\pi$ brain with the ability to perform stateful, long-horizon tasks that require remembering the past to decide the next subtask.

## 10. RLT: Precise Manipulation via Online RL (Mar 2026)
**Core Problem:** Foundation models often struggle with "the last millimeter" of precision in tasks like screw installation or charger insertion.
**Technical Contribution:** **RL Token (RLT)** adapts the VLA to expose a compact readout representation—an RL token—that serves as an interface for a tiny, high-speed online RL policy.
**Evolution:** Instead of fine-tuning the whole model, RLT allows for **local refinement** of VLA actions, improving success rates on contact-rich tasks in just a few hours of practice.

## 11. $\pi_{0.7}$: The Steerable Generalist (Apr 2026)
**Core Problem:** Prior models struggled with **compositional generalization**—recombining known skills to solve entirely new tasks out of the box.
**Technical Contribution:** **$\pi_{0.7}$** introduces **multi-modal context conditioning**, steering the model using detailed language instructions, generated subgoal images, and episode metadata (speed, quality).
**Evolution:** Build on the **Gemma 3 (4B)** backbone and the MEM architecture, $\pi_{0.7}$ represents the current state-of-the-art: a steerable model that can operate complex appliances like espresso machines out of the box.
