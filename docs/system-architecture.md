# Exoskeleton Control System — Architecture Notes

**Reference paper:** Luo et al., "Experiment-free exoskeleton assistance via learning in simulation," Nature 2024 (doi:10.1038/s41586-024-07382-4)

---

## Overall System Overview

The system has **three main stages**: biomechanical simulation, reinforcement learning in simulation, and real-time deployment on hardware. The key idea is that the RL controller never trains on a real person — it learns entirely in simulation, then transfers to the physical exoskeleton.

**Pipeline:**

OpenSim/Moco (biomechanical modeling) → Fast physics engine e.g. MuJoCo (RL training environment) → Trained policy deployed on embedded hardware

---

## Stage 1: Biomechanical Simulation (OpenSim / Moco)

**Purpose:** Generate and validate realistic human movement data — joint torques, muscle forces, GRF profiles, energy costs. This is the foundation that everything else builds on.

**Two approaches within OpenSim:**

| | IK + ID (Inverse Kinematics + Inverse Dynamics) | Moco (Optimal Control) |
|---|---|---|
| **What it does** | Takes real mocap data, computes joint angles (IK) and net joint torques (ID) | Solves for muscle activations/forces that produce or track a motion |
| **Input** | Marker data (.trc), GRF data (.mot), musculoskeletal model (.osim) | Model + objective function (e.g. minimize metabolic cost, track a motion) |
| **Output** | Joint angles, net joint moments | Muscle activations, forces, predicted/tracked kinematics |
| **Strengths** | Straightforward, well-established, good for analyzing existing data | Richer — can answer "what if" questions, can predict new motions |
| **Limitation** | Retrospective only — tells you what happened, not what would happen if conditions change | Computationally heavier, more complex to set up |

**Why Moco matters for this project:** The Nature paper uses motion tracking optimization (not simple IK/ID) to build the simulated environment. Moco can generate training scenarios across varied body parameters, gaits, and conditions — which is exactly what RL needs.

**Note:** OpenSim itself is too slow for millions of RL training episodes. The validated musculoskeletal model gets ported to a faster physics engine (e.g. MuJoCo) for RL training.

---

## Stage 2: Real-Time Deployment — Two Models

### Model 1: Activity Classifier

| | |
|---|---|
| **Input** | IMU data (accelerometers, gyroscopes on the user's body) |
| **Output** | Locomotion mode label — walk, run, incline, stairs, etc. + gait phase (heel strike, swing, toe-off) |
| **Architecture** | Typically CNN, LSTM, or even random forest on extracted features |
| **Speed** | Must run fast — real-time classification on embedded hardware |
| **Why needed** | The torque policy needs to know *what* the user is doing to apply appropriate assistance |

### Model 2: RL Torque Policy

| | |
|---|---|
| **Input** | Detected activity mode + real-time sensor state (joint angles, angular velocities, gait phase from IMUs) |
| **Output** | Torque command at the hip (or other assisted joint) |
| **Training** | Entirely in simulation — agent explores across thousands of simulated humans with varied body params, gaits, terrains |
| **Architecture** | Neural network (lightweight enough for embedded inference in milliseconds) |
| **Safety** | Never explores on a real person. Hardware-level safety limits (max torque caps, velocity limits) sit below the RL controller |

---

## Why Not Just Use IK/ID Directly for Control?

You *could* compute hip torques from IK/ID during normal walking, then have the exoskeleton apply a fraction of those torques (feedforward assistance). Early exoskeletons did this. The problem: **it's not adaptive.** It can't handle different users, speeds, terrains, or perturbations. The RL approach learns a reactive policy that adapts in real-time based on sensor feedback.

---

## Where Our Demo Fits

The OpenSim IK/ID and Moco work we're doing sits at the **foundation layer** — proving we can produce the biomechanical ground truth data that feeds the entire training pipeline. The classifier and RL policy are downstream tasks.

**Current pipeline:** Open-source mocap data → OpenSim IK/ID → Hip joint torque reproduction (demo)

**Target pipeline:** OpenSim/Moco simulation → MuJoCo training env → RL policy → Hardware deployment
