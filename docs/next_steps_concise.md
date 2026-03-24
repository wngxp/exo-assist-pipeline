# Proposal: Next Steps Toward RL-Based Exoskeleton Control

## Summary

We have completed the OpenSim/Moco tutorial phase. The 3D walking examples (MocoTrack + MocoInverse) run successfully on cortex, producing validated muscle activation profiles and joint kinematics. We are now ready to move toward integrating the exoskeleton model and eventually training an RL controller.

## Proposed 4-Step Plan

### Step 1: Baseline Human Walking Model (DONE)
- Ran MocoTrack (marker tracking + state tracking) and MocoInverse (with and without EMG) on a 3D walking model
- Produced validated muscle activation profiles, joint angles, and reserve actuator analysis
- Confirmed the musculoskeletal model produces physiologically realistic results
- **This serves as the "no exoskeleton" reference for comparison**

### Step 2: Exoskeleton Modeling in OpenSim
- Take Huang's SolidWorks specs (mass, inertia, joint locations, max torque) and represent the exoskeleton in OpenSim
- The exoskeleton is modeled as **additional bodies + torque actuators** attached to the hip joint of the human musculoskeletal model (not as a visual CAD overlay)
- This modifies the existing `.osim` file to create a new `human_with_exo.osim`

**Questions for Yang / Huang:**
- Which degrees of freedom does the exoskeleton assist? (hip flexion/extension only, or also adduction/abduction?)
- What are the actuator torque limits? (e.g., max 20 Nm per hip?)
- What is the total mass added to each leg segment?
- How is the exoskeleton physically attached — rigid connection at pelvis + thigh, or more complex?

### Step 3: Moco Validation with Exoskeleton
- Run MocoInverse with the combined human + exoskeleton model using the same walking data
- Compare muscle activations with vs. without exoskeleton assistance
- Expected result: hip flexor muscles (iliacus, psoas) should show reduced activation when the exoskeleton provides hip flexion/extension torque
- If reserve actuators remain small and muscle activations shift as expected, the combined model is validated

### Step 4: RL Training Framework
- Wrap the validated human + exoskeleton OpenSim model in a Python RL environment (Gym-style interface: `step()`, `reset()`, `observe()`)
- RL agent observes: joint angles, joint velocities, pelvis orientation (simulated IMU signals)
- RL agent controls: exoskeleton hip torque at each timestep
- Reward function: minimize metabolic cost (computed from muscle activations)
- Training with PPO (stable-baselines3 or similar) on cortex (RTX A5000)
- **Note:** OpenSim may be too slow for RL training (millions of steps needed). May need to convert the model to MuJoCo for speed (60-600x faster). This is an open question — need to check the Nature paper's pseudocode repo for details.

## Timeline Estimate

| Step | Duration | Dependencies |
|------|----------|-------------|
| Step 2: Exoskeleton modeling | 1-2 weeks | Huang's specs (mass, inertia, torque limits, attachment points) |
| Step 3: Moco validation | 1 week | Step 2 complete |
| Step 4: RL framework | 3-4 weeks | Step 3 complete, physics engine decision |

## Open Questions

1. **Physics engine for RL:** The Nature paper does not explicitly name which engine runs the RL training loop. OpenSim is likely too slow. We should review the pseudocode repo and supplementary materials to determine if we need MuJoCo or another fast simulator.
2. **Activity classifier:** Yang's proposed extension (adding a gait phase / activity classifier) — should this be designed in parallel with the RL work, or after the basic RL pipeline is running?
3. **Walking dataset for training at scale:** For RL we may want more diverse walking data. AddBiomechanics (CC BY 4.0) or Fukuchi 2018 (Figshare) are candidates, but download size is large (~1.5 GB+).