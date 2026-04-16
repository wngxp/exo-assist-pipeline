## Stage 2 - Iteration 1
Date:
Walker:
- DEP-RL local baseline

Exo setup:
- PPO exo policy
- current reward version
- current torque scaling/clipping

Result:
- walker-only episode length: ~116 to 1000 depending eval setup
- walker+exo episode length: 129
- walker+exo mean effort: 0.304452
- walker-only mean effort: ~0.327 (from earlier eval)
- mean |tau|: 1.458

Interpretation:
- effort decreased
- torque is nonzero and structured
- gait stability is worse than baseline
- likely issue is timing/coordination, not just torque magnitude