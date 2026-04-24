# outputs

This top-level folder exists to make the handoff structure explicit.

Current exception:

- the surviving RL scripts still write to `rl/rl_output/`
- those paths were left in place during cleanup to avoid breaking the active code path

If a future clean repo is created, shared outputs should move to a top-level output layout instead of living inside the code tree.
