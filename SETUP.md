# Exoskeleton Biomechanics Pipeline — Setup Guide

## Prerequisites

This guide walks you through connecting to the lab's computation machine (P920, hostname `wolong`) and setting up your development environment. The P920 runs Ubuntu and is where all OpenSim/Moco computation happens — your personal machine is just used for SSH access, file editing, and viewing plots remotely.

### 1. SSH Access

You need SSH to connect to `wolong` from your personal machine. The P920 is on the lab network at `10.24.31.48`.

**macOS / Linux:**
SSH is built-in. Open Terminal and test:
```bash
ssh wxp@10.24.31.48
```
Enter the password when prompted. If this works, you're good.

**Windows:**
Install one of the following:
- [Windows Terminal](https://aka.ms/terminal) (recommended, built into Windows 11)
- [PuTTY](https://www.putty.org/)

With Windows Terminal, the SSH command is the same as macOS/Linux.

### 2. SSH Config (Recommended)

Instead of typing the full IP every time, create a config file.

**macOS / Linux:**
```bash
mkdir -p ~/.ssh
nano ~/.ssh/config
```

**Windows (PowerShell):**
```powershell
notepad $env:USERPROFILE\.ssh\config
```

Add the following:
```
Host wolong
    HostName 10.24.31.48
    User <your-username>
```

Now you can connect with just `ssh wolong`.

### 3. X11 Forwarding (For Viewing Plots Remotely)

OpenSim Moco runs on `wolong` (headless), but you'll want to see matplotlib plots and visualizations on your local machine. X11 forwarding sends GUI windows over SSH.

**macOS:**
1. Install [XQuartz](https://www.xquartz.org/) (or `brew install --cask xquartz`)
2. **Log out and back in** after installing (required)
3. Connect with the `-X` flag:
```bash
ssh -X wolong
```
4. Test with `xclock` — a small clock window should appear on your Mac.

To make `-X` the default, add this to your SSH config:
```
Host wolong
    HostName 10.24.31.48
    User <your-username>
    ForwardX11 yes
```

**Windows:**
1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [Xming](http://www.straightrunning.com/XmingNotes/)
2. Launch VcXsrv/Xming before connecting
3. In PuTTY: Connection → SSH → X11 → Enable X11 forwarding
4. Or with Windows Terminal: `ssh -X wolong`

**Linux:**
X11 forwarding works out of the box:
```bash
ssh -X wolong
```

### 4. Remote Access via Tailscale (Optional)

To access `wolong` from outside the lab (e.g., from home), Tailscale is already installed on the P920. This has been tested and confirmed working, including X11 forwarding for remote plot visualization.

1. Install [Tailscale](https://tailscale.com/download) on your personal machine
2. Sign in with the same Tailscale account used on the P920 (ask the team for credentials)
3. Make sure Tailscale is running on both machines — check with `tailscale status`
4. Add a second host entry to your SSH config:
```
Host wolong-ts
    HostName 100.103.182.116
    User <your-username>
    ForwardX11 yes
```
5. Connect with `ssh wolong-ts` from anywhere
6. Test X11 forwarding with `xclock` — a clock window should appear on your local machine

**Note:** The P920 may show a DNS health warning from Tailscale — this does not affect SSH connectivity.

### 5. GitHub Setup

The project repo is at: `https://github.com/wngxp/exo-assist-pipeline`

**Clone the repo on your personal machine:**
```bash
git clone https://github.com/wngxp/exo-assist-pipeline.git
```

**Clone the repo on wolong:**
```bash
git clone https://github.com/wngxp/exo-assist-pipeline.git
```

If pushing to GitHub from `wolong`, you'll need a Personal Access Token (GitHub no longer allows password auth):
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. Use the token as your password when git prompts you

To avoid re-entering credentials every push:
```bash
git config --global credential.helper store
```
This saves the token locally after the first use.

**Note:** `wolong` is behind China's firewall, so GitHub operations may be slow. For pip installs, use the Tsinghua mirror for faster downloads:
```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6. Conda Environment on wolong

The OpenSim + Moco Python environment is already set up on `wolong`. Activate it before running any scripts:
```bash
conda activate opensim
```

This gives you:
- Python 3.12
- OpenSim 4.5.2 (with Moco included)
- NumPy, matplotlib

To verify everything works:
```bash
python -c "import opensim as osim; print(osim.__version__); study = osim.MocoStudy(); print('Moco OK')"
```

Expected output:
```
4.5.2
Moco OK
```

### 7. VS Code (Recommended Editor)

For editing files on your personal machine:
1. Install [VS Code](https://code.visualstudio.com/)
2. Install the **Remote - SSH** extension
3. Open VS Code → `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows/Linux) → "Remote-SSH: Connect to Host" → select `wolong`

This lets you edit files directly on `wolong` with a local VS Code window — no need to copy files back and forth.

---

**Next:** Once connected and verified, proceed to the [OpenSim IK/ID Tutorial](opensim/ik-id-demo/README.md).
