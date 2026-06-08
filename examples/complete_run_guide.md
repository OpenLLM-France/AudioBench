# AudioLLM Benchmark – DGX Setup Guide

This guide explains how to run the audiollm_benchmark Docker container on the DGX server and properly configure your user environment.

📌 **Context**

- Use **DGX 1** (the first DGX).
- The second DGX is already in use.
- This setup assumes you are not the original container user, so additional steps are required.

🚀 **1. Run the Docker Container**

Start the container with the following command:

```bash
docker run -it --rm \
  -v /media:/media \
  -v /home:/home \
  --gpus all \
  --shm-size=4g \
  --user $(id -u):$(id -g) \
  --name audiollm \
  audiollm_benchmark
```

👤 **2. Create Your User Inside the Container**

Since you're not the default user, open another terminal and run:

```bash
docker exec -it --user root:root audiollm bash
```

Then create your user inside the container:

```bash
useradd -m -u YOUR_ID USERNAME -s /bin/sh
rm -rf /tmp
```

🔎 **What is YOUR_ID?**

Replace `YOUR_ID` with your Linux user ID:
```bash
id -u
```

Replace `USERNAME` with your username:
```bash
whoami
```

Example:
```bash
useradd -m -u 1003 Alex -s /bin/sh
```

🏠 **3. Set Environment Variables (Main Terminal)**

Go back to the terminal where you executed `docker run` and configure:

```bash
export HOME=<YOUR-HOME-PATH>
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
```

📦 **4. Install Required Python Packages**

⚠️ **Important:**
You must install Python packages as **root** inside the container (in the terminal where you ran `docker exec`).

Otherwise, packages will install in `.local` and cause permission issues.

Example:
```bash
pip install flow_judge
pip install latex2sympy2
pip install qwen_omni_utils
```
Install any other required dependencies depending on your use case.

📁 **5. Additional Environment Variables (Optional)**

If needed, you can also define:

```bash
export MODELS_FOLDER=<YOUR-MODELS-PATH>
export DATA_FOLDER=<YOUR-DATA-PATH>
```
*Adjust paths if necessary.*

📊 **6. Running Evaluation**

To run an evaluation, ensure you are in the `AudioBench` repository root and use the following command:

```bash
python src/run_evaluations.py --config_path path/to/your/config.yaml
```

🎯 **Example Use Case**

Currently preparing:
- **VoxLingua107** dataset for language identification tasks.

🛠 **Common Issues**

❌ **Permission Denied during pip install**
If you see errors like:
`ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: '/.local'`
It means you installed packages as a non-root user.

✔ **Solution:** Install packages in the **root shell** inside the container.

✅ **Summary Workflow**

1. `docker run`
2. `docker exec` as root
3. Create your user with correct UID
4. Set `HOME` and `TMPDIR`
5. Install dependencies as root
6. Start working 🚀

If anything breaks, check:
- Your UID mapping
- Package installation location
- Environment variables

---
## Available Guides
- [Adding a New Model](adding_new_model.md)
- [Supported Datasets](supported_datasets.md)
- [Supported Models](supported_models.md)
