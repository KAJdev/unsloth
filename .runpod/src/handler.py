import runpod
import subprocess
import os


def handler(job):
    job_input = job["input"]
    args = job_input.get("args", {})
    creds = job_input.get("credentials", {})

    cmd = ["python", "-m", "unsloth", "train"]

    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(map(str, value))
        else:
            cmd.extend([flag, str(value)])

    env = os.environ.copy()
    if creds.get("hf_token"):
        env["HF_TOKEN"] = creds["hf_token"]
    if creds.get("wandb_api_key"):
        env["WANDB_API_KEY"] = creds["wandb_api_key"]

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

    return {
        "status": "Training complete",
        "output_dir": args.get("output_dir", "outputs"),
    }


runpod.serverless.start({"handler": handler})
