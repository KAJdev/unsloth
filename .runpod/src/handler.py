import runpod
import subprocess
import os

import train


def handler(job):
    job_input = job["input"]

    # turn job_input into a class with attributes
    class JobInput:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getattr__(self, item):
            return None

        def __setattr__(self, key, value):
            if key not in self.__dict__:
                self.__dict__[key] = value
            else:
                raise AttributeError(f"Cannot set attribute {key}")

        def get(self, key, default=None):
            return getattr(self, key, default)

    job_input = JobInput(**job_input)

    try:
        train.run(job_input)
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

    return {
        "status": "Training complete",
        "output_dir": job_input.get("output_dir", "outputs"),
    }


runpod.serverless.start({"handler": handler})
