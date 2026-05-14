import os
import subprocess
import time

class ChaosInjector:
    def __init__(self, manifests_dir="tests/manifests/chaos-experiments"):
        self.manifests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'manifests', 'chaos-experiments'))

    def apply_chaos(self, experiment_file):
        """Applies a Chaos Mesh yaml manifest using kubectl."""
        file_path = os.path.join(self.manifests_dir, experiment_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Chaos manifest not found: {file_path}")
            
        print(f"[*] Injecting chaos: {experiment_file}")
        subprocess.run(["kubectl", "apply", "-f", file_path], check=True)
        # Allow time for the chaos daemon to enforce the rule
        time.sleep(5) 

    def remove_chaos(self, experiment_file):
        """Removes a Chaos Mesh yaml manifest."""
        file_path = os.path.join(self.manifests_dir, experiment_file)
        print(f"[*] Removing chaos: {experiment_file}")
        subprocess.run(["kubectl", "delete", "-f", file_path], check=True)
        # Allow time for the system to stabilize
        time.sleep(10)