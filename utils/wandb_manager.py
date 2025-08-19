import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class WandbManager:
    """
    A generic manager for Weights & Biases to handle experiment tracking,
    logging, and resuming runs. This version buffers logs and sends them in batches.
    """

    def __init__(self, project_name: str, entity: Optional[str] = None, tracking_file: str = "experiment_tracking.json"):
        """
        Initializes the WandbManager.

        Args:
            project_name (str): The name of the wandb project.
            entity (Optional[str]): The wandb entity (username or team). Defaults to None.
            tracking_file (str): The local JSON file to track experiment runs.
        """
        if not WANDB_AVAILABLE:
            self.is_active = False
            print("Warning: wandb is not installed. WandbManager will be inactive.")
            return

        self.is_active = True
        self.project_name = project_name
        self.entity = entity
        self.tracking_file = tracking_file
        self.run = None
        self.api = wandb.Api()
        self.exp_num = None
        
        # Buffers for pending data
        self._pending_metrics = {}
        self._pending_images = {}
        
        print(f"WandbManager initialized for project '{self.project_name}'")

    def _load_tracking_data(self) -> Dict[str, Any]:
        """Loads experiment tracking data from the local JSON file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read tracking file '{self.tracking_file}': {e}")
        return {}

    def _save_tracking_data(self, data: Dict[str, Any]):
        """Saves tracking data to the local JSON file."""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True)
        except IOError as e:
            print(f"Warning: Could not write to tracking file '{self.tracking_file}': {e}")

    def get_experiment_number(self) -> int:
        """
        Extracts the experiment number from the run name (e.g., "E001 my-experiment").

        Args:
            run_name: The name of the experiment run.

        Returns:
            The experiment number as an integer.
        """
        if self.exp_num is not None:
            return self.exp_num
        
        raise ValueError("Experiment number is not set. Please initialize a run first.")

    def get_latest_experiment_number(self) -> int:
        """
        Determines the latest experiment number by checking the local tracking file
        and the wandb server.

        Returns:
            The highest experiment number found.
        """
        local_max = 0
        tracking_data = self._load_tracking_data()
        if tracking_data:
            numbers = [info.get('number', 0) for info in tracking_data.values()]
            if numbers:
                local_max = max(numbers)

        wandb_max = 0
        try:
            project_path = f"{self.entity}/{self.project_name}" if self.entity else self.project_name
            runs = self.api.runs(project_path)
            for run in runs:
                if run.name.startswith('E') and ' ' in run.name:
                    try:
                        num = int(run.name.split(' ')[0][1:])
                        wandb_max = max(wandb_max, num)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Warning: Could not fetch runs from wandb API: {e}. Relying on local tracking.")

        return max(local_max, wandb_max)

    def get_next_run_name(self, base_name: str) -> str:
        """
        Generates the name for the next experiment run (e.g., "E001 my-experiment").

        Args:
            base_name: The base name for the experiment.

        Returns:
            The formatted run name with the next experiment number.
        """
        if self.exp_num is not None:
            next_number =  self.exp_num
        else:
            next_number = self.get_latest_experiment_number() + 1
        self.exp_num = f"E{next_number:03d}"
        return f"{self.exp_num} {base_name}"

    def get_run_id_by_name(self, run_name: str) -> Optional[str]:
        """
        Finds the wandb run ID for a given run name from the local tracking file.

        Args:
            run_name: The name of the experiment to find.

        Returns:
            The wandb run ID if found, otherwise None.
        """
        tracking_data = self._load_tracking_data()
        return tracking_data.get(run_name, {}).get('wandb_id')

    def init(self, run_name: str, config: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None, resume_id: Optional[str] = None):
        """
        Initializes a new wandb run or resumes an existing one.

        Args:
            run_name: The name for the wandb run.
            config: A dictionary of hyperparameters.
            tags: A list of tags for the run.
            resume_id: The ID of a run to resume. If 'auto', it will try to find the ID from the run_name.
        """
        if not self.is_active:
            return

        if resume_id == 'auto':
            resume_id = self.get_run_id_by_name(run_name)
            if resume_id:
                print(f"Found run ID '{resume_id}' for run name '{run_name}'. Attempting to resume.")
            else:
                print(f"Could not find run ID for '{run_name}'. Starting a new run.")

        try:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config,
                tags=tags,
                id=resume_id,
                resume="allow"  # Allows resuming if id is found, otherwise starts a new run
            )
            if self.run:
                print(f"Wandb run '{self.run.name}' initialized. URL: {self.run.url}")
                tracking_data = self._load_tracking_data()
                exp_number = int(run_name.split(' ')[0][1:])
                tracking_data[self.run.name] = {
                    'number': exp_number,
                    'wandb_id': self.run.id,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_tracking_data(tracking_data)
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            self.is_active = False

    def add_metrics(self, data: Dict[str, Any]):
        """
        Adds a dictionary of metrics to a temporary buffer.
        These metrics will be sent when send_log() is called.

        Args:
            data: The dictionary of metrics to add (e.g., {"loss": 0.1}).
        """
        if self.is_active and self.run:
            self._pending_metrics.update(data)

    def add_images(self, image_data: Dict[str, Any]):
        """
        Adds a dictionary of images to a temporary buffer.
        The value for each key can be a single image or a list of images.

        Args:
            image_data: A dictionary where keys are names and values are image data
                        (e.g., numpy array, PIL image, path, or a list of these).
        """
        if self.is_active and self.run:
            self._pending_images.update(image_data)

    def send_log(self, step: Optional[int] = None):
        """
        Sends all buffered metrics and images to the wandb run and clears the buffers.

        Args:
            step: The custom step for this log operation.
        """
        if not self.is_active or not self.run:
            return

        if not self._pending_metrics and not self._pending_images:
            return  # Nothing to log

        log_payload = {}
        log_payload.update(self._pending_metrics)

        if self._pending_images:
            try:
                wandb_image_payload = {}
                for key, value in self._pending_images.items():
                    if isinstance(value, list):
                        # Handle a list of images for one key
                        wandb_image_payload[key] = [wandb.Image(item) for item in value]
                    else:
                        # Handle a single image
                        wandb_image_payload[key] = wandb.Image(value)
                log_payload.update(wandb_image_payload)
            except Exception as e:
                print(f"Warning: Failed to create wandb.Image objects: {e}")

        if not log_payload:
            self._pending_metrics.clear()
            self._pending_images.clear()
            return

        try:
            self.run.log(log_payload, step=step, commit=True)
        except Exception as e:
            print(f"Error sending log to wandb: {e}")
        finally:
            self._pending_metrics.clear()
            self._pending_images.clear()

    def finish(self):
        """Finishes the current wandb run."""
        if self.is_active and self.run:
            print(f"Finishing wandb run '{self.run.name}'.")
            self.run.finish()
            self.run = None
