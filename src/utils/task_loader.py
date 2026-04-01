"""
Task Configuration Loader
Load and manage different task configurations for materials discovery
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TaskLoader:
    """
    Load and manage task configurations

    Supports:
        - YAML configuration files (current)
        - CSV batch tasks (future)
    """

    def __init__(self, task_file: str = "config/tasks.yaml"):
        """
        Initialize TaskLoader

        Args:
            task_file: Path to task configuration file (YAML or CSV)
        """
        self.task_file = Path(task_file)

        if not self.task_file.exists():
            raise FileNotFoundError(
                f"Task configuration file not found: {self.task_file}\n"
                f"Please create it or check the path."
            )

        # Determine file type and load
        if self.task_file.suffix in ['.yaml', '.yml']:
            self.tasks = self._load_yaml()
        elif self.task_file.suffix == '.csv':
            # Future: CSV support for batch tasks
            self.tasks = self._load_csv()
        else:
            raise ValueError(
                f"Unsupported task file format: {self.task_file.suffix}\n"
                f"Supported: .yaml, .yml, .csv"
            )

    def _load_yaml(self) -> Dict[str, Any]:
        """Load tasks from YAML file"""
        logger.info(f"Loading tasks from {self.task_file}")

        with open(self.task_file, 'r') as f:
            data = yaml.safe_load(f)

        tasks = data.get('tasks', {})

        if not tasks:
            raise ValueError(
                f"No tasks found in {self.task_file}\n"
                f"Expected 'tasks:' section in YAML"
            )

        logger.info(f"Loaded {len(tasks)} task(s): {', '.join(tasks.keys())}")
        return tasks

    def _load_csv(self) -> Dict[str, Any]:
        """
        Load tasks from CSV file (Future implementation)

        CSV Format:
            task_name, constraint_type, value, description
            task1, decomposition_energy_max, 0.1, Metastable
            task1, band_gap_min, 2.5, Wide bandgap
            task2, decomposition_energy_max, 0.0, Stable only

        Returns:
            Dict of task configurations
        """
        import csv

        logger.warning(
            "CSV task loading is not yet implemented. "
            "Please use YAML format for now."
        )

        # Future implementation placeholder
        tasks = {}

        with open(self.task_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_name = row['task_name']
                if task_name not in tasks:
                    tasks[task_name] = {
                        'name': task_name,
                        'description': '',
                        'constraints': {},
                        'objective': {'targets': {}, 'weights': {}}
                    }

                # Parse constraint from row
                # TODO: Implement constraint parsing logic
                pass

        return tasks

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """
        Get a specific task configuration

        Args:
            task_name: Name of the task (key in tasks.yaml)

        Returns:
            Task configuration dict with keys:
                - name: Human-readable name
                - description: Task description
                - constraints: Property constraints dict
                - objective: Optimization targets and weights

        Raises:
            ValueError: If task not found
        """
        if task_name not in self.tasks:
            available = ', '.join(self.tasks.keys())
            raise ValueError(
                f"Task '{task_name}' not found.\n"
                f"Available tasks: {available}\n"
                f"Use --list-tasks to see all available tasks."
            )

        task = self.tasks[task_name]
        logger.info(f"Selected task: {task.get('name', task_name)}")

        return task

    def list_tasks(self) -> list:
        """
        List all available task names

        Returns:
            List of task names (keys)
        """
        return list(self.tasks.keys())

    def get_task_info(self, task_name: str) -> str:
        """
        Get formatted information about a task

        Args:
            task_name: Task name

        Returns:
            Formatted string with task details
        """
        task = self.get_task(task_name)

        lines = [
            f"Task: {task.get('name', task_name)}",
            f"Description: {task.get('description', 'N/A')}",
            "",
            "Constraints:"
        ]

        constraints = task.get('constraints', {})
        for name, definition in constraints.items():
            enabled = definition.get('enabled', True)
            status = "✓" if enabled else "✗ (disabled)"
            desc = definition.get('description', name)
            lines.append(f"  [{status}] {desc}")

        return '\n'.join(lines)

    def get_constraints(self, task_name: str) -> Dict[str, Any]:
        """
        Get constraints for a task, filtering out disabled constraints

        Args:
            task_name: Task name

        Returns:
            Dict of enabled constraints only
        """
        task = self.get_task(task_name)
        all_constraints = task.get('constraints', {})

        # Filter out disabled constraints
        enabled_constraints = {}
        for name, definition in all_constraints.items():
            if definition.get('enabled', True):
                enabled_constraints[name] = definition
            else:
                logger.info(f"Constraint '{name}' is disabled, skipping")

        return enabled_constraints

    def get_objective(self, task_name: str) -> Dict[str, Any]:
        """
        Get optimization objective for a task

        Args:
            task_name: Task name

        Returns:
            Objective configuration dict
        """
        task = self.get_task(task_name)
        return task.get('objective', {})

    @classmethod
    def from_csv(cls, csv_file: str) -> 'TaskLoader':
        """
        Create TaskLoader from CSV file (Future implementation)

        Args:
            csv_file: Path to CSV file

        Returns:
            TaskLoader instance
        """
        return cls(task_file=csv_file)
