from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import json
import pandas as pd

class BaseRunner(ABC):
    """Abstract base class for benchmark orchestration."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    @abstractmethod
    def run(self, dataset: Any, limit: int = None) -> Dict[str, Any]:
        """Execute the benchmark against the provided dataset."""
        pass

    def save_results(self, filename_prefix: str, aggregate_stats: Dict[str, Any]):
        """Save results to JSON and CSV formats."""
        base_path = self.output_dir / filename_prefix
        
        # Save detailed JSON
        payload = {
            "aggregate_stats": aggregate_stats,
            "results": self.results
        }
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        # Save flat CSV if results exist
        if self.results:
            # Flatten metrics into top-level for CSV
            flat_results = []
            for r in self.results:
                flat_r = r.copy()
                metrics = flat_r.pop("metrics", {})
                for k, v in metrics.items():
                    flat_r[f"metric_{k}"] = v
                flat_results.append(flat_r)
                
            df = pd.DataFrame(flat_results)
            df.to_csv(f"{base_path}.csv", index=False)
            
        print(f"Results saved to {base_path}.json and .csv")
