"""Visualization utilities"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from src.core.structure import CrystalStructure


class ResultsVisualizer:
    """Visualize search results and statistics"""
    
    @staticmethod
    def plot_energy_distribution(
        structures: List[CrystalStructure],
        save_path: Optional[str] = None
    ):
        """
        Plot decomposition energy distribution
        
        Args:
            structures: List of structures
            save_path: Optional path to save figure
        """
        energies = [
            s.decomposition_energy
            for s in structures
            if s.decomposition_energy is not None and np.isfinite(s.decomposition_energy)
        ]
        
        if not energies:
            print("No energy data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(energies, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', label='Stability threshold')
        plt.axvline(x=0.1, color='orange', linestyle='--', label='Metastability threshold')
        
        plt.xlabel('Decomposition Energy (eV/atom)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Decomposition Energies', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_convergence(
        history: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot convergence over iterations
        
        Args:
            history: List of iteration statistics
            save_path: Optional path to save figure
        """
        iterations = [h['iteration'] for h in history]
        best_ed = [h['best_ed'] for h in history]
        avg_ed = [h['avg_ed'] for h in history]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Energy convergence
        plt.subplot(1, 2, 1)
        plt.plot(iterations, best_ed, 'o-', label='Best Ed', linewidth=2)
        plt.plot(iterations, avg_ed, 's-', label='Average Ed', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Decomposition Energy (eV/atom)', fontsize=12)
        plt.title('Energy Convergence', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot 2: Success rates
        plt.subplot(1, 2, 2)
        valid_rates = [h.get('valid_rate', 0) for h in history]
        metastable_rates = [h.get('metastable_rate', 0) for h in history]
        
        plt.plot(iterations, valid_rates, 'o-', label='Valid Rate', linewidth=2)
        plt.plot(iterations, metastable_rates, 's-', label='Metastable Rate', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Rate (%)', fontsize=12)
        plt.title('Success Rates', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_pareto_front(
        structures: List[CrystalStructure],
        property_name: str = 'bulk_modulus',
        save_path: Optional[str] = None
    ):
        """
        Plot Pareto front for multi-objective optimization
        
        Args:
            structures: List of structures
            property_name: Property to plot vs stability
            save_path: Optional path to save figure
        """
        eds = []
        properties = []
        
        for s in structures:
            if s.decomposition_energy is not None and property_name in s.properties:
                eds.append(s.decomposition_energy)
                properties.append(s.properties[property_name])
        
        if not eds:
            print("No data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Color by metastability
        colors = ['green' if ed < 0 else 'orange' if ed < 0.1 else 'red' 
                 for ed in eds]
        
        plt.scatter(eds, properties, c=colors, alpha=0.6, s=50)
        
        plt.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Stable')
        plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Metastable')
        
        plt.xlabel('Decomposition Energy (eV/atom)', fontsize=12)
        plt.ylabel(f'{property_name}', fontsize=12)
        plt.title(f'Pareto Front: Stability vs {property_name}', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
