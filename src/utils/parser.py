"""Utilities for parsing LLM outputs"""

import re
import logging
from typing import List, Optional

from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)


class StructureParser:
    """Parse LLM-generated structure strings"""
    
    @staticmethod
    def extract_poscar_blocks(text: str) -> List[str]:
        """
        Extract POSCAR blocks from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            List of POSCAR strings
        """
        # Pattern 1: Look for typical POSCAR structure
        # Comment line + scaling + 3 lattice vectors + species + counts + direct + positions
        
        poscar_blocks = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            # Skip "Structure N:" header lines
            if lines[i].strip().startswith('Structure ') and ':' in lines[i]:
                i += 1
                continue

            # Skip markdown code block markers (```) and other noise
            line_stripped = lines[i].strip()
            if line_stripped in ['```', '```poscar', '```POSCAR', '```python', ''] or line_stripped.startswith('```'):
                i += 1
                continue

            # Check if this looks like start of POSCAR
            if i + 8 < len(lines):
                # Try to parse as POSCAR
                block_lines = []
                start_i = i  # Save starting position

                # Comment line (chemical formula)
                block_lines.append(lines[i])
                i += 1

                # Check for scaling factor (should be a number)
                try:
                    float(lines[i].strip())
                    block_lines.append(lines[i])
                    i += 1
                except:
                    i = start_i + 1  # Reset and move to next line
                    continue
                
                # Try to get 3 lattice vectors
                lattice_ok = True
                for _ in range(3):
                    parts = lines[i].split()
                    if len(parts) == 3:
                        try:
                            [float(x) for x in parts]
                            block_lines.append(lines[i])
                            i += 1
                        except:
                            lattice_ok = False
                            break
                    else:
                        lattice_ok = False
                        break

                if not lattice_ok:
                    i = start_i + 1  # Reset and move to next line
                    continue
                
                # Species line
                species_line = lines[i]
                block_lines.append(species_line)
                i += 1

                # Counts line - parse to know how many positions to expect
                counts_line = lines[i]
                block_lines.append(counts_line)
                i += 1

                # Determine expected number of position lines
                try:
                    expected_positions = sum(int(x) for x in counts_line.split())
                except:
                    i = start_i + 1  # Reset and move to next line
                    continue

                # Coordinate type
                block_lines.append(lines[i])
                i += 1

                # Positions - collect exactly the expected number
                position_count = 0
                while i < len(lines) and position_count < expected_positions:
                    line = lines[i].strip()
                    if not line:
                        break

                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            [float(x) for x in parts[:3]]
                            block_lines.append(lines[i])
                            position_count += 1
                            i += 1
                        except:
                            break
                    else:
                        break

                # Validate we got complete POSCAR with correct number of positions
                # POSCAR format: 1 (formula) + 1 (scale) + 3 (lattice) + 1 (elements) + 1 (counts) + 1 (coord type) + N (positions) = 8 + N
                if position_count == expected_positions and len(block_lines) >= 8 + expected_positions:
                    poscar_blocks.append('\n'.join(block_lines))
                    logger.debug(f"Extracted POSCAR block with {expected_positions} atoms")
                else:
                    logger.warning(f"Rejected incomplete POSCAR: declared {expected_positions} atoms but only found {position_count} coordinate lines. Formula: {block_lines[0] if block_lines else 'unknown'}")
                    i = start_i + 1  # Move to next line
            else:
                i += 1
        
        return poscar_blocks
    
    @staticmethod
    def parse_structures(text: str) -> List[CrystalStructure]:
        """
        Parse crystal structures from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            List of parsed CrystalStructure objects
        """
        structures = []
        
        # Extract POSCAR blocks
        poscar_blocks = StructureParser.extract_poscar_blocks(text)
        
        logger.info(f"Found {len(poscar_blocks)} POSCAR blocks in response")
        
        # Parse each block
        for i, poscar_str in enumerate(poscar_blocks):
            try:
                struct = CrystalStructure.from_poscar(poscar_str)
                structures.append(struct)
                logger.debug(f"Successfully parsed structure {i+1}: {struct.formula}")
            except Exception as e:
                logger.warning(f"Failed to parse POSCAR block {i+1}: {e}")
        
        return structures
    
    @staticmethod
    def validate_structure_string(poscar_str: str) -> bool:
        """
        Quick validation of POSCAR string format
        
        Args:
            poscar_str: POSCAR string
            
        Returns:
            True if valid format
        """
        try:
            lines = [l.strip() for l in poscar_str.split('\n') if l.strip()]
            
            # Minimum lines needed
            if len(lines) < 9:
                return False
            
            # Line 2 should be scaling factor
            float(lines[1])
            
            # Lines 3-5 should be lattice vectors
            for i in range(2, 5):
                parts = lines[i].split()
                if len(parts) != 3:
                    return False
                [float(x) for x in parts]
            
            # Line 7 should be counts (integers)
            counts = [int(x) for x in lines[6].split()]
            
            # Line 8 should be coordinate type
            coord_type = lines[7].lower()
            if not (coord_type.startswith('d') or coord_type.startswith('c')):
                return False
            
            # Should have positions
            total_atoms = sum(counts)
            if len(lines) < 8 + total_atoms:
                return False
            
            return True
            
        except Exception:
            return False