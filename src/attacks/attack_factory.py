"""
Attack Factory for RAG Attack Framework

This module provides a factory function to instantiate different attack
classes based on string keywords. This allows for easy selection of attack
types without directly importing specific attack classes.

Supported attacks:
- "poison_rag": PoisonedRAG attack implementation
- "corrupt_rag": CorruptRAG attack implementation
"""

from typing import Union
from config.config import AttackConfiguration
from .poisoned_rag_attack import PoisonedRAG
from .corrupt_rag_attack import CorruptRAG


def get_attack_class(
    attack_type: str,
    config: AttackConfiguration
):
    """
    Factory function to create attack instances based on attack type keyword.
    
    Args:
        attack_type: String keyword identifying the attack type
            - "poison_rag": Returns PoisonedRAG attack instance
            - "corrupt_rag": Returns CorruptRAG attack instance
        config: Configuration object for the attack (must match the attack
            type)
        
    Returns:
        Instance of the requested attack class
        
    Raises:
        ValueError: If attack_type is not supported
    """
    attack_map = {
        "poison_rag": PoisonedRAG,
        "corrupt_rag": CorruptRAG
    }

    config_map = {
        "poison_rag": config.poisoned_rag_attack_config,
        "corrupt_rag": config.corrupt_rag_attack_config
    }

    if attack_type not in attack_map:
        available_attacks = ", ".join(attack_map.keys())
        raise ValueError(
            f"Unsupported attack type '{attack_type}'. "
            f"Available attacks: {available_attacks}"
        )

    attack_class = attack_map[attack_type]
    config = config_map[attack_type]
    return attack_class(config)
