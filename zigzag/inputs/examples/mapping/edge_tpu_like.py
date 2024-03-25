mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("K", 8),
            "D2": ("C", 8),
            "D3": ("OX", 4),
            "D4": ("OY", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    },
    "Add": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 8),
            "D2": ("C", 1),
            "D3": ("OX", 1),
            "D4": ("OY", 1),
        },
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    },
    "Pooling": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 8),
            "D2": ("C", 1),
            "D3": ("OX", 1),
            "D4": ("OY", 1),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    },
}
