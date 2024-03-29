mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("OX", 8), "D3": ("OY", 4)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    },
    "Add": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("OX", 1), "D3": ("OY", 1)},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        "operand_precision": {"O": 16, "O_final": 8, "X": 8, "Y": 8},
    },
    "Pooling": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("OX", 1), "D3": ("OY", 1)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    },
}
