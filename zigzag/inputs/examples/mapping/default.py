mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    }
}
