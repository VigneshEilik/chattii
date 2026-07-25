class TransformerVisualizer:
    """
    Renders ASCII charts of attention score matrices and causal masks.
    """

    @staticmethod
    def render_attention_mask(seq_len: int = 16) -> str:
        """
        Generates an ASCII visualization grid of causal mask boundaries.
        """
        lines = ["Causal Mask Boundary Mapping ('.' = open, '#' = blocked):"]
        for r in range(seq_len):
            row_str = []
            for c in range(seq_len):
                if c <= r:
                    row_str.append(".")
                else:
                    row_str.append("#")
            lines.append(f"  Row {r:02d}: " + " ".join(row_str))
        return "\n".join(lines)
