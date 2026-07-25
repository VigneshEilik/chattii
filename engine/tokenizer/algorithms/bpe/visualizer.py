from typing import List


class BPEVisualizer:
    """
    Renders training progress metrics as ASCII graphs in the terminal.
    """

    @staticmethod
    def render_chart(
        label: str,
        x_values: List[int],
        y_values: List[float],
        height: int = 10,
        width: int = 40,
    ) -> str:
        """
        Renders a scaled ASCII line graph.
        """
        if not y_values or not x_values or len(x_values) != len(y_values):
            return "No data points to render."

        max_y = max(y_values)
        min_y = min(y_values)
        range_y = max_y - min_y if max_y != min_y else 1.0

        grid = [[" " for _ in range(width)] for _ in range(height)]

        for idx, y in enumerate(y_values):
            x_coord = (
                int(idx * (width - 1) / (len(y_values) - 1)) if len(y_values) > 1 else 0
            )
            y_coord = int((y - min_y) * (height - 1) / range_y)

            # Clamp coordinates
            x_coord = min(max(x_coord, 0), width - 1)
            y_coord = min(max(y_coord, 0), height - 1)

            grid[y_coord][x_coord] = "*"

        lines = [f"\n--- {label} ---"]

        for row_idx in reversed(range(height)):
            val = min_y + (row_idx * range_y / (height - 1))
            val_str = f"{val:8.3f} |" if isinstance(val, float) else f"{int(val):8d} |"
            row_chars = "".join(grid[row_idx])
            lines.append(f"{val_str}{row_chars}")

        # X-axis scale indicators
        lines.append(" " * 9 + "+" + "-" * width)

        spacing = width - 16
        if spacing > 0:
            lines.append(
                " " * 9 + f"{x_values[0]:<8d}" + " " * spacing + f"{x_values[-1]:>8d}"
            )
        else:
            lines.append(" " * 9 + f"{x_values[0]:d} ... {x_values[-1]:d}")

        return "\n".join(lines)
