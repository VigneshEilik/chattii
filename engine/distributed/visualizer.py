from engine.distributed.cluster.topology import ClusterTopology


class DistributedTopologyVisualizer:
    """Renders ASCII layouts of cluster nodes and scaling efficiencies."""

    @staticmethod
    def draw_ascii_topology() -> str:
        topo = ClusterTopology.get_topology()
        node_name = topo["node"]["hostname"]
        ip = topo["node"]["ip_address"]
        gpu_count = topo["gpu_count"]

        grid = [
            "=" * 50,
            f" Distributed Cluster: {node_name} ({ip})",
            "=" * 50,
        ]

        if gpu_count == 0:
            grid.append(" [CPU Only Execution Node]")
        else:
            for idx, dev in enumerate(topo["devices"]):
                grid.append(
                    f"  +-- GPU {idx}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)"
                )

        grid.append("=" * 50)
        return "\n".join(grid)
