"""
wmin.config.py

Config module of wmin

Author: Mark N. Costantini
Note: several functions are taken from validphys.config
Date: 11.11.2023
"""

class WminConfig():
    """
    WminConfig class
    """

    def parse_wminpdfset(self, name):
        """PDF set used to generate the weight minimization grid"""
        return self.parse_pdf(name)
    
    def produce_wmin_grid_indices(
        self,
        n_replicas,
        use_same_wmin_param_per_replica=False,
        wmin_grid_index_default=1,
    ):
        """
        Produce wmin_grid_indices over which to collect.
        Note: allows for different random wmin parametrisations for each Monte Carlo replica.
        """
        if use_same_wmin_param_per_replica:
            return [
                {"wmin_grid_index": wmin_grid_index_default} for i in range(n_replicas)
            ]
        else:
            return [{"wmin_grid_index": i} for i in range(n_replicas)]

    def produce_all_wmin_collect_indices(
        self,
        n_replicas,
        use_same_trval_split_per_replica=False,
        trval_index_default=1,
        use_same_wmin_param_per_replica=False,
        wmin_grid_index_default=1,
    ):
        """ """

        if use_same_trval_split_per_replica and use_same_wmin_param_per_replica:
            return [
                {
                    "replica_index": i,
                    "trval_index": trval_index_default,
                    "wmin_grid_index": wmin_grid_index_default,
                }
                for i in range(n_replicas)
            ]
        elif use_same_trval_split_per_replica:
            return [
                {
                    "replica_index": i,
                    "trval_index": trval_index_default,
                    "wmin_grid_index": i,
                }
                for i in range(n_replicas)
            ]
        elif use_same_wmin_param_per_replica:
            return [
                {
                    "replica_index": i,
                    "trval_index": i,
                    "wmin_grid_index": wmin_grid_index_default,
                }
                for i in range(n_replicas)
            ]
        else:
            return [
                {"replica_index": i, "trval_index": i, "wmin_grid_index": i}
                for i in range(n_replicas)
            ]
