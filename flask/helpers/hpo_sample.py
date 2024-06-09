# create class hpo_network
import random
from helpers.hpo import process_nodes, process_edges, read_hpo_from_json


class Network:

    def __init__(self):
        graph_id, meta, nodes, edges, property_chain_axioms = read_hpo_from_json()
        self.nodes = process_nodes(nodes)
        self.edges = process_edges(edges)

        # parents dictionary
        # structure: edge = [sub, obj] where sub is the child and obj is the parent
        self.my_parents = {}
        for edge in self.edges:
            parent = edge[1]
            child = edge[0]
            if child in self.my_parents:
                self.my_parents[child].append(parent)
            else:
                self.my_parents[child] = [parent]

    def get_all_ancestors(self, hpo_id):
        """
        :param hpo_id: HPO id
        :return: List of HPO ids that are ancestors of the given hpo_id
        """
        # find ancestors of hpo_id
        ancestors = []
        last_ancestors = [hpo_id]
        while last_ancestors:
            ancestors.extend(last_ancestors)
            next_ancestors = []
            for ancestor in last_ancestors:
                for edge in self.edges:
                    if edge[0] == ancestor:
                        next_ancestors.append(edge[1])
            last_ancestors = next_ancestors

        # remove duplicates without changing the order
        ancestors = list(dict.fromkeys(ancestors))

        return ancestors

    def sample_from_proper_ancestors(self, hpo_id, sample_size=1) -> list:
        """
        Sample from the ancestors of the given hpo_id, excluding the hpo_id itself
        :param hpo_id:
        :param sample_size:
        :return: the list of sampled HPO ids
        """

        ancestors = self.get_all_ancestors(hpo_id)
        ancestors.remove(hpo_id)
        # get random sample of ancestors
        sample = random.sample(ancestors, sample_size)
        return sample

    def sample_noisy(self, hpo_ids, sample_size) -> list:
        """
        :param hpo_ids: HPO id list
        :param sample_size: number of samples
        :return: List of sampled HPO ids that are NOT ancestors of the given hpo_ids
        """

        # get all HPO ids that are not ancestors of the given hpo_ids
        all_hpo_ids = [node["id"] for node in self.nodes]

        all_ancestors = set()
        for hpo_id in hpo_ids:
            all_ancestors.update(self.get_all_ancestors(hpo_id))

        non_ancestors = list(set(all_hpo_ids) - all_ancestors)

        # get random sample of non-ancestors
        sample = random.sample(non_ancestors, sample_size)

        return sample

    def sample_patient_phenotype(self, list_hpo_ids, precise, imprecise, noisy) -> list:
        """
        This is an alternative way to do the sampling. It is not currently used.
        The difference is that precise and imprecise samples cannot intersect.
        :param list_hpo_ids:
        :param precise:
        :param imprecise:
        :param noisy:
        :return: list of sampled HPO ids
        """

        # precise, imprecise, noisy are the number of samples for each category
        # choose which hpo_ids belong to precise, imprecise, and noisy (unique)

        # assert precise + imprecise <= len(list_hpo_ids)
        assert precise + imprecise <= len(list_hpo_ids)

        precise_samples = random.sample(list_hpo_ids, precise)
        list_hpo_ids = list(set(list_hpo_ids) - set(precise_samples))

        imprecise_samples = random.sample(list_hpo_ids, imprecise)
        list_hpo_ids = list(set(list_hpo_ids) - set(imprecise_samples))

        samples = precise_samples

        for imprecise_hpo in imprecise_samples:
            samples.extend(self.sample_from_proper_ancestors(imprecise_hpo, 1))

        samples.extend(self.sample_noisy(list_hpo_ids, noisy))

        return samples

    def sample_patient_phenotype_v2(self, list_hpo_ids, precise, imprecise, noisy) -> list:
        """
        This is the latest version of the sampling function
        :param list_hpo_ids:
        :param precise:
        :param imprecise:
        :param noisy:
        :return: list of sampled HPO ids
        """

        # precise, imprecise, noisy are the number of samples for each category
        # choose which hpo_ids belong to precise, imprecise, and noisy (unique)
        precise = min(precise, len(list_hpo_ids))
        imprecise = min(imprecise, len(list_hpo_ids))

        precise_samples = random.sample(list_hpo_ids, precise)
        imprecise_samples = random.sample(list_hpo_ids, imprecise)

        samples = precise_samples
        for imprecise_hpo in imprecise_samples:
            samples.extend(self.sample_from_proper_ancestors(imprecise_hpo, 1))

        samples.extend(self.sample_noisy(list_hpo_ids, noisy))
        return samples

    def sample_from_random_strategy(self, list_hpo_ids, strategies) -> list:

        # strategies tuples of 3 elements: (precise, imprecise, noisy),

        # random strategy
        strategy = random.choice(strategies)
        return self.sample_patient_phenotype_v2(list_hpo_ids, strategy[0], strategy[1], strategy[2])

    # returns a list of hpo_ids that are ancestors of the given hpo_ids that are at most max_ancestral_depth away
    def get_imprecision_pool(self, hpo_ids, max_ancestral_depth) -> list:

        pool = set()
        for hpo_id in hpo_ids:

            this_level_ancestors = [hpo_id]
            for i in range(max_ancestral_depth):
                next_level_ancestors = []
                for ancestor in this_level_ancestors:
                    if ancestor in self.my_parents:
                        next_level_ancestors.extend(self.my_parents[ancestor])

                pool.update(next_level_ancestors)
                this_level_ancestors = next_level_ancestors

                if not this_level_ancestors:
                    break

        return list(pool)




# initialize the network
#network = Network()

# sample from the network
#print( network.sample_from_proper_ancestors(4322, 5))

#print( network.sample_patient_phenotype([4322, 1234, 5678, 91011], 2, 2, 2))