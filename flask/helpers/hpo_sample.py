# create class hpo_network
import random

from hpo import process_nodes, process_edges, read_hpo_from_json
class Network:

    def __init__(self):
        graph_id, meta, nodes, edges, property_chain_axioms = read_hpo_from_json()
        self.nodes = process_nodes(nodes)
        self.edges = process_edges(edges)

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

    def sample_from_proper_ancestors(self, hpo_id, sample_size = 1) -> list:

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

        # precise, imporice, noisy are the number of samples for each category
        # choose which hpo_ids belong to precise, imprecise, and noisy (unique)

        precise_samples = random.sample(list_hpo_ids, precise)
        # ????
        list_hpo_ids = list(set(list_hpo_ids) - set(precise_samples))

        imprecise_samples = random.sample(list_hpo_ids, imprecise)
        list_hpo_ids = list(set(list_hpo_ids) - set(imprecise_samples))

        samples = precise_samples

        for imprecise_hpo in imprecise_samples:
            samples.extend(self.sample_from_proper_ancestors(imprecise_hpo, 1))

        samples.extend(self.sample_noisy(list_hpo_ids, noisy))

        return samples

    def sample_patient_phenotype_v2(self, list_hpo_ids, precise, imprecise, noisy) -> list:

        # precise, imporice, noisy are the number of samples for each category
        # choose which hpo_ids belong to precise, imprecise, and noisy (unique)

        precise_samples = random.sample(list_hpo_ids, precise)
        # ????
        #list_hpo_ids = list(set(list_hpo_ids) - set(precise_samples))

        imprecise_samples = random.sample(list_hpo_ids, imprecise)
        list_hpo_ids = list(set(list_hpo_ids) - set(imprecise_samples))

        samples = precise_samples

        for imprecise_hpo in imprecise_samples:
            samples.extend(self.sample_from_proper_ancestors(imprecise_hpo, 1))

        samples.extend(self.sample_noisy(list_hpo_ids, noisy))

        return samples




# initialize the network
#network = Network()

# sample from the network
#print( network.sample_from_proper_ancestors(4322, 5))

#print( network.sample_patient_phenotype([4322, 1234, 5678, 91011], 2, 2, 2))