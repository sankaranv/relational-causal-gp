import numpy as np
import json
import pandas as pd

class CovidData:

    def __init__(self) -> None:
        self.num_states = 3
        self.num_businesses_per_town = 3
        self.num_towns_per_state = 3

    def write_data_to_json(self, data_dir = 'data/'):
        dataset = {'state': self.states, 'town': self.towns, 'business': self.businesses}
        with open(f"{data_dir}covid_data.json", 'w') as f:
            json.dump(dataset, f)
        with open(f"{data_dir}covid_skeleton.json", 'w') as f:
            json.dump(self.relational_skeleton, f)

    def reset(self):
        self.states = {'policy': []}
        self.towns = {'policy': [], 'prevalence': []}
        self.businesses = {'occupancy': []}
        self.relational_skeleton = {}

    def relational_skeleton_to_adj_matrix(self):
        
        # Collect names of all entities
        self.state_names = list(self.relational_skeleton.keys())
        self.town_names = [town for _, state in self.relational_skeleton.items() for town in state.keys()]
        self.business_names = [business for _, state in self.relational_skeleton.items() for _, town in state.items() for business in town]
        self.entity_names = self.state_names + self.town_names + self.business_names
        
        # Extract adjacency matrices
        self.adj_matrices = {}

        self.adj_matrices['contains'] = pd.DataFrame(np.zeros((len(self.state_names), len(self.town_names))))
        self.adj_matrices['contains']['states'] = self.state_names
        self.adj_matrices['contains'].set_index('states', inplace = True)
        self.adj_matrices['contains'].columns = self.town_names

        self.adj_matrices['resides'] = pd.DataFrame(np.zeros((len(self.town_names), len(self.business_names))))
        self.adj_matrices['resides']['towns'] = self.town_names
        self.adj_matrices['resides'].set_index('towns', inplace = True)
        self.adj_matrices['resides'].columns = self.business_names

        for state, towns in self.relational_skeleton.items():
            for town, businesses in towns.items():
                self.adj_matrices['contains'].loc[state,town] = 1
                for business in businesses:
                    self.adj_matrices['resides'].loc[town, business] = 1
    
    def generate_data(self):
        self.reset()

        for i in range(self.num_states):
            # Sample state policy
            state_name = f"s{i}"
            self.relational_skeleton[state_name] = {}
            state_policy = np.random.normal(0, 0.5)
            self.states['policy'].append(state_policy)

            for j in range(self.num_towns_per_state):

                # Sample town policy per state
                town_name = f"t{i}{j}"
                self.relational_skeleton[state_name][town_name] = []
                town_policy = state_policy + np.random.normal(0, 0.2)
                self.towns['policy'].append(town_policy)

                num_businesses_per_town = np.random.poisson(np.abs(town_policy)) + 1
                for k in range(self.num_businesses_per_town):
                    # Sample business occupancy per town
                    business_name = f"b{i}{j}{k}"
                    self.relational_skeleton[state_name][town_name].append(business_name)
                    business_occupancy = town_policy + np.random.normal(0, 0.5)
                    self.businesses['occupancy'].append(business_occupancy)
                
                # Sample town prevalence per state
                if np.quantile(self.businesses['occupancy'], 0.75) > 0:
                    town_prevalence = state_policy - np.sin(town_policy * 4) + np.random.normal(0, 0.2)
                else:
                    town_prevalence = np.random.normal(-1., 0.2)
                self.towns['prevalence'].append(town_prevalence)         
