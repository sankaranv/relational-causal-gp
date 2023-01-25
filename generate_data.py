import numpy as np
import json
import pandas as pd
import os

class CovidData:

    def __init__(self):
        self.num_states = 3
        self.num_businesses_per_town = 3
        self.num_towns_per_state = 3

        self.reset()
        self.define_schema()
        self.define_causal_structure()

    def write_data_to_json(self, data_dir = 'data/covid/'):
        
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        # Write dataset
        dataset = {'state': self.states, 'town': self.towns, 'business': self.businesses}
        with open(f"{data_dir}data.json", 'w') as f:
            json.dump(dataset, f, indent = 4)

        # Write relational skeleton
        with open(f"{data_dir}skeleton.json", 'w') as f:
            json.dump(self.relational_skeleton, f, indent = 4)

        # Write relational schema
        schema = {'entities': self.entities, 'relations': self.relations}
        with open(f"{data_dir}schema.json", 'w') as f:
            json.dump(schema, f, indent = 4)

        # Write causal structure
        with open(f"{data_dir}structure.json", 'w') as f:
            json.dump(self.causal_edges, f, indent = 4)

    def read_data_from_json(self, dataset_path = 'data/covid/'):
        
        # Load relational skeleton
        with open(f"{dataset_path}skeleton.json", 'r') as f:
            skeleton = json.load(f)
        self.relational_skeleton = skeleton
        self.collect_instance_names()

        # Load dataset
        with open(f"{dataset_path}data.json", 'r') as f:
            data = json.load(f)
        self.states = data['state']
        self.towns = data['town']
        self.businesses = data['business']

        # Load relational schema and causal structure
        with open(f"{dataset_path}schema.json", 'r') as f:
            schema = json.load(f)
        self.entities = schema['entities']
        self.relations = schema['relations']

        # Load causal structure
        with open(f"{dataset_path}structure.json", 'r') as f:
            structure = json.load(f)  
        self.causal_edges = structure

    def reset(self):
        self.states = {'policy': []}
        self.towns = {'policy': [], 'prevalence': []}
        self.businesses = {'occupancy': []}
        self.relational_skeleton = {}
        self.state_names = []
        self.town_names = []
        self.business_names = []
        self.instance_names = []

    def collect_instance_names(self):
        
        # Collect names of all entities
        self.state_names = list(self.relational_skeleton.keys())
        self.town_names = [town for _, state in self.relational_skeleton.items() for town in state.keys()]
        self.business_names = [business for _, state in self.relational_skeleton.items() for _, town in state.items() for business in town]
        self.instance_names = self.state_names + self.town_names + self.business_names

    def relational_skeleton_to_adj_matrix(self, return_type = 'dataframe'):
        
        # Extract adjacency matrices
        adj_matrices = {}

        adj_matrices['contains'] = pd.DataFrame(np.zeros((len(self.state_names), len(self.town_names))))
        adj_matrices['contains']['state'] = self.state_names
        adj_matrices['contains'].set_index('state', inplace = True)
        adj_matrices['contains'].columns = self.town_names

        adj_matrices['resides'] = pd.DataFrame(np.zeros((len(self.town_names), len(self.business_names))))
        adj_matrices['resides']['town'] = self.town_names
        adj_matrices['resides'].set_index('town', inplace = True)
        adj_matrices['resides'].columns = self.business_names

        for state, towns in self.relational_skeleton.items():
            for town, businesses in towns.items():
                adj_matrices['contains'].loc[state,town] = 1
                for business in businesses:
                    adj_matrices['resides'].loc[town, business] = 1
        
        if return_type == 'dataframe':
            return adj_matrices
        elif return_type == 'numpy':
            for relation, matrix in adj_matrices.items():
                adj_matrices[relation] = matrix.to_numpy()
            return adj_matrices
        else:
            print("Invalid return type")
    
    def define_schema(self):
        self.entities = {
                            'state': ['policy'],
                            'town': ['policy', 'prevalence'],
                            'business': ['occupancy']
                        }
        self.relations = {
                            'contains': {'type': 'many_to_one', 'from': 'state', 'to': 'town'},
                            'resides': {'type': 'many_to_one', 'from': 'town', 'to': 'business'}
                         }
        
    def define_causal_structure(self):
        self.causal_edges = {
                        'contains': [
                                {'from': ('state', 'policy'), 'to': ('town', 'policy')},
                                {'from': ('state', 'policy'), 'to': ('town', 'prevalence')}
                                ],
                        'resides': [
                                {'from': ('town', 'policy'), 'to': ('business', 'occupancy')},
                                {'from': ('business', 'occupancy'), 'to': ('town', 'prevalence')}
                                ],
                        'self': [
                                {'from': ('town', 'policy'), 'to': ('town', 'prevalence')}
                                ]                                
                        }

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
        
        self.collect_instance_names()  

if __name__ == "__main__":
    covid = CovidData()
    covid.generate_data()
    covid.write_data_to_json()
    adj_matrices = covid.relational_skeleton_to_adj_matrix(return_type = 'dataframe')
    print(adj_matrices['contains'])
    print(adj_matrices['resides'])

    

