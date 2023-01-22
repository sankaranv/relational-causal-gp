import json
from collections import namedtuple 
import torch
import numpy as np
import pandas as pd
import networkx as nx

RelationalNode = namedtuple('RelationalNode', 'entity attribute')
CausalEdge = namedtuple('CausalEdge', 'parent child')
InstanceNode = namedtuple('InstanceNode', 'entity attribute instance')

class RelationalSchema:
    '''
    Relational Schema
    '''
    def __init__(self) -> None:
        self.empty_schema()

    def empty_schema(self):
        self.entity_classes = [] # each entry is a name
        self.relationship_classes = [] # each entry is a dict with keys left and right
        self.attribute_classes = {} # each key is an entity/relationship class and each value is a set of attribute names
        self.cardinality = {} # each key is a (relationship class, entity class) tuple and value is 'one' or 'many'
        self.relations = {} # each key is a relationship class and value is an (entity class, entity class) tuple
    def load_from_file(self, path_to_json):
        with open(path_to_json, 'r') as f:
            schema_dict = json.load(f)
        self.entity_classes = schema_dict["entity_classes"]
        self.relationship_classes = schema_dict["relationship_classes"]
        self.attribute_classes = schema_dict["attribute_classes"]
        self.cardinality = schema_dict["cardinality"]
        self.relations = schema_dict["relations"]
        for relation in self.relations:
            self.relations[relation] = tuple(self.relations[relation])
        if not self.is_valid_schema():
            print("Schema is invalid, could not load from file")
            self.empty_schema()

    def save_to_file(self, path_to_json):
        if self.is_valid_schema():
            schema_dict = {
                "entity_classes": list(self.entity_classes),
                "relationship_classes": list(self.relationship_classes),
                "attribute_classes": self.attribute_classes,
                "cardinality": self.cardinality,
                "relations": self.relations
            }
            with open(path_to_json, 'w') as f:
                json.dump(schema_dict, f)
        else:
            print("Schema is invalid, could not write to file")

    def is_valid_schema(self):
        for entity in self.entity_classes:
            if entity not in self.attribute_classes or not isinstance(self.attribute_classes[entity], list):
                return False
        for relation in self.relationship_classes:
            if relation not in self.cardinality:
               return False
            else:
                for entity in self.cardinality[relation]:
                    if entity not in self.entity_classes or self.cardinality[relation][entity] not in ['one', 'many']:
                        return False
        return True


class RelationalSkeleton:
    '''
    Relational Skeleton
    '''
    def __init__(self, schema) -> None:
        self.empty_skeleton(schema)

    def empty_skeleton(self, schema):
        self.entity_instances = {}
        self.relationship_instances = {}
        for entity in schema.entity_classes:
            self.entity_instances[entity] = {"names": []}
            for attribute in schema.attribute_classes[entity]:
                self.entity_instances[entity][attribute] = []
        for relation in schema.relationship_classes:
            self.relationship_instances[relation] = []
        self.instance_type = {}

    def get_instance_type(self, instance):
        return self.instance_type[instance]

    def load_from_file(self, schema, path_to_json):
        with open(path_to_json, 'r') as f:
            skeleton_dict = json.load(f)
        self.entity_instances = skeleton_dict["entity_instances"]
        for entity in self.entity_instances:
            # Save entity types for all entities
            for name in self.entity_instances[entity]["names"]:
                self.instance_type[name] = entity
        self.relationship_instances = skeleton_dict["relationship_instances"]
        for relation in self.relationship_instances:
            self.relationship_instances[relation] = [tuple(e) for e in self.relationship_instances[relation]]
        if not self.is_valid_skeleton(schema):
            print("Skeleton is invalid for the given schema, could not load from file")
            self.empty_skeleton(schema)

    def save_to_file(self, schema, path_to_json):
        if self.is_valid_skeleton(schema):
            skeleton_dict = {
                "entity_instances": self.entity_instances,
                "relationship_instances": self.relationship_instances
            }
            with open(path_to_json, 'w') as f:
                json.dump(skeleton_dict, f)
        else:
            print("Skeleton is invalid for the given schema, could not write to file")                

    def is_valid_skeleton(self, schema):
        for entity in schema.entity_classes:
            if entity not in self.entity_instances or not isinstance(self.entity_instances[entity], dict):
                print(f"Entity {entity} in the schema is missing in the skeleton")
                return False
            if "names" not in self.entity_instances[entity]:
                print(f"Names are missing for entity {entity}")
                return False
            for attribute in schema.attribute_classes[entity]:
                if attribute not in self.entity_instances[entity]:
                    print(f"Attribute {entity}.{attribute} in the schema is missing in the skeleton")
                    return False
                if not isinstance(self.entity_instances[entity][attribute], list):
                    print(f"Values of {entity}.{attribute} are not in a list or missing")
                    return False
                if len(self.entity_instances[entity][attribute]) != len(self.entity_instances[entity]["names"]):
                    print(f"Number of values of {entity}.{attribute} are not equal to the number of instance names")
                    return False
        all_instance_names = self.instance_type.keys()
        for relation in schema.relationship_classes:
            if relation not in self.relationship_instances or not isinstance(self.relationship_instances[relation], list):
               return False
            else:
                for item in self.relationship_instances[relation]:
                    if not isinstance(item, tuple) or item[0] not in all_instance_names or item[1] not in all_instance_names:
                        return False
        return True

class RelationalCausalStructure:
    '''
    Relational Causal Structure
    '''
    def __init__(self, schema, edges = None) -> None:
        self.schema = schema
        self.edges = {} if edges is None else edges
        self.nodes = set()
        self.parents = {}

    def load_edges_from_file(self, path_to_json):
        with open(path_to_json, 'r') as f:
            self.edges = json.load(f)
        for relation in self.edges:
            for idx, edge_list in enumerate(self.edges[relation]):
                # Convert list representation of node from JSON to named tuple
                self.edges[relation][idx] = [RelationalNode(*node) for node in self.edges[relation][idx]]
                # Add nodes to node list
                self.nodes.update(self.edges[relation][idx])   
                # Convert list representation of edge to named tuple
                self.edges[relation][idx] = CausalEdge(*self.edges[relation][idx])
                # Update parents dict
                edge = self.edges[relation][idx]
                if edge.child not in self.parents:
                    self.parents[edge.child] = [edge.parent]
                else:
                    self.parents[edge.child].append(edge.parent)
                if edge.parent not in self.parents:
                    self.parents[edge.parent] = []

    def save_edges_to_file(self, path_to_json):
        with open(path_to_json, 'w') as f:
            json.dump(self.edges, f)

    def get_incoming_edges(self):
        incoming_edges = {}
        for entity, attributes in self.schema.attribute_classes.items():
            attribute_dict = {}
            for attribute in attributes:
                attribute_dict[attribute] = []
            incoming_edges[entity] = attribute_dict
        
        for relation, edge_list in self.edges.items():
            for edge in edge_list:
                incoming_edges[edge.child.entity][edge.child.attribute].append((relation, edge))
        return incoming_edges        

class RelationalSCM:
    def __init__(self, structure: RelationalCausalStructure) -> None:
        self.structure = structure
        self.functions = {}
        for node in self.structure.nodes:
            self.functions[node] = None


def create_adj_mat_dict(structure: RelationalCausalStructure, skeleton: RelationalSkeleton) -> dict:
    '''
    Creates adjacency matrices based on the relational skeleton
    '''
    adj_mat_dict = {}
    for relation_name, entity_edge in structure.schema.relations.items():
        num_entity1 = len(skeleton.entity_instances[entity_edge[0]]["names"])
        num_entity2 = len(skeleton.entity_instances[entity_edge[1]]["names"])
        adj_mat = pd.DataFrame([[False] * num_entity2] * num_entity1)
        adj_mat.index = skeleton.entity_instances[entity_edge[0]]["names"]
        adj_mat.columns = skeleton.entity_instances[entity_edge[1]]["names"]
        for instance_edge in skeleton.relationship_instances[relation_name]:
            adj_mat.loc[instance_edge[0], instance_edge[1]] = True
        adj_mat_dict[relation_name] = adj_mat
    return adj_mat_dict

# Naming convention for nodes in ground graph is instance.attribute
def get_name(instance, attribute):
    return '.'.join([instance, attribute])

def create_ground_graph(structure: RelationalCausalStructure, skeleton: RelationalSkeleton) -> dict:
    '''
    Creates an abstract ground graph
    '''

    # Set up nodes in ground graph and save attribute values in each node
    # There will be one node for each (entity instance, attribute name) pair
    ground_graph = nx.DiGraph()
    for entity in skeleton.entity_instances:
        attributes = structure.schema.attribute_classes[entity]  
        for idx, instance_name in enumerate(skeleton.entity_instances[entity]["names"]):
            for attribute_name in attributes:
                node_name = get_name(instance_name,attribute_name)
                attribute_value = skeleton.entity_instances[entity][attribute_name][idx]
                ground_graph.add_node(node_name, val = attribute_value)

    # Set up self edges
    if "self" in structure.edges:
        edge_list = structure.edges["self"]
        for self_edge in edge_list:
            if self_edge.parent.entity != self_edge.child.entity:
                print("Edge is marked as a self-edge in skeleton but is between different entities")
                break
            else:
                for instance_name in skeleton.entity_instances[self_edge.parent.entity]["names"]:
                   parent_node_name = get_name(instance_name,self_edge.parent.attribute)
                   child_node_name = get_name(instance_name,self_edge.child.attribute)
                   ground_graph.add_edge(parent_node_name, child_node_name) 

    # Set up all other edges
    for relation_type, edge_list in skeleton.relationship_instances.items():
        for instance_edge in edge_list:
            # Add all edges in ground graph corresponding to each edge in the relational skeleton
            entity_0 = skeleton.get_instance_type(instance_edge[0])
            entity_1 = skeleton.get_instance_type(instance_edge[1])
            # Add edges between entities
            for relational_edge in structure.edges[relation_type]:
                if relational_edge.parent.entity == entity_0 and relational_edge.child.entity == entity_1:
                    parent_node_name = get_name(instance_edge[0],relational_edge.parent.attribute)
                    child_node_name = get_name(instance_edge[1],relational_edge.child.attribute)
                    ground_graph.add_edge(parent_node_name, child_node_name)
                # Don't forget to consider the opposite direction, relational edges are not necessarily directed
                if relational_edge.parent.entity == entity_1 and relational_edge.child.entity == entity_0:
                    parent_node_name = get_name(instance_edge[1],relational_edge.parent.attribute)
                    child_node_name = get_name(instance_edge[0],relational_edge.child.attribute)
                    ground_graph.add_edge(parent_node_name, child_node_name)

    return ground_graph

def create_subgraph_for_ITE(ground_graph: nx.DiGraph, treatment: InstanceNode, outcome: InstanceNode, cutoff = 10):
    source = get_name(treatment.instance, treatment.attribute)
    target = get_name(outcome.instance, outcome.attribute)
    subgraph = nx.DiGraph()
    if nx.has_path(ground_graph, source, target):
        for path in nx.all_simple_edge_paths(ground_graph, source, target, cutoff):
            for edge in path:
                subgraph.add_edge(*edge)
    else:
        print(f"No directed path from {source} to {target}")
    return subgraph

if __name__ == "__main__":

    # Create relational schema
    schema = RelationalSchema()
    schema.load_from_file('example/covid_schema.json')

    # Create relational skeleton
    skeleton = RelationalSkeleton(schema)
    skeleton.load_from_file(schema, 'example/covid_skeleton.json')

    # Create relational structure
    structure = RelationalCausalStructure(schema)
    structure.load_edges_from_file('example/covid_structure.json')

    # Create adjacency matrices and ground graph
    adj_mat_dict = create_adj_mat_dict(structure, skeleton)
    ground_graph = create_ground_graph(structure, skeleton)

    # Get subgraphs
    treatment = InstanceNode("state", "policy", "s1")
    outcome = InstanceNode("town", "prevalence", "t2")
    subgraph = create_subgraph_for_ITE(ground_graph, treatment, outcome)
    print(subgraph.edges)