import torch
from typing import List, Tuple

def get_pooling_selection_matrices(degrees: torch.Tensor, pool_ratio: float, depth: int) -> Tuple[List[torch.Tensor], List[int]]:

    all_selection_matrices = []
    num_pooled_nodes_list = []

    selection_matrix = torch.eye(degrees.shape[0], dtype=torch.float32)
    orig_degrees = degrees.clone()
    orig_N = degrees.shape[0]
    N = orig_N
    
    for _ in range(depth): 

        # Create a pool of nodes based on the degree
        num_pooled_nodes = int(N * pool_ratio)
        _, indices = torch.topk(degrees, num_pooled_nodes,
                                        largest = True, sorted = False)
    
        print(f"Selected node indices (top {num_pooled_nodes} by degree): {indices.tolist()}")

        sorted_new_indices = torch.sort(indices).values
        degrees = degrees[sorted_new_indices]
        print(f"Sorted selected node indices: {sorted_new_indices.tolist()}")

        # Create a matrix that selects the pooled nodes among the original nodes
        new_selection_matrix = torch.zeros((num_pooled_nodes, N))
        for i, idx in enumerate(sorted_new_indices):
            new_selection_matrix[i, idx] = 1.0

        print("Selections: ", new_selection_matrix @ torch.arange(N, dtype = torch.float32).view(-1,))
        print("Expected: ", sorted_new_indices.float())
        # print("Original degrees of selected nodes: ", orig_degrees[sorted_new_indices].float())
        assert torch.allclose(new_selection_matrix @ torch.arange(N, dtype=torch.float32).view(-1,),
                            sorted_new_indices.float()), "Selection matrix is incorrect."
        

        selection_matrix = new_selection_matrix @ selection_matrix # cumulative selections
        print("Cumulative selections: ", selection_matrix @ torch.arange(orig_N, dtype=torch.float32).view(-1,))
        # print("Cumulative expected: ", sorted_indices[sorted_new_indices].float())

        # print("Original degrees of cumulatively selected nodes: ", orig_degrees[sorted_indices[sorted_new_indices]].float())
    

        print("\n*****************************************************\n"
            + f"Repeating the pooling to verify consistency...\n" + 
            "*****************************************************\n")

        all_selection_matrices.append(new_selection_matrix)
        num_pooled_nodes_list.append(num_pooled_nodes)
        N = num_pooled_nodes


    return all_selection_matrices, num_pooled_nodes_list