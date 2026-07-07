"""
Unit tests for StocksDataDiffusion cloning functionality.

Tests verify that StocksDataDiffusion objects can be properly cloned
using PyTorch Geometric's Batch.from_data_list() and to_data_list() methods.
This is critical for generating multiple samples per input during evaluation.
"""

import pytest
import torch
from torch_geometric.data import Batch

from graph_signal_diffusion.datasets.sp100.dataset import StocksDataDiffusion as SP100StocksDataDiffusion
from graph_signal_diffusion.datasets.sp500.dataset import StocksDataDiffusion as SP500StocksDataDiffusion


def create_sample_stocks_data(StocksDataDiffusion, num_stocks=10, num_timesteps=5, num_features=3):
    """Create a sample StocksDataDiffusion object for testing."""
    # Create simple graph structure
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.randn(3, 2)  # 3 edges, 2 features per edge
    
    # Create node features [N, T, F]
    x = torch.randn(num_stocks, num_timesteps, num_features)
    
    # Create closing prices [N, T, 1]
    close_price = torch.randn(num_stocks, num_timesteps, 1)
    
    # Create target values [N, T_future, 1]
    y = torch.randn(num_stocks, 1, 1)
    
    # Create future closing prices [N, T_future, 1]
    close_price_y = torch.randn(num_stocks, 1, 1)
    
    # Create timestamps and stock indices
    timestamp = torch.zeros(num_stocks, dtype=torch.long)
    stocks_index = torch.arange(num_stocks)
    
    # Create info dict
    info = {
        "Features": ["feature1", "feature2", "feature3"],
        "Target": "DailyLogReturn",
        "Num_nodes": num_stocks
    }
    
    return StocksDataDiffusion(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        close_price=close_price,
        y=y,
        close_price_y=close_price_y,
        timestamp=timestamp,
        stocks_index=stocks_index,
        info=info
    )


@pytest.mark.parametrize("StocksDataDiffusion", [SP100StocksDataDiffusion, SP500StocksDataDiffusion])
class TestStocksDataCloning:
    """Test cloning functionality for both SP100 and SP500 StocksDataDiffusion classes."""
    
    def test_init_with_no_arguments(self, StocksDataDiffusion):
        """Test that StocksDataDiffusion can be instantiated with no arguments."""
        data = StocksDataDiffusion()
        assert data is not None
        # Verify it's a proper Data object
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
    
    def test_single_graph_to_batch_and_back(self, StocksDataDiffusion):
        """Test converting a single graph to batch and back to data list."""
        # Create a single data object
        data = create_sample_stocks_data(StocksDataDiffusion)
        
        # Create batch from single data object
        batch = Batch.from_data_list([data])
        
        # Convert back to data list - this should not raise TypeError
        data_list = batch.to_data_list()
        
        # Verify we get back one object
        assert len(data_list) == 1
        
        # Verify attributes are preserved
        assert torch.allclose(data_list[0].x, data.x)
        assert torch.equal(data_list[0].edge_index, data.edge_index)
        assert torch.allclose(data_list[0].y, data.y)
    
    def test_multiple_graphs_to_batch_and_back(self, StocksDataDiffusion):
        """Test converting multiple graphs to batch and back to data list."""
        # Create multiple data objects
        data_list = [create_sample_stocks_data(StocksDataDiffusion) for _ in range(3)]
        
        # Create batch
        batch = Batch.from_data_list(data_list)
        
        # Convert back to data list
        recovered_list = batch.to_data_list()
        
        # Verify we get back the same number of objects
        assert len(recovered_list) == 3
        
        # Verify each object's attributes
        for original, recovered in zip(data_list, recovered_list):
            assert torch.allclose(recovered.x, original.x)
            assert torch.equal(recovered.edge_index, original.edge_index)
            assert torch.allclose(recovered.y, original.y)
    
    def test_clone_batch_for_multiple_samples(self, StocksDataDiffusion):
        """Test the cloning logic used in trainer for generating multiple samples per input."""
        # Create a batch of data
        data_list = [create_sample_stocks_data(StocksDataDiffusion) for _ in range(2)]
        batch = Batch.from_data_list(data_list)
        
        # Number of samples to generate per input
        n_samples_per_input = 3
        
        # Simulate the trainer's cloning logic
        # Unbatch, repeat each graph, rebatch
        unbatched_list = batch.to_data_list()
        cloned_list = [g for g in unbatched_list for _ in range(n_samples_per_input)]
        data_cloned = Batch.from_data_list(cloned_list)
        
        # Verify the batch size increased correctly
        assert data_cloned.num_graphs == batch.num_graphs * n_samples_per_input
        assert data_cloned.num_graphs == 2 * 3  # 2 original graphs * 3 samples each
        
        # Verify we can unbatch the cloned batch
        final_list = data_cloned.to_data_list()
        assert len(final_list) == 6
    
    def test_clone_single_graph_for_multiple_samples(self, StocksDataDiffusion):
        """Test cloning a single graph (non-batched) for multiple samples."""
        from torch_geometric.data import Data
        
        # Create a single data object
        data = create_sample_stocks_data(StocksDataDiffusion)
        
        # Number of samples to generate per input
        n_samples_per_input = 5
        
        # Simulate the trainer's cloning logic for single Data
        if isinstance(data, Data) and not isinstance(data, Batch):
            data_cloned = Batch.from_data_list([data] * n_samples_per_input)
        
        # Verify the batch has correct number of graphs
        assert data_cloned.num_graphs == n_samples_per_input
        
        # Verify we can unbatch
        cloned_list = data_cloned.to_data_list()
        assert len(cloned_list) == n_samples_per_input
        
        # Verify all clones have the same data
        for cloned in cloned_list:
            assert torch.allclose(cloned.x, data.x)
            assert torch.equal(cloned.edge_index, data.edge_index)
    
    def test_info_attribute_preserved(self, StocksDataDiffusion):
        """Test that info attribute is preserved during batching and unbatching."""
        data = create_sample_stocks_data(StocksDataDiffusion)
        original_info = data.info.copy()
        
        # Create batch
        batch = Batch.from_data_list([data])
        
        # Unbatch
        recovered = batch.to_data_list()[0]
        
        # Verify info is preserved
        assert hasattr(recovered, 'info')
        assert recovered.info == original_info
    
    def test_cloning_order(self, StocksDataDiffusion):
        """
        Test that cloning maintains correct order: samples grouped by input.
        
        For 3 inputs with 4 samples each, order should be:
        [input1_s1, input1_s2, input1_s3, input1_s4, 
         input2_s1, input2_s2, input2_s3, input2_s4,
         input3_s1, input3_s2, input3_s3, input3_s4]
        
        NOT:
        [input1_s1, input2_s1, input3_s1, input1_s2, ...]
        """
        # Create 3 distinguishable graphs with unique y values
        n_inputs = 3
        n_samples_per_input = 4
        
        data_list = []
        for i in range(n_inputs):
            data = create_sample_stocks_data(StocksDataDiffusion)
            # Set unique identifier in y values for each input
            data.y = torch.full_like(data.y, fill_value=float(i + 1))
            data_list.append(data)
        
        # Create batch
        batch = Batch.from_data_list(data_list)
        
        # Simulate trainer's cloning logic
        unbatched_list = batch.to_data_list()
        cloned_list = [g for g in unbatched_list for _ in range(n_samples_per_input)]
        data_cloned = Batch.from_data_list(cloned_list)
        
        # Unbatch and verify order
        final_list = data_cloned.to_data_list()
        
        # Expected order: [1,1,1,1, 2,2,2,2, 3,3,3,3]
        expected_order = []
        for input_idx in range(1, n_inputs + 1):
            expected_order.extend([input_idx] * n_samples_per_input)
        
        # Verify order by checking y values
        actual_order = [int(g.y[0, 0, 0].item()) for g in final_list]
        
        assert actual_order == expected_order, \
            f"Order mismatch!\nExpected: {expected_order}\nActual:   {actual_order}"
        
        print(f"    Cloning order verified: {actual_order}")
        print(f"    → Samples are grouped by input (correct for evaluation)")


def test_sp500_metadata_fields_batch_and_unbatch():
    """SP500 metadata labels should survive batching for trainer graph-key extraction."""
    data_list = [create_sample_stocks_data(SP500StocksDataDiffusion) for _ in range(3)]
    for data in data_list:
        data.network_id = 0
        data.dataset_name = "sp500"

    batch = Batch.from_data_list(data_list)
    assert hasattr(batch, "network_id")
    assert hasattr(batch, "dataset_name")

    if torch.is_tensor(batch.network_id):
        assert batch.network_id.tolist() == [0, 0, 0]
    else:
        assert list(batch.network_id) == [0, 0, 0]

    if isinstance(batch.dataset_name, list):
        assert batch.dataset_name == ["sp500", "sp500", "sp500"]
    else:
        assert str(batch.dataset_name) == "sp500"

    recovered = batch.to_data_list()
    assert len(recovered) == 3
    for sample in recovered:
        assert sample.network_id == 0
        assert sample.dataset_name == "sp500"


if __name__ == "__main__":
    # Run tests for quick verification
    print("Testing SP100 StocksDataDiffusion...")
    test_suite = TestStocksDataCloning()
    
    for dataset_class, name in [(SP100StocksDataDiffusion, "SP100"), (SP500StocksDataDiffusion, "SP500")]:
        print(f"\n=== Testing {name} ===")
        try:
            test_suite.test_init_with_no_arguments(dataset_class)
            print("✓ test_init_with_no_arguments passed")
            
            test_suite.test_single_graph_to_batch_and_back(dataset_class)
            print("✓ test_single_graph_to_batch_and_back passed")
            
            test_suite.test_multiple_graphs_to_batch_and_back(dataset_class)
            print("✓ test_multiple_graphs_to_batch_and_back passed")
            
            test_suite.test_clone_batch_for_multiple_samples(dataset_class)
            print("✓ test_clone_batch_for_multiple_samples passed")
            
            test_suite.test_clone_single_graph_for_multiple_samples(dataset_class)
            print("✓ test_clone_single_graph_for_multiple_samples passed")
            
            test_suite.test_info_attribute_preserved(dataset_class)
            print("✓ test_info_attribute_preserved passed")
            
            test_suite.test_cloning_order(dataset_class)
            print("✓ test_cloning_order passed")
            
            print(f"\n✅ All tests passed for {name}!")
        except Exception as e:
            print(f"\n❌ Test failed for {name}: {e}")
            raise
